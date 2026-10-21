"""Train, construct fixed references, and evaluate ARC-VAD."""
import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clip import clip
from data import FeatureDataset, label_vector
from model import ARCVAD
from objectives import training_loss
from evaluate import evaluate


def load_config(path):
    return json.loads(Path(path).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def task_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('clipmodel.')}


def load_checkpoint(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith('clipmodel.') for k in result.missing_keys):
        raise ValueError(f'Incompatible checkpoint: {result}')
    return checkpoint


def train(model, config, args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    training = config['training']
    if args.stage == 2 and not args.checkpoint:
        raise ValueError('Stage II requires --checkpoint from Stage I')
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint)
    model.set_stage(args.stage)
    if model.use_vlsr:
        if not args.references: raise ValueError('Stage II requires --references')
        model.load_references(args.references)
    if config['dataset'] == 'ucf':
        loaders = [DataLoader(FeatureDataset(config,'train',normal),batch_size=training['batch_size'],shuffle=True,drop_last=True) for normal in [True,False]]
    else:
        loaders = [DataLoader(FeatureDataset(config,'train'),batch_size=training['batch_size'],shuffle=True)]
    if any(len(loader)==0 for loader in loaders): raise ValueError('No training batches; check manifest and batch size')
    testset = FeatureDataset(config, 'test')
    scale = training['stage2_lr_scale'] if args.stage == 2 else 1.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=training['lr']*scale,weight_decay=training['weight_decay'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,training['milestones'],training['lr_decay'])
    (out/'config.json').write_text(json.dumps(config,indent=2))
    best = -float('inf')
    start = time.perf_counter()
    for epoch in range(training[f'stage{args.stage}_epochs']):
        model.train()
        for batches in zip(*loaders):
            visual = torch.cat([batch[0] for batch in batches]).to(args.device)
            targets = torch.cat([batch[1] for batch in batches]).to(args.device)
            lengths = torch.cat([batch[2] for batch in batches])
            text,binary,categories = model(visual,None,config['prompts'],lengths)
            loss = training_loss(text,binary,categories,targets,lengths,training['text_loss_weight'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        metrics = evaluate(model,testset,config,args.device)
        metrics.update(epoch=epoch+1,stage=args.stage,seed=config['seed'])
        with (out/'metrics.jsonl').open('a') as stream: stream.write(json.dumps(metrics)+'\n')
        print(json.dumps(metrics),flush=True)
        if metrics[training['selection']] > best:
            best = metrics[training['selection']]
            torch.save(dict(model_state_dict=task_state(model),epoch=epoch+1,stage=args.stage,config=config,metrics=metrics),out/'best.pt')
        if training['rollback_to_best']:
            # Native model-only rollback: optimizer/scheduler are not reset.
            load_checkpoint(model,out/'best.pt')
    (out/'timing.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,includes='training, evaluation and checkpoint writing'),indent=2))


@torch.no_grad()
def build_references(model, config, args):
    if not args.checkpoint: raise ValueError('Reference construction requires a Stage-I checkpoint')
    load_checkpoint(model,args.checkpoint)
    model.set_stage(1)
    model.eval()
    dataset = FeatureDataset(config,'train',False)
    groups = {}
    for i in range(len(dataset)):
        visual,target,length = dataset[i]
        visual = visual[None].to(args.device)
        x = model.encode_video(visual,None,torch.tensor([length]))
        score = model.classifier(x+model.mlp2(x)).squeeze(-1)[0,:length]
        indices = score.topk(max(1,length//16)).indices
        vector = F.normalize(x[0,indices].mean(0),dim=-1).cpu()
        row = dataset.table.iloc[i]
        video = str(row['video_id']) if 'video_id' in row else re.sub(r'__\d+$','',Path(row['path']).stem)
        groups.setdefault(video,[]).append((vector,target))
    vectors,labels = [],[]
    for video,items in sorted(groups.items()):
        if len(items) != 10: raise ValueError(f'{video}: expected ten crop rows, found {len(items)}; supply video_id in manifest')
        if not all(torch.equal(items[0][1],t) for _,t in items): raise ValueError(f'Inconsistent crop labels: {video}')
        vector = F.normalize(torch.stack([v for v,_ in items]).mean(0),dim=0)
        for cls in items[0][1].nonzero().flatten().tolist():
            if cls == 0: raise ValueError('Normal videos are excluded from retrieval gallery')
            vectors.append(vector);labels.append(cls)
    tokens = clip.tokenize(config['descriptions']).to(args.device)
    descriptions = F.normalize(model.clipmodel.encode_text(model.clipmodel.encode_token(tokens),tokens).float(),dim=-1).cpu()
    bundle = dict(gallery=torch.stack(vectors),labels=torch.tensor(labels),descriptions=descriptions,
                  classes=config['classes'],source_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True);torch.save(bundle,output)
    print(f'Saved {len(vectors)} category references from {len(groups)} training videos.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','build-references','evaluate'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',type=int,choices=[1,2],default=1)
    parser.add_argument('--checkpoint')
    parser.add_argument('--references')
    parser.add_argument('--output',default='runs/default')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--clip-checkpoint',default='ViT-B/16')
    args=parser.parse_args();config=load_config(args.config)
    if args.seed is not None:config['seed']=args.seed
    seed_everything(config['seed'])
    model=ARCVAD(config,args.device,args.clip_checkpoint).to(args.device)
    if args.command=='train':train(model,config,args)
    elif args.command=='build-references':build_references(model,config,args)
    else:
        if not args.checkpoint:raise ValueError('Evaluation requires --checkpoint')
        load_checkpoint(model,args.checkpoint);model.set_stage(args.stage)
        if model.use_vlsr:
            if not args.references:raise ValueError('Stage-II evaluation requires --references')
            model.load_references(args.references)
        print(json.dumps(evaluate(model,FeatureDataset(config,'test'),config,args.device),indent=2))


if __name__=='__main__':main()

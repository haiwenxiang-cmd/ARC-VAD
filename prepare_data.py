"""Prepare portable official VadCLIP lists and annotations (features downloaded separately)."""
import argparse
import csv
import io
from pathlib import Path
import urllib.request

BASE = 'https://raw.githubusercontent.com/nwpu-zxr/VadCLIP/c41067f07d252efcda18008bea367886070c33b0/list/'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=['xd','ucf'],required=True)
    parser.add_argument('--train-root',required=True,help='Local directory containing training crop .npy files')
    parser.add_argument('--test-root',required=True,help='Local directory containing test .npy files')
    parser.add_argument('--output',default='data')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    for split,suffix,root in [('train','',args.train_root),('test','test',args.test_root)]:
        name=f'{args.dataset}_CLIP_rgb{suffix}.csv'
        text=urllib.request.urlopen(BASE+name,timeout=60).read().decode('utf-8')
        rows=list(csv.DictReader(io.StringIO(text)))
        missing=[]
        for row in rows:
            row['path']=str(Path(root).resolve()/Path(row['path'].replace('\\','/')).name)
            if not Path(row['path']).is_file():missing.append(row['path'])
        if missing:raise FileNotFoundError(f'{len(missing)} missing features; first: {missing[0]}')
        target=out/f'{args.dataset}_{split}.csv'
        with target.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['path','label']);writer.writeheader()
            writer.writerows({k:r[k] for k in ['path','label']} for r in rows)
        print(target,len(rows))
    suffix='_ucf' if args.dataset=='ucf' else ''
    for stem in ['gt','gt_segment','gt_label']:
        name=stem+suffix+'.npy'
        with urllib.request.urlopen(BASE+name,timeout=120) as response, (out/name).open('wb') as stream:
            while True:
                chunk=response.read(1024*1024)
                if not chunk:break
                stream.write(chunk)
        print(out/name)


if __name__=='__main__':main()

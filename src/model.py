"""ARC-VAD: shared-interface calibration and local support refinement."""
import torch
import torch.nn.functional as F
from backbone import CLIPVAD
from refinement import variation, semantic, combine


class ARCVAD(CLIPVAD):
    def __init__(self, config, device='cuda', clip_checkpoint='ViT-B/16'):
        cfg = config['model']
        super().__init__(len(config['classes']), 512, 256, 512, 1, cfg['layers'], cfg['window'], 10, 10, device, clip_checkpoint)
        self.alpha = torch.nn.Parameter(torch.tensor(cfg['alpha']), requires_grad=False)
        self.agrc = cfg['agrc']
        self.use_vlsr = False
        self.refinement_config = cfg
        self.gallery = None
        self.descriptions = None

    def encode_video(self, visual, padding_mask, lengths):
        xm = super().encode_video(visual, padding_mask, lengths)
        if not self.agrc:
            return xm
        positions = torch.arange(self.visual_length, device=visual.device)[None].expand(visual.shape[0], -1)
        anchor = (visual.float().permute(1, 0, 2) + self.frame_position_embeddings(positions).permute(1, 0, 2)).permute(1, 0, 2)
        return anchor + self.alpha * xm

    def load_references(self, path):
        bundle = torch.load(path, map_location=self.device, weights_only=False)
        self.gallery = {k: bundle[k] for k in ['gallery', 'labels']}
        self.descriptions = bundle['descriptions']
        if self.descriptions.shape != (self.num_class, 512):
            raise ValueError('Reference categories do not match model')

    def set_stage(self, stage):
        self.use_vlsr = stage == 2 and self.refinement_config['vlsr']
        predictors = ('classifier.', 'mlp1.', 'mlp2.', 'text_prompt_embeddings.')
        for name, param in self.named_parameters():
            param.requires_grad = not name.startswith('clipmodel.') and name != 'alpha' and (stage == 1 or name.startswith(predictors))

    def forward(self, visual, padding_mask, text, lengths, *, single_video=False):
        x = self.encode_video(visual, padding_mask, lengths)
        text_original = self.encode_textprompt(text)
        if self.use_vlsr:
            if self.gallery is None:
                raise RuntimeError('Load the training-video references before Stage II')
            cfg = self.refinement_config
            varied = variation(x, cfg['groups'], cfg['group_width'], cfg['gate_temperature'], cfg['beta'])
            valid = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
            raw = visual[valid][None] if single_video else visual

            def reference(features, lens):
                mask = torch.arange(features.shape[1], device=features.device)[None] < lens[:, None]
                score = self.classifier(features + self.mlp2(features)).squeeze(-1)
                context = F.normalize((score * mask)[:, None, :] @ features, dim=-1)
                category_text = text_original.detach()[None] + context
                category_text = F.normalize(category_text + self.mlp1(category_text), dim=-1)
                matches = F.normalize(raw.float(), dim=-1) @ self.descriptions.T
                return score, category_text, matches, self.gallery

            if single_video:
                semantic_features = x.clone()
                n = int(valid.sum())
                if n:
                    semantic_features[valid] = semantic(x[valid][None], lengths.new_tensor([n]), reference, cfg['beta'], cfg['groups'])[0]
            else:
                semantic_features = semantic(x, lengths, reference, cfg['beta'], cfg['groups'])
            x = combine(x, varied, semantic_features, visual, lengths, cfg['beta'])
        binary_logits = self.classifier(x + self.mlp2(x))
        context = binary_logits.permute(0, 2, 1) @ x
        context = context / context.norm(dim=-1, keepdim=True)
        context = context.expand(x.shape[0], text_original.shape[0], x.shape[2])
        category_text = text_original[None].expand(x.shape[0], -1, -1) + context
        category_text = category_text + self.mlp1(category_text)
        xn = x / x.norm(dim=-1, keepdim=True)
        tn = category_text / category_text.norm(dim=-1, keepdim=True)
        category_logits = xn @ tn.permute(0, 2, 1).type(xn.dtype) / .07
        return text_original, binary_logits, category_logits

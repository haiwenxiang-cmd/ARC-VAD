"""Parameter-free variation and semantic refinement at the shared visual interface."""
import torch
import torch.nn.functional as F


def variation(x, k=12, width=32, temperature=1., beta=1.):
    b, t, d = x.shape
    groups = x.reshape(b, t, d // width, width)
    motion = (groups[:, 1:] - groups[:, :-1]).norm(dim=-1)
    if t == 1:
        motion = x.new_zeros(b, 1, d // width)
    indices = motion.mean(1).topk(k, dim=-1).indices
    expanded = torch.cat([motion, motion[:, -1:]], 1) if t > 1 else motion
    weights = (expanded.gather(2, indices[:, None].expand(-1, t, -1)) / temperature).softmax(-1) * k
    gate = x.new_zeros(b, t, d // width).scatter(2, indices[:, None].expand(-1, t, -1), weights)
    return x + beta * (groups * gate[..., None]).reshape_as(x)


def semantic(x, lengths, reference, beta=1., k_groups=12):
    b, t, d = x.shape
    lens = lengths.to(x.device).long().clamp(0, t)
    mask = torch.arange(t, device=x.device)[None] < lens[:, None]
    with torch.no_grad():
        score, text, teacher, bank = reference(x, lens)
        k = (lens // 16).clamp_min(1)
        km = max(1, t // 16)
        select = torch.arange(km, device=x.device)[None] < k[:, None]
        ranked_indices = score.masked_fill(~mask, -torch.inf).topk(km, dim=1).indices
        chosen = x.gather(1, ranked_indices[..., None].expand(-1, -1, d))
        query = F.normalize((chosen * select[..., None]).sum(1) / k[:, None], dim=-1)
        similarities = query @ bank['gallery'].T / .07
        prior = []
        for cls in range(1, text.shape[1]):
            values = similarities[:, bank['labels'] == cls]
            if values.shape[-1] == 0:
                raise ValueError(f'Gallery has no references for category {cls}')
            prior.append(values.topk(min(3, values.shape[-1]), dim=-1).values.mean(-1))
        prior = torch.stack(prior, -1).softmax(-1)
        prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
        ranked = teacher[:, :, 1:].masked_fill(~mask[..., None], -torch.inf).topk(km, dim=1).values
        video_text = torch.where(select[..., None], ranked, 0.).sum(1) / k[:, None]
        prior = prior * (video_text / .07).softmax(-1)
        prior = prior / prior.sum(-1, keepdim=True).clamp_min(1e-8)
        positive = F.normalize((text[:, 1:] * prior[..., None]).sum(1), dim=-1)
        normal = text[:, 0]
        anomaly = score.sigmoid() * mask
        denominator = F.avg_pool1d(mask[:, None].float(), 17, stride=1, padding=8).squeeze(1)
        anomaly = F.avg_pool1d(anomaly[:, None], 17, stride=1, padding=8).squeeze(1)
        anomaly = anomaly / denominator.clamp_min(1. / 17)
        groups = x.reshape(b, t, 16, 32)
        differences = (groups[:, 1:] - groups[:, :-1]).norm(dim=-1)
        relative = differences / (groups[:, 1:].norm(dim=-1) + groups[:, :-1].norm(dim=-1)).clamp_min(1e-6)
        transitions = mask[:, 1:] & mask[:, :-1]
        motion = (relative * transitions[..., None]).sum(1) / transitions.sum(1).clamp_min(1)[:, None]
        salience = motion * (positive - normal).reshape(b, 16, 32).norm(dim=-1)
        salience = salience / salience.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        indices = salience.topk(k_groups, -1).indices
        weight = (salience.gather(1, indices).softmax(-1) * k_groups).clamp(max=1.)
        gate = torch.zeros_like(salience).scatter(1, indices, weight)
        gate = gate[..., None].expand(-1, -1, 32).reshape(b, 1, d)
        local = (teacher[:, :, 1:] / .07).softmax(-1) * prior[:, None]
        local = local / local.sum(-1, keepdim=True).clamp_min(1e-8)
        local = local / local.sum(-1, keepdim=True).clamp_min(1e-8)
        local = F.avg_pool1d((local * mask[..., None]).transpose(1, 2), 17, stride=1, padding=8).transpose(1, 2)
        local = local / denominator[..., None].clamp_min(1. / 17)
        local = local / local.sum(-1, keepdim=True).clamp_min(1e-8)
        local_positive = F.normalize(local @ text[:, 1:], dim=-1)
        target = anomaly[..., None] * local_positive + (1. - anomaly[..., None]) * normal[:, None]
    return x + beta * mask[..., None] * gate * (x.norm(dim=-1, keepdim=True) * target)


def combine(x, varied, semantic_features, raw, lengths, beta=1.):
    valid = torch.arange(x.shape[1], device=x.device)[None] < lengths.to(x.device)[:, None]
    with torch.no_grad():
        alignment = F.cosine_similarity(x.float(), raw.float(), dim=-1).clamp(0., 1.)[..., None].square()
        groups = x.reshape(*x.shape[:2], 16, 32)
        delta = (varied - x).reshape_as(groups)
        gain = (delta * groups).sum(-1) / (beta * groups.square().sum(-1)).clamp_min(1e-12)
        concentration = ((gain.max(-1).values - 1.) / 11.).clamp(0., 1.)[..., None]
        weight = torch.maximum(alignment, concentration)
    refined = x + weight * (varied - x) + (1. - weight) * (semantic_features - x)
    return torch.where(valid[..., None], refined, varied)

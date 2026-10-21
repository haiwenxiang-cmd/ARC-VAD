"""MIL and text-separation objectives inherited from VadCLIP."""
import torch
import torch.nn.functional as F


def training_loss(text, binary_logits, category_logits, targets, lengths, text_weight):
    category_bags, binary_bags = [], []
    for i, length in enumerate(lengths):
        n = int(length)
        k = min(n, n // 16 + 1)
        category_bags.append(category_logits[i, :n].topk(k, dim=0).values.mean(0))
        binary_bags.append(binary_logits[i, :n].sigmoid().squeeze(-1).topk(k).values.mean())
    target_distribution = targets / targets.sum(1, keepdim=True)
    category_loss = -(target_distribution * F.log_softmax(torch.stack(category_bags), dim=1)).sum(1).mean()
    binary_loss = F.binary_cross_entropy(torch.stack(binary_bags), 1. - targets[:, 0])
    normal = text[0] / text[0].norm(dim=-1, keepdim=True)
    separation = text.new_zeros(1)
    for category in text[1:]:
        separation = separation + torch.abs(normal @ (category / category.norm(dim=-1, keepdim=True)))
    separation = separation / (len(text)-1)
    return category_loss + binary_loss + text_weight * separation.squeeze()

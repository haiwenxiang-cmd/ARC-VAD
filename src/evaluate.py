"""Frame-level detection and category-aware temporal localization."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from data import split_features


def category_probabilities(binary, categories, temperature=2., hierarchical_weight=.1):
    # Hierarchical readout adapted from DSANet (MIT); see THIRD_PARTY_NOTICES.md.
    abnormal = torch.sigmoid(binary / temperature)
    distribution = (categories / temperature).softmax(-1)[:, 1:]
    distribution = distribution / (distribution.sum(-1, keepdim=True) + 1e-12)
    hierarchical = torch.cat([1. - abnormal, abnormal * distribution], dim=-1)
    return (1.-hierarchical_weight) * categories.softmax(-1) + hierarchical_weight * hierarchical


@torch.no_grad()
def evaluate(model, dataset, config, device):
    model.eval()
    category_responses, detection_c, detection_a = [], [], []
    for i in range(len(dataset)):
        features = dataset.features(i)
        visual, lengths = split_features(features)
        _, binary, categories = model(visual.to(device), None, config['prompts'], lengths, single_video=True)
        binary = binary.reshape(-1, 1)[:len(features)]
        categories = categories.reshape(-1, model.num_class)[:len(features)]
        if not torch.isfinite(binary).all() or not torch.isfinite(categories).all():
            raise FloatingPointError(f'Nonfinite valid prediction for video {i}')
        detection_c.append(np.repeat(binary.sigmoid().squeeze(-1).cpu().numpy(), 16))
        detection_a.append(np.repeat((1.-categories.softmax(-1)[:, 0]).cpu().numpy(), 16))
        scores = category_probabilities(binary, categories, **config['readout'])
        category_responses.append(np.repeat(scores.cpu().numpy(), 16, axis=0))
    gt = np.load(config['data']['gt'], allow_pickle=False).reshape(-1)
    c, a = np.concatenate(detection_c), np.concatenate(detection_a)
    if len(c) != len(gt):
        raise ValueError(f'Frame label length {len(gt)} != predicted length {len(c)}; check feature/list/annotation alignment')
    segments = np.load(config['data']['gt_segments'], allow_pickle=True)
    labels = np.load(config['data']['gt_labels'], allow_pickle=True)
    if len(segments) != len(dataset) or len(labels) != len(dataset):
        raise ValueError('Temporal annotation order/length must match the test manifest')
    if config['dataset'] == 'xd':
        from utils.xd_detectionMAP import getDetectionMAP
    else:
        from utils.ucf_detectionMAP import getDetectionMAP
    values, thresholds = getDetectionMAP(category_responses, segments, labels, excludeNormal=False)
    return dict(AUC_C=100*roc_auc_score(gt,c), AP_C=100*average_precision_score(gt,c),
                AUC_A=100*roc_auc_score(gt,a), AP_A=100*average_precision_score(gt,a),
                mAP=[float(x) for x in values], IoU=[float(x) for x in thresholds], AVG_mAP=float(np.mean(values)))

"""Feature manifests and the temporal sampling used by VadCLIP."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def label_vector(label, config):
    labels = str(label).split('-') if config['dataset'] == 'xd' else [str(label)]
    target = torch.tensor([float(name in labels) for name in config['classes']])
    if not target.any():
        raise ValueError(f'Unrecognized video label: {label}')
    return target


def sample_features(features, size=256):
    n = len(features)
    if n == 0 or features.ndim != 2 or features.shape[1] != 512:
        raise ValueError(f'Expected nonempty [T,512] features, got {features.shape}')
    if n > size:
        boundaries = np.linspace(0, n, size + 1, dtype=np.int32)
        sampled = np.zeros((size, features.shape[1]), dtype=np.float32)
        for i in range(size):
            sampled[i] = features[boundaries[i]:boundaries[i+1]].mean(0) if boundaries[i] != boundaries[i+1] else features[boundaries[i]]
        return sampled, size
    return np.pad(features, ((0, size-n), (0, 0))).astype(np.float32), n


def split_features(features, size=256):
    n = len(features)
    if n <= 0:
        raise ValueError('Empty feature sequence')
    # Retain the native evaluation window convention, including a trailing
    # zero-length window for lengths exactly divisible by the window size.
    windows = n // size + 1
    result = np.zeros((windows, size, features.shape[1]), dtype=np.float32)
    lengths = []
    for i in range(windows):
        part = features[i*size:(i+1)*size]
        result[i, :len(part)] = part
        lengths.append(len(part))
    return torch.from_numpy(result), torch.tensor(lengths, dtype=torch.long)


class FeatureDataset(Dataset):
    def __init__(self, config, split, normal=None):
        self.config = config
        self.root = Path(config['data']['feature_root'])
        self.table = pd.read_csv(config['data'][split + '_list'])
        if not {'path', 'label'} <= set(self.table.columns):
            raise ValueError('Manifest requires path,label columns')
        if normal is not None:
            keep = self.table.label.map(lambda x: bool(label_vector(x, config)[0]))
            self.table = self.table[keep if normal else ~keep].reset_index(drop=True)

    def __len__(self):
        return len(self.table)

    def features(self, index):
        name = Path(str(self.table.iloc[index]['path']))
        path = name if name.is_absolute() else self.root / name
        return np.load(path, allow_pickle=False).astype(np.float32)

    def __getitem__(self, index):
        features, length = sample_features(self.features(index))
        return torch.from_numpy(features), label_vector(self.table.iloc[index]['label'], self.config), length

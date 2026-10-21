# ARC-VAD

Implementation of ARC-VAD for fine-grained weakly supervised video anomaly
detection on XD-Violence and UCF-Crime.

ARC-VAD combines anchor-guided global residual calibration (AGRC) and
variation-guided local support refinement (VLSR) at the shared visual interface
of a CLIP-based two-branch detector. Stage I learns the shared representation;
Stage II freezes its generator and adapts the predictors with VLSR enabled.

![ARC-VAD framework](assets/framework.png)

[Vector framework diagram](assets/framework.pdf)

## Installation

Python 3.10 or later is required. The reference environment uses PyTorch 2.3.0
and CUDA 12.1. Install the matching PyTorch build for your GPU, then:

```bash
pip install -r requirements.txt
```

The frozen CLIP ViT-B/16 checkpoint is downloaded by the CLIP loader on first
use. An existing checkpoint can be supplied with `--clip-checkpoint PATH`.

## Data

Use the pre-extracted 512-dimensional CLIP features and annotations distributed
with [VadCLIP](https://github.com/nwpu-zxr/VadCLIP). Download the training and
test features separately and prepare the lists without changing their order:

```bash
python prepare_data.py --dataset xd --train-root /path/to/XDTrainClipFeatures --test-root /path/to/XDTestClipFeatures
python prepare_data.py --dataset ucf --train-root /path/to/UCFTrainClipFeatures --test-root /path/to/UCFTestClipFeatures
```

The preparation script retrieves the official video lists and annotations and
replaces feature paths with paths on your machine. The configuration files
`configs/xd.json` and `configs/ucf.json` specify the lists and annotation files.
Each feature file has shape `[T, 512]`. Training manifests contain ten crop
rows per video. A custom manifest may include `video_id` to identify crops of
the same video; otherwise the trailing `__<crop-index>` is removed from filenames.
XD labels retain all positive categories, e.g. `B1-B2-0`.

## Training and evaluation

Run the following commands from the repository root. XD-Violence:

```bash
python src/main.py train --config configs/xd.json --stage 1 --output runs/xd/stage1
python src/main.py build-references --config configs/xd.json --checkpoint runs/xd/stage1/best.pt --output runs/xd/references.pt
python src/main.py train --config configs/xd.json --stage 2 --checkpoint runs/xd/stage1/best.pt --references runs/xd/references.pt --output runs/xd/stage2
python src/main.py evaluate --config configs/xd.json --stage 2 --checkpoint runs/xd/stage2/best.pt --references runs/xd/references.pt
```

For UCF-Crime, replace `configs/xd.json` and `runs/xd` with `configs/ucf.json`
and `runs/ucf`. Stage I runs for 10 epochs and Stage II for 5 epochs. The
default seed is 234; use `--seed 42` or `--seed 243` on both training commands
and keep separate output/reference paths for each run.

The prediction input is `Xm` without AGRC, `X0 + alpha * Xm` with AGRC, and
the VLSR-refined representation during Stage II. Set `model.agrc` and
`model.vlsr` in a copy of the dataset configuration to run the four main
component configurations. Rebuild references from the corresponding Stage-I
model when VLSR is enabled.

## Evaluation settings

The configurations give all scalar settings, the class order, and the actual
fixed category descriptions. References are constructed from abnormal training
videos using video-level labels and remain fixed during Stage II and inference.
Each video descriptor is assigned to all its positive categories.

Checkpoint selection uses the configured coarse-grained metric at the end of
each epoch: category-aware AP (`AP_A`) on XD-Violence and binary-branch AUC
(`AUC_C`) on UCF-Crime. The selected model weights are restored between epochs;
the optimizer and scheduler states continue. Selection uses the benchmark test
split. The final category readout uses the temperature and interpolation weight
specified in the configuration. Both coarse-grained branch metrics and mAP at
IoU thresholds 0.1--0.5 are returned from the same checkpoint.

Training uses 256-snippet sampling or zero-padding. At evaluation, complete
videos are split into native windows; semantic guidance aggregates valid
snippets across windows of the same video. Variation statistics retain the
window's padded positions, while semantic aggregation uses valid positions.
Frame expansion, proposal scoring and temporal NMS follow the included
dataset-specific evaluation functions.

The source distribution contains code and configurations. Dataset features,
generated galleries, and trained weights are not bundled.

## Acknowledgments and license

The implementation builds on [VadCLIP](https://github.com/nwpu-zxr/VadCLIP),
[CLIP](https://github.com/openai/CLIP), and the hierarchical readout in
[DSANet](https://github.com/lessiYin/DSANet). See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the included licenses.

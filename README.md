# Aggregate_Project

Instance segmentation of aggregate materials (gravel vs sand) — thesis pipeline.

## Layout

- `experiment/00EDA.ipynb` — data pipeline: Roboflow bronze → cleaned silver →
  group-aware gold split → quality audits → offline augmentation (`gold_aug`)
- `experiment/01model_yolov26n.ipynb` — training: smoke test → autobatch →
  optimizer bake-off (MuSGD/AdamW/SGD) → data arms (gold vs gold_aug) → holdout
- `experiment/train.yaml` — augmentation + schedule policy (source of truth)
- `experiment/yolo26n-seg-p2.yaml` — custom P2-head variant (Segment26P2 patch)
- `experiment/results/` — metrics, seeds, dataset checksums

## Datasets

Images/labels are not in git. Two ways to get them:

1. **Release asset** (recommended): download `datasets.tar.gz` from
   [v0.1-data](https://github.com/Telotubbies/Aggregate_Project/releases/tag/v0.1-data)
   and extract into `experiment/data/`.
2. **Re-derive**: run `00EDA.ipynb` end-to-end (needs `ROBOFLOW_API_KEY`).

`gold` train/valid/test = 753/226/105. `gold_aug` train = 3012 (4x offline
expansion incl. dark/rain environmental variants). File-level sha256 manifest:
`experiment/results/dataset_checksums.csv`.

## Environment

```
pip install -r experiment/requirements.txt   # + torch for your GPU (CUDA or ROCm)
jupyter notebook experiment/00EDA.ipynb
```

GPU training validated on RX 7800 XT via ROCm/WSL; any CUDA/ROCm box works —
the model notebook auto-detects and falls back to CPU for smoke tests.

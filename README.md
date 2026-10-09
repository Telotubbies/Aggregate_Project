# Aggregate Material Segmentation and Material-State Classification

Instance-segmentation pipeline for classifying aggregate deliveries
(**gravel** vs **sand**) from truck-camera footage. Built on Ultralytics
**YOLO26n-seg**, evaluated under convention-aware protocols on a
leakage-free, video-grouped split, and deployed as a per-video
decision-state system that emits exactly one material label per event.

This repository accompanies a thesis project; every experiment is
reproducible and all reported numbers trace to committed artifacts.

---

## 1. Results at a glance

Final checkpoint: `experiment/runs/final_best/weights/best.pt`
(sha256 `9077148c9e89ec6a`, pinned in `results/model_card.json`).

| metric | value |
|---|---|
| val mask mAP50-95 | 0.558 |
| **test mask mAP50-95 (Ultralytics)** | **0.383** |
| test mask mAP50 | 0.528 |
| custom strict protocol | 0.388 |
| custom sieved protocol | 0.472 |
| Boundary IoU (Cheng et al., CVPR 2021) | 0.295 |
| per-class, sieved | gravel 0.654 / sand 0.290 |
| operating point (conf 0.25) | TP 179 / FP 94 / FN 90, F1 0.661 |
| inference latency @640 | p50 5.7 ms, p95 6.4 ms (~176 FPS, RTX 3080 Ti) |

Video-level decisions on three unseen thesis clips (stride 5):

| clip | decision | sand fraction | margin |
|---|---|---|---|
| 0a48349e | sand | 1.000 | 1.00 |
| 32842c3f | gravel | 0.005 | 0.99 |
| 5d81688a | gravel | 0.000 | 1.00 |

Five decision rules (mass vote, frame majority, top-20 % mass, median
frame, EMA) agree unanimously; no clip produces contradictory states.

> Reporting note: earlier exploratory runs showed ~0.79 "mAP50"; that figure
> is **validation** mAP50, not test mAP50-95, and older random-frame splits
> leaked videos across splits. The numbers above are the correct,
> comparable headline metrics.

---

## 2. Repository layout

```
Aggregate_Project/
└── experiment/
    ├── 00EDA.ipynb                  # data pipeline: bronze -> silver -> gold,
    │                                #   group-aware split, quality audits
    ├── 01model_yolov26n.ipynb       # ablation history: smoke -> autobatch ->
    │                                #   optimizer bake-off -> data arms -> P2
    ├── final_pipeline.ipynb         # PRIMARY DELIVERABLE: formal
    │                                #   train -> eval -> deploy notebook
    ├── methods_survey.md            # literature survey w/ verdicts per method
    ├── deploy/
    │   └── material_state.py        # production decision module (Ultralytics)
    ├── soup_eval.py                 # model-soup evaluation script
    ├── archive/
    │   ├── experiments/             # ablation + alternate-architecture nbs
    │   │   ├── 02_improvements.ipynb / 02improve_yolo.ipynb
    │   │   ├── 03fn_reduce.ipynb    #   inference-time FN-reduction study
    │   │   ├── 04solov2.ipynb       #   SOLOv2-R50 comparison
    │   │   └── solov2_gold.py       #   COCO conversion + mmdet runner
    │   └── configs/                 # auxiliary/experimental configs
    ├── data/                        # committed datasets (see §3)
    ├── results/                     # all metric artifacts, caches, logs,
    │                                #   model_card.json
    ├── runs/                        # training outputs (gitignored; see §6)
    └── weights/                     # base pretrained checkpoints
```

## 3. Datasets

Datasets are committed in-repo under `experiment/data/`:

| split | content | counts |
|---|---|---|
| `bronze` | raw Roboflow export (random frame split — leaked, retained for provenance only) | — |
| `silver` | cleaned labels | — |
| `gold` | **video-grouped split** (group key = video UUID) | 753 train / 226 valid / 105 test |
| `gold_clahe` | CLAHE A/B variant | — |
| `gold_aug` | offline 4x expansion (incl. dark/rain variants) | 3012 train |
| `gold_aug_clean` | sliver-free aug variant (clean960 arm) | — |
| `gold/*.json` | COCO-format conversions (mmdetection) | — |
| `thesis_videos` | 3 real delivery clips for state-decision validation | 56 s / 20 s / 21 s @30 FPS |

Fallback for shallow clones: `datasets.tar.gz` on the
[v0.1-data release](https://github.com/Telotubbies/Aggregate_Project/releases/tag/v0.1-data),
or re-run `00EDA.ipynb` end-to-end (requires `ROBOFLOW_API_KEY`).
File-level integrity: `experiment/results/dataset_checksums.csv`.

**Leakage control.** The original Roboflow export split video frames
randomly across train/valid/test, inflating metrics. `gold` groups all
frames of a source video into a single split; `2be0c9f6...` (sand, 315
frames) is pinned to train.

## 4. Evaluation protocols

All test numbers are produced by one harness (`final_pipeline.ipynb` §4):

- **strict** — COCO-style matching; every GT instance counts.
- **sieved** — sliver instances (area < 0.5 % or aspect > 8:1) removed
  from GT and predictions (aerial-imagery convention). +9 pt over strict;
  79/348 test GT are slivers.
- **biou** — Boundary IoU, ±2 % band around GT boundaries
  (canonical implementation; earlier project values ~0.64 used a different
  band definition and are not comparable).

## 5. Deployment

`experiment/deploy/material_state.py` streams a video, accumulates
confidence-weighted mask mass per class behind a debounced
present/absent gate, and emits **one** decision per clip:
`gravel | sand | ambiguous | no-data`. It never returns competing states;
low-margin or empty-evidence events are flagged for human review.

```bash
cd experiment/deploy
python material_state.py \
    --video ../data/thesis_videos/<clip>.mp4 \
    --model ../runs/final_best/weights/best.pt \
    --stride 5
```

Decision rule: `argmax_class Σ_frames Σ_instances conf × mask_area`,
operating point conf 0.25 / area 0.5 % (sieved), margin = |sand_frac−0.5|×2.

## 6. Reproducing

```bash
pip install -r experiment/requirements.txt   # plus torch for your GPU
jupyter notebook experiment/final_pipeline.ipynb
```

- Environment validated: Python 3.12, torch 2.11.0+cu130,
  ultralytics 8.4.171, RTX 3080 Ti (12 GB). Earlier bring-up used ROCm.
- The training cell is **skip-guarded** — re-running the notebook reuses
  `runs/final_best/weights/best.pt` without spending GPU time.
- `runs/` and `*.pt`/`*.onnx` are gitignored; the final checkpoint is
  identified by the sha256 in `results/model_card.json`.
- SOLOv2 comparison needs a separate env (torch 2.1.2+cu121, mmcv 2.1.0,
  mmdet 3.3.0, numpy<2) — see `archive/experiments/04solov2.ipynb`.
- Seeds pinned (`seed=42`); per-run seeds logged in `results/seeds.json`.

## 7. Key findings

1. **Recipe** — AdamW lr 1e-3 on raw `gold` beat MuSGD/SGD and every
   offline-augmentation arm; selected on validation only.
2. **P2 head** — stride-4 detection lost to the P3 baseline
   (val 0.526 vs 0.557); warm-transfer remapping made it worse (0.366).
3. **FN is domain/convention-bound** — taxonomy: no_pred 38 %, conf_cut
   32 %, weak/iou_low/cls_swap the rest. Threshold tuning has ~2 pt F1
   headroom only.
4. **Inference-time ceiling ~+1.3 pt** — temporal boost + flip union +
   model ensemble reached strict 0.368 / sieved 0.462; SAHI and
   upscale inference degraded results.
5. **Architecture swap does not help** — SOLOv2-R50 (46 M params) scored
   below YOLO26n (2.7 M): strict 0.345 vs 0.355, with 2.4x the FP.
6. **Model soup** — weight averaging across recipes degrades the model
   (soup_eval.py); checkpoints land in different basins.
7. **Video state is stable** — clean clips give margin ≥ 0.99 under all
   five aggregation rules; ambiguity handling is implemented but untested
   on mixed-material footage (no labeled clips exist).

## 8. Known limitations / remaining gaps

- Sand recall is the residual weakness (sieved sand AP 0.290): unseen
  scenes and ambiguous pile boundaries; ~100 test FN persist across all
  methods.
- Edge-device latency (Jetson-class) not yet benchmarked.
- No labeled video events exist, so event-level accuracy is unmeasured;
  the three validation clips are unlabeled but decisive.
- Drift monitoring, re-annotation loop, and multi-seed significance
  testing are documented as gaps in `final_pipeline.ipynb` §8.

## 9. References

- Cheng et al., *Boundary IoU*, CVPR 2021 — §4 `biou` protocol.
- Kirillov et al., *PointRend*, CVPR 2020 — void-region convention.
- Wortsman et al., *Model Soups*, ICML 2022 — `soup_eval.py`.
- Lu et al., *Resour. Conserv. Recycl.* 2022; FEQS cargo-volume
  (*Appl. Sci.* 2019); M2CAI temporal smoothing — §7 decision design.
- Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019 —
  `results/model_card.json`.
- Sculley et al., *Hidden Technical Debt in ML Systems*, NeurIPS 2015 —
  §8 gap framework.
- Full method-by-method survey with per-technique verdicts:
  `experiment/methods_survey.md`.

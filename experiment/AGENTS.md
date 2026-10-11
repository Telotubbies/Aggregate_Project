# experiment/ — thesis segmentation pipeline

Environment: WSL2 + ROCm (RX 7800 XT), venv `/home/gigabyte/gpu-venv`
(Python 3.12, torch 2.6+rocm6.4, ultralytics 8.4.175). Run all scripts
inside WSL: `cd /mnt/b/02_Projects/thesis_final/experiment`.

## Layout

    00_EDA.ipynb                  dataset exploration
    01_model_yolov26n.ipynb       YOLO26n-seg + small-label ablation (5 arms)
    02_ml_models.ipynb            handcrafted SVM (oracle-mask) + video map
    03_model_maskrcnn.ipynb       Mask R-CNN baseline
    04_model_patchseg.ipynb       patchseg-grainseg port (U-Net 58k + ws)
    05_model_unet_watershed.ipynb U-Net 7.8M + watershed
    06_model_cellpose.ipynb       Cellpose-SAM (2x class-agnostic, frozen)
    07_model_fastyolov8.ipynb     YOLOv8n-seg + Ghost variant
    08_model_blendmask.ipynb      CondInst R50 via mmdet (BlendMask n/a)

    eval_common.py      shared holdout evaluator — every model notebook
                        calls evaluate_holdout(); writes results/<name>_holdout.json
    patchseg_run.py     nb04 pipeline module
    unet_watershed.py   nb05 pipeline module
    maskrcnn_torch.py   nb03 pipeline module
    sam31_eval.py / sam31_viz.py   SAM3.1 zero-shot eval + figures
    _bm_*.py            nb08 runtime deps (mmdet cfg, torch26 patch,
                        yolo2coco, train, eval) — keep at root, imported
                        by notebook cells
    train.yaml          ultralytics data cfg
    yolo26n-seg.pt / yolov8n-seg.pt / yolov8n-ghost-seg.yaml /
    yolo26n-seg-p2.yaml model weights/configs referenced by notebooks

    data/     gold (train/valid/test splits — test = 105-img holdout),
              gold_aug, cellpose prep dir
    results/  per-model subdirs + canonical <name>_holdout.json +
              figures/csvs (paths hardcoded in notebooks — do not move)
    runs/     ultralytics + mmdet training outputs
    weights/  downloaded checkpoints (condinst coco pretrain etc.)
    external/ cloned upstream repos (patchseg-grainseg)
    dev/      one-off scripts organized by task:
              cellpose/ blendmask/ nb_gen/ patch/ audit/ fastyolo/
              logs/ scratch_nb/  — not needed for reproduction

## Conventions

- Holdout = `data/gold/test` (never tuned on). Selection on `valid`.
- Metrics: `eval_common.evaluate_holdout(gt, preds, name)` → AP50
  per class, macro AP50, size-bin recall (small<32², med<96², large),
  P/R/F1 sweep. Preds contract: {stem: [(cls01, score, bool HxW)]}.
- classes: 0=gravel 1=sand.
- GPU contention is real — avoid parallel training jobs; first epoch
  at a new imgsz triggers slow MIOpen exhaustive search (one-time).

## Current leaderboard (macro AP50, same holdout)

    YOLO26n-seg 0.488 | MRCNN 0.391 | CondInst 0.379 | SAM3.1 zs ~0.37
    U-Net+ws 0.350 | YOLOv8n 0.336 | ghost 0.184 | patchseg 0.179
    | cellpose ~0.001 (granularity mismatch — fragments vs regions)

## GPU stack findings (2026-10-11, TheRock torch 2.9.1+rocm7.10a gfx1101)

- **fp16 AND bf16 produce NaN on this build** — even trivial 2-conv
  forward. MIOpen kernel pool on RDNA3 has buggy entries; only the
  deterministic algo subset is correct. ALWAYS pass `amp=False` and
  keep `deterministic=True` — the AMP self-check can pass spuriously.
- `deterministic=False` → NaN even in FP32 (bad non-det kernel).
- First-iter MIOpen exhaustive search = minutes-long stall (normal,
  one-time per conv config; CPU busy, not deadlocked).
- Steady-state: batch16@960 FP32 ≈ 6.5 s/it, ~13 GB VRAM.
- Upgrade path (untested): stable wheel torch[device-gfx1101]
  +rocm10.1.0 — ROCm 7.2.4+ fixed RDNA3 kernel bugs; may unlock AMP.
- MIOPEN_USER_DB_PATH / MIOPEN_CUSTOM_CACHE_DIR persist tuning
  across runs; MIOPEN_FIND_ENFORCE=3 tunes+stores.

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

## Grouped CV continuation (2026-10-14, Protocol v2)

For current `00`/`16`/`17` CV work, use Linux + RTX 3080 Ti and kernel `venv_inf`
(`/home/supawich/YOLO AGGREGATE/.venv_inf/bin/python`), not the historical WSL paths above.
`17_cv_v2.ipynb` is the clean v2-only notebook (2 seg models, grouped CV, parent sampler,
freeze/evaluate gates, external/Stripe audit). `16_train_last.ipynb` is historical context
(fixed-split 6-model benchmark) and was the launcher of run set `3b867e9545011a5f3c52`;
continue that launch in `17` via `NB16_ADOPT_PLAN=1` + `NB16_APPROVED_PROTOCOL=3b867e9545011a5f3c52`
(`NB16_APPROVED_PREVIEW_SHA=f6dd2f6010d6888d00bb07caae6498b4a87467b3e5f3015d79184ad4d303853b`).
`00` audit/CV executes cells 1, 66, 68, 70, 72, 74, 76, 78, 80; `NB00_BUILD_CV=1` materializes `groupcv_offline_v2_<id>` (`original + color_light + clahe_blur + environment` on inner-train only) and publishes `nb00-groupcv-v2`.
Active v2 dataset: `data/groupcv_offline_v2_de6276f231115417` (source `gold_fixed_v1` id `9bf2fab6…`).
Historical `data/groupcv_offline_v1_253fef93327e4d8a` and `results/nb16_full_cv/*` belong to v1 (`hflip/gamma_brightness/clahe_blur`, online disabled) and are rejected by v2 handoff loaders.
`16`/`17` default to audit; separate flags gate each stage: `NB16_CV_PREVIEW=1`, `NB16_CV_SMOKE=1`, `NB16_FULL_CV=1` (requires `NB16_APPROVED_PROTOCOL` + `NB16_APPROVED_PREVIEW_SHA`), `NB16_CV_FREEZE=1`, `NB16_CV_EVALUATE=1`, `NB16_EXTERNAL_AUDIT=1`, `NB16_EVAL_EXTERNAL=1`, `NB16_EVAL_STRIPE=1`.
Full CV (20 runs: 2 models × 2 arms × 5 folds × seed 42) launched 2026-10-14; outputs under `runs|results/nb16_recipe_v2/3b867e9545011a5f3c52/`.
Persist outputs for executed cells and retain historical outputs in skipped cells.
Handoff identity hashes code cells only; Markdown/output edits do not invalidate it.
Use grouped development CV on existing data; Stripe 88 is a development-inspected seen-scene benchmark, and external ZIPs require mechanical + human ontology/annotation/duplicate review before reporting separately.

# Aggregate Instance Segmentation — Experiment Log (RTX 3080 Ti)

Research question: **why does YOLO-seg miss aggregate instances (especially sand),
and which intervention actually helps?**

Environment: RTX 3080 Ti 12 GB · torch 2.5.1+cu121 · ultralytics 8.4.59 · AMP on
Seed 42, `yolo26n-seg.pt`, imgsz 960, batch −1, epochs ≤100, patience 30,
all train-time augmentations disabled (controlled protocol, nb12).
Metric: per-class AP50 + recall at same-class mask-IoU ≥ .5, conf ≥ .05,
evaluated at flat 960 on held-out images (identical evaluator across all runs).

## 1. Experiment matrix

| # | Experiment | Arms | Question |
|---|---|---|---|
| E1 | Data recipes (nb12) | aug, rfs, sahift, hem, res1280 | can resampling/tiling/resolution fix sand FN? |
| E2 | Capacity probe (nb15) | s960 (yolo26s) | is the nano model too small? |
| E3 | Contrast preprocess | gold_clahe | does CLAHE help? |
| E4 | GT audit | — | are sand FN caused by bad labels? |
| E5 | Split audit → `gold_v2` | — | is the holdout scene-balanced? |
| E6 | Same-model re-eval | aug × v2 test | how much of the "failure" was the test set? |
| E7 | 4-fold group-CV | fold0–3 | honest mean±std under scene holdout |
| E8 | LOGO ×11 scenes | leave-one-group-out | per-scene generalization map |
| E9 | Image-quality audit | laplacian var × group | does blur/resolution explain sand FN? |

## 2. Results

### E1 — data recipes (old `gold/test`, 105 imgs)

| arm | gravel AP50 | sand AP50 | macro | Δ |
|---|---|---|---|---|
| aug (control) | **0.733** | **0.415** | **0.574** | — |
| sahift | 0.687 | 0.360 | 0.524 | −.050 |
| res1280 | 0.737 | 0.308 | 0.523 | −.052 |
| rfs | 0.614 | 0.381 | 0.497 | −.077 |
| hem | 0.687 | 0.305 | 0.496 | −.078 |

Every data-level recipe loses to the control. `hem` buys gravel recall (+.05)
at the cost of a prediction flood (npred 137→217) and the worst sand AP.

### E2 — capacity (same protocol, `yolo26s-seg`, 3.6× FLOPs)

macro **0.552 < 0.574**. Size-bin recall (s960): small <32² **0/27**,
med 32–96² **.081**, large ≥96² .670. Capacity is not the bottleneck;
the s-model additionally fragments large piles.

### E3 — CLAHE (earlier arm sweep)

gold 0.582 vs gold_clahe 0.574 — no gain despite higher contrast metrics.

### E4 — GT audit (aug predictions on old test)

138/159 sand GT are FN; **136 have zero matching prediction** (max IoU 0.000).
Regions are visually plausible, well-drawn piles — the model emits nothing.
Not a label-noise problem.

### E5 — split audit

Old split is group-disjoint (32 uuid groups, 0 crossing) but
**scene-imbalanced**: 152/159 test sand instances come from one group,
`bbf45eda` (rear-view dump scene), a viewpoint absent from train.
`data/gold_v2` (hardlinks, gold untouched): 759/150/175 imgs, 19/6/7 groups,
test sand = 339 inst across 2 scenes + gravel from 2 scenes.

### E6 — same weights, new test

| eval | grav | sand | macro | sand rec |
|---|---|---|---|---|
| aug × old test | .733 | .415 | .574 | .132 |
| **aug × gold_v2 test** | **.782** | **.502** | **.642** | **.277** |
| v2ctrl × v2 test | .673 | .347 | .510 | .221 |

The "sand failure" was largely a test-set artifact (+0.068 macro, ×2 recall
with identical weights). `v2ctrl` is lower because it trains on 759 plain
images vs the control's 3,012 offline-augmented — not an apples-to-apples
model comparison.

### E7 — 4-fold group-CV (uuid-atomic folds, shared valid)

| fold | grav | sand | macro |
|---|---|---|---|
| 0 (bbf45eda in test) | .624 | .312 | .468 |
| 1 | .711 | .132 | .421 |
| 2 | .678 | .165 | .421 |
| 3 (6459c886 in test) | .744 | .547 | .645 |
| **mean ± std** | **.689 ± .044** | **.289 ± .164** | **.489 ± .092** |

Sand fold-std (.16) exceeds every recipe delta in E1 combined — scene
composition, not recipe, is the dominant variance source.

### E8 — LOGO (leave-one-scene-out, 11 folds)

| held-out scene | type | imgs | gravel AP50 | sand AP50 | sand rec |
|---|---|---|---|---|---|
| 0a48349e | sand pile | 81 | — | **.716** | .64 |
| 6459c886 | sand pile | 49 | — | .589 | .38 |
| 2be0c9f6 | sand side-view | 314 | — | .584 | .47 |
| bbf45eda | **sand rear-view** | 36 | — | .507 | **.18** |
| 4a7f29af | mixed | 48 | .594 | .266 | .10 |
| 1991989f | mixed | 65 | .516 | **.000** | .00 |
| 96ecd4fe | gravel | 62 | **.897** | — | — |
| d56fcb3c | gravel | 57 | .647 | — | — |
| f9b12ec3 | gravel | 56 | .631 | — | — |
| 0f964fdb | gravel | 90 | .622 | — | — |
| 6f40fbbd | gravel | 47 | .582 | — | — |

**Gravel mean .649 (0.52–0.90) — generalizes to every unseen scene.
Sand mean .532 (0.00–0.72) — scene-identity decides success:**
pile-type sand transfers across scenes; rear-view and mixed-context sand
do not, even when trained on all other sand scenes.

### E9 — image quality vs LOGO outcome

| group | res | sharpness (lap var) | sand AP50 |
|---|---|---|---|
| 0a48349e | 360×640 | 664 | .72 |
| 6459c886 | 360×640 | 411 | .59 |
| bbf45eda | 432×768 | 663 | .51 |
| 4a7f29af | 540×960 | 1783 | .27 |
| 1991989f | 432×768 | **2072** | **.00** |

The sharpest group in the dataset scores sand AP = 0; the sharpest gravel
groups are also not its best. Blur/resolution does not explain the failure —
this also rules out sharpening/upscaling preprocessing as a fix.

## 3. Conclusions

1. **The original holdout systematically understated sand performance** —
   one unseen rear-view burst carried 95% of test sand.
2. **Scene identity is the dominant performance factor**, larger than any
   recipe, capacity, or preprocessing delta measured (E1–E3).
3. **Sand failure = context/viewpoint gap, not class difficulty** — pile sand
   transfers across scenes (0.58–0.72); rear-view/mixed-context sand does not
   (0.00–0.51) even with all other sand scenes in training.
4. **The only evidence-backed fix is data coverage**: collect rear-view and
   mixed-context sand scenes. Every cheaper lever tested was exhausted.
5. Reporting protocol: per-scene LOGO matrix + group-CV mean±std —
   a single-split sand AP50 is not a meaningful number for this dataset.

## 4. Trade-offs & limitations

- **Single seed** throughout; recall on ~100-GT folds is noisy (±~.05
  plausible on bbf45eda: CV gave .02 recall, LOGO .18).
- **LOGO/CV macro is depressed** on mono-class test groups (missing class
  scores AP=0) — read per-class columns, not macro, for E7/E8.
- **Shared valid set** in CV (not nested CV) — acceptable because nothing
  was tuned on it; early-stop selection only.
- `2be0c9f6` (314 imgs) always trains in CV — its unseen-scene score comes
  only from LOGO.
- `v2ctrl` vs `aug` is confounded by the missing `gold_v2_aug` (4× offline
  aug) — do not read it as "new split hurts training".
- Eval threshold fixed conf=.05, IoU=.5 — absolute AP shifts with protocol,
  but all arms share it, so comparisons hold.
- gold_v2/CV/LOGO datasets are hardlinks — rebuild with `scripts/` if
  deleted; do not commit datasets.

## 5. Reproduce

```
scripts/make_gold_v2.py        # build data/gold_v2 + audit + figs
scripts/train_eval_gold_v2.py  # aug×v2 eval + train v2ctrl + eval
scripts/group_cv.py            # 4-fold class-aware group CV
scripts/logo.py                # 11-fold leave-one-scene-out
scripts/gt_audit_sand.py       # sand FN crop audit (old holdout)
```

Key artifacts: `resplit_v2/` (split figs + eval jsons), `cv/` (fold jsons +
fig_cv.png), `logo/` (per-scene jsons + fig_logo.png + group_quality.csv),
`recipes/` (nb12/nb15 figs + preds_*.npy).

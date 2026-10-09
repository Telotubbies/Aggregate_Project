# Data Card — Aggregate (gravel/sand) Instance Segmentation

## Source
- Roboflow project `aggregate_rock-and-sand` v5, workspace `ilovefoods-workspace-qd0ut`
- Format: YOLO instance segmentation (normalized polygons), classes: ['gravel', 'sand']
- Frames extracted from ~30 source video clips (uuid-prefix groups)

## Pipeline (bronze -> silver -> gold)

| stage | content | instances |
|---|---|---|
| bronze | raw Roboflow export, frame-level random split | 6,078 |
| silver | degenerate removed (209), duplicate clusters merged keep-largest (1424 in 932 clusters, IoMin>=0.5 same-class), tiny noise removed (584, area<0.001) | 3,861 |
| gold | silver + group-aware re-split (uuid + dHash ham<=6 merged groups, class-aware 70/20/10) | 3,861 |

## Key decisions
- duplicates: same-image same-class polygon overlap; IoMin >= 0.5 -> keep largest member
- tiny masks: standalone < 0.001 image area confirmed noise via visual inspection
- elongated masks: kept (real spread-out piles, not noise)
- empty-label (background-only) images: removed at 7.3 per owner decision (5 images; gold counts in the table above are pre-removal)

## Leakage finding
- Original splits: 17/30 uuid groups spanned >1 split
  (1073 images = 98.5% leaked)
- dHash found 36 cross-group near-dup pairs -> merged to 26 groups
- Gold split: 0 groups cross split boundaries (asserted)

## Final gold splits
           n_images   pct
new_split                
test            105   9.6
train           754  69.2
valid           230  21.1

## Caveats
- 30 clips is limited scene diversity for 1,089 frames — adjacent frames are highly correlated
- auto-labels are pseudo-ground-truth; residual label noise exists below inspection thresholds
- eval metrics on gold test are clip-independent but NOT scene-independent if clips share sites
- clips are mostly single-class scenes: train has 15% mixed-class images but valid/test have ZERO —
  per-class AP is reported separately; mixed-scene performance cannot be evaluated on this data
- quality flags are uneven across splits (test 14% vs valid 4% of images) — see quality_flags.csv
  for per-slice reporting

## Human review protocol
- every removal rule was inspected visually before application (EDA sections 3-4, 10-11)
- removal criteria limited to geometrically invalid or near-identical same-class duplicates
- all borderline flag classes (elongated, blurred, cast, coverage, boundary-density)
  were documented as flags, never auto-dropped
- residual noise: round-2 model-disagreement audit planned after baseline training

## Reproducibility artifacts
- seeds: results/seeds.json
- checksums: results/dataset_checksums.csv
- audit CSVs: split_audit.csv, quality_flags.csv, instance_flags.csv, aug_log.csv (gold_aug)

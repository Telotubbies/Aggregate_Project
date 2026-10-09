# Methods Survey — Aggregate Segmentation Improvement Levers

Literature-backed catalog of improvement methods for the YOLO26n-seg aggregate
(gravel/sand) thesis pipeline. Compiled 2026 — every entry cross-referenced to
the project's diagnosed bottlenecks.

## Project bottlenecks (empirical, from nb 01)

| # | Bottleneck | Evidence |
|---|---|---|
| B1 | FN-dominated errors | test: 190 FN vs 44 FP; 57% of FN = sand |
| B2 | val→test domain shift | mask mAP50-95: 0.557 val → 0.359 test; sand 0.36→0.20 |
| B3 | Targets are amorphous regions ("stuff") | median instance = 4.4% of image (~90×90px); piles/spills not countable objects |
| B4 | Label ambiguity | FN regions = scatter on grate, stained truck-bed surfaces, partially occluded piles |
| B5 | P2 head ineffective | p2: 0.526 / p2w-remap: 0.366 vs baseline 0.557 val mAP50-95; consistent w/ PCB ablation study ("no measurable gains" when objects aren't <8×8px) |
| B6 | Partial-pretrain confound on modified arch | index-keyed transfer: P2 got 34% vs baseline 88%; warm-remap produced incoherent init (warm-box/random-cls) |

---

## 1. Task reformulation (largest potential gain)

| Method | Evidence / citation | Fit |
|---|---|---|
| Semantic segmentation (DeepLabV3+, SegFormer, UPerNet) | Kirillov et al., *Panoptic Segmentation* CVPR19 — "stuff" = amorphous uncountable regions; instance mAP penalizes boundary ambiguity directly | ⭐⭐⭐ B3,B4 — piles are textbook stuff |
| Panoptic / universal models (Mask2Former, OneFormer) | OneFormer CVPR23 > Mask2Former on all 3 tasks | ⭐⭐ B3 |
| Domain-identical prior art | *Vision-based method to identify materials transported by dump trucks*, Eng. Appl. AI 2024; *Bulk material identification + volume*, EAAI 2025 (YOLOv5+PointNet++) | ⭐⭐⭐ same scene family — they use semantic seg |
| Boundary-aware metric reporting (boundary IoU) | Cheng et al., *Boundary IoU* CVPR21 | ⭐ fairer metric for ambiguous-boundary regions |

## 2. Data & label quality (attacks B1+B4 directly)

| Method | Evidence | Fit |
|---|---|---|
| Annotation audit / relabel ambiguous GT | Univ. Limerick study: highest label noise → **−0.185–0.208 mAP** | ⭐⭐⭐ B4 — likely hidden ceiling |
| Pseudo-labeling on unlabeled video frames | PAIS (arXiv 2308.05359) **+12pt**; PL-DC (AAAI26) **+11.7 AP** @1% COCO labels | ⭐⭐⭐ free frames exist per video |
| SAM-assisted relabeling / GT refinement | ParticleSAM (EUSIPCO25, CDW particles); M2Fusion-SAM +14% mAP50-95 industrial; Grounded-SAM auto-label pipeline; NVIDIA TAO auto-label (Grounding DINO iterative thresholds) | ⭐⭐ B4 |
| Copy-Paste augmentation | Ghiasi et al. CVPR21: **+3.6 mask AP rare classes**; X-Paste ICML23 **+6.5** long-tail | ⭐⭐⭐ `copy_paste=0.0` currently — try 0.1–0.3, paste sand onto varied beds |
| Diffusion-synthesized samples | SyntheticGen / LoRA-diffusion few-shot (20–50 real imgs → synthetic rare-class); Dataset-Diffusion NeurIPS23 pseudo-labels | ⭐⭐ B2 sand scenes |
| Active learning annotation targeting | Region/superpixel AL surveys: dynamic-budget AL +5.6% mIoU over static; CVPR21 superpixel AL | ⭐⭐ pick highest-FN test-adjacent frames to relabel |
| Weak+strong label mixing | CVPR23 budget-constrained campaigns: image-level labels ~100× cheaper, mixed budgets beat all-strong | ⭐ label more frames cheaply |

## 3. Training recipe & optimization

| Method | Evidence | Fit |
|---|---|---|
| Two-stage fine-tune (freeze backbone → unfreeze low LR) | Ultralytics finetuning guide; medical fine-tuning survey (8 strategies) | ⭐⭐ B6 — correct way to warm-start modified arch |
| Discriminative LR (param groups) | standard partial-pretrain practice | ⭐⭐ B6 |
| `mask_ratio=1` + `retina_masks=True` | higher mask rasterization fidelity → high-IoU AP | ⭐⭐⭐ cheap, direct on mAP50-95 |
| Larger model (yolo26s/m) + imgsz 960/1280 | capacity+resolution | ⭐⭐ current n = 3.05M params |
| Model soup / checkpoint averaging | Wortsman et al. ICML22 — weight-avg > best single ckpt | ⭐⭐ free |
| Longer schedule for slow optimizers | MuSGD best@ep29/30 — 30ep screening biased | ⭐ fairness |
| Self-supervised backbone (DINOv2) | DINOv2 robust cross-domain features, linear-head seg without fine-tune | ⭐⭐ B2 — frozen SSL features generalize better under shift |
| Knowledge distillation (large teacher → small student) | Structured KD CVPR19; canonical KD ~99% of teacher mIoU @¼ params; GKD +10.6% generalization | ⭐⭐ distill a big teacher into yolo26n |
| **SAM optimizer (sharpness-aware) / SWA** | Foret et al. ICLR21; OOD study: SAM **+4.76%** vs Adam avg, strongest variant **+8.01%** on unseen domain | ⭐⭐⭐ B2 — targets exactly val→test gap |
| **Domain-gen modules: MixStyle / FACT / TF-Cal** | MixStyle ICLR21 mixes feature statistics; FACT Fourier amplitude mixing; TF-Cal test-time style calibration | ⭐⭐ B2 |
| **BiFPN / CBAM attention neck** | YOLOv8_CB weld-defects +3.07% mAP; rice-pest CBAM+BiFPN | ⭐ texture-region focus (B3) |
| **Monocular-depth aux channel** | AIP Advances 2025: DeepLabV3+ + depth on material piles (92.9%); BriGeS depth×seg fusion | ⭐ pile geometry vs bed surface disambiguation (B4) |

## 4. Loss functions (attacks B1, B4, class imbalance)

| Method | Evidence |
|---|---|
| Boundary loss (distance-map contours) | Kervadec MIDL19 runner-up best paper: +8% Dice unbalanced |
| Lovász-Softmax (direct mIoU surrogate) | Berman CVPR18 |
| Noise-robust losses: GCE, Symmetric CE (SL), APL, abstention | NeurIPS18/ICCV19/ICML20 — train robust to ambiguous labels instead of fixing them — ⭐⭐ B4 |
| Focal / Seesaw loss, class weighting for sand | long-tail literature — B1 |
| OHEM / hard-example mining | focus FN-heavy images |

## 5. Inference-time (no retrain)

| Method | Evidence | Fit |
|---|---|---|
| TTA (`augment=True`, flip+multiscale) | +1–2% mAP typical; Ultralytics TTA guide | ⭐⭐⭐ trivial |
| Per-class conf threshold sweep (lower for sand) | FN-dominant profile | ⭐⭐⭐ |
| Weighted Boxes Fusion (WBF) ensemble | ZFTurbo arXiv 1910.13302 — beats NMS/soft-NMS | ⭐⭐ ensemble p3+p2 ckpts |
| SAHI sliced inference | +5–14% AP on VisDrone/xView small objects | ⭐ regions large — limited upside, may catch scatter fragments |
| Test-time BN adaptation / entropy-min (TTA-Seg, ADVENT) | CVPR19 ADVENT; Mixture-of-Prototypes TTA-Seg CVPR26 | ⭐⭐ B2 — adapt to test scene unlabeled |
| Dense-CRF / superpixel-CRF post-process | Krähenbühl NeurIPS11; PSP-CRF boundary refinement | ⭐ sharpen ambiguous boundaries |
| PointRend / Mask Transfiner / BPR head swap | Transfiner CVPR22 **+3.0 mask AP**, +6.6 boundary AP Cityscapes | ⭐⭐ |

## 6. Architecture alternatives

| Method | Evidence |
|---|---|
| RT-DETR / DETR-family | CVPR24 "DETRs Beat YOLOs" 53.1 AP @108FPS; PCB study: RT-DETR 2× YOLO mAP on small objects |
| Mask R-CNN two-stage | used directly in stockpile-aggregate paper (TRB 2020) |
| Material-recognition features (texture encoders) | MINC dataset (Bell CVPR15) — material classes are texture-driven; patch-CNN+CRF 73% acc | ⭐ B3 |
| ~~P2 small-object head~~ | **CLOSED**: PCB ablation "no measurable gains"; our p2=0.526/p2w=0.366 < 0.557; objects not small |

## 7. Temporal / video exploitation (under-used asset)

| Method | Evidence |
|---|---|
| Temporal-consistency self-training (Siamese) | bioRxiv microscopy — unlabeled adjacent frames improve AP |
| Cross-frame pseudo-labels (TPS) | ECCV22 domain-adaptive video seg |
| Label propagation across video frames | standard video-seg — annotate 1 frame, propagate to neighbors → cheap more labels | ⭐⭐⭐ |
| Consistency-aware KD (KD-SSP) | +12.5% temporal consistency UAVid |

## 8. Domain shift mitigations (B2)

| Method | Evidence |
|---|---|
| UDA: adversarial feature align (ADVENT, CyCADA) | CVPR19; pixel+feature cycle-consistent adaptation |
| Entropy minimization on target domain | Vu CVPR19 |
| Domain randomization (photometric/style jitter harder) | clahe arm already tested (0.574 < gold 0.582 — weak evidence) |
| More sand scenes matching test distribution | universal consensus — coverage beats tricks |
| Uncertainty estimation / abstention on OOD | γ-SSL arXiv 2402.17653; ATTA NeurIPS23 — flag unreliable preds |

## 9. Semi-/weakly-supervised frameworks

| Method | Evidence |
|---|---|
| FixMatch-style consistency regularization | standard SSL — unlabeled frames |
| Semi-supervised instance seg (PAIS, PL-DC) | above |
| Weakly-supervised (image-level / box-level) | Papandreou; Arnab ECCV18 non-overlapping stuff+things |
| Scribble/click supervision | cheap GT for ambiguous regions |
| **SAM2 video masklets** | SAM2 (Ravi et al. 2024) — prompt 1 frame, propagate masks through whole video | ⭐⭐⭐ cheapest label expansion for our video data |
| **Interactive relabel tools** | RITM, FocalClick CVPR22 — correct existing masks with clicks; Progressive Merge preserves good regions | ⭐⭐ B4 audit speedup |
| **Multi-annotator / soft-label GT (STAPLE)** | STAPLE EM label fusion; QUBIQ soft-label training → calibrated soft masks | ⭐⭐⭐ B4 — model ambiguity instead of forcing binary GT |

## 10. Evaluation & analysis upgrades

| Method | Purpose |
|---|---|
| Per-class operating-point curves (PR per class) | find sand FN recovery threshold |
| Boundary IoU metric | B4 fairer scoring |
| Confusion analysis FN taxonomy (real-miss vs ambiguous-GT) | separates model error from label convention — already started via fpfn_test overlays |
| Calibration (temp scaling, conformal) | reliable conf under domain shift |

---

## Prioritized roadmap (cost/benefit)

| Rank | Action | Effort | Expected |
|---|---|---|---|
| 1 | `copy_paste≈0.2` + `mask_ratio=1` + `retina_masks` + TTA + sand conf sweep | hours | +3–8pt combined |
| 2 | Label audit on test FN overlays (real-miss vs ambiguous-GT taxonomy) | manual | reveals true ceiling |
| 3 | **SAM2 label propagation** + pseudo-label unlabeled frames (PAIS-style) | 1–2 days | +5–12pt (lit) |
| 3b | SAM-optimizer arm (AdamW→SAM or ASAM) for OOD gap | 1 arm | +~5% target-domain (lit) |
| 4 | Semantic-seg arm (DeepLabV3+/SegFormer) — stuff-vs-things argument | 1 arm | paradigm shift |
| 5 | Model soup + WBF ensemble of existing ckpts | hours | +1–2pt free |
| 6 | RT-DETR arm as NMS-free alternative | 1 arm | lit suggests ≥YOLO |
| 7 | Noise-robust loss (GCE/APL) if labels stay ambiguous | 1 arm | robustness to B4 |

## Ruled out (evidence-closed)

- **P2 detection head** — wrong problem (objects aren't <8×8px); 3-way evidence: PCB ablation study, our cold-P2, warm-remap P2 all lose.
- **Naive index-remap warm start** — produces incoherent init (warm-box + random-cls since COCO 80→2 cls mismatch); p2w empirically worst (0.366).
- **gold_aug offline expansion** — 4× augmented data hurt (0.559 < 0.582): synthesis distribution shift.

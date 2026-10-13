# Final Model v1 — stripe split benchmark (`gold_ws` / `bench_ws`)

วันที่: 2026-10-14

รายงานฉบับนี้รวมผล benchmark ชุด `bench_ws` — ชุดที่ YOLO n/s ได้ macro AP50 ~90% —
และประกาศ **Final Model v1** พร้อมเหตุผลการเลือก

## 1. Dataset / Split

| รายการ | ค่า |
|---|---|
| eval split | `data/gold_ws/test` — **88 ภาพ** |
| train source | `data/gold_ws_aug` — train **3,510** ภาพ (aug K=3/ภาพ + hard-group oversample K=5) |
| base split `gold_ws` | train 812 / valid 184 / test 88 |
| classes | `0=gravel`, `1=sand` |
| data yaml | `data/gold_ws/data.yaml`, `data/gold_ws_aug/data.yaml` |

> **"stripe split"** = ใน `gold_ws` เฟรมเรียงตำแหน่ง (pos) จากคลิปเดียวกันถูกแบ่งสลับลง
> train/valid/test ดู `results/gold_ws/frame_assignment.csv` ดังนั้นผลชุดนี้เป็น
> **seen-scene benchmark** — ฉากเดียวกันปรากฏข้าม split ไม่ใช่ unseen-site generalization

**Evaluation protocol** (`scripts/bench_common.py`): conf ≥ 0.05, match ด้วย
mask-IoU ≥ 0.5 (same class), per-class AP50 + macro AP50, evaluator เดียวกันทุกโมเดล

## 2. โมเดลทั้งหมดในชุดนี้

| โมเดล | ประเภท | checkpoint | train time |
|---|---|---|---|
| YOLO26n-seg | instance seg | `runs/ws/final/weights/best.pt` (6.3 MB) | 1.12 h |
| YOLO26s-seg | instance seg | `runs/bench_ws/yolo26s/weights/best.pt` (23 MB) | 1.54 h |
| Mask R-CNN R50-FPN v2 | two-stage instance seg | `results/bench_ws/mrcnn_best.pt` (178 MB) | 1.10 h |
| U-Net + watershed | semantic → instances | `results/bench_ws/unet_best.pt` (30 MB) | 0.99 h |
| YOLO26n-det (aux) | detection เท่านั้น | `runs/bench_ws/yolo26n_det/weights/best.pt` (20 MB) | — |

## 3. ผลบน `gold_ws/test` (mask AP50)

| โมเดล | gravel AP50 | sand AP50 | **macro AP50** | box AP50 (macro) |
|---|---:|---:|---:|---:|
| **Mask R-CNN** | 0.9644 | 0.9129 | **0.9386** | 0.925 |
| **YOLO26s-seg** | 0.9331 | 0.8806 | **0.9068** | 0.897 |
| **YOLO26n-seg** | 0.9419 | 0.8565 | **0.8992** | 0.889 |
| U-Net + watershed | 0.2856 | 0.2992 | 0.2924 | 0.269 |

aux: `yolo26n_det` (det-only) best valid box mAP50 = **0.9216** @ epoch 9 —
ไม่อยู่ใน seg leaderboard เพราะไม่มี mask

### Per-class detail (`*_eval.json`)

| โมเดล | class | P | R | F1 | TP | preds | GT |
|---|---|---:|---:|---:|---:|---:|---:|
| YOLO26n-seg | gravel | 0.947 | 0.884 | 0.914 | 168 | 200 | 181 |
| YOLO26n-seg | sand | 0.879 | 0.808 | 0.842 | 82 | 109 | 99 |
| YOLO26s-seg | gravel | 0.871 | 0.895 | 0.883 | 168 | 224 | 181 |
| YOLO26s-seg | sand | 0.914 | 0.747 | 0.822 | 86 | 121 | 99 |
| Mask R-CNN | gravel | 0.971 | 0.923 | 0.946 | 174 | 257 | 181 |
| Mask R-CNN | sand | 0.853 | 0.879 | 0.866 | 92 | 167 | 99 |
| U-Net | gravel | 0.403 | 0.503 | 0.447 | 113 | 511 | 181 |
| U-Net | sand | 0.369 | 0.586 | 0.453 | 60 | 277 | 99 |

### Size-bin recall (`table_sizebins.csv`)

| โมเดล | class | large | medium | small |
|---|---|---:|---:|---:|
| YOLO26n-seg | gravel | 0.953 | 0.899 | 0.833 |
| YOLO26n-seg | sand | 0.962 | 0.871 | 0.312 |
| YOLO26s-seg | gravel | 0.972 | 0.884 | 0.667 |
| YOLO26s-seg | sand | 0.962 | 0.903 | 0.500 |
| Mask R-CNN | gravel | 0.981 | 0.957 | 0.667 |
| Mask R-CNN | sand | 0.962 | 0.935 | 0.812 |
| U-Net | gravel | 0.726 | 0.522 | 0.000 |
| U-Net | sand | 0.865 | 0.419 | 0.125 |

## 4. Efficiency (`table_efficiency.csv`, RTX 3080 Ti)

| โมเดล | Params | GFLOPs | Latency | FPS | Size | Pi5 est. |
|---|---:|---:|---:|---:|---:|---:|
| **YOLO26n-seg** | 3.05 M | 22.87 | 6.26 ms | **159.7** | 6.6 MB | ~250 ms / 4.0 FPS |
| YOLO26s-seg | 11.44 M | 83.44 | 7.63 ms | 131.1 | 23.4 MB | ~458 ms / 2.2 FPS |
| Mask R-CNN | 46.36 M | 128.24* | 37.36 ms | 26.8 | 185.8 MB | ~3362 ms / 0.3 FPS |
| U-Net + ws | 7.76 M | 48.3 | 147.06 ms** | 6.8 | 31.1 MB | ~8824 ms / 0.11 FPS |

\* GFLOPs เฉพาะ backbone \*\* รวม watershed post-processing / Pi5 = estimate ไม่ใช่ผลวัดจริง

## 5. Training recipes

| โมเดล | base weights | imgsz | epochs (best) | batch | optimizer | aug |
|---|---|---|---|---|---|---|
| YOLO26n-seg | `weights/yolo26n-seg.pt` | 960 | 62 run, best ~54 | -1 | auto (SGD) | ทุก aug ปิด (mosaic/flip/mixup/copy_paste/HSV/geo = 0) |
| YOLO26s-seg | `weights/yolo26s-seg.pt` | 960 | 47 run, best 8 | -1 | auto | เหมือน n |
| YOLO26n-det | `weights/yolo26n.pt` | 960 | 44 run, best 9 | -1 | auto | เหมือน n |
| Mask R-CNN | `maskrcnn_resnet50_fpn_v2` (torchvision) | — | 12, best ep 4 (val AP50 0.9394) | 2 | SGD m=0.9 wd=5e-4 lr=0.005 step@8,11 | offline aug เท่านั้น |
| U-Net | scratch, 3 cls (bg/gravel/sand) | 512 | 40, best ep 10 (val 0.3956) | 8 | Adam lr=1e-3, BCE+Dice | offline aug เท่านั้น |

ทุกรัน seed=42, deterministic, patience 30 (YOLO)

## 6. Final Model v1 — การเลือก

**เกณฑ์**: deployment จริงเป้าหมายคือ edge/desktop real-time → ชั่งน้ำหนัก AP50 + ขนาด + latency

| ตำแหน่ง | โมเดล | เหตุผล |
|---|---|---|
| 🥇 **Final Model v1** | **YOLO26n-seg** (`runs/ws/final/weights/best.pt`) | macro AP50 0.8992 ต่ำกว่า s เพียง 0.0076 แต่เบากว่า 3.7× (6.6 MB, 3.05M params), เร็วสุด 159.7 FPS (RTX3080Ti), Pi5 ~4 FPS — เหมาะ deploy สุด |
| accuracy winner (YOLO) | YOLO26s-seg | AP50 สูงสุดใน YOLO (0.9068), sand AP50 ดีกว่า n (+0.024) — ใช้ถ้าต้องการ accuracy เพิ่มและรับ latency/ขนาดได้ |
| accuracy winner (overall) | Mask R-CNN | AP50 สูงสุด (0.9386), sand small-bin ดีสุด — แต่ 185.8 MB / 26.8 FPS / Pi5 ~0.3 FPS ไม่เหมาะ edge เก็บไว้เป็น reference |
| — | U-Net + watershed | AP50 0.2924 ต่ำกว่ามาก + ช้าสุด (watershed) — ตัดออก |

> ถ้า deployment target คือ server/GPU ที่ไม่จำกัดทรัพยากร ให้ใช้ **YOLO26s-seg** แทน

### Demo videos (YOLO-seg overlay บน 3 คลิปจริง)

วิดีโอ overlay mask + label (`gravel/sand conf`) render บนคลิปต้นฉบับจาก
`data/thesis_videos/` — อยู่ใน `results/bench_ws/video/`:

| clip | video |
|---|---|
| 32842c3f (595f @29fps, 540×960) | [▶ 32842c3f.mp4](../results/bench_ws/video/32842c3f.mp4) |
| 5d81688a (623f @30fps, 960×540) | [▶ 5d81688a.mp4](../results/bench_ws/video/5d81688a.mp4) |
| 0a48349e (1664f @29fps, 360×640) | [▶ 0a48349e.mp4](../results/bench_ws/video/0a48349e.mp4) |

<video src="../results/bench_ws/video/32842c3f.mp4" width="360" controls></video>
<video src="../results/bench_ws/video/5d81688a.mp4" width="360" controls></video>
<video src="../results/bench_ws/video/0a48349e.mp4" width="360" controls></video>

(`.avi` ไฟล์ดิบขนาดใหญ่ในโฟลเดอร์เดียวกันไม่ได้ commit — เหลือไว้เฉพาะ mp4)

## 7. Caveats

1. ผลชุดนี้เป็น **seen-scene (stripe) split** — ฉาก/คลิปเดียวกันปรากฏทั้ง train และ test
   ตัวเลข ~0.90–0.94 จึงเป็น easy-capped benchmark ไม่ใช่ unseen-site accuracy
   (LOGO/unseen-group diagnostic เคยตกลงเหลือ macro ~0.325 — ดู `docs/model_comparison.md` §3)
2. `bench_ws` เป็น protocol เดิม แยกจาก grouped-CV ใหม่ (`results/nb16_full_cv`, `gold_fixed_v1`)
   — ห้ามเอาตัวเลขสองชุดมาเทียบกันโดยตรง
3. มี seed เดียว (42) — ยังไม่ได้วัด uncertainty
4. latency Pi5 เป็น estimate เท่านั้น

## 8. Artifacts / Reproduce

- results: `results/bench_ws/` — สรุปเนื้อหาโฟลเดอร์:
  - `*_eval.json` — ผลเต็ม (per-class AP50/P/R/F1/tp + train_log) ของ 4 โมเดล: `yolo26n`, `yolo26s`, `mrcnn`, `unet`
  - `*_test_eval.json` — ผล test-only เวอร์ชันสั้นของทุกโมเดล
  - `table_models.csv` / `table_tradeoff.csv` / `table_box.csv` — leaderboard รวม mask AP50, P/R/F1 และ box AP50
  - `table_efficiency.csv` / `table_rpi5*.csv` / `efficiency.json` — params, GFLOPs, latency/FPS (3080 Ti + Pi5 est.)
  - `table_sizebins.csv` — recall แยก size bin (large/med/small)
  - `per_uuid_*.csv` — recall แยกราย uuid group (unseen-group diagnostic)
  - `fig16_*.png` / `fig_*.png` — tradeoff, per-uuid, pred grid, train curves, data split
  - `video/` — demo overlay mp4 3 คลิป (+ .avi ดิบที่ไม่ได้ commit)
  - `*_best.pt` / `*_last.pt` — checkpoint mrcnn/unet (`*.pt` ถูก gitignore); `preds_*.npy` — prediction cache (gitignored)
  - `yolo26s_log.json` — training log ของ yolo26s run
- scripts: `scripts/bench_common.py`, `bench_eval_yolo.py`, `bench_yolo26s.py`, `bench_mrcnn.py`, `bench_unet.py`, `make_gold_ws_aug.py`

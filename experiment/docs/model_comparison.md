# รายงานเปรียบเทียบโมเดลและข้อจำกัด

อัปเดตล่าสุด: 2026-10-13

ขอบเขต: รายงานนี้รวบรวมผลโมเดลหลายตัวจากหลาย protocol ของโปรเจกต์ aggregate segmentation (gravel vs sand) โดยแยกชัดเจนว่าตัวเลขใดเป็น seen-scene benchmark, grouped holdout หรือผล grouped CV ที่ยังรันไม่ครบ

## สรุปสั้น

- โปรเจกต์นี้เคยประเมินโมเดลอย่างน้อย **9 ตัว/variant**: YOLO26n-seg, YOLO26s-seg, Mask R-CNN, CondInst R50, SAM3.1 zero-shot, U-Net + watershed, YOLOv8n-seg, YOLOv8n-Ghost-seg, PatchSeg และ Cellpose-SAM
- ตัวเลขประมาณ **0.90–0.94 macro AP50** มาจาก `gold_ws` ซึ่งเป็น **seen-scene / development-inspected benchmark** ไม่ใช่หลักฐาน generalization ไปยังคลิปหรือไซต์ใหม่
- Leaderboard บน `gold/test` เดิมให้ YOLO26n-seg สูงสุดที่ **0.488 macro AP50** แต่ชุดนี้เคยถูกใช้ระหว่าง development แล้ว จึงไม่ใช่ untouched confirmatory test
- LOGO/unseen-group diagnostic ต่ำกว่ามาก: groupwise macro AP50 เฉลี่ยประมาณ **0.325** (ช่วง 0.254–0.448) ในไฟล์ `results/logo/logo_summary.csv`
- Grouped CV ใหม่บน `gold_fixed_v1` กำลังรันแบบ 60 runs; ณ snapshot ที่ commit รายงานนี้มีผลครบเฉพาะ `yolo26n_seg` fold 0 ทั้งสอง data arms จึง **ยังสรุปผล CV ทั้งหมดไม่ได้**

---

## 1. รายชื่อโมเดลที่เคยประเมิน

ตารางนี้คือ leaderboard เดิมจาก notes ของโปรเจกต์บน holdout เดียวกัน (`data/gold/test`) โดยใช้ macro AP50 เป็นหลัก ตัวเลขเป็น historical/archived comparison และไม่ควรผสานกับ protocol ใหม่โดยตรง

| โมเดล | ประเภท | macro AP50 เดิม | บทบาทในงาน |
|---|---|---:|---|
| YOLO26n-seg | instance segmentation | 0.488 | baseline/primary deployment candidate เดิม |
| Mask R-CNN | two-stage instance segmentation | 0.391 | upper-capacity baseline |
| CondInst R50 | instance segmentation | 0.379 | alternative architecture |
| SAM3.1 zero-shot | prompt/foundation model | ~0.37 | zero-shot diagnostic |
| U-Net + watershed | semantic/instance separation | 0.350 | classical segmentation pipeline |
| YOLOv8n-seg | instance segmentation | 0.336 | cross-version lightweight YOLO |
| YOLOv8n-Ghost-seg | lightweight instance segmentation | 0.184 | low-compute variant |
| PatchSeg | patch/grain segmentation | 0.179 | domain-adjacent method |
| Cellpose-SAM | cell/instance segmentation | ~0.001 | granularity mismatch diagnostic |

อ้างอิงเดิม: `experiment/AGENTS.md` และ `experiment/results/cv/cv_summary.csv`

---

## 2. Seen-scene benchmark (`gold_ws`)

ผลด้านล่างมาจาก `experiment/results/bench_ws/table_models.csv` และ efficiency table ซึ่งเป็น benchmark ที่เฟรม train/test มีฉากจากคลิปเดียวกัน จึงถือเป็น seen-scene/easy-capped result

| โมเดล | gravel AP50 | sand AP50 | macro AP50 | Params | Latency RTX 3080 Ti | ขนาดโมเดล |
|---|---:|---:|---:|---:|---:|---:|
| Mask R-CNN | 0.964 | 0.913 | **0.939** | 46.36 M | 37.36 ms | 185.8 MB |
| YOLO26s-seg | 0.933 | 0.881 | **0.907** | 11.44 M | 7.63 ms | 23.4 MB |
| YOLO26n-seg | 0.942 | 0.856 | **0.899** | 3.05 M | 6.26 ms | 6.6 MB |
| U-Net + watershed | 0.286 | 0.299 | **0.292** | 7.76 M | 147.06 ms | 31.1 MB |

การตีความ:

- **Mask R-CNN ชนะด้าน AP50** แต่หนักและช้ากว่า YOLO มาก
- **YOLO26n** ให้ trade-off ดีที่สุดสำหรับ deployment บน GPU desktop (ประมาณ 160 FPS ใน benchmark นี้)
- **YOLO26s** ได้ AP50 สูงกว่า `n` เล็กน้อย แต่ใช้พารามิเตอร์และ latency เพิ่มขึ้น
- ตัวเลขชุดนี้ไม่ควรอ้างเป็น unseen-site accuracy เพราะฉากเดียวกันปรากฏข้าม split

Artifacts:

- `experiment/results/bench_ws/table_models.csv`
- `experiment/results/bench_ws/table_tradeoff.csv`
- `experiment/results/bench_ws/table_efficiency.csv`

---

## 3. Seen-scene กับ unseen-group diagnostic

| ชุดประเมิน | gravel AP50 | sand AP50 | หมายเหตุ |
|---|---:|---:|---|
| `gold_ws` seen-scene test | 0.9419 | 0.8565 | macro AP50 = 0.8992 |
| Unseen LOGO mean (per-class report) | 0.5610 | 0.4436 | ไม่ใช่ groupwise macro โดยตรง |
| Unseen LOGO best per class | 0.8966 | 0.7157 | ค่าสูงสุดเกิดบน group ต่างกัน |

จาก `experiment/results/logo/logo_summary.csv` เมื่อคำนวณแบบ groupwise:

| metric | mean | min | max |
|---|---:|---:|---:|
| gravel AP50 | 0.408 | 0.000 | 0.8966 |
| sand AP50 | 0.242 | 0.000 | 0.7157 |
| macro AP50 | 0.325 | 0.254 | 0.448 |
| sand recall | 0.162 | 0.000 | 0.642 |

ข้อสรุป: คะแนน ~0.90 จึงเป็นผลของ benchmark ที่เห็นฉากคล้ายกันมาก เมื่อ hold out ทั้ง group/clip ผลตกลงชัดเจน โดยเฉพาะ sand

Artifacts:

- `experiment/results/ws/table_seen_vs_unseen.csv`
- `experiment/results/logo/logo_summary.csv`

---

## 4. Grouped CV ใหม่บน `gold_fixed_v1`

Grouped CV protocol ปัจจุบันใช้:

- dataset: `gold_fixed_v1`
- 5 grouped folds
- 6 recipes: `yolo26n_seg`, `yolo26s_seg`, `mrcnn`, `unet_ws`, `yolo26n_det`, `yolo26s_det`
- 2 data arms: `control`, `offline_aug`
- seed เดียว: `42`
- รวม **60 runs**
- batch size 2, FP32, AMP off
- budget ต่อ run เทียบ control 30 epochs ด้วย optimizer steps ที่เท่ากันระหว่างสอง arms
- checkpoint/threshold เลือกจาก inner validation เท่านั้น
- outer heldout ใช้เฉพาะ evaluation หลังเลือก checkpoint/threshold แล้ว

Snapshot ผลที่ commit ไว้ (fold 0 เท่านั้น):

| run | selected epoch | inner-val objective | heldout box AP | heldout mask AP | heldout mask AP50 | runtime |
|---|---:|---:|---:|---:|---:|---:|
| `yolo26n_seg` / control | 22 | inner-valid mask AP | 0.1754 | 0.1863 | 0.3309 | 17.4 min |
| `yolo26n_seg` / offline_aug | 11 | inner-valid mask AP | 0.1774 | 0.1954 | 0.3306 | 20.5 min |

อ่านผลชั่วคราวได้เพียงว่าใน fold 0 offline augmentation มี heldout mask AP สูงกว่า control ประมาณ **+0.0091** แต่ยังเป็นเพียง 1 fold, 1 seed และโมเดลเดียว จึงยังไม่ใช่ข้อสรุประดับทดลองทั้งหมด

Artifacts:

- `experiment/results/nb16_full_cv/27290ad076700f73261a/plan.json`
- `experiment/results/nb16_full_cv/27290ad076700f73261a/yolo26n_seg_fold0_seed42_control.json`
- `experiment/results/nb16_full_cv/27290ad076700f73261a/yolo26n_seg_fold0_seed42_offline_aug.json`

---

## 5. ข้อจำกัดสำคัญ

1. **ตัวเลขต่าง protocol เทียบกันไม่ได้โดยตรง** — `gold_ws` เป็น seen-scene benchmark, `gold/test` เป็น holdout เดิม, LOGO เป็น group-heldout diagnostic และ `gold_fixed_v1` CV เป็น protocol ใหม่ที่มี budget และ fold policy ต่างกัน
2. **ผล ~0.90–0.94 AP50 ไม่ใช่ unseen generalization** — benchmark นี้มีฉากจากคลิปเดียวกันข้าม train/test จึงเอื้อให้โมเดลจำพื้นหลัง มุมกล้อง และรูปทรงกองวัสดุได้
3. **test เดิมไม่ใช่ untouched confirmatory test** — เคยใช้ระหว่าง development/inspection แล้ว จึงเหมาะกับ development evidence มากกว่า final confirmation
4. **grouped CV ใหม่ยังไม่ครบ** — ณ snapshot นี้มีผลครบเพียง `yolo26n_seg` fold 0 สอง arms; ยังไม่ควรสรุป winner หรือ effect ของ augmentation
5. **มี seed เดียว** — ลดเวลารันแต่ไม่พอสำหรับวัด uncertainty จาก random initialization/ordering
6. **model comparison ไม่ได้ control ทุกอย่าง** — optimizer, pretrained weights, input size, framework และ post-processing ต่างกันตาม native recipe ของแต่ละโมเดล
7. **offline augmentation ถูกจำกัดเฉพาะ inner train เท่านั้น** — นี่คุม leakage ได้ แต่ augmented distribution ยังมาจากฉากเดิมและอาจไม่ช่วย unseen scene
8. **budget ปัจจุบันต่ำกว่า recipe เดิม** — full CV ใช้ 30 virtual epochs และ batch 2 เพื่อคุม matched steps และ VRAM ไม่ใช่ best-possible budget ของแต่ละโมเดล
9. **ไม่มีภาพ negative/empty-label ในชุด benchmark ปัจจุบัน** — จึงยังไม่ได้วัด false-positive บนฉากว่างหรือฉากที่ไม่มีวัสดุ
10. **label ambiguity ยังเป็น ceiling สำคัญ** — pile boundary, scatter บน grate และพื้นผิวรถบรรทุกทำให้ GT ไม่ชัดเจน โดยเฉพาะ sand
11. **Sand เป็นคลาสที่อ่อนแอที่สุด** — ผลหลายชุดแสดง sand AP/recall ต่ำกว่า gravel และใน CV fold 0 อยู่ใกล้ศูนย์
12. **Cellpose/SAM-like models ถูก metric ลงโทษจาก granularity mismatch** — โมเดล fragment-level เทียบกับ region-level GT ไม่ยุติธรรม ตัวเลขต่ำจึงไม่ได้หมายถึง foundation model ใช้ไม่ได้ทุกกรณี
13. **latency วัดบน RTX 3080 Ti เท่านั้น** — ตัวเลข Raspberry Pi 5 เป็น estimate ไม่ใช่ผลวัดจริง และ Jetson/edge deployment ยังไม่ได้ benchmark
14. **ยังไม่มี external site/event validation** — grouped CV เป็น development evidence ภายใน dataset เดิม ไม่ได้พิสูจน์ performance บนสถานที่ เวลา หรือกล้องใหม่
15. **event-level decision ยังไม่มี labeled clips** — การตัดสินระดับวิดีโอยังประเมินแบบ qualitative/unlabeled ไม่ได้วัด event accuracy แบบมี ground truth

---

## 6. วิธีใช้ผลอย่างปลอดภัย

- ใช้ `gold_ws` เพื่อเปรียบเทียบโมเดลในฉากคุ้นเคยและ sanity-check implementation เท่านั้น
- ใช้ `logo_summary.csv` และ grouped CV สำหรับคำถามเรื่อง generalization
- ใช้ YOLO26n เป็นตัวเลือก deployment เริ่มต้นจาก trade-off ขนาด/ความเร็ว แต่ต้องรอผล grouped CV ครบก่อนประกาศ winner
- ใช้ Mask R-CNN เป็น high-capacity reference เท่านั้น จนกว่าจะมีหลักฐาน unseen-group ที่ดีกว่าอย่างมีนัยสำคัญ
- รายงานทุกคะแนนพร้อมชื่อ protocol, split policy, metric และจำนวน seeds/folds เสมอ

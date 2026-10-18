# แผนการวิจัย: การแบ่งส่วนวัสดุมวลรวม (gravel/sand) และการตัดสินสถานะวัสดุจากภาพวิดีโอ

ฉบับ v0.2 (2026-10-18; ปรับหลัง Notebook 01) — เอกสารนี้กำหนดวิธีการทั้งหมดก่อนเริ่มการทดลอง (pre-registration ภายในโครงการ) การเปลี่ยนแปลงภายหลังต้องบันทึกในหัวข้อ 15 พร้อมเหตุผล

---

## 1. ขอบเขตและวัตถุประสงค์

งานนี้ออกแบบขั้นตอนการวิจัยใหม่สำหรับแบบจำลอง instance segmentation (Ultralytics YOLO26-seg) ที่ใช้จำแนกวัสดุมวลรวมสองชนิด ได้แก่ หิน (gravel) และทราย (sand) จากภาพกล้องที่ติดตั้งบริเวณจุดเทวัสดุของรถบรรทุก ผลลัพธ์ของระบบในการใช้งานจริงคือสถานะของวัสดุ ณ ช่วงเวลาหนึ่ง (ไม่มีวัสดุ / หิน / ทราย / ผสม) ไม่ใช่ขอบเขตของวัตถุแต่ละชิ้น

วัตถุประสงค์มีดังนี้

1. สร้างชุดข้อมูลระดับ Silver ที่ตรวจสอบได้ พร้อมการแบ่งชุดข้อมูลที่ป้องกันการรั่วไหลของข้อมูล (data leakage)
2. ศึกษา preprocessing และ augmentation ที่สอดคล้องกับสภาพการทำงานจริง รวมถึงสถานการณ์ฝนตก แสงน้อย และฝุ่นฟุ้ง
3. สร้าง baseline ที่ทำซ้ำได้ และทดลองปรับปรุงแบบควบคุมตัวแปร (optimizer, attention module, quantization)
4. ประเมินผลสองระดับ คือ ระดับ instance (มาตรฐาน COCO) และระดับสถานะของเฟรมและวิดีโอ (ตรงกับการใช้งานจริง)
5. ประเมินบนชุด holdout ภายนอกหนึ่งครั้งหลังจากตรึงแบบจำลองแล้ว และเสนอขั้นตอนการถ่ายภาพที่เป็นมาตรฐาน

ข้อจำกัดเชิงกระบวนการ

- งานวิเคราะห์ ประมวลผล ฝึก ประเมิน และ benchmark ทั้งหมดอยู่ใน Jupyter notebook สี่ไฟล์ ไม่มีสคริปต์ Python แยก (อนุญาตเฉพาะฟังก์ชันช่วยภายใน cell)
- ทุก cell ที่รันต้องเก็บ output ไว้ใน notebook เป็นหลักฐาน
- คำอธิบาย Markdown ใช้ภาษาไทยทางการ คำศัพท์เทคนิคใช้ภาษาอังกฤษตามมาตรฐาน
- ทุกการตัดสินใจเชิงวิธีการมีหัวข้อย่อย Method / Rationale / Reference / Implementation / Limitation

---

## 2. คำถามวิจัย

| # | คำถาม | Notebook ที่ตอบ |
|---|---|---|
| RQ1 | กลยุทธ์การแบ่งชุดข้อมูลแบบใดให้ค่าประมาณ generalization ที่ป้องกันการรั่วไหลได้ดีที่สุด | 01 |
| RQ2 | preprocessing วิธีใดปรับปรุงภาพโดยไม่ทำลายเนื้อผิวและสีของวัสดุ | 02 |
| RQ3 | augmentation วิธีใดเพิ่มประสิทธิภาพและความทนทานของ segmentation | 02 |
| RQ4 | AdamW และ MuSGD ต่างกันอย่างไรภายใต้เงื่อนไขที่ควบคุม | 03 |
| RQ5 | ชั้นต่าง ๆ ของ YOLO แปลงภาพวัสดุอย่างไร และ feature map จริงแสดงอะไร | 03 |
| RQ6 | ชั้นใดใช้ทรัพยากรคำนวณมากที่สุด และ FLOPs สัมพันธ์กับ latency จริงอย่างไร | 03 |
| RQ7 | quantization ให้ trade-off ระหว่างความแม่นยำและประสิทธิภาพอย่างไร บน RTX 3080 Ti และ Raspberry Pi 5 | 03 |
| RQ8 | CBAM หรือ ECA ให้ผลดีกว่า baseline อย่างวัดได้หรือไม่ | 03 |
| RQ9 | การทำภาพให้เป็นมาตรฐานช่วย generalization บนข้อมูล holdout หรือไม่ | 04 |
| RQ10 | การตั้งค่าใดสมดุลที่สุดระหว่างความแม่นยำ ความทนทาน ต้นทุนการคำนวณ และความเป็นไปได้ในการใช้งาน | 03, 04 |
| RQ11 | ความแม่นยำระดับสถานะ (frame/video state) ต่างจากความแม่นยำระดับ instance อย่างไร | 03, 04 |

---

## 3. สภาพแวดล้อมและทรัพยากร

| รายการ | ค่า |
|---|---|
| GPU | NVIDIA GeForce RTX 3080 Ti 12 GB (sm_86), driver 580.178.04 |
| CPU / RAM | 20 threads / 31 GB |
| Environment | `implement/research/.venv` (uv), Python 3.12.3 |
| แพ็กเกจหลัก | torch 2.14.1+cu130, torchvision 0.29.1, ultralytics 8.4.180, albumentations 2.0.8, sahi 0.12.8, onnx 1.23.2, onnxruntime 1.31.0, openvino 2026.4.1, pycocotools 2.0.11, shapely 2.2.0, imagehash 4.3.2 |
| รายการแพ็กเกจทั้งหมด | `implement/research/requirements.lock.txt` (181 รายการ) |
| Jupyter kernel | `aggregate-research` |
| แพ็กเกจที่ต้องติดตั้งเพิ่มใน Notebook 03 | TensorRT (FP16/INT8 บน 3080 Ti), ncnn (export สำหรับ RPi5) |
| อุปกรณ์ปลายทาง | RTX 3080 Ti (วัดในเครื่องนี้) และ Raspberry Pi 5 (ผู้วิจัยวัดเองบนบอร์ดจริงด้วย cell ที่เตรียมไว้) |
| พื้นที่ดิสก์ | ประมาณ 171 GB ว่าง |

หมายเหตุ: ultralytics 8.4.180 ต่างจากงานเดิม (8.4.171) และ torch ต่างจากงานเดิม (2.11.0) ผลจึงเทียบกับงานเดิมได้เชิงแนวโน้มเท่านั้น

---

## 4. ผลการสำรวจข้อมูลเบื้องต้น (ก่อนเริ่ม Notebook 01)

### 4.1 ชุดพัฒนา (`implement/data/bronze`)

- ที่มา: Roboflow `ilovefoods-workspace-qd0ut/aggregate_rock-and-sand` เวอร์ชัน 5, CC BY 4.0, export 2026-10-08
- ไฟล์ทั้งหมดตรงกับ `experiment/data/bronze` ทุกไบต์ (`diff -rq` ไม่พบความต่าง) จึงเป็นชุดเดียวกับงานเดิม
- 1,089 ภาพ: train 751 / valid 229 / test 109; label ครบทุกภาพ; มีภาพที่ไม่มี annotation 5 ภาพ
- คลาส: 0 = gravel, 1 = sand
- README ของ Roboflow ระบุว่า "3 versions of each source image" แต่ไม่ได้ระบุ augmentation ใด ๆ

ขนาดภาพ (สูง × กว้าง) มีหลายแบบ ซึ่งสัมพันธ์กับแหล่งวิดีโอ

| ขนาด | จำนวน |
|---|---:|
| 640 × 360 | 511 |
| 960 × 540 | 251 |
| 768 × 432 | 200 |
| 432 × 768 | 90 |
| 540 × 960 | 21 |
| อื่น ๆ (1706 × 960, 1440 × 1080, 284 × 160 ฯลฯ) | 16 |

กลุ่มแหล่งที่มา (group key = UUID ของวิดีโอจากชื่อไฟล์) และคลาสที่ปรากฏ

| Group | ภาพ | train/valid/test เดิม | gravel inst. | sand inst. | ลักษณะ |
|---|---:|---|---:|---:|---|
| 2be0c9f6 | 315 | 215/68/32 | 0 | 2179 | sand ล้วน, ใหญ่ที่สุด (29%) |
| 0f964fdb | 90 | 62/18/10 | 366 | 0 | gravel ล้วน |
| 0a48349e | 81 | 57/16/8 | 0 | 423 | sand ล้วน, เป็นคลิปใน `thesis_videos` |
| 96ecd4fe | 66 | 46/14/6 | 183 | 0 | gravel ล้วน |
| 1991989f | 65 | 43/14/8 | 480 | 68 | ผสม (gravel เด่น) |
| d56fcb3c | 57 | 39/12/6 | 268 | 0 | gravel ล้วน |
| f9b12ec3 | 56 | 38/12/6 | 217 | 0 | gravel ล้วน |
| 6459c886 | 49 | 36/9/4 | 0 | 200 | sand ล้วน |
| 4a7f29af | 48 | 34/11/3 | 71 | 115 | ผสม |
| 6f40fbbd | 47 | 33/9/5 | 324 | 0 | gravel ล้วน |
| fa45e6fb | 42 | 28/10/4 | 234 | 0 | gravel ล้วน |
| 07451999 | 38 | 27/7/4 | 137 | 6 | gravel เด่น |
| bbf45eda | 36 | 23/10/3 | 4 | 164 | sand เด่น |
| b029e315 | 31 | 22/6/3 | 278 | 0 | gravel ล้วน |
| 8f435cb5 | 28 | 20/5/3 | 152 | 0 | gravel ล้วน |
| 5d81688a | 21 | 15/4/2 | 78 | 0 | gravel ล้วน, เป็นคลิปใน `thesis_videos` |
| ภาพเดี่ยว/กลุ่มเล็ก 14 กลุ่ม | 21 | กระจาย | — | — | ภาพนิ่งจากแหล่งต่าง ๆ |

ข้อสังเกตสำคัญ

1. **การแบ่งเดิมของ Roboflow รั่วไหลระดับวิดีโอ** ทุกวิดีโอหลักปรากฏทั้งใน train, valid และ test
2. **คลาสผูกกับวิดีโอ (confounding)** วิดีโอส่วนใหญ่มีวัสดุชนิดเดียว แบบจำลองจึงอาจเรียนรู้ฉากหรือรถแทนเนื้อวัสดุ
3. **กลุ่มที่มีทรายมีน้อยมาก** มีเพียงประมาณ 5 วิดีโอ (2be0c9f6, 0a48349e, 6459c886, bbf45eda, 4a7f29af) ทำให้การแบ่งแบบกลุ่มมีความแปรปรวนสูง และจำกัดจำนวน fold ที่ทำได้
4. **ความละเอียดภาพผูกกับแหล่งที่มา** อาจเป็นทางลัด (shortcut) ให้แบบจำลองใช้แยกคลาส
5. **ชื่อไฟล์ซ้ำ 143 ชื่อ (stem เดียวกัน 2–3 ไฟล์)** จากการตรวจด้วยตาและผลต่างพิกเซลเฉลี่ย 23–72 ระดับสี พบว่าไม่ใช่ภาพ augment ของภาพเดียวกัน แต่เป็นเฟรมต่างเวลาของวิดีโอเดียวกัน ซึ่งน่าจะเกิดจากการตัดเฟรมวิดีโอมากกว่าหนึ่งครั้ง มี 65 คู่ที่อยู่ข้าม split เดิม การแบ่งตามกลุ่มวิดีโอแก้ปัญหานี้ได้ ข้อสรุปนี้ต้องยืนยันด้วย perceptual hash ใน Notebook 01
6. **ไม่มีภาพที่ไม่มีวัสดุ (negative)** นอกจาก 5 ภาพที่ label ว่าง จึงวัด false positive บนฉากว่างจากชุดพัฒนาไม่ได้

### 4.2 ชุด holdout (`implement/data/bronze/holdout`)

| ไฟล์ | จำนวนภาพ (โดยประมาณ) | คลาสในไฟล์ | การใช้ |
|---|---:|---|---|
| Piled Materials.v2i.yolo26.zip | ~1,095 (train/valid/test ของ Roboflow) | Rock, Sand | ประเมิน L1 และ L2; map Rock→gravel, Sand→sand |
| Separate Rock-Sand.v6i.yolo26.zip | ~182 | Road Cone, Rock, Sand, Scraper | ประเมิน L1 และ L2; map Rock→gravel, Sand→sand, คลาสอื่นเป็น ignore |
| S02.zip | 66 | ไม่มี label (1280 × 570) | วิเคราะห์เชิงคุณภาพ การกระจายของ prediction และ false positive |

ไฟล์ zip ทั้งสามตรงกับไฟล์ที่ notebook 18 ของงานเดิมเคยใช้ประเมินแล้ว ชุด holdout นี้จึงไม่ใช่ข้อมูลที่ยังไม่เคยเห็นโดยสมบูรณ์ ต้องเปิดเผยในวิทยานิพนธ์ และห้ามใช้ผลการประเมินเดิมเป็นเกณฑ์เลือกวิธีใด ๆ ในงานนี้

### 4.3 วิดีโอสำหรับประเมินระดับเวลา (`experiment/data/thesis_videos`)

| คลิป | ความยาว | ความสัมพันธ์กับชุดพัฒนา |
|---|---|---|
| 0a48349e | 56 วินาที | เป็นกลุ่มเดียวกับภาพ label 81 ภาพ (sand) |
| 5d81688a | 21 วินาที | เป็นกลุ่มเดียวกับภาพ label 21 ภาพ (gravel) |
| 32842c3f | 20 วินาที | ไม่อยู่ในชุดพัฒนา (unseen) |

ถ้าจะใช้ 0a48349e และ 5d81688a ประเมินระดับเวลาโดยไม่รั่วไหล ต้องกำหนดให้สองกลุ่มนี้อยู่ใน test ตั้งแต่การแบ่งข้อมูล (หัวข้อ 7.4)

---

## 5. บทเรียนจากงานเดิม

| # | สิ่งที่พบ | แหล่ง | ผลต่อแผน |
|---|---|---|---|
| 1 | ผลบน val (mask mAP50-95 0.557) สูงกว่าบน test (0.359) มาก | `results/p2_ablation.csv` | val ต้องแบ่งแบบกลุ่มด้วย ไม่ใช่เฉพาะ test |
| 2 | ประเมินแบบ leave-one-group-out ได้ macro AP50 เฉลี่ย 0.325 (0.254–0.448) เทียบกับ 0.90 บน seen-scene | `results/logo/logo_summary.csv` | ตัวเลขจาก split ที่รั่วไหลใช้อ้าง generalization ไม่ได้ |
| 3 | offline augmentation เพิ่ม AP50 บน holdout (Piled 0.049→0.090, Separate 0.122→0.155; 5 runs) แต่ลด val ภายใน (0.559 เทียบกับ 0.582) | `results/nb18_external/…/summary_clean_curated.csv`, `results/data_arms.csv` | ใช้ augmentation ระดับกลาง และวัดความทนทานด้วยชุด val-C แยก |
| 4 | arm `aug_strong` ได้คะแนนต่ำกว่า | `results/improve_arms.csv` | ยืนยันว่าไม่ควรใช้ augmentation รุนแรง |
| 5 | เทียบ optimizer จาก seed เดียว 30 epochs; MuSGD ได้ค่าดีที่สุดที่ epoch 29/30 (ยังไม่ลู่เข้า) | `results/optimizer_bakeoff.csv` | ใช้งบการฝึกที่ลู่เข้า, lr ตามสูตรของแต่ละ optimizer, 3 seeds |
| 6 | P2 head โอน pretrained weights ได้ 34% เทียบกับ 88% ของ baseline | `docs/methods_survey.md` (B6) | การแทรก CBAM/ECA ต้องรายงานสัดส่วน weights ที่โอนได้ |
| 7 | P2 head, SAHI, ขยายภาพตอน inference, model soup และ CLAHE ไม่ช่วย | `README.md`, `results/data_arms.csv` | ไม่ทำซ้ำเป็นการทดลองหลัก ทดสอบยืนยันสั้น ๆ และอ้างผลเดิม |
| 8 | FN เป็นข้อผิดพลาดหลัก (no_pred 38%, conf_cut 32%) และ sand อ่อนที่สุด | `results/fn_rca.csv` | วิเคราะห์ FN/FP แยกตามขนาดและคลาส และกำหนด threshold จาก val |
| 9 | มีหลาย protocol การประเมินปนกัน (strict/sieved/void) | `results/eval3_protocols.csv` | ใช้ evaluator หลักตัวเดียว และรายงานแบบ strict คู่กับแบบ ignore |
| 10 | GT ส่วนที่เป็นชิ้นเล็กหรือชิ้นเรียวยาว (sliver) 79/348 ชิ้นใน test เดิม; ตัดออกแล้วคะแนนเพิ่มประมาณ 9 จุด | `README.md` | กำหนดกฎ Silver ล่วงหน้าและวิเคราะห์ความไวต่อค่า threshold (หัวข้อ 6) |
| 11 | grouped CV fold 0 ได้ heldout mask AP เพียง 0.19 | `docs/model_comparison.md` | ต้องรายงานช่วงความเชื่อมั่น ไม่ใช่ค่าจากชุดทดสอบเดียว |
| 12 | งานเดิมรันบน ROCm ด้วย batch 2, FP32 และปิด AMP | `docs/model_comparison.md` | บน CUDA ใช้ AMP และหาขนาด batch ด้วย autobatch |
| 13 | ขอบเขตกองวัสดุไม่ชัดเจน และเป็นเพดานของคะแนน | `docs/methods_survey.md` (B4) | รายงาน Boundary IoU เพิ่ม |
| 14 | การตัดสินระดับวิดีโอไม่เคยวัดเทียบกับ label | `README.md` §8 | เพิ่มการประเมินระดับ L3 ด้วย label ช่วงเวลา (หัวข้อ 9.3) |

---

## 6. ชุดข้อมูล Silver

### 6.1 หลักการ

- กฎการทำความสะอาดกำหนดจากข้อมูลและการตรวจด้วยตาเท่านั้น ห้ามใช้ผลของแบบจำลองหรือผลบน test เป็นเกณฑ์
- ใช้กฎเดียวกันกับทุก split และกับชุด holdout
- เก็บชุด bronze ไว้โดยไม่แก้ไข Silver เป็นชุดใหม่ที่มีเวอร์ชันและ checksum

### 6.2 ขั้นตอน

1. **ตรวจความถูกต้อง:** ตรวจรูปแบบไฟล์ label, class ID, พิกัดในช่วง [0, 1], จำนวนจุดยอดอย่างน้อย 3 จุด และหา polygon ที่ตัดกันเอง (ตรวจด้วย `shapely.is_valid`)
2. **ซ่อม polygon:** ใช้ `shapely.make_valid` และเก็บเฉพาะชิ้นส่วนหลักเมื่อ polygon แตกเป็นหลายชิ้น
3. **ตัดเศษและหนาม (morphological opening):** ทำ `buffer(-ε).buffer(+ε)` เพื่อตัดส่วนที่แคบกว่า 2ε
4. **รวม annotation ซ้ำ:** instance คลาสเดียวกันที่ IoMin ≥ τ_dup ถือเป็นการ label ซ้ำ ให้เก็บชิ้นที่ใหญ่ที่สุด
5. **ลบ instance เล็ก:** ลบชิ้นที่มีพื้นที่น้อยกว่า a_min (สัดส่วนของพื้นที่ภาพ) หรือมีอัตราส่วนด้านยาวต่อด้านสั้นมากกว่า r_max

### 6.3 การวิเคราะห์ความไวต่อค่า threshold

| พารามิเตอร์ | ค่าที่ทดสอบ |
|---|---|
| a_min | 0.05%, 0.1%, 0.25%, 0.5% ของพื้นที่ภาพ |
| r_max | 6, 8, 10, ไม่จำกัด |
| ε | 0, 0.25%, 0.5% ของด้านสั้นของภาพ |
| τ_dup | 0.5, 0.7, 0.9 |

สำหรับแต่ละค่า รายงานจำนวนและสัดส่วนของ instance ที่ถูกลบ แยกตามคลาสและกลุ่ม การเปลี่ยนแปลงของพื้นที่ mask รวม และภาพ overlay ของสิ่งที่ถูกลบ (สีแดงเส้นประ)

เกณฑ์เลือกค่าที่ใช้จริง

1. ขนาดเล็กที่สุดที่แบบจำลองมีโอกาสแยกได้: feature map ระดับ P3 มี stride 8 ที่ input 640 จึงมีความละเอียด 1 cell ต่อ 8 × 8 พิกเซล หรือประมาณ 0.016% ของพื้นที่ภาพ 640 × 640 ค่า a_min ควรมากกว่านี้หลายเท่า
2. ตำแหน่งที่จำนวน instance ที่ถูกลบเริ่มเพิ่มชันในกราฟความไว
3. การตรวจด้วยตา ภาพ overlay ต้องไม่แสดงว่ากฎลบบริเวณวัสดุจริงที่มีขนาดใหญ่

### 6.4 การจัดการ instance เล็กในการประเมิน

- **ชุดฝึก:** ลบ instance ที่ต่ำกว่าเกณฑ์ออกจาก label
- **ชุดประเมิน (val, test, holdout):** ไม่ลบ แต่กำหนดเป็น ignore region ตามแนวทาง `iscrowd` ของ COCO ผลคือ GT ส่วนนี้ไม่นับเป็น FN และ prediction ที่ตรงกับส่วนนี้ไม่นับเป็น FP
- **การรายงาน:** รายงานสอง protocol คือ strict (นับ GT ทุกชิ้น) และ ignore (ใช้ ignore region) เพื่อแสดงผลของชิ้นเล็กต่อคะแนนอย่างโปร่งใส

---

## 7. การแบ่งชุดข้อมูล

### 7.1 Group key

ใช้ regex ดึง UUID ของวิดีโอจากชื่อไฟล์ ตัด suffix การดาวน์โหลดซ้ำ เช่น `-1-` ออก ภาพนิ่งที่ไม่มี UUID ของวิดีโอให้เป็นกลุ่มเดี่ยว จากนั้นรวมกลุ่มเพิ่มเมื่อพบภาพเกือบซ้ำข้ามกลุ่ม (pHash Hamming distance ≤ d_nd โดยกำหนด d_nd จากการกระจายของระยะภายในวิดีโอเดียวกัน) ด้วย union-find

### 7.2 กลยุทธ์ที่เปรียบเทียบ

| กลยุทธ์ | คำอธิบาย | สิ่งที่วัด |
|---|---|---|
| S0 Random (Roboflow เดิม) | สุ่มระดับภาพ | สัดส่วนกลุ่มและคู่ภาพเกือบซ้ำที่ข้าม split |
| S1 GroupShuffleSplit | สุ่มระดับกลุ่ม | สมดุลของคลาสและขนาดของแต่ละ split |
| S2 StratifiedGroupKFold | แบ่งระดับกลุ่มโดยคุมสัดส่วนคลาส | สมดุลของคลาสและความแปรปรวนระหว่าง fold |
| S3 Constrained group split | ค้นหาการแบ่งระดับกลุ่มภายใต้เงื่อนไขที่กำหนด (หัวข้อ 7.4) ด้วย seed คงที่ | ใช้เป็นการแบ่งหลักที่เสนอ |

ตัวชี้วัดของการแบ่ง

- จำนวนกลุ่มและคู่ภาพเกือบซ้ำที่อยู่ข้าม split (ต้องเป็นศูนย์)
- ความต่างของสัดส่วนคลาสระหว่าง split (Jensen–Shannon divergence)
- สัดส่วนภาพของแต่ละ split เทียบกับเป้าหมาย
- จำนวนกลุ่มของแต่ละคลาสในแต่ละ split

### 7.3 สัดส่วน

ชุดข้อมูลมีกลุ่มทรายเพียงประมาณ 5 กลุ่ม การทดสอบสัดส่วนหลายแบบจึงมีประโยชน์จำกัด เป้าหมายคือ test 15–20% และ val 15–20% ของภาพ โดยให้ทุก split มีกลุ่มของทั้งสองคลาสอย่างน้อยหนึ่งกลุ่ม และรายงานสัดส่วนจริงที่ได้ (การแบ่งระดับกลุ่มทำให้ได้สัดส่วนตรงเป้าไม่ได้)

### 7.4 เงื่อนไขของการแบ่งหลัก (S3)

1. กลุ่ม 0a48349e และ 5d81688a อยู่ใน test เพื่อให้ประเมินวิดีโอใน `thesis_videos` ได้โดยไม่รั่วไหล (กำหนดล่วงหน้าเพราะการประเมินระดับ L3 ต้องใช้)
2. กลุ่ม 2be0c9f6 (29% ของภาพ) อยู่ใน train เพราะถ้าอยู่ใน val หรือ test จะครอบงำค่าเฉลี่ยของ split นั้น
3. val และ test มีกลุ่มที่มี sand เด่นอย่างน้อยหนึ่งกลุ่ม และกลุ่มผสมรวมกันอย่างน้อยหนึ่งกลุ่ม
4. ค้นหาจากการสุ่มกำหนดกลุ่มที่เหลือ 10,000 ครั้งด้วย seed 42 แล้วเลือกการแบ่งที่สัดส่วนภาพใกล้เป้าหมายที่สุดภายใต้เงื่อนไขข้างต้น บันทึกการแบ่งที่ได้คะแนนใกล้เคียงไว้ด้วยเพื่อแสดงความไวของการแบ่ง

### 7.5 Grouped cross-validation (สำหรับการยืนยันขั้นสุดท้าย)

ข้อมูลที่ไม่ใช่ test แบ่งเป็น 4 fold ด้วย StratifiedGroupKFold จำนวน fold จำกัดด้วยกลุ่มทรายที่เหลือ ใช้ fold นี้เฉพาะยืนยันผลของการตั้งค่าที่เลือกแล้วเทียบกับ baseline ไม่ใช้คัดกรอง arm

### 7.6 การตรึง test

หลังจากผ่านด่านตรวจของ Notebook 01 รายชื่อภาพใน test จะถูกบันทึกพร้อม SHA-256 ใน `manifests/` และ Notebook 02 และ 03 จะไม่โหลดภาพ test ยกเว้นใน cell ประเมินผลสุดท้ายที่ระบุชัดเจน

---

## 8. Preprocessing และ augmentation (Notebook 02)

### 8.1 ชุดภาพตัวอย่าง

เลือก 8 ภาพจากชุด train ด้วยวิธีที่บันทึกไว้ ได้แก่ ภาพละหนึ่งภาพจาก 8 ช่องของตาราง (คลาส: gravel / sand / ผสม) × (ความสว่าง: ต่ำ / กลาง / สูง) × (ความละเอียด: ต่ำ / สูง) ภายในแต่ละช่องเลือกภาพที่ใกล้ค่า median ของช่องนั้นที่สุด ใช้ seed 42 และไม่ใช้ภาพ val หรือ test

### 8.2 Preprocessing ที่ทดสอบ

| วิธี | พารามิเตอร์ | แหล่งอ้างอิง | สิ่งที่วัด |
|---|---|---|---|
| Baseline (ภาพดิบ + letterbox 640) | — | Ultralytics docs | — |
| Letterbox เทียบกับ resize ที่ไม่รักษาอัตราส่วน | 640 | Ultralytics docs | การบิดเบือนรูปทรงของ polygon |
| Gray-world white balance | — | Buchsbaum 1980 | ΔE2000 ของสีวัสดุ, การเปลี่ยนแปลงของ histogram ช่อง a*b* |
| CLAHE บนช่อง L (LAB) | clip 1.5/2/3, grid 8 | Zuiderveld 1994; Pizer et al. 1987 | RMS contrast, entropy, การขยาย noise |
| Gamma / auto-exposure | γ จากค่าเฉลี่ยความสว่าง | มาตรฐานการประมวลผลภาพ | ความสว่างเฉลี่ยและการตัดปลาย histogram (clipping) |
| Denoise (fastNlMeans หรือ bilateral) | h ต่ำ | Buades et al. 2005 | Laplacian variance และ GLCM ก่อน/หลัง (การสูญเสียเนื้อผิว) |
| Unsharp mask | σ 1, amount 0.5 | มาตรฐานการประมวลผลภาพ | การขยาย noise และ halo |

ตัวชี้วัดคุณภาพภาพที่ไม่ต้องใช้ GT: ความสว่างเฉลี่ย, RMS contrast, Laplacian variance (ความคมชัด), ค่า GLCM contrast/homogeneity (เนื้อผิว), ΔE2000 ของบริเวณวัสดุก่อน/หลัง (ความคงอยู่ของสี) และสัดส่วนพิกเซลที่ถูกตัดปลาย

ผลต่อแบบจำลอง: ทดสอบเฉพาะวิธีที่ผ่านเกณฑ์คุณภาพ ด้วย arm ฝึกเดียวต่อวิธี และวัดบน val ส่วน CLAHE มีผลเดิมว่าไม่ช่วย (0.574 เทียบกับ 0.582) จึงทดสอบยืนยันสั้น ๆ เท่านั้น

### 8.3 การกำหนดระดับความแรงของ augmentation

ใช้แนวทางระดับความแรง 5 ระดับของ ImageNet-C (Hendrycks & Dietterich, ICLR 2019) และ COCO-C (Michaelis et al., 2019) โดยนิยามระดับ 1–5 ของแต่ละสถานการณ์

- **การฝึก:** สุ่มจากระดับ 1–3 (ระดับกลาง)
- **การประเมินความทนทาน (val-C):** ใช้ครบทั้ง 5 ระดับ

### 8.4 Augmentation ที่ทดสอบ

| สถานการณ์ / วิธี | แบบจำลองและพารามิเตอร์สำหรับการฝึก (ระดับ 1–3) | แหล่งอ้างอิง | ความสอดคล้องกับสภาพจริง |
|---|---|---|---|
| ฝน | เส้นสายฝน (streak) ความหนาแน่นต่ำถึงปานกลาง ทำมุม 75–105° ความยาว 8–22 px, blur เล็กน้อย, contrast ลด 5–15% | Tremblay et al., IJCV 2021; Halder et al., ICCV 2019 | ลานเทวัสดุกลางแจ้ง; ไม่ใช่ physics-based rendering เต็มรูป |
| แสงน้อย (gamma) | `I = β·(α·I)^γ` โดย α~U(0.9,1), β~U(0.7,0.9), γ~U(1.4,2.2) และ noise แบบ Gaussian–Poisson ระดับต่ำ | Lv et al., arXiv 1908.00682 (ใช้ γ~U(1.5,5), β~U(0.5,1)) | ช่วงเช้ามืดและเย็น; งานนี้ใช้ครึ่งล่างของช่วงในงานต้นฉบับ |
| ฝุ่นฟุ้ง | Koschmieder `I = J·t + A(1−t)` โดย t~U(0.65,0.85), airlight A สีเหลืองน้ำตาล, t สม่ำเสมอหรือไล่ระดับแนวตั้ง | Sakaridis et al., IJCV 2018; Shi et al., Sensors 2022 | ฝุ่นจากการเทวัสดุ; ไม่มีแผนที่ความลึกจริง |
| Photometric ทั่วไป | HSV (ค่าเริ่มต้นของ Ultralytics), JPEG quality 60–90, Gaussian noise σ ≤ 5 | Ultralytics docs; Buslaev et al., Information 2020 | การบีบอัดวิดีโอและ sensor noise |
| Blur | motion blur kernel 3–7, Gaussian σ 0.1–1.2 | Hendrycks & Dietterich 2019 | การสั่นของกล้องและวัสดุที่เคลื่อนที่ |
| Geometric | fliplr 0.5, หมุน ±10°, translate 0.1, scale 0.5; flipud เฉพาะเมื่อ EDA ยืนยันว่ามุมกล้องไม่มีทิศบน-ล่างที่มีความหมาย | Ultralytics docs | กล้องติดตั้งตายตัว มุมมองค่อนข้างคงที่ |
| Mosaic / MixUp | mosaic 1.0 (ปิด 10 epoch สุดท้าย), mixup 0 | Bochkovskiy et al., 2020; Zhang et al., ICLR 2018 | Mosaic เป็นค่าเริ่มต้นของ Ultralytics |
| Copy-Paste | copy_paste 0.3, mode flip | Ghiasi et al., CVPR 2021 | วางวัสดุบนพื้นรถหลายแบบ |

การตรวจความถูกต้องของ mask หลังการแปลง

- การแปลงเชิงเรขาคณิตต้องแปลง polygon ด้วย matrix เดียวกับภาพ แล้วตรวจด้วยการ rasterize mask เทียบกับภาพที่แปลงแล้ว
- polygon ที่หลุดขอบภาพต้อง clip และตรวจ `is_valid`
- การแปลงด้านแสงและสีต้องไม่เปลี่ยน polygon

### 8.5 การแยก offline และ online

| ประเภท | วิธี | ใช้กับ |
|---|---|---|
| Offline (สร้างไฟล์ใหม่) | ฝน, แสงน้อย, ฝุ่น, photometric ทั่วไป และ blur | train เท่านั้น; ภาพที่สร้างใช้ group เดียวกับภาพต้นฉบับ |
| Online (ระหว่างฝึก) | geometric, HSV, mosaic, copy-paste | train เท่านั้น |
| ชุดประเมินความทนทาน val-C | ฝน, แสงน้อย, ฝุ่น ระดับ 1–5 | ใช้เลือกวิธี; ไม่ใช้ฝึก |
| ไม่มีการแปลง | — | val, test, holdout (ยกเว้น preprocessing ที่เลือกใช้กับทุกชุดเหมือนกัน) |

ข้อจำกัดของ val-C: arm ที่ฝึกด้วยตัวสร้างภาพตัวเดียวกับ val-C จะได้คะแนนสูงเกินจริงบน val-C ผลต่อสภาพจริงต้องดูจาก holdout ใน Notebook 04

### 8.6 การควบคุมงบการฝึก

ทุก arm ได้จำนวน optimizer step เท่ากัน (iteration-matched) arm ที่ขยายข้อมูลแบบ offline K เท่าจะฝึกจำนวน epoch ลดลงเป็นสัดส่วน 1/(1+K) เพื่อให้จำนวน step เท่ากับ arm ควบคุม และรายงานจำนวน epoch, step และเวลาจริงของทุก arm

### 8.7 รายการ arm ของ Notebook 02

ค่าคงที่ของทุก arm: YOLO26n-seg pretrained บน COCO, imgsz 640, AdamW (lr0 0.001), batch จาก autobatch ที่ตรึงแล้ว, จำนวน step เท่ากัน, split S3

| Arm | การตั้งค่า | บทบาท |
|---|---|---|
| A0 | ปิด augmentation ทั้งหมด | ขอบล่าง |
| A1 | ค่าเริ่มต้นของ Ultralytics (online) | baseline หลัก |
| A2 | A1 + geometric ระดับกลาง | |
| A3 | A1 + offline photometric ทั่วไปและ blur | |
| A4 | A1 + offline ฝน | |
| A5 | A1 + offline แสงน้อย | |
| A6 | A1 + offline ฝุ่น | |
| A7 | A1 + offline ฝน + แสงน้อย + ฝุ่น (สถานการณ์ละหนึ่ง variant) | ผลรวมของสภาพแวดล้อม |
| A8 | A1 + copy_paste 0.3 | |
| A9 | รวมเฉพาะ arm ที่ช่วยได้เมื่อทดสอบเดี่ยว | สูตรที่เลือก |

ลำดับการทดลอง

1. คัดกรอง A0–A8 ด้วย seed 42
2. นำ 3 arm ที่ดีที่สุด และ A1 มารันเพิ่มเป็น 3 seeds (42, 43, 44)
3. สร้าง A9 แล้วรัน 3 seeds

เกณฑ์ตัดสิน

- เกณฑ์หลัก: mask AP50–95 บน val
- เกณฑ์รอง: mPC บน val-C (ค่าเฉลี่ยของ mask AP50–95 ทุกสถานการณ์และทุกระดับ) และ relative robustness mPC/AP_clean
- arm หนึ่งถือว่าช่วยได้เมื่อค่าเฉลี่ยของ 3 seeds สูงกว่า A1 เกินความแปรปรวนระหว่าง seed (ส่วนเบี่ยงเบนมาตรฐาน) ของ A1

---

## 9. การประเมินผล

### 9.1 L1: ระดับ instance (มาตรฐาน COCO)

- Evaluator หลักคือ pycocotools บน GT จาก Silver พร้อม ignore region (หัวข้อ 6.4) ตรวจเทียบผลกับ `model.val()` ของ Ultralytics บน val อย่างน้อยหนึ่งครั้ง และอธิบายความต่างถ้ามี
- Metric หลักสำหรับการเลือกแบบจำลองคือ mask AP@[0.50:0.95] บน val ตามมาตรฐาน COCO
- Metric ประกอบ: mask/box AP50, AP75, AP_S/M/L, AP ราย class, precision/recall/F1 ที่ operating point, TP/FP/FN, confusion matrix ของคลาส และ Boundary IoU (Cheng et al., CVPR 2021)
- Operating point: เลือก confidence threshold ราย class จาก val โดยให้ F1 สูงสุด ใช้ IoU 0.5 สำหรับการจับคู่ และใช้ NMS IoU ค่าเริ่มต้นของ Ultralytics ตามที่บันทึกไว้
- การเลือก checkpoint: ใช้ `best.pt` ของ Ultralytics ซึ่งเลือกจาก fitness บน val และบันทึกนิยามของ fitness ในเวอร์ชันที่ติดตั้ง

### 9.2 L2: ระดับสถานะของเฟรม

Ground truth: คลาส c ถือว่าปรากฏในเฟรมเมื่อพื้นที่ mask รวมของคลาสนั้นมีค่าอย่างน้อย a_min ของ Silver จากนั้นแปลงเป็น 4 สถานะ ได้แก่ none, gravel, sand และ mixed

การทำนาย

- คะแนนของคลาส c คำนวณเป็น `S_c = Σ (confidence × สัดส่วนพื้นที่ mask)` ของ instance คลาส c ในเฟรม
- คลาส c ถือว่าปรากฏเมื่อ `S_c ≥ θ_c`
- เลือก θ_c จาก val เท่านั้น ด้วยเกณฑ์ macro-F1 สูงสุด

Metric

- AP ระดับภาพราย class (Tabernik et al., J. Intell. Manuf. 2020)
- balanced accuracy (Brodersen et al., ICPR 2010), macro-F1, Cohen's κ และ confusion matrix 4 × 4
- ช่วงความเชื่อมั่น 95% ด้วย cluster bootstrap ระดับกลุ่ม

ข้อจำกัด

- สถานะ none ไม่มีตัวอย่างในชุดพัฒนา ต้องใช้เฟรมที่ไม่มีวัสดุจากวิดีโอหรือจาก S02
- เพราะคลาสผูกกับวิดีโอ การประเมิน L2 ต้องใช้การแบ่งแบบกลุ่มเท่านั้น
- L2 ตอบคำถามที่ง่ายกว่า L1 ค่าที่สูงกว่าจึงไม่ได้แปลว่าแบบจำลองดีกว่า ต้องรายงานคู่กับ L1 เสมอ (Maier-Hein et al., Nature Methods 2024)

### 9.3 L3: ระดับเวลา (วิดีโอ)

Ground truth: label ช่วงเวลา (clip, t_start, t_end, state) ของ 3 คลิปใน `thesis_videos` ได้มาตามขั้นตอนต่อไปนี้

1. ผู้ช่วยวิจัย (Claude) ร่าง label จากการดูเฟรมที่สุ่มทุก 0.5 วินาที **ก่อน** ดูผลของแบบจำลอง และบันทึกเป็น `draft_v0`
2. สร้างวิดีโอ overlay ที่แสดง timeline ของสถานะร่าง เพื่อให้ผู้วิจัยตรวจทาน
3. ผู้วิจัยแก้ไขและยืนยันเป็น `v1` จากนั้นตรึง label ไว้
4. ระบุในวิทยานิพนธ์ว่าผู้ label ไม่ใช่ผู้ตรวจอิสระ

การทำนาย: ใช้สถานะรายเฟรมตามหัวข้อ 9.2 แล้วปรับให้เรียบด้วย EMA หรือ k-of-n hysteresis โดยเลือกพารามิเตอร์ล่วงหน้า ไม่ปรับจากผลบนคลิปทดสอบ

Metric

- segmental F1@{10, 25, 50} และ Edit score (Lea et al., CVPR 2017; Abu Farha & Gall, CVPR 2019)
- frame-wise accuracy
- จำนวนครั้งที่สถานะเปลี่ยนต่อนาที (flicker)
- เวลาตั้งแต่วัสดุปรากฏจนระบบตัดสินถูก (time-to-decision)
- ความถูกต้องรายคลิป พร้อม Wilson interval

ข้อจำกัด: มีเพียง 3 คลิป ผล L3 จึงเป็นหลักฐานเชิงสาธิตมากกว่าเชิงสถิติ

### 9.4 สถิติ

- ผลระหว่าง arm รายงานค่าเฉลี่ย ± ส่วนเบี่ยงเบนมาตรฐานจาก 3 seeds และแสดงค่ารายตัว
- ผลบน test และ holdout รายงานช่วงความเชื่อมั่น 95% ด้วย cluster bootstrap ระดับกลุ่ม (10,000 รอบ)
- การเปรียบเทียบแบบคู่ใช้ paired cluster bootstrap ของผลต่าง
- ไม่สรุปว่าวิธีใดดีกว่า ถ้าช่วงความเชื่อมั่นของผลต่างคร่อมศูนย์
- รายงาน effect size คู่กับช่วงความเชื่อมั่นเสมอ

---

## 10. Notebook 03: การฝึก สถาปัตยกรรม และ benchmark

### 10.1 Baseline

- ตรวจโครงสร้างจริงของ YOLO26n-seg ในเวอร์ชันที่ติดตั้ง ได้แก่ yaml, ชนิดของ module ราย index, จำนวน parameter และ GFLOPs ที่ 640 จากนั้นจึงออกแบบการแก้ไข
- สูตรการฝึกใช้ augmentation จาก A9 และ AdamW lr0 0.001 จำนวน epoch กำหนดจาก smoke test ให้ loss ลู่เข้า พร้อม patience และบันทึก batch, imgsz, seed และ hardware
- ตรวจการทำซ้ำได้: ฝึก 2 seeds และตั้งค่า `deterministic=True` ของ Ultralytics ความต่างของ mask AP บน val ต้องอยู่ในช่วงที่รายงานได้ และบันทึกส่วนที่ไม่ deterministic บน GPU
- ผลลัพธ์ที่บันทึก: `best.pt`, `last.pt`, `args.yaml`, `results.csv`, กราฟ loss และ metric, ผลประเมิน L1/L2 บน val และ manifest ของ environment

### 10.2 Optimizer: AdamW เทียบกับ MuSGD

- MuSGD มีใน `ultralytics/optim/muon.py` ของเวอร์ชันที่ติดตั้ง ต้องตรวจการทำงานจริงก่อนใช้
- ค่าคงที่: สถาปัตยกรรม, split, imgsz, augmentation, จำนวน step และการประเมิน
- lr0 และ momentum ใช้ตามสูตรที่ Ultralytics แนะนำของแต่ละ optimizer และบันทึกความต่างของ schedule
- ใช้ 3 seeds ต่อ optimizer
- สิ่งที่เปรียบเทียบ: กราฟการลู่เข้า, ความเสถียร (ความแปรปรวนของ loss), mask AP บน val, เวลาฝึก และหน่วยความจำ GPU สูงสุด ส่วนผลบน test รายงานเฉพาะการตั้งค่าสุดท้าย

### 10.3 Feature map ของแต่ละชั้น

- ใช้ภาพตัวอย่างหนึ่งภาพจาก train ที่เลือกตามหัวข้อ 8.1
- ลงทะเบียน forward hook กับทุก module ใน `model.model` และกับ module ย่อยของ C3k2 (หรือ module ที่เทียบเท่าในโครงสร้างจริง)
- บันทึกของแต่ละชั้น: index, ชนิดของ module, shape ของ input และ output, stride เทียบกับภาพ และสถิติของ activation
- ภาพที่สร้าง: grid ของ 16 channel ที่มี activation เฉลี่ยสูงสุด, ค่าเฉลี่ยของทุก channel และภาพรวมของทุกชั้นในรูปเดียว
- attention module: แสดงค่า channel weight ของ CBAM และ ECA (หลัง sigmoid) และ spatial attention map ของ CBAM แยกจาก feature ที่ออกจาก module
- ระบุสิ่งที่ภาพเหล่านี้สรุปไม่ได้ เช่น activation สูงไม่ได้แปลว่าเป็นเหตุผลของการตัดสินใจ (Zeiler & Fergus, ECCV 2014)
- ไฟล์จัดเก็บตามรูปแบบ `feature_maps/<run>/<idx>_<module>/`

### 10.4 ต้นทุนการคำนวณ

ระดับชั้น

- จำนวน parameter
- FLOPs ด้วย thop โดยระบุ convention (1 MAC = 2 FLOPs)
- shape ของ input และ output
- หน่วยความจำของ activation (จำนวนสมาชิก × ขนาดของชนิดข้อมูล)
- latency ราย module ด้วย CUDA event (warm-up 50 รอบ วัด 200 รอบ รายงาน median)

ระดับทั้งแบบจำลอง

- latency p50 และ p95 ที่ batch 1, throughput ที่ batch 8 และ 16
- เปรียบเทียบ FP32 กับ FP16 และหน่วยความจำสูงสุด (`torch.cuda.max_memory_allocated`)
- ใช้ `torch.cuda.synchronize` ทุกครั้งก่อนอ่านเวลา และแยกเวลาของ preprocessing, inference และ postprocessing

การวิเคราะห์: ความสัมพันธ์อันดับ (Spearman) ระหว่าง FLOPs กับ latency รายชั้น และอธิบายส่วนที่ไม่สอดคล้องด้วย memory access, kernel launch overhead และ kernel implementation (Ma et al., ECCV 2018 — ShuffleNet V2)

### 10.5 Quantization

| อุปกรณ์ | Runtime / ความละเอียดเชิงตัวเลข | การประเมินความแม่นยำ | ผู้วัด latency |
|---|---|---|---|
| RTX 3080 Ti | PyTorch FP32 (อ้างอิง), PyTorch FP16, ONNX Runtime CUDA FP32, TensorRT FP16, TensorRT INT8 | ในเครื่องนี้, test ชุดเดียวกัน | Claude ในเครื่องนี้ |
| Raspberry Pi 5 | ONNX FP32, NCNN FP16, OpenVINO FP32, OpenVINO INT8 | ในเครื่องนี้ด้วย runtime เดียวกันบน CPU x86 (ระบุว่าตัวเลขบน ARM อาจต่างเล็กน้อย) | ผู้วิจัยบนบอร์ดจริงด้วย cell ที่เตรียมให้ |

- INT8 calibration ใช้ภาพจาก train 300–500 ภาพ แบบ stratified ตามกลุ่ม และไม่ใช้ val หรือ test
- สิ่งที่รายงาน: mask/box AP, L2 macro-F1, ขนาดไฟล์, latency p50/p95, throughput, หน่วยความจำ และผลต่างเชิงตัวเลขของ output เทียบกับ FP32 (ค่าเฉลี่ยของผลต่างสัมบูรณ์ของ logit และ mask IoU ของ prediction ที่จับคู่กัน)
- บันทึก operation ที่ runtime ไม่รองรับ และ fallback ที่เกิดขึ้น
- อ้างอิง: Jacob et al., CVPR 2018; Nagel et al., arXiv 2106.08295; Ultralytics export docs

### 10.6 Attention และสถาปัตยกรรม

| Arm | การแก้ไข | หมายเหตุ |
|---|---|---|
| B | baseline (ผลจาก 10.1) | |
| C1 | CBAM หลังชั้น P3 ของ neck | CBAM มีในเวอร์ชันที่ติดตั้ง (Woo et al., ECCV 2018) |
| C2 | ECA หลังชั้น P3 | เขียน module เองใน cell และลงทะเบียนกับ parser (Wang et al., CVPR 2020) ตรวจ forward/backward ก่อนใช้ |
| C3 | CBAM ที่ P3, P4 และ P5 | |
| C4 | ECA ที่ P3, P4 และ P5 | |
| F1 (exploratory) | การรวม feature แบบอื่น เช่น weighted fusion | ทำเฉพาะเมื่อ EDA ชี้ว่าขนาดวัตถุหรือ error สนับสนุน |
| SAHI (inference เท่านั้น) | sliced inference บน val | งานเดิมพบว่าทำให้ผลแย่ลง จึงทดสอบยืนยันเท่านั้น |

- ทุก arm รัน 3 seeds ด้วยงบการฝึกเท่ากับ B
- รายงานสัดส่วน pretrained weights ที่โอนได้ (จำนวน tensor และ parameter ที่โอนได้ต่อทั้งหมด) และออกแบบการแทรก module ให้ index ของชั้นเดิมโอน weights ได้มากที่สุด
- ทำ ablation ทีละการแก้ไข ไม่รวมหลายการแก้ไขพร้อมกัน
- เลือกการตั้งค่าที่ดีที่สุดจาก val และรายงานต้นทุนเพิ่มเป็น parameter, GFLOPs และ latency

### 10.7 การยืนยันขั้นสุดท้ายและเปิดผล test

1. นำการตั้งค่าที่เลือกและ baseline มาประเมินด้วย grouped CV 4 fold (หัวข้อ 7.5)
2. ตรึงการตั้งค่า, checkpoint และ threshold ราย class บันทึกเป็น `configs/frozen_inference.yaml` พร้อม SHA-256 ของ weights
3. ประเมินบน test ครั้งเดียว ทั้ง L1 และ L2 พร้อมช่วงความเชื่อมั่น และแสดงตัวอย่างข้อผิดพลาด
4. สร้างตารางเปรียบเทียบสุดท้ายของทุก arm (ค่าบน val) และบรรทัดของผล test

---

## 11. Notebook 04: Holdout และการทำภาพให้เป็นมาตรฐาน

### 11.1 ลำดับ

1. ตรวจว่า `configs/frozen_inference.yaml` มีอยู่และ hash ตรงกับ weights ถ้าไม่ตรงให้หยุดการทำงาน
2. EDA ของ holdout (ไม่ใช้ผลของแบบจำลอง) ได้แก่ ขนาดภาพ, ความสว่าง, contrast, สี, ความคมชัด, คลาส, ขนาด instance และเนื้อผิว
3. วัด domain shift เทียบกับชุดพัฒนา ด้วยระยะห่างของการกระจายของสถิติภาพ และระยะห่างใน embedding space ของ backbone ที่ตรึงแล้ว
4. ตรวจการซ้อนทับกับชุดพัฒนาด้วย SHA-256 และ pHash
5. map คลาสตามตารางในหัวข้อ 4.2 และใช้กฎ Silver เดียวกันกับชุดพัฒนา
6. ประเมิน L1 และ L2 ครั้งเดียว ด้วย config ที่ตรึงแล้ว
7. ประเมิน L3 บน 3 คลิปด้วย label v1

### 11.2 การทำภาพให้เป็นมาตรฐาน

ส่วน A: ข้อเสนอการควบคุมการถ่ายภาพ (physical acquisition)

- แสง: การวัดความสว่างของฉาก และแหล่งแสงเสริมที่ติดตั้งตายตัว
- การตั้งค่ากล้อง: ล็อก exposure และ white balance
- การปรับเทียบสี: แผ่นเทียบสี (เช่น ColorChecker) ติดตั้งในมุมภาพ
- ความละเอียด ระยะทำงาน และมุมกล้อง
- การตรวจคุณภาพภาพก่อนเข้าระบบ ได้แก่ ความสว่าง ความคมชัด และการตัดปลายของ histogram
- อ้างอิงแนวทางการวัดคุณลักษณะกล้อง เช่น EMVA 1288 (ตรวจรายละเอียดก่อนอ้าง)

ส่วน B: การปรับภาพด้วยซอฟต์แวร์ที่ทำได้กับข้อมูลที่มี ใช้เฉพาะวิธีที่ Notebook 02 เลือกไว้แล้ว ได้แก่ gray-world white balance, การปรับ exposure และ CLAHE ถ้าผ่านเกณฑ์ ห้ามปรับพารามิเตอร์ใหม่จากผลบน holdout

ข้อจำกัด: ภาพไม่มีแผ่นเทียบสีหรือข้อมูลการวัดแสง จึงไม่อ้างว่าภาพผ่านการปรับเทียบสี (calibrated) การปรับใน ส่วน B เป็นเพียงการทำให้สถิติของภาพใกล้กันขึ้นเท่านั้น

### 11.3 การเปรียบเทียบภาพดิบกับภาพที่ปรับมาตรฐาน

- เลือกภาพตัวอย่างด้วยวิธีเดียวกับหัวข้อ 8.1 แต่สุ่มจาก holdout
- แสดงภาพเดิม, สถิติคุณภาพภาพ, ภาพหลังปรับ, overlay ของ GT และของ prediction, ผลต่าง และคะแนน
- ประเมินทั้งชุดสองรอบ (ภาพดิบ และภาพที่ปรับมาตรฐาน) ด้วย config เดียวกัน รายงานผลต่างแบบคู่พร้อมช่วงความเชื่อมั่น

---

## 12. Reproducibility และการจัดเก็บ

### 12.1 โครงสร้างไดเรกทอรี

```
implement/research/
├── RESEARCH_PLAN.md
├── requirements.lock.txt
├── notebooks/
│   ├── 01_eda_and_dataset_splitting.ipynb
│   ├── 02_preprocessing_and_augmentation.ipynb
│   ├── 03_yolo_training_architecture_and_benchmarking.ipynb
│   └── 04_holdout_standardization_and_final_evaluation.ipynb
├── data/                      # ชุดข้อมูลที่สร้างขึ้น ชื่อมี hash ของเวอร์ชัน
│   ├── silver_<hash>/
│   ├── dev_<split>_<hash>/
│   ├── offline_aug_<arm>_<hash>/
│   ├── val_c_<hash>/
│   └── holdout_<name>_<hash>/
├── manifests/                 # รายชื่อภาพ, group, split, SHA-256, เวอร์ชันชุดข้อมูล
├── configs/                   # data yaml, recipe ของแต่ละ arm, model yaml, frozen_inference.yaml
├── labels_temporal/           # label ช่วงเวลาของวิดีโอ (draft_v0, v1)
├── outputs/
│   ├── 01_eda/
│   ├── 02_preprocessing_augmentation/
│   ├── 03_yolo_experiments/
│   │   ├── baseline/
│   │   ├── optimizer_comparison/
│   │   ├── architecture_ablations/
│   │   ├── quantization/
│   │   ├── feature_maps/
│   │   ├── layer_benchmarks/
│   │   └── evaluation/
│   └── 04_holdout/
└── reports/                   # ตารางสรุป, บรรณานุกรม (references.bib), experiment_log.md
```

### 12.2 กติกา

- **ชื่อ run:** `<notebook>_<arm>_s<seed>_<config-hash-8>` ถ้าโฟลเดอร์มีอยู่แล้วและมีไฟล์ผลครบ cell จะข้ามการรัน (ป้องกันการรันซ้ำ) ถ้าผลไม่ครบจะ resume จาก `last.pt` และไม่เขียนทับโดยไม่แจ้ง
- **cell ที่ใช้เวลานาน:** ติดป้าย `[EXPENSIVE ~xx นาที GPU]` ใน Markdown และมี flag `RUN_TRAINING` ที่ต้องเปิดเอง
- **ทุก run บันทึก:** `run_manifest.json` ที่มี dataset hash, config ทั้งหมด, seed, git commit, เวอร์ชันของแพ็กเกจ, ข้อมูล GPU/driver, เวลาเริ่ม-จบ และ SHA-256 ของ weights
- **Seed:** ค่าหลัก 42 seed เพิ่มเติมคือ 43 และ 44 ใช้ `deterministic=True` ของ Ultralytics และ seed ของ numpy, random และ torch
- **output ของ notebook:** รันทุก cell และเก็บ output ใน `.ipynb` ใช้ `jupyter nbconvert --execute --inplace` ด้วย kernel `aggregate-research`
- **บันทึกการทดลอง:** `reports/experiment_log.md` เพิ่มทุกการทดลองตามรูปแบบ hypothesis → setup → result → decision รวมถึงการทดลองที่ล้มเหลว
- **Git:** ไม่ commit weights และชุดข้อมูลขนาดใหญ่ (ตาม `.gitignore`) แต่ commit notebook, manifest, config และรายงาน

### 12.3 ด่านตรวจ (gates)

| ด่าน | เงื่อนไขก่อนไปขั้นถัดไป |
|---|---|
| G1 (จบ 01) | ไม่มีกลุ่มหรือคู่ภาพเกือบซ้ำข้าม split; ทุก split มีทั้งสองคลาส; Silver ผ่านการตรวจความถูกต้องทุกข้อ; checksum ตรงกับ manifest; ผู้วิจัยตรวจทานการแบ่ง |
| G2 (จบ 02) | offline aug ผ่านการตรวจการตรงกันของ mask; val-C สร้างครบ; การแบ่งไม่ถูกแก้ไข; มี shortlist และสูตร A9 |
| G3 (ก่อนฝึกจริงใน 03) | smoke test 1–2 epoch ผ่าน; ประมาณเวลาจริง; evaluator ตรงกับ `model.val()` |
| G4 (ก่อนเปิดผล test) | ตรึง config และ threshold แล้ว พร้อม hash |
| G5 (ก่อน 04) | `frozen_inference.yaml` มีอยู่; label ช่วงเวลา v1 ยืนยันแล้ว |

---

## 13. งบประมาณการคำนวณ (ประมาณการ)

| ส่วน | จำนวน run | เวลาต่อ run (ประมาณ) | รวม |
|---|---:|---:|---:|
| 02 คัดกรอง A0–A8 | 9 | 15–20 นาที | ~3 ชม. |
| 02 ยืนยัน 4 arm × 2 seeds เพิ่ม + A9 × 3 | 11 | 15–20 นาที | ~3.5 ชม. |
| 03 baseline และการทดสอบการทำซ้ำได้ | 2 | 20 นาที | ~0.7 ชม. |
| 03 optimizer (MuSGD 3 + AdamW ใช้ร่วมกับ baseline) | 4 | 20 นาที | ~1.3 ชม. |
| 03 attention C1–C4 × 3 seeds | 12 | 20–25 นาที | ~4.5 ชม. |
| 03 grouped CV 4 fold × 2 การตั้งค่า | 8 | 20 นาที | ~2.7 ชม. |
| Benchmark, quantization, feature map | — | — | ~1.5 ชม. |
| **รวม** | **~46** | | **~17 ชม.** |

ตัวเลขนี้เป็นการประมาณ จะปรับตามเวลาจริงจาก smoke test ใน G3 ถ้าเกินงบ จะลด seed ของขั้นคัดกรอง แต่ไม่ลดขั้นยืนยัน

---

## 14. บรรณานุกรมและสถานะการตรวจสอบ

สัญลักษณ์: [ตรวจแล้ว] = ยืนยันรายละเอียดบรรณานุกรมและข้อความที่อ้างจากต้นฉบับในขั้นวางแผนแล้ว; [รอตรวจ] = จะยืนยันจากต้นฉบับก่อนใส่ใน notebook ถ้ายืนยันไม่ได้จะไม่อ้าง

### การแบ่งข้อมูลและ leakage
- [ตรวจแล้ว] Kapoor, S., & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. doi:10.1016/j.patter.2023.100804.
- [ตรวจแล้ว] Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913–929. doi:10.1111/ecog.02881.
- [ตรวจแล้ว] Barz, B., & Denzler, J. (2020). Do we train on test data? Purging CIFAR of near-duplicates. *Journal of Imaging*, 6(6), 41. doi:10.3390/jimaging6060041.
- [ตรวจแล้ว] Zauner, C. (2010). Implementation and benchmarking of perceptual image hash functions. Diploma thesis, Upper Austria University of Applied Sciences, Hagenberg.
- [ตรวจแล้ว] Geirhos, R., et al. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673.
- [ตรวจแล้ว] Hasler, D., & Süsstrunk, S. (2003). Measuring colourfulness in natural images. *Proc. SPIE* 5007, 87–95.
- [รอตรวจ] Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*.
- [ตรวจแล้ว] scikit-learn documentation: StratifiedGroupKFold (stratification แบบ best-effort; ใกล้ GroupKFold เมื่อมีกลุ่มใหญ่จำนวนน้อย).

### Preprocessing และการปรับเทียบสี
- [รอตรวจ] Buchsbaum, G. (1980). A spatial processor model for object colour perception. *J. Franklin Institute*.
- [รอตรวจ] Pizer, S. M., et al. (1987). Adaptive histogram equalization and its variations. *CVGIP*.
- [รอตรวจ] Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. *Graphics Gems IV*.
- [รอตรวจ] Buades, A., Coll, B., & Morel, J.-M. (2005). A non-local algorithm for image denoising. *CVPR*.
- [รอตรวจ] EMVA Standard 1288 (การวัดคุณลักษณะกล้อง).

### Augmentation และความทนทาน
- [ตรวจแล้ว] Hendrycks, D., & Dietterich, T. (2019). Benchmarking neural network robustness to common corruptions and perturbations. *ICLR*. arXiv:1903.12261.
- [ตรวจแล้ว] Michaelis, C., et al. (2019). Benchmarking robustness in object detection: Autonomous driving when winter is coming. arXiv:1907.07484 (15 corruptions × 5 ระดับ, นิยาม mPC).
- [ตรวจแล้ว] Tremblay, M., Halder, S. S., de Charette, R., & Lalonde, J.-F. (2021). Rain rendering for evaluating and improving robustness to bad weather. *IJCV*, 129(2), 341–360. arXiv:2009.03683.
- [ตรวจแล้ว] Halder, S. S., Lalonde, J.-F., & de Charette, R. (2019). Physics-based rendering for improving robustness to rain. *ICCV*.
- [ตรวจแล้ว] Lv, F., Li, Y., & Lu, F. (2019/2020). Attention guided low-light image enhancement with a large scale low-light simulation dataset. arXiv:1908.00682 (สมการ `β·(α·I)^γ`, α~U(0.9,1), β~U(0.5,1), γ~U(1.5,5), noise Gaussian–Poisson).
- [ตรวจแล้ว: ชื่อเรื่องและแบบจำลอง Koschmieder] Sakaridis, C., Dai, D., & Van Gool, L. (2018). Semantic foggy scene understanding with synthetic data. *IJCV*. ค่า β ที่ใช้ใน Foggy Cityscapes ยังไม่ได้ยืนยัน
- [ตรวจแล้ว: สีเหลือง/แดงของภาพฝุ่นทราย] Shi, F., et al. (2022). Sand dust images enhancement based on red and blue channels. *Sensors*. (PMC8914657; ตรวจรายชื่อผู้แต่งอีกครั้ง)
- [ตรวจแล้ว] Ghiasi, G., et al. (2021). Simple Copy-Paste is a strong data augmentation method for instance segmentation. *CVPR*, 2918–2928. arXiv:2012.07177.
- [ตรวจแล้ว] Ultralytics. YOLO data augmentation guide (ค่าเริ่มต้นของ hyperparameter). https://docs.ultralytics.com/guides/yolo-data-augmentation/
- [รอตรวจ] Bochkovskiy, A., Wang, C.-Y., & Liao, H.-Y. M. (2020). YOLOv4. arXiv:2004.10934 (Mosaic).
- [รอตรวจ] Zhang, H., et al. (2018). mixup: Beyond empirical risk minimization. *ICLR*.
- [รอตรวจ] Buslaev, A., et al. (2020). Albumentations: Fast and flexible image augmentations. *Information*.

### สถาปัตยกรรม attention และ optimizer
- [รอตรวจ] Ultralytics YOLO26 documentation (โครงสร้างแบบจำลอง, MuSGD).
- [รอตรวจ] Woo, S., et al. (2018). CBAM: Convolutional block attention module. *ECCV*.
- [รอตรวจ] Wang, Q., et al. (2020). ECA-Net: Efficient channel attention for deep convolutional neural networks. *CVPR*.
- [รอตรวจ] Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization. *ICLR* (AdamW).
- [รอตรวจ] Jordan, K., et al. (2024). Muon optimizer (บทความออนไลน์) และ Liu, J., et al. (2025). Muon is scalable for LLM training. arXiv:2502.16982.
- [รอตรวจ] Akyon, F. C., et al. (2022). Slicing aided hyper inference and fine-tuning for small object detection. *ICIP* (SAHI).

### Feature map, ประสิทธิภาพ และ quantization
- [รอตรวจ] Zeiler, M. D., & Fergus, R. (2014). Visualizing and understanding convolutional networks. *ECCV*.
- [รอตรวจ] Ma, N., et al. (2018). ShuffleNet V2: Practical guidelines for efficient CNN architecture design. *ECCV*.
- [รอตรวจ] Dehghani, M., et al. (2022). The efficiency misnomer. *ICLR*.
- [รอตรวจ] Jacob, B., et al. (2018). Quantization and training of neural networks for efficient integer-arithmetic-only inference. *CVPR*.
- [รอตรวจ] Nagel, M., et al. (2021). A white paper on neural network quantization. arXiv:2106.08295.
- [รอตรวจ] Ultralytics export / TensorRT / OpenVINO / NCNN / Raspberry Pi documentation.

### การประเมินผล
- [ตรวจแล้ว] Maier-Hein, L., et al. (2024). Metrics reloaded: Recommendations for image analysis validation. *Nature Methods*, 21(2), 195–212. doi:10.1038/s41592-023-02151-z.
- [ตรวจแล้ว] Tabernik, D., Šela, S., Skvarč, J., & Skočaj, D. (2020). Segmentation-based deep-learning approach for surface-defect detection. *J. Intell. Manuf.*, 31(3), 759–776 (segmentation + decision network; ประเมินแบบ per-image ด้วย AP).
- [ตรวจแล้ว] Lea, C., et al. (2017). Temporal convolutional networks for action segmentation and detection. *CVPR*, 156–165 (segmental F1).
- [ตรวจแล้ว] Abu Farha, Y., & Gall, J. (2019). MS-TCN: Multi-stage temporal convolutional network for action segmentation. *CVPR*, 3575–3584 (F1@{10,25,50}, Edit).
- [ตรวจแล้ว] Brodersen, K. H., Ong, C. S., Stephan, K. E., & Buhmann, J. M. (2010). The balanced accuracy and its posterior distribution. *ICPR*, 3121–3124.
- [ตรวจแล้ว: มีอยู่จริงเท่านั้น] Driouache, Milpied, & Motamedi (2024). Vision-based method to identify materials transported by dump trucks. *EAAI*, 135. ยังไม่ได้อ่านฉบับเต็ม จึงไม่อ้างวิธีหรือตัวเลข
- [รอตรวจ] Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV* (นิยาม AP และ area range).
- [รอตรวจ] Cheng, B., et al. (2021). Boundary IoU: Improving object-centric image segmentation evaluation. *CVPR*.
- [รอตรวจ] Efron, B., & Tibshirani, R. (1993). *An Introduction to the Bootstrap*.
- [รอตรวจ] Bouthillier, X., et al. (2021). Accounting for variance in machine learning benchmarks. *MLSys*.

### Domain shift และการรายงาน
- [รอตรวจ] Recht, B., et al. (2019). Do ImageNet classifiers generalize to ImageNet? *ICML*.
- [รอตรวจ] Koh, P. W., et al. (2021). WILDS: A benchmark of in-the-wild distribution shifts. *ICML*.
- [รอตรวจ] Mitchell, M., et al. (2019). Model cards for model reporting. *FAT\**.
- [รอตรวจ] Gebru, T., et al. (2021). Datasheets for datasets. *CACM*.
- [รอตรวจ] Pineau, J., et al. (2021). Improving reproducibility in machine learning research. *JMLR*.

---

## 15. บันทึกการเปลี่ยนแปลงแผน

| วันที่ | การเปลี่ยนแปลง | เหตุผล |
|---|---|---|
| 2026-10-18 | ฉบับร่าง v0.1 | — |
| 2026-10-18 | Notebook 01: รวมกลุ่มจากภาพเกือบซ้ำ (pHash d ≤ 8) 30 → 26 กลุ่ม; 5d81688a รวมกับ 0f964fdb (เหตุการณ์เดียวกัน) จึง pin กลุ่มที่รวมแล้วไว้ใน test | ตรวจพบด้วย pHash และยืนยันด้วยตา (รูปที่ 2, 9); ถ้าไม่รวม การ pin 5d81688a ไว้ใน test จะรั่วไหลไปยัง 0f964fdb ใน train |
| 2026-10-18 | เงื่อนไข S3 (หัวข้อ 7.4 ข้อ 3): เปลี่ยนจาก "val และ test มีกลุ่ม mixed" เป็น "train และ test มีกลุ่ม mixed; val มีกลุ่ม sand ≥ 20 เฟรม; train มีกลุ่มที่มี sand ≥ 20 เฟรมอย่างน้อย 3 กลุ่ม" | ข้อมูลมีกลุ่ม mixed เพียง 2 กลุ่ม เงื่อนไขเดิมทำให้ train มีเฟรม mixed เพียง 7 เฟรม |
| 2026-10-18 | Grouped CV ใช้ 3 fold แทน 4 fold (หัวข้อ 7.5) | จำนวนกลุ่มที่มี sand ≥ 20 เฟรมในชุดพัฒนา (ไม่นับกลุ่มที่ pin ไว้ใน train) มีเพียง 3 กลุ่ม |
| 2026-10-18 | Silver (หัวข้อ 6.2 ข้อ 4): รวม annotation ซ้ำด้วย union แทนการเก็บชิ้นที่ใหญ่ที่สุด และแยกเกณฑ์ระหว่างป้ายฝึก (ตัดเฉพาะชิ้นที่เล็กกว่าหนึ่ง cell ของ P3) กับ GT ประเมิน (ignore เมื่อพื้นที่ < 0.1% หรือ aspect > 10) | รูปที่ 12 ของ Notebook 01: polygon ซ้อนกันมักเป็นทั้งกองกับส่วนย่อย; การลบชิ้นเล็กออกจากป้ายฝึกจะสอนให้พิกเซลวัสดุจริงเป็นพื้นหลัง |
| 2026-10-18 | ตัดการอ้าง Pech-Pacheco et al. (2000) สำหรับ variance of Laplacian | ยืนยันได้เพียงว่ามีบทความอยู่จริง แต่ยืนยันเนื้อหาที่อ้างไม่ได้ |
| 2026-10-18 | การรวมกลุ่ม: ภาพนิ่งที่ไม่มี UUID แต่ละภาพเป็นหน่วยของตัวเอง และผูกกับกลุ่มวิดีโอที่ใกล้ที่สุดเพียงกลุ่มเดียว (union-find ใช้เฉพาะวิดีโอกับวิดีโอ) | รอบแรก union-find รวม 07451999 (gravel) กับ 2be0c9f6 (sand) ผ่านภาพนิ่ง "download" และภาพนิ่ง "download" สามภาพที่ชื่อเหมือนกันเป็นคนละภาพ ทำให้เกิดคู่ภาพเกือบซ้ำข้าม split 7 คู่ (ด่าน G1 จับได้) |
| 2026-10-18 | เงื่อนไข S3 และจำนวน fold นับ "กลุ่มที่มีเฟรม sand ≥ 20 เฟรม" แทน "กลุ่มที่มี ≥ 20 ภาพ" | 07451999 มี 38 ภาพแต่มี sand เพียงไม่กี่เฟรม ทำให้ fold ของ CV สร้างไม่ได้ |
| 2026-10-18 | Silver: ตัดกฎ aspect > 10 ออกจากเกณฑ์ ignore ของชุดประเมิน (เหลือเฉพาะพื้นที่ < 0.1%) | การตรวจตัวอย่าง 120 ชิ้น (Notebook 01 หัวข้อ 10.1) พบ 7/30 ชิ้นในชั้น ignore ที่เป็นแถบหินบนตะแกรงชัดเจน (พื้นที่ 0.7–3%) ซึ่งถูกจับโดยกฎ aspect เท่านั้น |
| 2026-10-18 | การตรวจตัวอย่าง Silver ทำโดยผู้ช่วยวิจัย (Claude) จากภาพตัด ไม่ใช่ผู้ตรวจที่เป็นมนุษย์อิสระ | ผู้วิจัยมอบหมาย; ต้องระบุในวิทยานิพนธ์ และผู้วิจัยแก้ไขหรือเพิ่มผู้ตรวจได้ใน `manifests/silver_audit_review_v1.csv` |
| 2026-10-18 | ลดงบ GPU จาก ~17 เป็น ~10 ชั่วโมง: offline arm ใช้ K = 1 (A7 ใช้ K = 3), attention 4 arm × 2 seed, grouped CV ใช้เฉพาะยืนยัน baseline เทียบกับการตั้งค่าที่เลือก (seed 42), baseline ของ Notebook 03 ใช้ run ของ Notebook 02 ซ้ำ | ผู้วิจัยกำหนดว่า 23 ชั่วโมงมากเกินไป |
| 2026-10-18 | CBAM ใช้ implementation ตามบทความ (avg+max pool, MLP r = 16, spatial 7×7) แทน CBAM ของ Ultralytics; CBAM และ ECA เริ่มต้นเป็น identity (2·sigmoid, ชั้นสุดท้ายเริ่มที่ศูนย์) | CBAM ใน `ultralytics/nn/modules/conv.py` ใช้ channel attention แบบ RTMDet ไม่ตรงกับบทความ; การเริ่มต้นแบบ identity ป้องกันการทำลาย feature ของแบบจำลองที่ pretrain แล้ว (เป็นการปรับจากบทความที่ต้องรายงาน) |
| 2026-10-18 | MuSGD ใช้ค่าจาก `train_args` ของ checkpoint ทางการ (lr0 0.0054, lrf 0.05, momentum 0.937, wd 0.00064, warmup 1 epoch ที่ batch 128) | เป็นแหล่งค่าที่ตรวจสอบได้จากผู้พัฒนา |
| 2026-10-18 | Notebook 04: standardization ด้วยซอฟต์แวร์กำหนดล่วงหน้าเป็น S1 = gray-world + gamma ไปยัง median L\* ของ train; smoothing ของ L3 กำหนดล่วงหน้า EMA α = 0.3 และ hysteresis 3-of-5 | ป้องกันการปรับพารามิเตอร์จากผลบน holdout และวิดีโอ |

---

## 16. ลำดับการทำงาน

1. ผู้วิจัยตรวจทานเอกสารนี้
2. Notebook 01 → ด่าน G1 (หยุดให้ผู้วิจัยตรวจทานการแบ่ง)
3. Notebook 02 ส่วน visualization ของ preprocessing และ augmentation → สร้างชุดข้อมูล offline และ val-C → ด่าน G2
4. Notebook 02 ส่วน ablation (ฝึกจริง)
5. ร่าง label ช่วงเวลา `draft_v0` และวิดีโอ overlay ส่งให้ผู้วิจัยตรวจ (ทำขนานกับขั้นที่ 4)
6. Notebook 03 → ด่าน G3 → การทดลอง → ด่าน G4 → เปิดผล test
7. Notebook 04 → ด่าน G5 → EDA ของ holdout → ประเมินครั้งเดียว
8. ตรวจทานทุก artifact, citation และบันทึกการทดลอง

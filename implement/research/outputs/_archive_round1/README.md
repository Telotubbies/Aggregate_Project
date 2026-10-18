# Round 1 (confirmatory) — ผลที่เปิด test และ holdout ครั้งแรก (18 ตุลาคม 2569 14:00–14:09)

ผลในโฟลเดอร์นี้เป็นผล confirmatory ของรอบแรก: สูตร A9 (ซึ่งไม่ได้รวม A3_photometric เนื่องจาก bug ใน `compose_a9`),
YOLO26n-seg 640, AdamW 20 epoch แบบ iteration-matched, seed 42 การประเมินรอบที่สองหลังจากนี้เป็น post-hoc analysis

- `notebooks/` สำเนาของ notebook ทั้ง 5 ไฟล์พร้อม output ณ เวลาที่เปิดผล
- `03_evaluation/` TEST_OPENED.json, test_summary.json, dets_test.json.gz, ตาราง test และตาราง ablation ของรอบแรก
- `04_holdout/` HOLDOUT_OPENED.json, ผล holdout, ผลวิดีโอและ GIF
- `03.5_best_model_evaluation/` การประเมินโดยละเอียดของแบบจำลองรอบแรก (val, val-C, test)
- `configs/` frozen_inference.yaml และ aug_recipe_selected.json ของรอบแรก

from pathlib import Path
import time, json
import torch
from ultralytics import YOLO
EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
t0 = time.perf_counter()
m = YOLO(str(EXP/'weights'/'yolo26s-seg.pt'))
m.train(data=str(EXP/'data'/'gold_ws_aug'/'data.yaml'), imgsz=960,
        batch=-1, epochs=100, patience=30, seed=42, optimizer='auto',
        mosaic=0.0, fliplr=0.0, flipud=0.0, mixup=0.0, copy_paste=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
        scale=0.0, shear=0.0, perspective=0.0,
        project=str(EXP/'runs'), name='bench_ws/yolo26s')
json.dump({'hours': round((time.perf_counter()-t0)/3600,2)},
          open(EXP/'results'/'bench_ws'/'yolo26s_log.json','w'))
print('S DONE')

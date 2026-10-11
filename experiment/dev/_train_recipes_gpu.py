import os, sys
os.environ.setdefault('WANDB_MODE', 'offline')
from pathlib import Path
from ultralytics import YOLO
import torch

if os.environ.get('FORCE_AMP'):
    import ultralytics.engine.trainer as _t
    _t.check_amp = lambda *a, **k: True   # name bound in trainer ns;
                                        # check fails on TheRock —
                                        # WATCH first epoch for NaN
    print('FORCE_AMP: trainer.check_amp bypassed', flush=True)
# DO NOT set cudnn.benchmark on TheRock gfx1101 — benchmark mode picks
# the fastest kernel, and the fastest kernels are the broken ones
# (they skip compute → 9it/s + garbage 7e-44 losses). Default MIOpen
# heuristic choice is the only correct path observed.

EXP = Path(r'B:\02_Projects\thesis_final\experiment')
ARMS = {
    # arm          data                                  imgsz  batch
    'aug':    (EXP / 'data' / 'gold_aug',               960, 16),
    'sahift': (EXP / 'data' / 'recipes' / 'aug_sahift', 640, 64),
}
KW = dict(batch=16, epochs=50, patience=15, seed=42, optimizer='auto',
          fliplr=0.0, flipud=0.0, mosaic=0.0, mixup=0.0,
          hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
          scale=0.0, shear=0.0, perspective=0.0, erasing=0.0,
          copy_paste=0.0, device=0, workers=4, cache='ram',
          amp=False,          # TheRock gfx1101: fp16/bf16 conv NaN
          deterministic=True,   # aug-6 known-good; det flag was never
                                # the problem — benchmark mode was
          project=str(EXP / 'runs'))


def main():
    print('device check:', torch.cuda.get_device_name(0), flush=True)
    for arm, (data, sz, bs) in ARMS.items():
        best = EXP / 'runs' / 'recipes' / arm / 'weights' / 'best.pt'
        if best.exists():
            print(arm, 'cached'); continue
        print(f'===== train {arm} | {data} @ {sz} bs{bs} =====', flush=True)
        m = YOLO(str(EXP / 'yolo26n-seg.pt'))
        m.train(data=str(data / 'data.yaml'), imgsz=sz, batch=bs,
                name=f'recipes/{arm}', **{k: v for k, v in KW.items()
                                        if k != 'batch'})
        del m; torch.cuda.empty_cache()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()

"""GT audit: extract every FN'd sand GT on holdout -> crop grid + stats."""
import sys, json
from pathlib import Path
import numpy as np, cv2, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
TEST = EXP / 'data' / 'gold' / 'test'
OUT = EXP / 'results' / 'recipes'
CLS = ['gravel', 'sand']
COL = {0: (60, 200, 60), 1: (60, 140, 255)}  # BGR

stems = [p.stem for p in sorted((TEST / 'images').iterdir())]

def load_gt(stem):
    ip = next((TEST / 'images').glob(stem + '.*'))
    im = cv2.imread(str(ip)); H, W = im.shape[:2]
    masks, cls = [], []
    for ln in (TEST / 'labels' / (stem + '.txt')).read_text().splitlines():
        p = ln.split()
        xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [xy.astype(np.int32)], 1)
        masks.append(m.astype(bool)); cls.append(int(p[0]))
    return im, masks, cls

def iou(a, b):
    i = (a & b).sum(); u = (a | b).sum()
    return i / max(u, 1)

preds = np.load(OUT / 'preds_aug.npy', allow_pickle=True).item()

rows = []
crops = []
for st in stems:
    im, gm_, gc_ = load_gt(st)
    H, W = im.shape[:2]
    for gi, (c, m) in enumerate(zip(gc_, gm_)):
        if c != 1:                       # sand GT only
            continue
        pl = preds.get(st, [])
        sand = [q for q in pl if q[0] == 1]
        grav = [q for q in pl if q[0] == 0]
        iou_s = max([iou(q[2], m) for q in sand], default=0)
        iou_g = max([iou(q[2], m) for q in grav], default=0)
        hit = iou_s >= .5
        cat = ('TP' if hit else
               'sand-pred-low-IoU' if iou_s > 0 else
               'wrong-class (gravel pred)' if iou_g >= .3 else
               'invisible (no pred)')
        area = int(m.sum())
        rows.append({'stem': st, 'gi': gi, 'area_px': area,
                     'iou_sand': round(iou_s, 3), 'iou_gravel': round(iou_g, 3),
                     'cat': cat})
        if not hit:
            ys, xs = np.where(m)
            y0, y1 = ys.min(), ys.max(); x0, x1 = xs.min(), xs.max()
            pad = int(max(y1 - y0, x1 - x0) * .8) + 10
            cy0, cy1 = max(0, y0 - pad), min(H, y1 + pad)
            cx0, cx1 = max(0, x0 - pad), min(W, x1 + pad)
            crops.append({'stem': st, 'gi': gi, 'cat': cat, 'area': area,
                          'iou_s': iou_s, 'iou_g': iou_g,
                          'crop': (cy0, cy1, cx0, cx1), 'mask': m,
                          'pl': pl})

df = pd.DataFrame(rows)
print('=== sand GT on holdout ===')
print('total:', len(df), '| TP:', (df.cat == 'TP').sum(),
      '| FN:', (df.cat != 'TP').sum())
print('\nFN breakdown by category:')
print(df[df.cat != 'TP'].cat.value_counts().to_string())
print('\nFN area stats:')
print(df[df.cat != 'TP'].area_px.describe().round(0).to_string())
print('\nFN max-sand-IoU dist (0=model sees nothing):')
print(df[df.cat != 'TP'].iou_sand.describe().round(3).to_string())
df.to_csv(OUT / 'gt_audit_sand.csv', index=False)

# ---- crop grid: all FN'd sand GTs, worst-first ----
crops.sort(key=lambda r: -r['area'])
N = min(len(crops), 36)
nc = 6; nr = int(np.ceil(N / nc))
fig, ax = plt.subplots(nr, nc, figsize=(3.2 * nc, 3.6 * nr))
for a in ax.flat:
    a.axis('off')
for i, r in enumerate(crops[:N]):
    a = ax.flat[i]
    cy0, cy1, cx0, cx1 = r['crop']
    im, _, _ = load_gt(r['stem'])
    crop = im[cy0:cy1, cx0:cx1].copy()
    cm = r['mask'][cy0:cy1, cx0:cx1]
    cnt, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(crop, cnt, -1, (255, 255, 0), 2)   # GT = yellow
    for c, s, m in r['pl']:
        pm = m[cy0:cy1, cx0:cx1]
        if pm.sum() == 0:
            continue
        cnt, _ = cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnt, -1, COL[c], 2)
    a.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    a.set_title(f"{r['stem'][:10]}#{r['gi']} {r['area']}px\n"
                f"{r['cat'][:18]} IoU_s={r['iou_s']:.2f}", fontsize=6)
    a.axis('off')
fig.suptitle(f'FN\'d sand GT crops — yellow=GT, green/blue=preds '
             f'({len(crops)} FN of {len(df)} sand GT)')
plt.tight_layout()
plt.savefig(OUT / 'fig_gt_audit_sand.png', dpi=150)
print('\nsaved', OUT / 'fig_gt_audit_sand.png', '| crops:', len(crops))

"""Leave-One-Group-Out on top-10 uuid groups — per-scene generalization.
- test = one big group entirely held out (never in train/valid)
- valid = all 22 non-top10 small groups (shared, ~80 imgs) — same for every fold
- train = remaining 9 top-10 groups
- yolo26n-seg, nb12 protocol, flat-960 eval, same-class IoU>=.5 AP50
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch, shutil
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
GOLD = EXP / 'data' / 'gold'
LDIR = EXP / 'data' / 'logo'
OUT = EXP / 'results' / 'logo'; OUT.mkdir(parents=True, exist_ok=True)
RUNS = EXP / 'runs'
CLS = ['gravel', 'sand']
DEV = 'cuda:0'

# ---------- group table ----------
grp = {}
for split in ('train', 'valid', 'test'):
    for lt in (GOLD / split / 'labels').glob('*.txt'):
        u = lt.stem.split('-')[0]
        d = grp.setdefault(u, dict(imgs=0, grav=0, sand=0))
        d['imgs'] += 1
        for ln in lt.read_text().splitlines():
            if ln.strip():
                d['grav' if ln.split()[0] == '0' else 'sand'] += 1
GT = pd.DataFrame(grp).T.infer_objects()
for c in ('imgs', 'grav', 'sand'):
    GT[c] = GT[c].astype(int)

TOP10 = list(GT.sort_values('imgs', ascending=False).index[:10])
if 'bbf45eda' not in TOP10:
    TOP10.append('bbf45eda')  # key rear-view sand scene — research question
VALID_G = [u for u in GT.index if u not in TOP10]
print('held-out folds:', TOP10, flush=True)
print('valid pool:', len(VALID_G), 'groups /', GT.loc[VALID_G, 'imgs'].sum(), 'imgs',
      flush=True)

# ---------- materialize per-fold datasets ----------
def link(src, dst):
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

grp_files = {}
for split in ('train', 'valid', 'test'):
    for f in (GOLD / split / 'images').iterdir():
        grp_files.setdefault(f.stem.split('-')[0], []).append(f)

for g in TOP10:
    d = LDIR / g
    if d.exists():
        continue
    for name, groups in (('train', [u for u in TOP10 if u != g]),
                         ('valid', VALID_G), ('test', [g])):
        for kind in ('images', 'labels'):
            (d / name / kind).mkdir(parents=True, exist_ok=True)
        for u in groups:
            for f in grp_files[u]:
                link(f, d / name / 'images' / f.name)
                lt = f.parent.parent / 'labels' / (f.stem + '.txt')
                if lt.exists():
                    link(lt, d / name / 'labels' / lt.name)
    (d / 'data.yaml').write_text(
        f"path: {d}\ntrain: train/images\nval: valid/images\n"
        "test: test/images\nnc: 2\nnames: ['gravel', 'sand']\n")

# ---------- eval (nb12 protocol) ----------
def build_gt(test_dir):
    gt = {}
    for ip in sorted((test_dir / 'images').iterdir()):
        im = cv2.imread(str(ip)); H, W = im.shape[:2]
        masks, cls = [], []
        for ln in (test_dir / 'labels' / (ip.stem + '.txt')).read_text().splitlines():
            p = ln.split()
            xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [xy.astype(np.int32)], 1)
            masks.append(m.astype(bool)); cls.append(int(p[0]))
        gt[ip.stem] = (im, masks, cls)
    return gt

def iou_mat(pm, gm):
    if not len(pm) or not len(gm):
        return np.zeros((len(pm), len(gm)))
    P = np.stack(pm); G = np.stack(gm)
    i = (P[:, None] & G[None]).reshape(len(P), len(G), -1).sum(-1)
    u = (P[:, None] | G[None]).reshape(len(P), len(G), -1).sum(-1)
    return i / np.maximum(u, 1)

def eval_fold(preds, GT_, conf=0.05):
    ngt = {c: sum(cl.count(c) for _, _, cl in GT_.values()) for c in (0, 1)}
    out = {'ngt': ngt}
    for c, cn in enumerate(CLS):
        det = []
        for st, pl in preds.items():
            det += [{'st': st, 's': s, 'm': m}
                    for c_, s, m in pl if c_ == c and s >= conf]
        det.sort(key=lambda d: -d['s'])
        matched = {k: np.zeros(len(v[1]), bool) for k, v in GT_.items()}
        tp = np.zeros(len(det))
        for i, d in enumerate(det):
            _, gm_, gc_ = GT_[d['st']]
            sel = [j for j, cc in enumerate(gc_) if cc == c]
            if sel:
                M = iou_mat([d['m']], [gm_[j] for j in sel])[0]
                j = sel[int(M.argmax())]
                if M.max() >= .5 and not matched[d['st']][j]:
                    tp[i] = 1; matched[d['st']][j] = 1
        ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
        rec = np.concatenate([[0], ctp / max(ngt[c], 1), [1]])
        prec = np.concatenate([[0], ctp / np.maximum(ctp + cfp, 1), [0]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        out[cn] = {'AP50': round(float(np.trapezoid(prec, rec)), 4),
                   'recall': round(float(tp.sum() / max(ngt[c], 1)), 3),
                   'tp': int(tp.sum()), 'npred': len(det)}
    out['AP50_macro'] = round(np.mean([out[c]['AP50'] for c in CLS]), 4)
    return out

def predict_flat(model, im, conf=0.05, imgsz=960):
    r = model.predict(im, imgsz=imgsz, conf=conf,
                      retina_masks=True, device=DEV, verbose=False)[0]
    out = []
    if r.masks is None:
        return out
    H, W = im.shape[:2]
    for c_, s_, m_ in zip(r.boxes.cls.cpu().numpy().astype(int),
                          r.boxes.conf.cpu().numpy(),
                          r.masks.data.cpu().numpy()):
        mb = cv2.resize((m_ > 0.5).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append((int(c_), float(s_), mb))
    return out

TRAIN_KW = dict(batch=-1, epochs=100, patience=30, seed=42,
                optimizer='auto', fliplr=0.0, flipud=0.0, mosaic=0.0,
                mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                erasing=0.0, copy_paste=0.0, project=str(RUNS))

for g in TOP10:
    rj = OUT / f'{g}.json'
    if rj.exists():
        print(g, 'cached', flush=True); continue
    best = RUNS / 'logo' / g / 'weights' / 'best.pt'
    if not best.exists():
        print(f'===== LOGO train, held-out={g} =====', flush=True)
        t0 = time.perf_counter()
        m = YOLO(str(EXP / 'weights' / 'yolo26n-seg.pt'))
        m.train(data=str(LDIR / g / 'data.yaml'), imgsz=960,
                name=f'logo/{g}', **TRAIN_KW)
        print(g, 'hours:', round((time.perf_counter() - t0) / 3600, 2), flush=True)
        del m; torch.cuda.empty_cache()
    GT_g = build_gt(LDIR / g / 'test')
    m = YOLO(str(best))
    preds = {s: predict_flat(m, v[0]) for s, v in tqdm(GT_g.items(), desc=g)}
    res = eval_fold(preds, GT_g)
    res['held_out'] = g
    res['test_imgs'] = int(GT.loc[g, 'imgs'])
    json.dump(res, open(rj, 'w'), indent=1)
    del m; torch.cuda.empty_cache()
    print(g, json.dumps(res), flush=True)

rows = []
for g in TOP10:
    r = json.load(open(OUT / f'{g}.json'))
    rows.append(dict(group=g, imgs=r['test_imgs'],
                     ngt_g=r['ngt']['0'], ngt_s=r['ngt']['1'],
                     grav=r['gravel']['AP50'], sand=r['sand']['AP50'],
                     s_rec=r['sand']['recall'], macro=r['AP50_macro']))
df = pd.DataFrame(rows)
df.to_csv(OUT / 'logo_summary.csv', index=False)
print(df.to_string(index=False), flush=True)
print('DONE', flush=True)

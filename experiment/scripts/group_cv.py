"""5-fold class-aware group CV on gold uuid groups.
- groups atomic (uuid prefix); 2be0c9f6 always in train (314 imgs > fold size)
- valid = fixed 6 groups (v2 valid) shared across folds -> fair early-stop
- assignment: seeded greedy minimizing imgs+per-class instance imbalance;
  rejects seeds where any fold lacks a class -> deterministic, uuid-aware
- trains yolo26n-seg (nb12 protocol) per fold, evals on fold test (flat-960,
  same-class IoU>=.5 AP50), reports mean+-std
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch, shutil
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
GOLD = EXP / 'data' / 'gold'
CV = EXP / 'data' / 'cv'
OUT = EXP / 'results' / 'cv'; OUT.mkdir(parents=True, exist_ok=True)
RUNS = EXP / 'runs'
CLS = ['gravel', 'sand']
DEV = 'cuda:0'
K = 4  # foldable sand lives in ~4 groups; k=5 leaves thin-sand folds (metric noise)

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

VALID_GROUPS = {'0a48349e', '96ecd4fe', 'e9eebbd1', '5079fbab', '7a600bac', '7b6b3d3f'}
ALWAYS_TRAIN = {'2be0c9f6'}          # 314 imgs — larger than any fold
foldable = [u for u in GT.index if u not in VALID_GROUPS | ALWAYS_TRAIN]
print(f'groups: {len(GT)} | foldable: {len(foldable)} | valid: {len(VALID_GROUPS)}',
      flush=True)

# ---------- seeded class-aware fold assignment ----------
def assign(seed):
    rng = np.random.RandomState(seed)
    order = list(foldable)
    rng.shuffle(order)
    # largest-first within shuffle: sort by size desc, keep a jittered order
    order.sort(key=lambda u: -GT.loc[u, 'imgs'] * (1 + 0.1 * rng.rand()))
    folds = [[] for _ in range(K)]
    tot = GT.loc[foldable, ['imgs', 'grav', 'sand']].sum().values
    tgt = tot / K
    cur = np.zeros((K, 3))
    for u in order:
        # pick fold with max deficit across the 3 metrics (normalized)
        deficit = (tgt - cur) / tgt
        f = int(deficit.sum(1).argmax())
        folds[f].append(u)
        cur[f] += GT.loc[u, ['imgs', 'grav', 'sand']].values
    return folds

folds = None
for seed in range(10000):
    cand = assign(seed)
    ok = all(
        GT.loc[f, 'sand'].sum() >= 40 and GT.loc[f, 'grav'].sum() >= 40
        and 20 <= GT.loc[f, 'imgs'].sum() <= 250
        for f in cand if f)
    if ok and all(len(f) for f in cand):
        folds = cand; used_seed = seed; break
assert folds, 'no valid assignment in 10k seeds'
print('assignment seed:', used_seed, flush=True)
fold_tbl = {f'fold{k}': folds[k] for k in range(K)}
for k in range(K):
    s = GT.loc[folds[k], ['imgs', 'grav', 'sand']].sum()
    print(f'fold{k}: {len(folds[k])} groups | {s.imgs} imgs | grav {s.grav} | sand {s.sand}',
          flush=True)
json.dump({k: list(v) for k, v in fold_tbl.items()},
          open(OUT / 'folds.json', 'w'), indent=1)

# ---------- materialize datasets (hardlinks) ----------
def link(src, dst):
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

stem2grp, grp_files = {}, {}
for split in ('train', 'valid', 'test'):
    for f in (GOLD / split / 'images').iterdir():
        u = f.stem.split('-')[0]
        stem2grp[f.stem] = u
        grp_files.setdefault(u, []).append(f)

if CV.exists():
    shutil.rmtree(CV)
for k in range(K):
    test_g = set(folds[k])
    train_g = (set(foldable) - test_g) | ALWAYS_TRAIN
    for name, groups in (('train', train_g), ('valid', VALID_GROUPS),
                         ('test', test_g)):
        for kind in ('images', 'labels'):
            (CV / f'fold{k}' / name / kind).mkdir(parents=True, exist_ok=True)
        for u in groups:
            for f in grp_files[u]:
                link(f, CV / f'fold{k}' / name / 'images' / f.name)
                lt = f.parent.parent / 'labels' / (f.stem + '.txt')
                if lt.exists():
                    link(lt, CV / f'fold{k}' / name / 'labels' / lt.name)
    (CV / f'fold{k}' / 'data.yaml').write_text(
        f"path: {CV / f'fold{k}'}\ntrain: train/images\n"
        "val: valid/images\ntest: test/images\n"
        "nc: 2\nnames: ['gravel', 'sand']\n")

# ---------- evaluator (nb12 protocol, parametrized test dir) ----------
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

def eval_fold(preds, GT, conf=0.05):
    ngt = {c: sum(cl.count(c) for _, _, cl in GT.values()) for c in (0, 1)}
    out = {'ngt': ngt}
    for c, cn in enumerate(CLS):
        det = []
        for st, pl in preds.items():
            det += [{'st': st, 's': s, 'm': m}
                    for c_, s, m in pl if c_ == c and s >= conf]
        det.sort(key=lambda d: -d['s'])
        matched = {k: np.zeros(len(v[1]), bool) for k, v in GT.items()}
        tp = np.zeros(len(det))
        for i, d in enumerate(det):
            _, gm_, gc_ = GT[d['st']]
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

for k in range(K):
    tag = f'fold{k}'
    rj = OUT / f'{tag}.json'
    if rj.exists():
        print(tag, 'cached', flush=True); continue
    best = RUNS / 'cv' / tag / 'weights' / 'best.pt'
    if not best.exists():
        print(f'===== train {tag} =====', flush=True)
        t0 = time.perf_counter()
        m = YOLO(str(EXP / 'weights' / 'yolo26n-seg.pt'))
        m.train(data=str(CV / tag / 'data.yaml'), imgsz=960,
                name=f'cv/{tag}', **TRAIN_KW)
        print(tag, 'train hours:', round((time.perf_counter() - t0) / 3600, 2),
              flush=True)
        del m; torch.cuda.empty_cache()
    GT_k = build_gt(CV / tag / 'test')
    m = YOLO(str(best))
    preds = {s: predict_flat(m, v[0]) for s, v in tqdm(GT_k.items(), desc=tag)}
    res = eval_fold(preds, GT_k)
    json.dump(res, open(rj, 'w'), indent=1)
    del m; torch.cuda.empty_cache()
    print(tag, json.dumps(res), flush=True)

# ---------- summary ----------
rows = []
for k in range(K):
    r = json.load(open(OUT / f'fold{k}.json'))
    rows.append(dict(fold=k, grav=r['gravel']['AP50'], sand=r['sand']['AP50'],
                     macro=r['AP50_macro'],
                     ngt_sand=r['ngt']['1']))
df = pd.DataFrame(rows)
df.loc['mean'] = df[['grav', 'sand', 'macro']].mean().round(4)
df.loc['std'] = df[['grav', 'sand', 'macro']].std().round(4)
df.to_csv(OUT / 'cv_summary.csv')
print(df.to_string(), flush=True)
print('DONE', flush=True)

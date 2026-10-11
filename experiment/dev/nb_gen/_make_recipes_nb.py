import json

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None,
                     "outputs": [], "source": s.splitlines(True)}

cells = [
md("""# 12 — Instance-Separation Training Recipes (gold_aug · YOLO26n-seg · 960 · NMS/e2e)

**Focus.** Layer surgery (nb10) proved missed GTs get ~zero classification
response at their responsible level — a *representation* failure, not a
decode/threshold failure. This notebook attacks that at the **training
level**, with techniques that change *how instances are presented/
sampled during training*. Base dataset: `gold_aug` (geo+photo+dark/rain
offline aug, 3,012 imgs). Model/recipe: `yolo26n-seg.pt`, imgsz 960,
batch -1, optimizer auto, patience 30, seed 42, all train-time aug off
(same protocol as `runs/size_abl/all` — only the data recipe changes).

## Literature — instance-handling-during-training techniques (~20)

| # | Technique | Ref | How it acts on instances | Arm here |
|---|-----------|-----|--------------------------|----------|
| 1 | Copy-Paste aug | Ghiasi+ CVPR'21 (arXiv:2012.07177) | paste instances across images | `cp` |
| 2 | Repeat-Factor Sampling | Gupta+ LVIS CVPR'19 (arXiv:1908.03195) | oversample imgs w/ rare class | `rfs` |
| 3 | Instance-aware RFS | IRFS NeurIPS'23 (arXiv:2305.08069) | oversample by instance count | inside `rfs` |
| 4 | Seesaw Loss | Wang+ CVPR'21 | rebalance pos/neg grad per class | discuss only |
| 5 | FASA feat-aug | Zang+ ICCV'21 | augment rare-class features | — |
| 6 | SAHI fine-tuning | Akyon+ ICIP'22 (arXiv:2202.06934) | train on sliced tiles | `sahift` |
| 7 | SNIP scale training | Singh&Davis CVPR'18 | train each obj at right scale | via `sahift` |
| 8 | FPN multi-level assign | Lin+ CVPR'17 | instance→level by size | built-in YOLO |
| 9 | TaskAligned assigner | Feng+ ICCV'21 (TOOD) | score-aware pos assignment | built-in YOLO26 |
| 10 | ATSS adaptive assign | Zhang+ CVPR'20 | per-instance pos/neg split | — |
| 11 | O2F one-to-few assign | Niu+ TITS'24 | soften one2one assignment | YOLO26 built-in |
| 12 | Hard-example mining | Shrivastava+ CVPR'16 (OHEM) | train on missed instances | `hem` |
| 13 | Dataset cartography | Swayamdipta+ EMNLP'20 | confidence/var per example | basis of `hem` |
| 14 | Deep watershed energy | Bai&Urtasun CVPR'17 | separate touching instances | nb05 analog |
| 15 | Discriminative embed | De Brabandere+ '17 | pixel-embed clustering | — |
| 16 | BCNet bilayer | Ke+ CVPR'21 | occluder/occludee layers | — |
| 17 | CondInst dynamic conv | Tian+ ECCV'20 | per-instance mask head | nb08 tested |
| 18 | SOLOv2 matrix NMS | Wang+ NeurIPS'20 | grid-instance + fast NMS | — |
| 19 | PolarMask contour | Xie+ CVPR'20 | polar instance repr | — |
| 20 | Weather/env aug | Halder+ ICCV'19; WACV'24 | domain-shift instances | via gold_aug |

## Arms (toggleable)

| arm | data recipe | why it addresses the FN mechanism |
|-----|-------------|-----------------------------------|
| `aug` | gold_aug as-is (control) | env/domain coverage baseline |
| `cp` | gold_aug + 753 copy-paste variants | more instance placements/contexts |
| `rfs` | gold_aug + sand-image duplication ×2 | instance-count balance (IRFS-lite) |
| `sahift` | gold_aug sliced 640px tiles ov=0.2 | small instances get dedicated pixels (SAHI-FT) |
| `hem` | gold_aug + ×2 imgs w/ baseline-missed GT | train where the model is weak (OHEM/cartography) |
| `res1280` | gold_aug, imgsz=1280 | more pixels per instance (control for sahift) |

Owner decisions logged: cp included (re-tests nb00 rejection), base =
gold_aug stack, P2-head arm dropped, arms toggleable until GPU returns."""),

code("""import json, shutil, time
from pathlib import Path
import numpy as np, cv2, torch, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm

for _c in ('B:\\\\02_Projects\\\\thesis_final\\\\experiment',
           '/mnt/b/02_Projects/thesis_final/experiment',
           '/b/02_Projects/thesis_final/experiment'):
    if (Path(_c) / 'data' / 'gold' / 'test').exists():
        EXP = Path(_c)
        break
TEST = EXP / 'data' / 'gold' / 'test'
AUG = EXP / 'data' / 'gold_aug'
REC = EXP / 'data' / 'recipes'
OUT = EXP / 'results' / 'recipes'; OUT.mkdir(parents=True, exist_ok=True)
RUNS = EXP / 'runs' / 'recipes'
CLS = ['gravel', 'sand']
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
GPU = torch.cuda.is_available()
print('device:', DEV, '| EXP:', EXP)

# ---------- GT + eval (identical protocol to nb11) ----------
stems = [p.stem for p in sorted((TEST / 'images').iterdir())]

def load_gt(stem):
    ip = next((TEST / 'images').glob(stem + '.*'))
    im = cv2.imread(str(ip)); H, W = im.shape[:2]
    masks, cls = [], []
    lt = TEST / 'labels' / (stem + '.txt')
    for ln in lt.read_text().splitlines():
        p = ln.split()
        xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [xy.astype(np.int32)], 1)
        masks.append(m.astype(bool)); cls.append(int(p[0]))
    return im, masks, cls

GT = {s: load_gt(s) for s in stems}
NGT = {c: sum(cl.count(c) for _, _, cl in GT.values()) for c in (0, 1)}
print('test GT:', NGT)

def iou_mat(pm, gm):
    if not len(pm) or not len(gm):
        return np.zeros((len(pm), len(gm)))
    P = np.stack(pm); G = np.stack(gm)
    i = (P[:, None] & G[None]).reshape(len(P), len(G), -1).sum(-1)
    u = (P[:, None] | G[None]).reshape(len(P), len(G), -1).sum(-1)
    return i / np.maximum(u, 1)

def eval_preds(preds, conf=0.05):
    out = {}
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
        rec = np.concatenate([[0], ctp / max(NGT[c], 1), [1]])
        prec = np.concatenate([[0], ctp / np.maximum(ctp + cfp, 1), [0]])
        for i in range(len(prec) - 2, -1, -1):
            prec[i] = max(prec[i], prec[i + 1])
        out[cn] = {'AP50': round(float(np.trapz(prec, rec)), 4),
                   'recall': round(float(tp.sum() / max(NGT[c], 1)), 3),
                   'tp': int(tp.sum()), 'npred': len(det)}
    out['AP50_macro'] = round(np.mean([out[c]['AP50'] for c in CLS]), 4)
    return out

# ---------- label io helpers (train-set polygon ops) ----------
def read_lbl(p):
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            t = ln.split()
            if len(t) >= 7:
                out.append((int(t[0]),
                            np.asarray(t[1:], np.float32).reshape(-1, 2)))
    return out

def write_lbl(p, polys):
    lines = []
    for c, xy in polys:
        if len(xy) >= 3:
            lines.append(str(c) + ' ' + ' '.join(f'{v:.6f}' for v in xy.ravel()))
    p.write_text('\\n'.join(lines))

def link_or_copy(src, dst):
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

def clone_split(src_dir, dst_dir):
    for kind in ('images', 'labels'):
        (dst_dir / kind).mkdir(parents=True, exist_ok=True)
        for f in (src_dir / kind).iterdir():
            link_or_copy(f, dst_dir / kind / f.name)

def write_yaml(d):
    (d / 'data.yaml').write_text(
        f'train: train/images\\nval: valid/images\\nnc: 2\\n'
        f\"names: ['gravel', 'sand']\\n\")

def build_base(name):
    \"\"\"data/recipes/<name> = gold_aug clone (all splits).\"\"\"
    d = REC / name
    if (d / 'data.yaml').exists():
        return d
    for split in ('train', 'valid', 'test'):
        clone_split(AUG / split, d / split)
    write_yaml(d)
    return d"""),

md("""## §1 Dataset builders (CPU, idempotent — run once, cache on disk)"""),

md("""### §1a `cp` — Copy-Paste (Ghiasi+ CVPR'21) — **DROPPED**

Built and verified, then rejected on inspection: ~31% of variants were
dead duplicates (all paste attempts rejected by the 5%-mask-occlusion /
12%-area-cap / lower-half placement guards), and surviving pastes were
mostly tiny edge slivers with visible cut-edge / lighting / missing-
contact-shadow artifacts. This domain's amorphous piles lack the clean
object boundaries copy-paste needs — confirms nb00's earlier rejection.
Generator removed; `data/recipes/aug_cp` left on disk for audit."""),

md("""### §1b `rfs` — Instance-aware oversampling (RFS/IRFS)

RFS (Gupta+ LVIS'19) repeats images containing rare classes; IRFS
(NeurIPS'23) weights by *instance count*. 2-class data → image-level
≈ instance-level. Implement: every train image containing ≥1 sand
instance is duplicated once (all 4 variants incl. augs) →
`data/recipes/aug_rfs`."""),

code("""def build_rfs(overwrite=False):
    d = REC / 'aug_rfs'
    if (d / 'data.yaml').exists() and not overwrite:
        print('cached'); return d
    base = build_base('aug_rfs')
    # original stems containing sand
    sand_stems = []
    for lt in sorted((AUG / 'train' / 'labels').glob('*.txt')):
        if '_aug' in lt.stem:
            continue
        if any(ln.split()[0] == '1' for ln in lt.read_text().splitlines()):
            sand_stems.append(lt.stem)
    n_files = 0
    for st in sand_stems:
        # duplicate every variant of this original (itself + _aug1..3)
        for f in list((base / 'train' / 'images').glob(st + '*')):
            ni = f.with_name(f.stem + '_r1.jpg')
            nl = base / 'train' / 'labels' / (f.stem + '_r1.txt')
            if not ni.exists():
                link_or_copy(f, ni)
                link_or_copy(base / 'train' / 'labels' / (f.stem + '.txt'), nl)
                n_files += 1
    n_dup_imgs = len(sand_stems)
    print(f'sand imgs: {n_dup_imgs}, dup files written: {n_files}')
    return base"""),

md("""### §1c `sahift` — SAHI fine-tuning (Akyon+ ICIP'22)

Slice every gold_aug train image (540×960) into **320px** tiles
(overlap 0.25 → ~8 tiles/img), remap polygons via rasterize+contour
clip (proper rect clip, not vertex clamp), drop fragments <256px²
(paper's ignore-region analogue). valid/test untouched full-frame →
`data/recipes/aug_sahift`. Train imgsz=640 → each 320px tile is
**upscaled 2× at train time** — that's the actual SAHI-FT mechanism:
instances get ~2× the pixel budget so features can fire on sand/small."""),

code("""def build_sahift(overwrite=False, tile=320, ov=0.25, min_px=256):
    d = REC / 'aug_sahift'
    if (d / 'data.yaml').exists() and not overwrite:
        print('cached'); return d
    if d.exists():
        shutil.rmtree(d)
    for split in ('valid', 'test'):                 # eval splits = full gold
        clone_split(AUG / split, d / split)
    for kind in ('images', 'labels'):
        (d / 'train' / kind).mkdir(parents=True, exist_ok=True)

    def clip_poly(px, tw, th):
        \"\"\"Proper rect-clip: rasterize -> contour -> clean polygon.\"\"\"
        m = np.zeros((th, tw), np.uint8)
        q = px.copy()
        q[:, 0] = np.clip(q[:, 0], 0, tw - 1)
        q[:, 1] = np.clip(q[:, 1], 0, th - 1)
        cv2.fillPoly(m, [q.astype(np.int32)], 1)
        cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        if not cnt:
            return None, 0
        c_big = max(cnt, key=cv2.contourArea)[:, 0, :].astype(np.float32)
        return c_big, int(m.sum())

    n_tiles = n_inst = 0
    for ip in tqdm(sorted((AUG / 'train' / 'images').glob('*.jpg')),
                   desc='tiling'):
        im = cv2.imread(str(ip)); H, W = im.shape[:2]
        polys = read_lbl(AUG / 'train' / 'labels' / (ip.stem + '.txt'))
        step = int(tile * (1 - ov))
        xs = list(range(0, max(W - tile, 0) + 1, step))
        ys = list(range(0, max(H - tile, 0) + 1, step))
        if xs[-1] + tile < W: xs.append(W - tile)
        if ys[-1] + tile < H: ys.append(H - tile)
        for yi, y in enumerate(ys):
            for xi, x in enumerate(xs):
                t = im[y:y + tile, x:x + tile]
                th, tw = t.shape[:2]
                tpolys = []
                for c, xy in polys:
                    px = xy * [W, H] - [x, y]
                    if (px[:, 0].max() < 0 or px[:, 1].max() < 0 or
                            px[:, 0].min() >= tw or px[:, 1].min() >= th):
                        continue
                    q, area = clip_poly(px, tw, th)
                    if q is None or area < min_px or len(q) < 3:
                        continue                    # drop sliver fragments
                    tpolys.append((c, q / [tw, th]))
                stem = f'{ip.stem}_t{yi}{xi}'
                cv2.imwrite(str(d / 'train' / 'images' / (stem + '.jpg')), t)
                write_lbl(d / 'train' / 'labels' / (stem + '.txt'), tpolys)
                n_tiles += 1; n_inst += len(tpolys)
    write_yaml(d)
    print(f'tiles: {n_tiles}, instances kept: {n_inst}')
    return d"""),

md("""### §1d `hem` — Hard-example mining (OHEM / Dataset Cartography)

Predict train split with the **baseline** best.pt; mark images whose GT
has ≥1 missed instance (no pred ≥0.5 mask-IoU 0.5). Duplicate those
images ×2 (all aug variants) → `data/recipes/aug_hem`. This is the
cartography 'hard region' oversampled — targets the exact failure mode
(dead feature response on sand/small)."""),

code("""def build_hem(overwrite=False):
    d = REC / 'aug_hem'
    cache = OUT / 'train_preds_baseline.npy'
    base_model_p = EXP / 'runs' / 'size_abl' / 'all' / 'weights' / 'best.pt'
    if cache.exists():
        tr_preds = np.load(cache, allow_pickle=True).item()
    else:
        from ultralytics import YOLO
        bm = YOLO(str(base_model_p))
        tr_preds = {}
        for ip in tqdm(sorted((AUG / 'train' / 'images').glob('*.jpg')),
                       desc='baseline train-scan'):
            if '_aug' in ip.stem:
                continue                        # score originals only
            im = cv2.imread(str(ip))
            r = bm.predict(im, imgsz=960, conf=0.05, retina_masks=True,
                           device=DEV, verbose=False)[0]
            pl = []
            if r.masks is not None:
                H, W = im.shape[:2]
                for c_, s_, m_ in zip(r.boxes.cls.cpu().numpy().astype(int),
                                      r.boxes.conf.cpu().numpy(),
                                      r.masks.data.cpu().numpy()):
                    mb = cv2.resize((m_ > .5).astype(np.uint8), (W, H),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
                    pl.append((int(c_), float(s_), mb))
            tr_preds[ip.stem] = pl
        np.save(cache, np.array(tr_preds, dtype=object), allow_pickle=True)

    hard = []
    for stem, pl in tr_preds.items():
        lt = AUG / 'train' / 'labels' / (stem + '.txt')
        ip = next((AUG / 'train' / 'images').glob(stem + '.*'))
        im = cv2.imread(str(ip)); H, W = im.shape[:2]
        gm, gc = [], []
        for c, xy in read_lbl(lt):
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [(xy * [W, H]).astype(np.int32)], 1)
            gm.append(m.astype(bool)); gc.append(c)
        miss = False
        for j, (c, m) in enumerate(zip(gc, gm)):
            same = [p for p in pl if p[0] == c]
            if not same or iou_mat([p[2] for p in same], [m]).max() < .5:
                miss = True; break
        if miss:
            hard.append(stem)
    print(f'hard images: {len(hard)} / {len(tr_preds)}')

    base = build_base('aug_hem')
    n = 0
    for st in hard:
        for f in list((base / 'train' / 'images').glob(st + '*')):
            if '_h1' in f.stem:
                continue
            ni = f.with_name(f.stem + '_h1.jpg')
            nl = base / 'train' / 'labels' / (f.stem + '_h1.txt')
            if not ni.exists():
                link_or_copy(f, ni)
                link_or_copy(base / 'train' / 'labels' / (f.stem + '.txt'), nl)
                n += 1
    print('dup files written:', n)
    return base"""),

md("""## §2 Build all recipe datasets (CPU work — runs even without GPU)"""),

code("""BUILDERS = {'rfs': build_rfs,
            'sahift': build_sahift, 'hem': build_hem}
# 'cp' dropped: ~30% variants ended as dead dupes and pasted slivers
# looked fake (edge fragments, lighting/shadow mismatch) — matches
# nb00's earlier rejection of copy-paste on this domain.
DATA = {}
for name, fn in BUILDERS.items():
    try:
        DATA[name] = fn()
        print(name, '->', DATA[name])
    except Exception as e:
        print(name, 'FAILED:', e)
DATA['aug'] = AUG            # control arm = gold_aug itself
DATA['res1280'] = AUG        # same data, different imgsz"""),

md("""## §2.5 Polygon-level visualization — what each arm did to the GT

Left = source GT polygons (gravel=green, sand=orange), right = what the
training split actually contains after the recipe. All figures saved to
`results/recipes/fig_viz_*.png`."""),

code("""COL = {0: (60, 200, 60), 1: (60, 140, 255)}     # BGR gravel/sand

def draw_polys(im, polys, thick=2, alpha=0.25, hi=None):
    out = im.copy()
    ov = out.copy()
    for i, (c, xy) in enumerate(polys):
        px = (xy * [im.shape[1], im.shape[0]]).astype(np.int32)
        cv2.fillPoly(ov, [px], COL[c])
        cv2.polylines(out, [px], True,
                      COL[c], 4 if (hi is not None and i in hi) else thick)
    return cv2.addWeighted(ov, alpha, out, 1 - alpha, 0)

def show(ax, im, title):
    ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    ax.set_title(title, fontsize=8); ax.axis('off')

# ---------- pick a demo stem present in all arms ----------
orig_stems = sorted(p.stem for p in (AUG / 'train' / 'images').glob('*.jpg')
                    if '_aug' not in p.stem)
demo = max(orig_stems, key=lambda s: len(read_lbl(
    AUG / 'train' / 'labels' / (s + '.txt'))))
d_im = cv2.imread(str(AUG / 'train' / 'images' / (demo + '.jpg')))
d_p  = read_lbl(AUG / 'train' / 'labels' / (demo + '.txt'))
H, W = d_im.shape[:2]

fig, ax = plt.subplots(2, 4, figsize=(20, 9))

# ===== row 1: cp + sahift =====
show(ax[0, 0], draw_polys(d_im, d_p), f'ORIG {demo[:18]}\\n{len(d_p)} inst')

ax[0, 1].axis('off')
ax[0, 1].text(0.05, 0.6, 'CP — DROPPED', fontweight='bold', fontsize=11,
            transform=ax[0, 1].transAxes)
ax[0, 1].text(0.05, 0.35,
            'verified: 31% dead dupes,\\npasted slivers look fake\\n(edge frags, light/shadow)\\n= nb00 rejection confirmed',
            fontsize=8, transform=ax[0, 1].transAxes)

# sahift: tile grid on original
tile, step = 320, int(320 * 0.75)
g_im = draw_polys(d_im, d_p)
for x in range(0, W, step):
    cv2.rectangle(g_im, (x, 0), (min(x + tile, W) - 1, H - 1), (255, 0, 255), 1)
for y in range(0, H, step):
    cv2.rectangle(g_im, (0, y), (W - 1, min(y + tile, H) - 1), (255, 0, 255), 1)
show(ax[0, 2], g_im, 'SAHIFT: tile grid (magenta=320px tiles)')

# sahift: one tile w/ clipped polys from generated dataset
tiles = sorted((REC / 'aug_sahift' / 'train' / 'images').glob(demo + '_t*.jpg'))
if tiles:
    tp_ = tiles[0]
    t_im = cv2.imread(str(tp_))
    t_p = read_lbl(REC / 'aug_sahift' / 'train' / 'labels' / (tp_.stem + '.txt'))
    show(ax[0, 3], draw_polys(t_im, t_p),
         f'TILE {tp_.stem[-4:]}: {len(t_p)} clipped inst')
else:
    ax[0, 3].axis('off')

# ===== row 2: rfs + hem =====
def count_cls(d):
    cnt = {0: 0, 1: 0}
    for lt in (d / 'train' / 'labels').glob('*.txt'):
        for ln in lt.read_text().splitlines():
            if ln.strip():
                cnt[int(ln.split()[0])] += 1
    return cnt

c0 = count_cls(AUG)
cR = count_cls(REC / 'aug_rfs') if (REC / 'aug_rfs' / 'data.yaml').exists() else c0
x = np.arange(2)
ax[1, 0].bar(x - .2, [c0[0], c0[1]], .4, label='gold_aug')
ax[1, 0].bar(x + .2, [cR[0], cR[1]], .4, label='aug_rfs')
ax[1, 0].set_xticks(x); ax[1, 0].set_xticklabels(CLS)
ax[1, 0].set_title('RFS: instance counts\\n(sand dup x2)')
ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=.3, axis='y')
for i, v in enumerate([cR[0], cR[1]]):
    ax[1, 0].text(i + .2, v, str(v), ha='center', fontsize=8)

cH = count_cls(REC / 'aug_hem') if (REC / 'aug_hem' / 'data.yaml').exists() else c0
ax[1, 1].bar(x - .2, [c0[0], c0[1]], .4, label='gold_aug')
ax[1, 1].bar(x + .2, [cH[0], cH[1]], .4, label='aug_hem')
ax[1, 1].set_xticks(x); ax[1, 1].set_xticklabels(CLS)
ax[1, 1].set_title('HEM: instance counts\\n(hard-img dup x2)')
ax[1, 1].legend(fontsize=7); ax[1, 1].grid(alpha=.3, axis='y')
for i, v in enumerate([cH[0], cH[1]]):
    ax[1, 1].text(i + .2, v, str(v), ha='center', fontsize=8)

# hem: one hard image, GT colored found/missed vs baseline preds
tr_preds_p = OUT / 'train_preds_baseline.npy'
if tr_preds_p.exists():
    tr_preds = np.load(tr_preds_p, allow_pickle=True).item()
    hard_st = None
    for st, pl in tr_preds.items():
        lt = AUG / 'train' / 'labels' / (st + '.txt')
        gm, gc = [], []
        ip = next((AUG / 'train' / 'images').glob(st + '.*'))
        tim = cv2.imread(str(ip)); th, tw = tim.shape[:2]
        for c, xy in read_lbl(lt):
            m = np.zeros((th, tw), np.uint8)
            cv2.fillPoly(m, [(xy * [tw, th]).astype(np.int32)], 1)
            gm.append(m.astype(bool)); gc.append(c)
        st2, pl2 = st, pl
        for c, m in zip(gc, gm):
            same = [p for p in pl2 if p[0] == c]
            if not same or iou_mat([p[2] for p in same], [m]).max() < .5:
                hard_st = (st, tim, gm, gc); break
        if hard_st:
            break
    if hard_st:
        st, tim, gm, gc = hard_st
        out = tim.copy()
        for c, m in zip(gc, gm):
            same = [p for p in tr_preds[st] if p[0] == c]
            hit = same and iou_mat([p[2] for p in same], [m]).max() >= .5
            col = (60, 200, 60) if hit else (50, 50, 240)
            cnt, _ = cv2.findContours(m.astype(np.uint8),
                                      cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnt, -1, col, 3)
        show(ax[1, 2], out, f'HEM src {st[:16]}\\ngreen=found red=MISSED (baseline)')
    else:
        ax[1, 2].axis('off'); ax[1, 2].set_title('no hard img?')
else:
    ax[1, 2].axis('off'); ax[1, 2].set_title('baseline scan not run')

# legend
ax[1, 3].axis('off')
ax[1, 3].text(0.05, 0.9, 'LEGEND', fontweight='bold')
ax[1, 3].add_patch(plt.Rectangle((0.05, 0.7), 0.1, 0.08,
                   color=np.array(COL[0][::-1]) / 255))
ax[1, 3].text(0.2, 0.72, 'gravel GT polygon')
ax[1, 3].add_patch(plt.Rectangle((0.05, 0.55), 0.1, 0.08,
                   color=np.array(COL[1][::-1]) / 255))
ax[1, 3].text(0.2, 0.57, 'sand GT polygon')
ax[1, 3].text(0.05, 0.32, 'magenta = tile grid (sahift)')
ax[1, 3].text(0.05, 0.24, 'red outline = baseline-missed (hem)')

plt.tight_layout()
plt.savefig(OUT / 'fig_viz_arms.png', dpi=110)
plt.show()
print('saved', OUT / 'fig_viz_arms.png')"""),

md("""### §2.6 Per-instance close-up — sahift clipping detail"""),

code("""# zoom on ONE polygon that crosses a tile boundary -> shows clip behavior
tiles = sorted((REC / 'aug_sahift' / 'train' / 'images')
               .glob(demo + '_t*.jpg'))
if tiles:
    # find a tile whose label count differs from neighbors = boundary clip
    fig, ax = plt.subplots(1, min(len(tiles), 4) + 1, figsize=(18, 5))
    show(ax[0], draw_polys(d_im, d_p), f'full frame {len(d_p)} inst')
    for i, tp_ in enumerate(tiles[:4]):
        t_im = cv2.imread(str(tp_))
        t_p = read_lbl(REC / 'aug_sahift' / 'train' / 'labels'
                       / (tp_.stem + '.txt'))
        show(ax[i + 1], draw_polys(t_im, t_p),
             f'{tp_.stem[-5:]}: {len(t_p)} inst')
    plt.tight_layout()
    plt.savefig(OUT / 'fig_viz_sahift_zoom.png', dpi=110)
    plt.show()"""),

md("""## §3 Train — identical recipe to baseline `size_abl/all`

`yolo26n-seg.pt`, batch -1, epochs≤100, patience 30, seed 42,
optimizer auto, all train-time aug off (fliplr/hsv/mosaic/... = 0 —
same as baseline args.yaml). Only data (and imgsz for sahift/res1280)
change. **Requires GPU.** Cache-first: skips arms with existing best.pt."""),

code("""ARMS = {
    # name     data        imgsz
    'aug':     ('aug',      960),   # control: gold_aug effect alone
    'rfs':     ('aug_rfs',  960),
    'sahift':  ('aug_sahift', 640),
    'hem':     ('aug_hem',  960),
    'res1280': ('aug',     1280),
}
# toggle arms off by deleting keys above before GPU returns

TRAIN_KW = dict(batch=-1, epochs=100, patience=30, seed=42,
                optimizer='auto', fliplr=0.0, flipud=0.0, mosaic=0.0,
                mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                erasing=0.0, copy_paste=0.0,
                project=str(RUNS.parent))

if not GPU:
    print('NO GPU — training skipped. Re-run cell when cuda available.')
else:
    from ultralytics import YOLO
    for arm, (dk, sz) in ARMS.items():
        best = RUNS / arm / 'weights' / 'best.pt'
        if best.exists():
            print(arm, 'cached'); continue
        data_y = DATA.get(dk) or (REC / dk)
        print(f'===== train {arm} | data={data_y} imgsz={sz} =====')
        m = YOLO('yolo26n-seg.pt')
        m.train(data=str(data_y / 'data.yaml'), imgsz=sz,
                name=f'recipes/{arm}', **TRAIN_KW)
        del m; torch.cuda.empty_cache()"""),

md("""## §4 Eval — flat 960 inference (NMS-free e2e as shipped) on gold/test"""),

code("""from ultralytics import YOLO

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

res = {}
for arm in ARMS:
    best = RUNS / arm / 'weights' / 'best.pt'
    rj = OUT / f'{arm}.json'
    if rj.exists():
        res[arm] = json.load(open(rj)); continue
    if not best.exists():
        print(arm, 'no weights — skipped'); continue
    m = YOLO(str(best))
    preds = {s: predict_flat(m, GT[s][0]) for s in tqdm(stems, desc=arm)}
    res[arm] = eval_preds(preds)
    json.dump(res[arm], open(rj, 'w'), indent=1)
    np.save(OUT / f'preds_{arm}.npy',
            np.array({k: [(c, s, mm.astype(np.uint8)) for c, s, mm in v]
                      for k, v in preds.items()}, dtype=object),
            allow_pickle=True)
    del m; torch.cuda.empty_cache() if GPU else None
    print(arm, json.dumps(res[arm]))"""),

md("""## §5 Verdict"""),

code("""if res:
    tab = pd.DataFrame({k: {c: v[c]['AP50'] for c in CLS} |
                        {'macro': v['AP50_macro'],
                         'rec_g': v['gravel']['recall'],
                         'rec_s': v['sand']['recall']}
                        for k, v in res.items()}).T
    print(tab.round(3).to_string())
    tab[['macro']].plot.bar(figsize=(7, 4), legend=False)
    plt.ylabel('AP50_macro'); plt.grid(alpha=.3, axis='y')
    plt.title('instance-separation recipes vs baseline 0.566(nb11 metric)')
    plt.tight_layout()
    plt.savefig(OUT / 'fig_arms.png', dpi=120); plt.show()
    json.dump(res, open(OUT / 'recipes_results.json', 'w'), indent=1)
else:
    print('no arm results yet — waiting for GPU training')"""),

md("""## Limitations & protocol notes

- Each arm changes **one** data recipe on top of gold_aug — comparisons
  are arm-vs-arm and arm-vs-baseline(0.488 ultralytics / 0.566 nb-metric),
  not paired stats; single seed 42.
- `cp` **dropped after data verification** — re-test confirmed nb00's
  rejection: ~31% of variants were dead duplicates (occlusion/cap
  rejection), surviving pastes were edge slivers with visible blending
  artifacts. Pile-domain lacks the object boundaries copy-paste needs.
- `sahift` trains at 640 on tiles; eval is still flat 960 full-frame —
  if it helps, the gain is *feature-level*, not tiling-at-inference.
- `hem` uses baseline's own misses → mild self-confirmation bias;
  reported as exploratory.
- RFS dup of aug variants intentionally oversamples sand *and* its
  augmented forms (instance-count principle).
- All arms: no retrain-free tricks — pure training-level instance
  separation per the research question."""),
]

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
      "language": "python", "name": "python3"},
      "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
import os
out = r"B:\02_Projects\thesis_final\experiment\12_instance_recipes.ipynb"
json.dump(nb, open(out, "w", encoding='utf-8'), indent=1)
print("wrote", out)

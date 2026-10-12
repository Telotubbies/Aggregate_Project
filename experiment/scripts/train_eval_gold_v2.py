"""Train ctrl on gold_v2 + eval old-aug and new model on gold_v2/test.
Same protocol/metrics as nb12: flat-960 predict, same-class IoU>=.5 AP50."""
import json, time
from pathlib import Path
import numpy as np, cv2, torch
from tqdm import tqdm
from ultralytics import YOLO

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
V2 = EXP / 'data' / 'gold_v2'
OUT = EXP / 'results' / 'resplit_v2'; OUT.mkdir(exist_ok=True)
RUNS = EXP / 'runs'
TEST = V2 / 'test'
CLS = ['gravel', 'sand']
DEV = 'cuda:0'

# ---------- GT + evaluator (verbatim protocol from nb12) ----------
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

GT = {s: load_gt(s) for s in stems}
NGT = {c: sum(cl.count(c) for _, _, cl in GT.values()) for c in (0, 1)}
print('gold_v2 holdout:', len(GT), '| GT:', NGT, flush=True)

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
        out[cn] = {'AP50': round(float(np.trapezoid(prec, rec)), 4),
                   'recall': round(float(tp.sum() / max(NGT[c], 1)), 3),
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

def run_eval(tag, weights):
    rj = OUT / f'{tag}.json'
    if rj.exists():
        print(tag, 'cached:', json.load(open(rj)), flush=True)
        return
    m = YOLO(str(weights))
    preds = {s: predict_flat(m, GT[s][0]) for s in tqdm(stems, desc=tag)}
    res = eval_preds(preds)
    json.dump(res, open(rj, 'w'), indent=1)
    np.save(OUT / f'preds_{tag}.npy',
            np.array({k: [(c, s, mm.astype(np.uint8)) for c, s, mm in v]
                      for k, v in preds.items()}, dtype=object), allow_pickle=True)
    del m; torch.cuda.empty_cache()
    print(tag, json.dumps(res), flush=True)

# ---------- 1) old aug model x NEW test (isolates test-set effect) ----------
run_eval('aug_on_v2', RUNS / 'recipes' / 'aug' / 'weights' / 'best.pt')

# ---------- 2) train ctrl on gold_v2 (same protocol as nb12) ----------
best = RUNS / 'v2' / 'ctrl' / 'weights' / 'best.pt'
if not best.exists():
    TRAIN_KW = dict(batch=-1, epochs=100, patience=30, seed=42,
                    optimizer='auto', fliplr=0.0, flipud=0.0, mosaic=0.0,
                    mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0,
                    translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
                    erasing=0.0, copy_paste=0.0, project=str(RUNS))
    t0 = time.perf_counter()
    m = YOLO(str(EXP / 'weights' / 'yolo26n-seg.pt'))
    m.train(data=str(V2 / 'data.yaml'), imgsz=960, name='v2/ctrl', **TRAIN_KW)
    json.dump({'hours': round((time.perf_counter() - t0) / 3600, 2)},
              open(OUT / 'train_log_v2.json', 'w'))
    del m; torch.cuda.empty_cache()
else:
    print('v2/ctrl cached', flush=True)

# ---------- 3) new model x new test ----------
run_eval('v2ctrl_on_v2', RUNS / 'v2' / 'ctrl' / 'weights' / 'best.pt')
print('DONE', flush=True)

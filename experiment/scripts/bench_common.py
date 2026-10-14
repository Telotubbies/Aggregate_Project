"""nb17 shared: GT loading, IoU-match eval, FLOPs, latency for bench models.
Preds contract: {stem: [(cls01, score, bool HxW)]} — same as eval_common.
"""
from pathlib import Path
import json, time
import numpy as np, cv2, torch

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
GOLD = EXP / 'data' / 'gold'
WS = EXP / 'data' / 'gold_ws'
OUT = EXP / 'results' / 'bench_ws'
OUT.mkdir(parents=True, exist_ok=True)
CLS = ['gravel', 'sand']
DEV = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def load_gt(split_dir):
    """split_dir = gold_ws/test etc. -> {stem: (img_bgr, [masks], [cls])}"""
    out = {}
    for ip in sorted((split_dir / 'images').iterdir()):
        im = cv2.imread(str(ip)); H, W = im.shape[:2]
        masks, cls = [], []
        lt = split_dir / 'labels' / (ip.stem + '.txt')
        if lt.exists():
            for ln in lt.read_text().splitlines():
                p = ln.split()
                xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
                m = np.zeros((H, W), np.uint8)
                cv2.fillPoly(m, [xy.astype(np.int32)], 1)
                masks.append(m.astype(bool)); cls.append(int(p[0]))
        out[ip.stem] = (im, masks, cls)
    return out


def iou_mat(pm, gm):
    if not len(pm) or not len(gm):
        return np.zeros((len(pm), len(gm)))
    P = np.stack(pm); G = np.stack(gm)
    i = (P[:, None] & G[None]).reshape(len(P), len(G), -1).sum(-1)
    u = (P[:, None] | G[None]).reshape(len(P), len(G), -1).sum(-1)
    return i / np.maximum(u, 1)


def eval_preds(preds, GT, conf=0.05):
    """AP50 + recall + P/R at conf sweep -> dict per class + macro."""
    NGT = {c: sum(cl.count(c) for _, _, cl in GT.values()) for c in (0, 1)}
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
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        bi = int(f1.argmax())
        out[cn] = {'AP50': round(float(np.trapezoid(prec, rec)), 4),
                   'recall': round(float(tp.sum() / max(NGT[c], 1)), 3),
                   'P': round(float(prec[bi]), 3), 'R': round(float(rec[bi]), 3),
                   'F1': round(float(f1[bi]), 3),
                   'tp': int(tp.sum()), 'npred': len(det), 'gt': NGT[c]}
    out['AP50_macro'] = round(np.mean([out[c]['AP50'] for c in CLS]), 4)
    return out


def sizebin_recall(preds, GT, conf=0.05):
    """recall split by GT mask area: small<32^2, med<96^2, large."""
    bins = {}
    for st, (im, gm_, gc_) in GT.items():
        for c_, s_, m_ in preds.get(st, []):
            pass
        for j, g in enumerate(gm_):
            a = g.sum()
            b = 'small' if a < 1024 else ('med' if a < 9216 else 'large')
            hit = 0
            for c_, s_, m_ in preds.get(st, []):
                if c_ != gc_[j] or s_ < conf:
                    continue
                if iou_mat([m_], [g])[0, 0] >= .5:
                    hit = 1; break
            bins.setdefault((gc_[j], b), [0, 0])
            bins[(gc_[j], b)][1] += 1
            bins[(gc_[j], b)][0] += hit
    return {f'{CLS[c]}_{b}': round(tp / n, 3) if n else None
            for (c, b), (tp, n) in sorted(bins.items())}


def flops_params(model, size=(960, 960), dev=DEV):
    """GFLOPs + params via thop on a dummy input."""
    from thop import profile
    m = model
    if hasattr(m, 'model'):
        m = m.model
    x = torch.randn(1, 3, *size).to(dev)
    try:
        f, p = profile(m, (x,), verbose=False)
    except Exception as e:
        return None, None
    return round(f / 1e9, 2), round(p / 1e6, 2)


def latency_ms(fn, n_warm=10, n_iter=50):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return round((time.perf_counter() - t0) / n_iter * 1000, 2)

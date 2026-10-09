# ============================================================
# MODEL SOUP evaluation — average weights of compatible checkpoints,
# then score on the grouped test split with the SAME custom harness
# used in final_pipeline.ipynb (strict / sieved mask mAP50-95 + TP/FP/FN).
#
# Reference: Wortsman et al., "Model soups: averaging weights of
# multiple fine-tuned models" (ICML 2022) — uniform soup here;
# candidates auto-filtered to state_dicts identical in keys+shapes
# to runs/final_best (same architecture, any recipe).
# ============================================================
import json, time
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import torch
from ultralytics import YOLO

WORK = Path(__file__).resolve().parent
DS = WORK / 'data' / 'gold'
RUNS = WORK / 'runs'
RES = WORK / 'results'
REF = RUNS / 'final_best' / 'weights' / 'best.pt'
NAMES = {0: 'gravel', 1: 'sand'}

# ---------------- eval harness (identical to notebook) ----------------
TI = DS / 'test' / 'images'
TL = DS / 'test' / 'labels'
AREA_MIN, ASP = 0.005, 8.0
IOUS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def sieve(g):
    return g['area'] >= AREA_MIN and g['aspect'] <= ASP


def load_gt(lf, W, H):
    out = []
    if lf.exists():
        for ln in lf.read_text().strip().splitlines():
            p = ln.split()
            if len(p) < 7:
                continue
            xy = np.array(p[1:], float).reshape(-1, 2)
            px = (xy * [W, H]).astype(np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [px], 1)
            bw, bh = np.ptp(px[:, 0]), np.ptp(px[:, 1])
            out.append(dict(cls=int(p[0]), mask=m.astype(bool),
                            area=float(m.mean()),
                            aspect=float(max(bw, bh) / max(1, min(bw, bh)))))
    return out


def miou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.


GT, SHW = {}, {}
for ip in sorted(TI.glob('*.jpg')):
    im = cv2.imread(str(ip))
    H, W = im.shape[:2]
    SHW[ip.name] = (W, H)
    GT[ip.name] = load_gt(TL / (ip.stem + '.txt'), W, H)
print(f'test: {len(GT)} imgs | {sum(len(v) for v in GT.values())} GT insts')


def predict(model):
    PR = {}
    for nm, (W, H) in SHW.items():
        im = cv2.imread(str(TI / nm))
        r = model.predict(im, imgsz=640, conf=0.001, device=0,
                          retina_masks=True, verbose=False)[0]
        pr = []
        if r.masks is not None:
            for c, cf, mm in zip(r.boxes.cls.cpu().numpy().astype(int),
                                 r.boxes.conf.cpu().numpy(),
                                 r.masks.data.cpu().numpy()):
                mk = cv2.resize(mm, (W, H)) > 0
                if mk.mean() > 0:
                    pr.append(dict(cls=int(c), conf=float(cf), mask=mk))
        PR[nm] = pr
    return PR


def evaluate(PR, mode='strict', conf_op=0.25):
    aps = []
    for cls in (0, 1):
        n_gt = sum(1 for im in GT.values() for g in im if g['cls'] == cls
                   and (mode != 'sieved' or sieve(g)))
        aa = []
        for thr in IOUS:
            recs = []
            for nm in GT:
                keep = [g for g in GT[nm] if g['cls'] == cls
                        and (mode != 'sieved' or sieve(g))]
                used = set()
                for p in sorted([p for p in PR[nm] if p['cls'] == cls
                                 and (mode != 'sieved' or p['mask'].mean() >= AREA_MIN)],
                                key=lambda x: -x['conf']):
                    best, bj = 0., -1
                    for j, g in enumerate(keep):
                        if j in used:
                            continue
                        v = miou(p['mask'], g['mask'])
                        if v > best:
                            best, bj = v, j
                    if best >= thr:
                        used.add(bj)
                        recs.append((p['conf'], 1))
                    else:
                        recs.append((p['conf'], 0))
            recs.sort(key=lambda x: -x[0])
            tp = np.cumsum([r[1] for r in recs])
            fp = np.cumsum([1 - r[1] for r in recs])
            rec = tp / max(n_gt, 1)
            prec = np.divide(tp, np.maximum(tp + fp, 1))
            mrec = np.concatenate([[0], rec, [1]])
            mpre = np.concatenate([[0], prec, [0]])
            for i in range(len(mpre) - 2, -1, -1):
                mpre[i] = max(mpre[i], mpre[i + 1])
            aa.append(mpre[np.minimum(
                np.searchsorted(mrec, np.linspace(0, 1, 101), 'left'),
                len(mpre) - 1)].mean())
        aps.append(float(np.nanmean(aa)))
    tp = fp = fn = 0
    for nm in GT:
        used = set()
        keep = [g for g in GT[nm] if mode != 'sieved' or sieve(g)]
        for p in sorted([p for p in PR[nm] if p['conf'] >= conf_op
                         and (mode != 'sieved' or p['mask'].mean() >= AREA_MIN)],
                        key=lambda x: -x['conf']):
            best, bj = 0., -1
            for j, g in enumerate(keep):
                if j in used or g['cls'] != p['cls']:
                    continue
                v = miou(p['mask'], g['mask'])
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                used.add(bj)
                tp += 1
            else:
                fp += 1
        fn += len(keep) - len(used)
    return dict(mAP=float(np.mean(aps)), gravel=aps[0], sand=aps[1],
                TP=tp, FP=fp, FN=fn)


# ---------------- model soup ----------------
def state_of(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    m = ckpt.get('ema') or ckpt.get('model')
    sd = m.float().state_dict() if hasattr(m, 'state_dict') else m
    return sd, ckpt


def compatible(sd, ref_sd):
    return (set(sd) == set(ref_sd) and
            all(sd[k].shape == ref_sd[k].shape and
                sd[k].dtype == ref_sd[k].dtype for k in ref_sd))


def soup(ckpts, out_path):
    acc = None
    for p in ckpts:
        sd, _ = state_of(p)
        if acc is None:
            acc = {k: v.clone().double() for k, v in sd.items()}
        else:
            for k in acc:
                acc[k] += sd[k].double()
    n = len(ckpts)
    avg = {k: (v / n).to(torch.float32) for k, v in acc.items()}
    ref_sd, ref_ckpt = state_of(REF)
    ref_model = ref_ckpt.get('model')
    ref_model.load_state_dict(avg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in ref_ckpt.items() if k != 'ema'} |
               {'model': ref_model}, out_path)
    return out_path


def main():
    print('scanning checkpoints for shape-compatibility with final_best...')
    ref_sd, _ = state_of(REF)
    pool = {}
    for p in sorted(RUNS.glob('*/weights/*.pt')):
        if 'smoke' in str(p) or 'solov2' in str(p) or 'soup' in str(p):
            continue
        try:
            sd, _ = state_of(p)
            pool[str(p)] = compatible(sd, ref_sd)
        except Exception as e:
            print('  skip(unreadable):', p.name, e)
            pool[str(p)] = False
    compat = [p for p, ok in pool.items() if ok]
    print(f'compatible: {len(compat)}/{sum(1 for _ in pool)} scanned')
    for p in compat:
        print('  +', Path(p).parent.parent.name, Path(p).name)

    fb_best = str(REF)
    fb_last = str(RUNS / 'final_best' / 'weights' / 'last.pt')
    gold_bests = [p for p in compat
                  if p.endswith('best.pt') and
                  Path(p).parent.parent.name.startswith(('arm_gold', 'final', 'opt_'))]
    recipes = {
        'soup_pair_final': [fb_best, fb_last],
        'soup_gold_bests': sorted(gold_bests),
        'soup_all_compat': sorted(compat),
    }
    recipes = {k: [Path(x) for x in v if Path(x).exists()]
               for k, v in recipes.items() if len(v) > 1}

    rows = []
    base_m = YOLO(fb_best)
    base_pr = predict(base_m)
    for mode in ('strict', 'sieved'):
        r = evaluate(base_pr, mode)
        rows.append(dict(soup='FINAL_BEST (no soup)', n_ckpt=1,
                         protocol=mode, **r))
    del base_m
    torch.cuda.empty_cache()

    for name, ckpts in recipes.items():
        out = RUNS / name / 'weights' / 'best.pt'
        print(f'\n== {name}: averaging {len(ckpts)} ckpts ==')
        soup(ckpts, out)
        m = YOLO(str(out))
        pr = predict(m)
        for mode in ('strict', 'sieved'):
            r = evaluate(pr, mode)
            rows.append(dict(soup=name, n_ckpt=len(ckpts),
                             protocol=mode, **r))
            print(f'   {mode}: mAP={r["mAP"]:.4f} gravel={r["gravel"]:.3f} '
                  f'sand={r["sand"]:.3f} TP={r["TP"]} FP={r["FP"]} FN={r["FN"]}',
                  flush=True)
        mt = m.val(data=str(DS / 'data.yaml'), split='test', imgsz=640,
                   device=0, plots=False, verbose=False)
        rows[-1]['ultra_mAP50_95'] = round(float(mt.seg.map), 4)
        rows[-2]['ultra_mAP50_95'] = round(float(mt.seg.map), 4)
        del m, pr
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    df.to_csv(RES / 'soup_eval.csv', index=False)
    print('\n===== LEADERBOARD =====')
    print(df.to_string(index=False))


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'\ntotal {time.time() - t0:.0f}s')

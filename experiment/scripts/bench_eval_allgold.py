"""nb16: eval every bench model on ALL gold images (train+valid+test, 1084 imgs).
Seen-scene ceiling only — many frames were training data. Saves
preds_<n>_allgold.npy + allgold_<n>.json (per-class AP50/P/R/F1) + per-uuid csv."""
import json, sys, time
from pathlib import Path
import numpy as np, cv2, torch, pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from bench_common import EXP, GOLD, OUT, CLS, DEV, load_gt, eval_preds

GT = {}
for sp in ('train', 'valid', 'test'):
    GT.update(load_gt(GOLD / sp))
print('gold imgs:', len(GT), flush=True)

def save_eval(name, preds):
    np.save(OUT / f'preds_{name}_allgold.npy',
            np.array({k: [(c, s, m.astype(np.uint8)) for c, s, m in v]
                      for k, v in preds.items()}, dtype=object), allow_pickle=True)
    res = eval_preds(preds, GT)
    json.dump(res, open(OUT / f'allgold_{name}.json', 'w'), indent=1)
    # per-uuid recall
    from bench_common import iou_mat
    rec = {}
    for st, (im, gm_, gc_) in GT.items():
        grp = st.split('-')[0]
        for j, g in enumerate(gm_):
            hit = 0
            for c_, s_, m_ in preds.get(st, []):
                if c_ != gc_[j] or s_ < .05: continue
                if iou_mat([m_], [g])[0, 0] >= .5: hit = 1; break
            rec.setdefault((grp, gc_[j]), [0, 0])
            rec[(grp, gc_[j])][1] += 1; rec[(grp, gc_[j])][0] += hit
    df = pd.DataFrame([(g, CLS[c], tp, n) for (g, c), (tp, n) in rec.items()],
                      columns=['group', 'cls', 'tp', 'gt'])
    df['recall'] = df['tp'] / df['gt'].clip(lower=1)
    df.to_csv(OUT / f'per_uuid_allgold_{name}.csv', index=False)
    print(name, res['AP50_macro'], flush=True)

from ultralytics import YOLO
def yolo_preds(wpt, imgsz=960):
    m = YOLO(str(wpt)); out = {}
    for st, (im, gm_, gc_) in GT.items():
        r = m.predict(im, imgsz=imgsz, conf=.05, retina_masks=True,
                      device=DEV, verbose=False)[0]
        H, W = im.shape[:2]
        out[st] = [] if r.masks is None else [
            (int(c_), float(s_),
             cv2.resize((mm > .5).astype(np.uint8), (W, H),
                        interpolation=cv2.INTER_NEAREST).astype(bool))
            for c_, s_, mm in zip(r.boxes.cls.cpu().numpy().astype(int),
                                  r.boxes.conf.cpu().numpy(),
                                  r.masks.data.cpu().numpy())]
    del m; torch.cuda.empty_cache()
    return out

if __name__ == '__main__':
    which = sys.argv[1:] if len(sys.argv) > 1 else ['yolo26n','yolo26s','mrcnn','unet']
    for n in which:
        if n == 'yolo26n':
            save_eval(n, yolo_preds(EXP/'runs'/'ws'/'final'/'weights'/'best.pt'))
        elif n == 'yolo26s':
            save_eval(n, yolo_preds(EXP/'runs'/'bench_ws'/'yolo26s'/'weights'/'best.pt'))
        elif n == 'mrcnn':
            from bench_mrcnn import predict_masks
            from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
            mm = maskrcnn_resnet50_fpn_v2(weights=None, min_size=640, max_size=960)
            mm.load_state_dict(torch.load(OUT/'mrcnn_best.pt', weights_only=True))
            mm.to(DEV).eval()
            save_eval(n, {st: predict_masks(mm, GT[st][0]) for st in GT})
            del mm; torch.cuda.empty_cache()
        elif n == 'unet':
            from bench_unet import UNet, unet_instances
            um = UNet().to(DEV)
            um.load_state_dict(torch.load(OUT/'unet_best.pt', weights_only=True)); um.eval()
            save_eval(n, {st: unet_instances(um, GT[st][0]) for st in GT})
            del um; torch.cuda.empty_cache()

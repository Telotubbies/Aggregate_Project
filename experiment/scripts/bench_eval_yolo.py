"""eval a YOLO weights file on gold_ws/test -> preds_{name}.npy + {name}_eval.json"""
import json, sys
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, str(Path(__file__).parent))
from bench_common import EXP, WS, OUT, DEV, load_gt, eval_preds
from ultralytics import YOLO

name, wpt = sys.argv[1], sys.argv[2]

def predict_flat(model, im, conf=0.05, imgsz=960):
    r = model.predict(im, imgsz=imgsz, conf=conf, retina_masks=True,
                      device=DEV, verbose=False)[0]
    if r.masks is None:
        return []
    H, W = im.shape[:2]
    return [(int(c_), float(s_), cv2.resize((m_ > 0.5).astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST).astype(bool))
            for c_, s_, m_ in zip(r.boxes.cls.cpu().numpy().astype(int),
                                  r.boxes.conf.cpu().numpy(),
                                  r.masks.data.cpu().numpy())]

GT = load_gt(WS / 'test')
m = YOLO(wpt)
preds = {st: predict_flat(m, GT[st][0]) for st in GT}
np.save(OUT / f'preds_{name}.npy',
        np.array({k: [(c, s, mm.astype(np.uint8)) for c, s, mm in v]
                  for k, v in preds.items()}, dtype=object), allow_pickle=True)
res = eval_preds(preds, GT)
json.dump({'test': res}, open(OUT / f'{name}_eval.json', 'w'), indent=1)
print(name, json.dumps(res))

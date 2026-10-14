"""nb17 arm: torchvision Mask R-CNN R50-FPN v2 on gold_ws_aug.
YOLO-polygon labels -> per-instance masks+boxes. Standard COCO-finetune recipe:
SGD lr .005 mom .9 wd 5e-4, 12 epochs, step lr x0.1 @8,11, batch 4.
Saves best-by-valid-AP50 weights + test preds (npz) + train_log.json.
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bench_common import EXP, WS, OUT, CLS, DEV, load_gt, iou_mat, eval_preds

TRAIN = EXP / 'data' / 'gold_ws_aug'
EPOCHS, BS, LR = 12, 2, 0.005
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)


def yolo_to_targets(img_path, lt_path):
    im = cv2.imread(str(img_path))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    H, W = im.shape[:2]
    masks, boxes, labels = [], [], []
    if lt_path.exists():
        for ln in lt_path.read_text().splitlines():
            p = ln.split()
            xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [xy.astype(np.int32)], 1)
            if m.sum() < 4:
                continue
            ys, xs = np.where(m)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            boxes.append([x1, y1, x2, y2])
            masks.append(m); labels.append(int(p[0]) + 1)  # 0=bg
    t = {'boxes': torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
         'labels': torch.as_tensor(labels, dtype=torch.int64),
         'masks': torch.as_tensor(np.stack(masks) if masks else
                                  np.zeros((0, H, W), np.uint8), dtype=torch.uint8),
         'image_id': torch.tensor([hash(img_path.stem) % 2**31])}
    return torch.from_numpy(im).permute(2, 0, 1).float() / 255., t


class DS(Dataset):
    def __init__(self, split):
        self.d = split
        self.imgs = sorted((split / 'images').iterdir())
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        ip = self.imgs[i]
        return yolo_to_targets(ip, self.d / 'labels' / (ip.stem + '.txt'))


def collate(b):
    return tuple(zip(*b))


def predict_masks(model, im_bgr, conf=0.05):
    rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).to(DEV)
    with torch.no_grad():
        o = model([x])[0]
    out = []
    for lb, s, m in zip(o['labels'].cpu().numpy(), o['scores'].cpu().numpy(),
                        o['masks'].cpu().numpy()[:, 0]):
        if s < conf or lb < 1 or lb > 2:
            continue
        out.append((int(lb) - 1, float(s), m > 0.5))
    return out


def valid_ap50(model, GT):
    model.eval()
    preds = {st: predict_masks(model, GT[st][0]) for st in GT}
    model.train()
    return eval_preds(preds, GT)['AP50_macro']


def main():
    model = maskrcnn_resnet50_fpn_v2(weights='DEFAULT',
                                     min_size=640, max_size=960)
    model.to(DEV).train()
    ds = DS(TRAIN / 'train')
    dl = DataLoader(ds, BS, shuffle=True, num_workers=4,
                    collate_fn=collate, drop_last=True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                          lr=LR, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.MultiStepLR(opt, [8, 11], 0.1)
    GTv = dict(list(load_gt(TRAIN / 'valid').items())[:60])  # subset: light ckpt metric
    best_ap, best_ep, t0 = -1, -1, time.perf_counter()
    log = []
    for ep in range(EPOCHS):
        ls, nb = 0., 0
        for imgs, tgts in dl:
            imgs = [i.to(DEV) for i in imgs]
            tgts = [{k: v.to(DEV) for k, v in t.items()} for t in tgts]
            loss = sum(model(imgs, tgts).values())
            opt.zero_grad(); loss.backward(); opt.step()
            ls += float(loss); nb += 1
        sch.step()
        ap = valid_ap50(model, GTv)
        log.append(dict(epoch=ep + 1, loss=round(ls / max(nb, 1), 4), val_ap50=ap))
        print(f'ep{ep+1} loss {ls/max(nb,1):.3f} val AP50 {ap}', flush=True)
        if ap > best_ap:
            best_ap, best_ep = ap, ep + 1
            torch.save(model.state_dict(), OUT / 'mrcnn_best.pt')
    torch.save(model.state_dict(), OUT / 'mrcnn_last.pt')

    model.load_state_dict(torch.load(OUT / 'mrcnn_best.pt', weights_only=True))
    model.eval()
    GTt = load_gt(WS / 'test')
    preds = {st: predict_masks(model, GTt[st][0]) for st in GTt}
    np.save(OUT / 'preds_mrcnn.npy',
            np.array({k: [(c, s, m.astype(np.uint8)) for c, s, m in v]
                      for k, v in preds.items()}, dtype=object), allow_pickle=True)
    res = eval_preds(preds, GTt)
    json.dump({'test': res, 'best_ep': best_ep, 'best_val_ap50': best_ap,
               'train_log': log, 'hours': round((time.perf_counter() - t0) / 3600, 2)},
              open(OUT / 'mrcnn_eval.json', 'w'), indent=1)
    print('MRCNN DONE', json.dumps(res))


if __name__ == '__main__':
    main()

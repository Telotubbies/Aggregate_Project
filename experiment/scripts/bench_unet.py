"""nb17 arm: compact U-Net semantic seg (bg/gravel/sand) + watershed instances.
Semantic train at 512^2, Adam 1e-3, BCE+Dice, 40 epochs, batch 8.
Instances: per-class prob map -> dist-transform peaks -> skimage watershed.
Saves weights + test preds + unet_eval.json.
"""
import json, time
from pathlib import Path
import numpy as np, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage as ndi
from skimage.segmentation import watershed
import sys
sys.path.insert(0, str(Path(__file__).parent))
from bench_common import EXP, WS, OUT, CLS, DEV, load_gt, eval_preds

TRAIN = EXP / 'data' / 'gold_ws_aug'
IMGSZ, EPOCHS, BS, LR = 512, 40, 8, 1e-3
torch.manual_seed(42); np.random.seed(42)


def conv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(self, b=32, ncls=3):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = conv(3, b), conv(b, b*2), conv(b*2, b*4), conv(b*4, b*8)
        self.mid = conv(b*8, b*16)
        self.u4, self.d4 = nn.ConvTranspose2d(b*16, b*8, 2, 2), conv(b*16, b*8)
        self.u3, self.d3 = nn.ConvTranspose2d(b*8, b*4, 2, 2), conv(b*8, b*4)
        self.u2, self.d2 = nn.ConvTranspose2d(b*4, b*2, 2, 2), conv(b*4, b*2)
        self.u1, self.d1 = nn.ConvTranspose2d(b*2, b, 2, 2), conv(b*2, b)
        self.head = nn.Conv2d(b, ncls, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        m = self.mid(self.pool(e4))
        d4 = self.d4(torch.cat([self.u4(m), e4], 1))
        d3 = self.d3(torch.cat([self.u3(d4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def raster(ip, lt, sz):
    im = cv2.imread(str(ip)); H, W = im.shape[:2]
    sem = np.zeros((H, W), np.uint8)
    if lt.exists():
        for ln in lt.read_text().splitlines():
            p = ln.split()
            xy = np.asarray(p[1:], np.float32).reshape(-1, 2) * [W, H]
            cv2.fillPoly(sem, [xy.astype(np.int32)], int(p[0]) + 1)
    im = cv2.resize(im, (sz, sz))
    sem = cv2.resize(sem.astype(np.uint8), (sz, sz),
                     interpolation=cv2.INTER_NEAREST).astype(np.int64)
    return (torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            .permute(2, 0, 1).float() / 255., torch.from_numpy(sem))


class DS(Dataset):
    def __init__(self, split):
        self.d = split
        self.imgs = sorted((split / 'images').iterdir())
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, i):
        ip = self.imgs[i]
        return raster(ip, self.d / 'labels' / (ip.stem + '.txt'), IMGSZ)


def dice_loss(logits, y, ncls=3):
    p = F.softmax(logits, 1)
    oh = F.one_hot(y, ncls).permute(0, 3, 1, 2).float()
    i = (p * oh).sum((2, 3)); u = (p + oh).sum((2, 3))
    return 1 - (2 * i / u.clamp(min=1)).mean()


def unet_instances(model, im_bgr, conf=0.3):
    """semantic prob -> per-class watershed instances [(cls,score,mask)]."""
    H, W = im_bgr.shape[:2]
    x = torch.from_numpy(cv2.cvtColor(cv2.resize(im_bgr, (IMGSZ, IMGSZ)),
                         cv2.COLOR_BGR2RGB)).permute(2, 0, 1)[None].float() / 255.
    with torch.no_grad():
        pr = F.softmax(model(x.to(DEV)), 1)[0].cpu().numpy()
    pr = np.stack([cv2.resize(pr[c], (W, H)) for c in range(3)])
    out = []
    for c in (1, 2):
        fg = (pr[c] > conf).astype(np.uint8)
        if not fg.any():
            continue
        dist = ndi.distance_transform_edt(fg)
        peaks = (dist == ndi.maximum_filter(dist, 9)) & (dist > 2)
        mk, _ = ndi.label(peaks)
        if mk.max() == 0:
            mk, _ = ndi.label(fg)
        ws = watershed(-pr[c], mk, mask=fg.astype(bool))
        for k in range(1, ws.max() + 1):
            m = ws == k
            if m.sum() < 25:
                continue
            out.append((c - 1, float(pr[c][m].mean()), m))
    return out


def main():
    model = UNet().to(DEV)
    print('params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
    ds = DS(TRAIN / 'train')
    dl = DataLoader(ds, BS, shuffle=True, num_workers=4, drop_last=True)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    GTv = load_gt(TRAIN / 'valid')
    best, best_ep, t0 = -1, -1, time.perf_counter()
    log = []
    for ep in range(EPOCHS):
        model.train(); ls = 0.; nb = 0
        for x, y in dl:
            x, y = x.to(DEV), y.to(DEV)
            lg = model(x)
            loss = F.cross_entropy(lg, y) + dice_loss(lg, y)
            opt.zero_grad(); loss.backward(); opt.step()
            ls += float(loss); nb += 1
        model.eval()
        preds = {st: unet_instances(model, GTv[st][0]) for st in list(GTv)}
        ap = eval_preds(preds, GTv)['AP50_macro']
        log.append(dict(epoch=ep + 1, loss=round(ls / max(nb, 1), 4), val_ap50=ap))
        print(f'ep{ep+1} loss {ls/max(nb,1):.3f} val AP50 {ap}', flush=True)
        if ap > best:
            best, best_ep = ap, ep + 1
            torch.save(model.state_dict(), OUT / 'unet_best.pt')
    torch.save(model.state_dict(), OUT / 'unet_last.pt')

    model.load_state_dict(torch.load(OUT / 'unet_best.pt', weights_only=True))
    model.eval()
    GTt = load_gt(WS / 'test')
    preds = {st: unet_instances(model, GTt[st][0]) for st in GTt}
    np.save(OUT / 'preds_unet.npy',
            np.array({k: [(c, s, m.astype(np.uint8)) for c, s, m in v]
                      for k, v in preds.items()}, dtype=object), allow_pickle=True)
    res = eval_preds(preds, GTt)
    json.dump({'test': res, 'best_ep': best_ep, 'best_val_ap50': best,
               'train_log': log, 'hours': round((time.perf_counter() - t0) / 3600, 2)},
              open(OUT / 'unet_eval.json', 'w'), indent=1)
    print('UNET DONE', json.dumps(res))


if __name__ == '__main__':
    main()

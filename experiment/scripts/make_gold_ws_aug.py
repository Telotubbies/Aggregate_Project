"""data/gold_ws_aug/ — offline data expansion of gold_ws/train.

Recipe = nb00 §10.2 (proven arm: aug beat all controlled recipes), K=3
variants per train image + env variant, PLUS hard-group oversampling
(K=5) for FN-heavy scenes identified by LOGO
(class-/scene-balanced augmentation; cf. Cui et al. 2019, Ghiasi et al.
2021, Shorten & Khoshgoftaar 2019). valid/test copied untouched.
"""
from pathlib import Path
import shutil
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from hashlib import md5 as _md5

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
SRC = EXP / 'data' / 'gold_ws'
AUG_DIR = EXP / 'data' / 'gold_ws_aug'
OUT = EXP / 'results' / 'gold_ws'
OUT.mkdir(parents=True, exist_ok=True)

HARD = {'bbf45eda', '1991989f', '4a7f29af'}   # sand AP <= .27 in LOGO
K_AUG, K_HARD, SEED = 3, 5, 42

def md5(p):
    return _md5(p.read_bytes()).hexdigest()

def load_polys(lbl_path):
    out = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            out.append((int(parts[0]),
                        np.array(list(map(float, parts[1:]))).reshape(-1, 2)))
    return out

def shoelace_area(xs, ys):
    return 0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))

def clean_polys(polys, min_area=1e-5):
    keep = []
    for c, p in polys:
        p = np.clip(p, 0.0, 1.0)
        if len(np.unique(p, axis=0)) < 3 or shoelace_area(p[:, 0], p[:, 1]) < min_area:
            continue
        keep.append((c, p))
    return keep

def aug_hflip(img, polys):
    return cv2.flip(img, 1), [(c, np.c_[1 - p[:, 0], p[:, 1]]) for c, p in polys]

def aug_affine(img, polys, deg, scale, tx, ty):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    M[:, 2] += [tx * w, ty * h]
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
    newp = [(c, np.clip(np.c_[p * [w, h], np.ones(len(p))] @ M.T / [w, h], 0.0, 1.0))
            for c, p in polys]
    return out, newp

def aug_hsv(img, polys, dh, ds, dv):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + dh) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + ds), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + dv), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB), polys

def aug_dark(img, g, blue=0.10):
    lut = ((np.arange(256) / 255.0) ** g * 255).astype(np.uint8)
    out = cv2.LUT(img, lut).astype(np.float32)
    out[:, :, 0] *= (1 - blue)
    out[:, :, 2] *= (1 + blue * 0.5)
    return np.clip(out, 0, 255).astype(np.uint8)

def aug_rain(img, rng, density=0.0006, angle_deg=None, length=None):
    h, w = img.shape[:2]
    overlay = np.zeros_like(img)
    n = int(h * w * density)
    ang = np.deg2rad(angle_deg if angle_deg is not None else rng.uniform(75, 105))
    dx, dy = np.cos(ang), np.sin(ang)
    for _ in range(n):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        L = length if length is not None else rng.uniform(8, 22)
        cv2.line(overlay, (int(x0), int(y0)),
                 (int(x0 + dx * L), int(y0 + dy * L)),
                 (int(rng.uniform(160, 220)),) * 3, 1, cv2.LINE_AA)
    img = cv2.addWeighted(img, 1.0, cv2.GaussianBlur(overlay, (3, 3), 0), 0.6, 4)
    return cv2.GaussianBlur(img, (3, 3), 0)

def clahe_lab(img, clip=2.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2RGB)

def one_variant(img, polys, rng):
    ops = []
    if rng.random() < 0.5:
        img, polys = aug_hflip(img, polys); ops.append("hflip")
    if rng.random() < 0.6:
        img, polys = aug_affine(img, polys,
                                deg=rng.uniform(-12, 12),
                                scale=rng.uniform(0.8, 1.25),
                                tx=rng.uniform(-0.08, 0.08),
                                ty=rng.uniform(-0.08, 0.08))
        ops.append("affine")
    if rng.random() < 0.6:
        img, polys = aug_hsv(img, polys,
                             dh=rng.uniform(-7, 7),
                             ds=rng.uniform(-0.25, 0.25),
                             dv=rng.uniform(-0.25, 0.25))
        ops.append("hsv")
    if rng.random() < 0.4:
        g = rng.uniform(0.8, 1.25)
        img = cv2.LUT(img, ((np.arange(256) / 255.0) ** (1 / g) * 255).astype(np.uint8))
        ops.append(f"gamma={g:.2f}")
    if rng.random() < 0.3:
        img = clahe_lab(img); ops.append("clahe")
    if rng.random() < 0.15:
        s = rng.uniform(0.1, 1.2)
        img = cv2.GaussianBlur(img, (0, 0), s); ops.append(f"blur={s:.2f}")
    return img, clean_polys(polys), ";".join(ops) or "none"

def env_variant(img, polys, rng):
    ops = []
    do_dark, do_rain = rng.random() < 0.5, rng.random() < 0.5
    if not (do_dark or do_rain):
        do_dark = True
    if do_dark:
        img = aug_dark(img, g=rng.uniform(1.4, 2.2), blue=rng.uniform(0.05, 0.15))
        ops.append("dark")
    if do_rain:
        img = aug_rain(img, rng, density=rng.uniform(0.0003, 0.0009),
                       angle_deg=rng.uniform(75, 105))
        ops.append("rain")
    return img, clean_polys(polys), ";".join(ops)

def write_label(path, polys):
    lines = [f"{c} " + " ".join(f"{v:.6f}" for v in p.ravel()) for c, p in polys]
    path.write_text("\n".join(lines))

if AUG_DIR.exists():
    shutil.rmtree(AUG_DIR)
for split in ("train", "valid", "test"):
    (AUG_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (AUG_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
log = []
train_files = sorted((SRC / "train" / "images").iterdir())
n_hard = 0
for f in train_files:
    stem, ext = f.stem, f.suffix
    hard = stem.split('-')[0] in HARD
    n_hard += hard
    img = np.asarray(Image.open(f).convert("RGB"))
    polys = load_polys(SRC / "train" / "labels" / f"{stem}.txt")
    Image.fromarray(img).save(AUG_DIR / "train" / "images" / f.name)
    write_label(AUG_DIR / "train" / "labels" / f"{stem}.txt", polys)
    log.append({"image": f.name, "variant": "orig", "ops": "",
                "hard": hard, "n_inst": len(polys)})
    k = K_HARD if hard else K_AUG
    for i in range(1, k + 1):
        if i == k:
            out_img, out_p, ops = env_variant(img, polys, rng)
        else:
            out_img, out_p, ops = one_variant(img, polys, rng)
        name = f"{stem}_aug{i}{ext}"
        Image.fromarray(out_img).save(AUG_DIR / "train" / "images" / name)
        write_label(AUG_DIR / "train" / "labels" / f"{stem}_aug{i}.txt", out_p)
        log.append({"image": name, "variant": f"aug{i}", "ops": ops,
                    "hard": hard, "n_inst": len(out_p)})

for split in ("valid", "test"):
    for img_path in (SRC / split / "images").iterdir():
        shutil.copy2(img_path, AUG_DIR / split / "images" / img_path.name)
        lt = SRC / split / "labels" / (img_path.stem + ".txt")
        (AUG_DIR / split / "labels" / f"{img_path.stem}.txt").write_text(
            lt.read_text() if lt.exists() else "")

(AUG_DIR / "data.yaml").write_text(
    f"path: {AUG_DIR}\ntrain: train/images\nval: valid/images\n"
    "test: test/images\nnc: 2\nnames: ['gravel', 'sand']\n")
log_df = pd.DataFrame(log)
log_df.to_csv(AUG_DIR / "aug_log.csv", index=False)

# ---- verification ----
checks = {}
tr = AUG_DIR / "train"
n_img = len(list((tr / "images").iterdir()))
n_lbl = len(list((tr / "labels").iterdir()))
exp = len(train_files) + n_hard * K_HARD + (len(train_files) - n_hard) * K_AUG
checks["train images = expected count"] = n_img == exp
checks["label file per image"] = n_img == n_lbl
bad = 0
for lt in (tr / "labels").iterdir():
    for line in lt.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        vals = np.array(list(map(float, parts[1:])))
        if int(parts[0]) not in (0, 1) or len(parts) < 7 or ((vals < 0) | (vals > 1)).any():
            bad += 1
checks["all coords in [0,1], classes valid, >=3 pts"] = bad == 0
for s in ("valid", "test"):
    checks[f"{s} labels identical to gold_ws"] = all(
        md5(a) == md5(b) for a, b in zip(
            sorted((SRC / s / "labels").iterdir()),
            sorted((AUG_DIR / s / "labels").iterdir())))
for k, v in checks.items():
    print(f"  [{'PASS' if v else 'FAIL'}] {k}")

def inst_of(split_dir):
    g = s = 0
    for lt in (split_dir / "labels").glob("*.txt"):
        for ln in lt.read_text().splitlines():
            if ln.strip():
                g += ln.split()[0] == '0'; s += ln.split()[0] == '1'
    return g, s

g, s = inst_of(tr)
print(f"\n{len(train_files)} originals ({n_hard} hard) -> {n_img} train imgs "
      f"({n_img/len(train_files):.1f}x)")
print(f"train instances: gravel {g} | sand {s}")
hard_log = log_df[log_df.hard]
print(f"hard-group imgs: {n_hard} orig -> {len(hard_log)} total "
      f"(oversampled {len(hard_log)/max(n_hard,1):.1f}x)")
print(f"written: {AUG_DIR}")

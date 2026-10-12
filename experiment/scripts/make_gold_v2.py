"""Build data/gold_v2 — uuid-grouped balanced re-split + audit + viz."""
import csv
from pathlib import Path
from collections import Counter
import numpy as np, cv2, shutil, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

EXP = Path('/home/supawich/Desktop/Aggregate_Project/experiment')
G = EXP / 'data' / 'gold'
V2 = EXP / 'data' / 'gold_v2'
OUT = EXP / 'results' / 'resplit_v2'
OUT.mkdir(parents=True, exist_ok=True)
CLS = ['gravel', 'sand']

# ---------- 1. group table ----------
grp = {}  # uuid -> dict(split, imgs, grav, sand)
for split in ('train', 'valid', 'test'):
    for lt in (G / split / 'labels').glob('*.txt'):
        u = lt.stem.split('-')[0]
        d = grp.setdefault(u, dict(split=split, imgs=0, grav=0, sand=0))
        d['imgs'] += 1
        for ln in lt.read_text().splitlines():
            if ln.strip():
                d['grav' if ln.split()[0] == '0' else 'sand'] += 1
GT = pd.DataFrame(grp).T.infer_objects().sort_values('imgs', ascending=False)
for c in ('imgs', 'grav', 'sand'):
    GT[c] = GT[c].astype(int)
print(GT.to_string())

# ---------- 2. assignment (hand-tuned greedy on the 3 big sand groups,
#               gravel groups fill to ~70/15/15 of 1084 imgs) ----------
ASSIGN = {
    # sand-heavy groups
    '2be0c9f6': 'train',  '4a7f29af': 'train', '1991989f': 'train',
    '07451999': 'train',  '5d14519a': 'train', '04802d75': 'train',
    'fe3e6f0d': 'train',  '0a48349e': 'valid', '6459c886': 'test',
    'bbf45eda': 'test',
    # gravel-heavy groups
    '0f964fdb': 'train',  'd56fcb3c': 'train', '6f40fbbd': 'train',
    'fa45e6fb': 'train',  '8f435cb5': 'train', '5d81688a': 'train',
    '96ecd4fe': 'valid',  'e9eebbd1': 'valid', '5079fbab': 'valid',
    '7a600bac': 'valid',  '7b6b3d3f': 'valid',
    'f9b12ec3': 'test',   'b029e315': 'test',
    '68935cd9': 'test',   '2f4b2f2f': 'test', '9056e676': 'test',
}
# anything unassigned (rf-download singles etc.) -> train
for u in grp:
    ASSIGN.setdefault(u, 'train')

GT['v2'] = pd.Series({u: ASSIGN[u] for u in GT.index})

def comp(col):
    return GT.groupby(col)[['imgs', 'grav', 'sand']].sum()

print('\n=== BEFORE (gold) ===')
print(comp('split').to_string())
print('\n=== AFTER (gold_v2) ===')
print(comp('v2').to_string())

GT.to_csv(OUT / 'group_table.csv')

# ---------- 3. write dataset (hardlink) ----------
def link_or_copy(src, dst):
    try:
        dst.hardlink_to(src)
    except OSError:
        shutil.copy2(src, dst)

if V2.exists():
    shutil.rmtree(V2)
for split in ('train', 'valid', 'test'):
    for kind in ('images', 'labels'):
        (V2 / split / kind).mkdir(parents=True, exist_ok=True)

stem2v2 = {}
for u, row in GT.iterrows():
    src_split = row['split']; dst_split = row['v2']
    for f in (G / src_split / 'images').glob(u + '-*'):
        link_or_copy(f, V2 / dst_split / 'images' / f.name)
        lt = G / src_split / 'labels' / (f.stem + '.txt')
        if lt.exists():
            link_or_copy(lt, V2 / dst_split / 'labels' / lt.name)
        stem2v2[f.stem] = dst_split
    # rf-download singles have no '-' suffix pattern match too
    for f in (G / src_split / 'images').glob(u + '*'):
        if f.stem in stem2v2:
            continue
        link_or_copy(f, V2 / dst_split / 'images' / f.name)
        lt = G / src_split / 'labels' / (f.stem + '.txt')
        if lt.exists():
            link_or_copy(lt, V2 / dst_split / 'labels' / lt.name)
        stem2v2[f.stem] = dst_split

(V2 / 'data.yaml').write_text(
    'path: .\ntrain: train/images\nval: valid/images\ntest: test/images\n'
    "nc: 2\nnames: ['gravel', 'sand']\n")
pd.Series(stem2v2).to_csv(OUT / 'stem_to_split.csv')

for split in ('train', 'valid', 'test'):
    print(split, len(list((V2 / split / 'images').iterdir())), 'imgs')

# ---------- 4. figs ----------
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(['imgs', 'grav', 'sand']):
    before = comp('split')[metric].reindex(['train', 'valid', 'test'])
    after = comp('v2')[metric].reindex(['train', 'valid', 'test'])
    x = np.arange(3); w = .36
    ax[i].bar(x - w / 2, before.values, w, label='gold (old)')
    ax[i].bar(x + w / 2, after.values, w, label='gold_v2 (new)')
    for xi, (b, a) in enumerate(zip(before.values, after.values)):
        ax[i].text(xi - w / 2, b, str(int(b)), ha='center', va='bottom', fontsize=7)
        ax[i].text(xi + w / 2, a, str(int(a)), ha='center', va='bottom', fontsize=7)
    ax[i].set_xticks(x); ax[i].set_xticklabels(before.index)
    ax[i].set_title(metric); ax[i].legend(fontsize=7); ax[i].grid(alpha=.3, axis='y')
fig.suptitle('gold vs gold_v2 — split composition')
plt.tight_layout(); plt.savefig(OUT / 'fig_split_compare.png', dpi=150)

# per-group stacked bars, one panel per split, old vs new
fig, ax = plt.subplots(2, 3, figsize=(16, 10))
for k, (col, ttl) in enumerate([('split', 'gold (old)'), ('v2', 'gold_v2')]):
    for si, split in enumerate(('train', 'valid', 'test')):
        sub = GT[GT[col] == split].sort_values('imgs')
        left = np.zeros(len(sub))
        for ci, met in enumerate(['grav', 'sand']):
            ax[k][si].barh(sub.index, sub[met], left=left,
                           color=['#3cc83c', '#ff9933'][ci],
                           label=CLS[ci] if (k == 0 and si == 0) else None)
            left += sub[met].values
        top = sub['imgs'].max() if len(sub) else 0
        ax[k][si].set_title(f'{ttl} · {split} — {len(sub)} groups, '
                            f'{int(sub["imgs"].sum())} imgs, top group {top} imgs', fontsize=9)
        ax[k][si].tick_params(axis='y', labelsize=6)
ax[0][0].legend(loc='lower right')
fig.suptitle('uuid-group composition per split (green=gravel, orange=sand)')
plt.tight_layout(); plt.savefig(OUT / 'fig_groups.png', dpi=150)
print('figs saved ->', OUT)

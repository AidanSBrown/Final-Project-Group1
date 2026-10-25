#!/usr/bin/env python3
"""Download the ISIC 2018 challenge data (Codella et al. 2019) into <repo>/data.

    python3 code/setup_data.py                  # everything, into <repo>/data
    python3 code/setup_data.py --splits val test # only some splits
    python3 code/setup_data.py --out /path/to/data --keep-zips

Downloads ~16 GB from the official ISIC bucket, one at a time; each zip is deleted
after extraction unless --keep-zips. Peak disk use is ~26 GB (the 10.4 GB Task 1-2
training zip plus its extracted images). Safe to re-run: finished steps are skipped.
Standard library only.

Layout:
    task1_segmentation/{train,val,test}/images/ISIC_xxxxxxx.jpg
    task1_segmentation/{train,val,test}/masks/ISIC_xxxxxxx_segmentation.png
    task2_attributes/{train,val,test}/images/ISIC_xxxxxxx.jpg          (same images as task 1)
    task2_attributes/{train,val,test}/masks/ISIC_xxxxxxx_attribute_<name>.png
    task3_classification/{train,val,test}/images/ISIC_xxxxxxx.jpg
    task3_classification/{train,val,test}/labels.csv                  (one-hot, 7 classes)
    task3_classification/train/lesion_groupings.csv                   (lesion_id per image)
"""
import argparse
import os
import re
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / 'data'  # <repo>/data, wherever this is run from
BASE = 'https://isic-challenge-data.s3.amazonaws.com/2018/'
SPLITS = {'train': 'Training', 'val': 'Validation', 'test': 'Test'}
ATTRS = ['globules', 'milia_like_cyst', 'negative_network', 'pigment_network', 'streaks']
IMG = re.compile(r'ISIC_\d{7}\.jpg')
SEG = re.compile(r'ISIC_\d{7}_segmentation\.png')
ATT = re.compile(r'ISIC_\d{7}_attribute_\w+\.png')
CSV = re.compile(r'ISIC2018_Task3_\w+_GroundTruth\.csv')


def download(name, dest):
    if dest.exists():
        return
    part = dest.with_name(dest.name + '.part')
    print(f'downloading {name}', flush=True)
    with urllib.request.urlopen(BASE + name) as r, open(part, 'wb') as f:
        total, done = int(r.headers.get('Content-Length', 0)), 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total > 10e6:
                print(f'\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB', end='', flush=True)
    if total > 10e6:
        print()
    part.rename(dest)


def extract(zpath, pattern, dst, rename=None):
    """Extract only files whose base name matches pattern, flattened into dst."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            name = Path(info.filename).name
            if info.is_dir() or '__MACOSX' in info.filename or not pattern.fullmatch(name):
                continue
            with z.open(info) as src, open(dst / (rename or name), 'wb') as out:
                shutil.copyfileobj(src, out)
            n += 1
    if n == 0:
        sys.exit(f'error: no files matching {pattern.pattern} in {zpath.name}')
    return n


def mirror(src, dst):
    """Give dst the same files as src: hard links (no extra space), copies if unsupported."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        t = dst / p.name
        if not t.exists():
            try:
                os.link(p, t)
            except OSError:
                shutil.copy2(p, t)


def check(out, splits):
    """Every image has its masks/label and vice versa."""
    def ids(d, suffix):
        return {p.name[:12] for p in d.glob('*' + suffix)}

    ok = True
    for s in splits:
        t1, t2, t3 = (out / t / s for t in ('task1_segmentation', 'task2_attributes', 'task3_classification'))
        imgs = ids(t1 / 'images', '.jpg')
        pairs = [('task1 masks', ids(t1 / 'masks', '_segmentation.png'))]
        pairs += [(f'task2 {a}', ids(t2 / 'masks', f'_attribute_{a}.png')) for a in ATTRS]
        t3_imgs = ids(t3 / 'images', '.jpg')
        rows = (t3 / 'labels.csv').read_text().split()[1:]
        pairs.append(('task3 labels', {r.split(',')[0] for r in rows}))
        for label, got in pairs:
            expected = t3_imgs if label == 'task3 labels' else imgs
            if got != expected:
                ok = False
                print(f'  {s} {label}: {len(expected - got)} missing, {len(got - expected)} without image')
        print(f'{s:5s}  task1/2: {len(imgs)} images   task3: {len(t3_imgs)} images, {len(rows)} labels')
    print('all checks passed' if ok else 'ERROR: some images are missing masks or labels')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=DATA, help=f'output folder (default: {DATA})')
    ap.add_argument('--splits', nargs='+', choices=list(SPLITS), default=list(SPLITS))
    ap.add_argument('--keep-zips', action='store_true', help='keep downloaded zips in <out>/_zips')
    args = ap.parse_args()

    out = Path(args.out)
    zdir = out / '_zips'
    zdir.mkdir(parents=True, exist_ok=True)

    def fetch(name, pattern, dst, rename=None):
        done = zdir / (name + '.done')
        if done.exists():
            return
        zpath = zdir / name
        download(name, zpath)
        n = extract(zpath, pattern, dst, rename)
        print(f'  {n} files -> {dst.relative_to(out)}')
        if not args.keep_zips:
            zpath.unlink()
        done.touch()

    t1, t2, t3 = out / 'task1_segmentation', out / 'task2_attributes', out / 'task3_classification'
    for s in args.splits:
        S = SPLITS[s]
        fetch(f'ISIC2018_Task1-2_{S}_Input.zip', IMG, t1 / s / 'images')
        mirror(t1 / s / 'images', t2 / s / 'images')
        fetch(f'ISIC2018_Task1_{S}_GroundTruth.zip', SEG, t1 / s / 'masks')
        fetch(f'ISIC2018_Task2_{S}_GroundTruth{"_v3" if s == "train" else ""}.zip', ATT, t2 / s / 'masks')
        fetch(f'ISIC2018_Task3_{S}_Input.zip', IMG, t3 / s / 'images')
        fetch(f'ISIC2018_Task3_{S}_GroundTruth.zip', CSV, t3 / s, rename='labels.csv')
    if 'train' in args.splits:
        download('ISIC2018_Task3_Training_LesionGroupings.csv', t3 / 'train' / 'lesion_groupings.csv')

    sys.exit(0 if check(out, args.splits) else 1)


if __name__ == '__main__':
    main()

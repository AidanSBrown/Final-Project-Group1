# data/ — ISIC 2018 challenge data (Codella et al. 2019)

Not tracked by git. Build it from the repo root with:

```bash
python3 code/setup_data.py
```

This downloads ~16 GB from the official ISIC bucket (peak disk use ~26 GB) and checks
that every image has its masks/labels. Re-running skips finished steps; `--splits val test`
builds only some splits.

```
task1_segmentation/   {train,val,test}/images/ISIC_xxxxxxx.jpg
                      {train,val,test}/masks/ISIC_xxxxxxx_segmentation.png
task2_attributes/     {train,val,test}/images/ISIC_xxxxxxx.jpg      (same images as task 1)
                      {train,val,test}/masks/ISIC_xxxxxxx_attribute_<name>.png
task3_classification/ {train,val,test}/images/ISIC_xxxxxxx.jpg
                      {train,val,test}/labels.csv                  (one-hot, 7 classes)
                      train/lesion_groupings.csv                   (lesion_id per image)
_zips/                download bookkeeping (and zips, with --keep-zips)
```

| Task | Train | Val | Test |
|---|---|---|---|
| 1 Segmentation | 2,594 | 100 | 1,000 |
| 2 Attributes (5 masks/image) | 2,594 | 100 | 1,000 |
| 3 Classification (7 classes) | 10,015 | 193 | 1,512 |

## Notes

- `test` is a holdout, a validation set must be split from `train`.
- Tasks 1 and 2 use the same images (hard-linked, so no extra disk space).
- Masks are binary PNGs (0/255), same size as their image.
- Task 2 attributes: `globules`, `milia_like_cyst`, `negative_network`, `pigment_network`, `streaks`.
  Many masks are empty (in train, streaks has a non-empty mask in only 3.9% of images,
  negative network 7.3%), so use Dice/focal loss or oversampling.
- Task 3 `labels.csv` columns: `image,MEL,NV,BCC,AKIEC,BKL,DF,VASC` (one-hot).
- Task 3 train has 10,015 images of only 7,470 lesions: split by `lesion_id`
  (`train/lesion_groupings.csv`) to avoid leakage.
- Task 3 is heavily imbalanced (NV ≈ 67%); the challenge metric is balanced multiclass accuracy.

## Sources and licenses

- Tasks 1–2: ISIC 2018 challenge, https://challenge.isic-archive.com/data/#2018 — CC-0.
- Task 3: HAM10000, via the same ISIC 2018 page — CC-BY-NC.

## Data Citations
1. Codella et al., "Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the
   International Skin Imaging Collaboration (ISIC)", 2018, https://arxiv.org/abs/1902.03368
2. Tschandl, Rosendahl & Kittler, "The HAM10000 dataset...", Sci. Data 5, 180161 (2018),
   https://doi.org/10.1038/sdata.2018.161

# Face Recognition for the Pakistani Context — TechFest / Nemo's Deep Dive Round 2

Image or video in → every face **detected, aligned, cropped, quality-checked, embedded and identified**.
Built on top of three open-source projects instead of from scratch:

| Stage | Built on | Why |
|---|---|---|
| Detection + 5 landmarks | [InsightFace](https://github.com/deepinsight/insightface) `buffalo_l` (RetinaFace-family SCRFD-10GF) | State-of-the-art on WIDER FACE incl. small/occluded faces; gives the landmarks alignment needs |
| Recognition backbone | [AdaFace](https://github.com/mk-minchul/AdaFace) IR-101, WebFace12M | Loss adapts the margin to image quality, so it is designed for blurry / low-res / CCTV images (SoTA on IJB-S, TinyFace); PyTorch, so it can be fine-tuned |
| Baseline for comparison | InsightFace ArcFace R50 (WebFace600K) | Shows the effect of the model choice in the report |

Both model packs are licensed for **non-commercial research**, which fits a university competition.

## Pipeline

```
image / video frames
  └─ 1. Detect (SCRFD/RetinaFace)         bbox + 5 landmarks; 2nd pass on padded+upscaled image for tiny faces
     └─ 2. Align & crop                   similarity transform onto the ArcFace 112×112 template → crop returned to user
        └─ 3. Quality gate                reject: face < 24 px or detector conf < 0.55
           │                              repair: dark / low-contrast / overexposed → CLAHE + gamma
        └─ 4. Embed (AdaFace, fine-tuned) 512-d, flip test-time augmentation, norm-weighted fusion
           └─ 5. Match                    cosine similarity vs gallery templates; threshold calibrated on
                                          validation identities (EER point, or FAR = 1 %)
video: faces linked into tracks (IoU + embedding); identity decided on the quality-weighted mean embedding of the whole track
```

**Similarity metric — cosine:** AdaFace/ArcFace are trained with an angular-margin softmax on L2-normalised features, so identity is encoded in the *angle* between embeddings; cosine similarity is exactly the quantity the training objective optimises.
**Threshold:** chosen on the validation identities, never on test; `threshold.json` stores both the EER threshold (balanced FAR = FRR) and the FAR = 1 % threshold (security-oriented). Switch with `--operating_point`.

## Fine-tuning recipe (`scripts/train.py`)

- Freeze IR-101 except the last 6 residual blocks + embedding layer (≈30 M of 65 M params trainable, mostly the embedding layer) — small data, so the early layers stay fixed.
- New AdaFace margin head over our identities, initialised from class-mean embeddings.
- **Embedding-preservation loss** to a frozen copy of the original model → adapts to Pakistani faces without forgetting.
- Class-balanced sampling, OneCycle LR, mixed precision.
- Augmentation (`facerec/augment.py`): CCTV-style downscale, Gaussian + motion blur, JPEG, noise, lighting/colour cast, greyscale, **synthetic mask, dupatta/hijab/cap/turban band, sunglasses**.
- Best epoch chosen on validation EER; if fine-tuning never beats the pretrained model, the pretrained weights are kept.

## Evaluation (`scripts/evaluate.py`)

- Identity-disjoint splits (asserted in code); public-dataset identities only in train.
- Metrics: accuracy, precision, recall, F1, ROC/AUC, FAR, FRR, EER, TAR@FAR 1 % and 0.1 %, plus rank-1/5 and open-set DIR/FPIR.
- **Ablation:** A0 bbox crop → A1 +alignment → A2 +quality enhancement → A3 +flip TTA → A4 +fine-tuning (+ ArcFace reference).
- **Groups:** gender, age group, region, skin tone, source, and conditions (head covering, glasses, mask, facial hair, low light, CCTV/low-res, outdoor, clean).
- Figures: ROC (all ablation rows), DET curve, genuine/impostor score histogram.

## Run it

Everything is in **`notebooks/FaceRec_Pakistan_Colab.ipynb`** (T4 GPU). Upload `facerec_pk.zip` to `MyDrive/NemoFR/` and run top to bottom.

| Script | Purpose |
|---|---|
| `scripts/download_weights.py` | AdaFace + InsightFace weights |
| `scripts/import_google_form.py` | consenting volunteers from a Google Form → `raw/` |
| `scripts/import_halfface.py` | halfFace (South Asian, masked) → `raw/` (train only) |
| `scripts/prepare_data.py` | detect, align, quality metrics, splits, data report |
| `scripts/train.py` | fine-tuning |
| `scripts/evaluate.py` | ablation, groups, identification, threshold |
| `scripts/enroll_gallery.py` | register known people |
| `scripts/run_inference.py` | image / video / folder → crops + identities + JSON |
| `scripts/benchmark.py` | hidden benchmark in pairs, gallery/probe, or crop-only format |
| `app.py` | Gradio demo: recognise, enroll, 1:1 verify, threshold slider |

Raw data layout: `raw/<participant code>/*.jpg|*.mp4` + `raw/identities.csv` (`identity,gender,age_group,region,skin_tone,source,consent_id`). See `docs/CONSENT_AND_DATA_COLLECTION.md`.

## Optional extension: anti-spoofing
[Silent-Face-Anti-Spoofing](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing) (MiniFASNet) can be added as a check between steps 1 and 2 to reject printed photos / phone screens. Not part of the scored benchmark path, so it is left optional.

## Citations
Deng et al., *ArcFace*, CVPR 2019 · Deng et al., *RetinaFace*, CVPR 2020 · Guo et al., *SCRFD*, ICLR 2022 · Kim, Jain & Liu, *AdaFace*, CVPR 2022 · halfFace dataset, Mendeley Data `pk44mkx9vm`.

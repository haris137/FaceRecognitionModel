# [Team name] — Face Recognition for the Pakistani Context
*Nemo's Deep Dive, Round 2 · NED University CSIT TechFest '26 · submitted 22 Oct 2026*

## 1. Summary (5 lines)
Model, pipeline, data size, headline test results (EER, accuracy, TAR@FAR=1%), biggest improvement from the ablation.

## 2. Model choice
- AdaFace IR-101 (WebFace12M): why — quality-adaptive margin for low-quality images; strongest public results on IJB-S / TinyFace; PyTorch for fine-tuning.
- Compared against InsightFace ArcFace R50 → paste the "ref" row from `ablation.md`.

## 3. Pipeline design
Diagram (copy from README) + one paragraph per stage with the justification, then the ablation table showing what each stage adds:
→ paste `results/ablation.md`, insert `roc_ablation.png`.

Similarity metric and threshold: cosine (matches the angular-margin training objective); threshold from validation EER point = **[value]** (FAR 1 % threshold = [value]). Insert `score_distribution.png` and `det_curve.png`.

## 4. Data collection and consent
- Volunteers: [N] consenting adults via Google Form, participant codes, private storage, deletion by 30 Nov 2026, withdrawal route (see `docs/CONSENT_AND_DATA_COLLECTION.md`).
- Public data: halfFace ([N] identities, train only, cited).
- Diversity: paste tables from `processed/prepare_report.md` (gender, age, region, skin tone, conditions).
- Splits: train [x] / val [y] / test [z] identities, no overlap.

## 5. Fine-tuning
Recipe (frozen layers, distillation, augmentation list), training curve from `train_history.json`, val EER before → after.

## 6. Results
- Verification on unseen identities: accuracy, precision, recall, F1, FAR, FRR, EER, AUC, TAR@FAR.
- Identification: paste `identification.md`.

## 7. Performance across groups
Paste `subgroups.md`. Discuss the largest gaps honestly (e.g. masks, low light, small groups with few pairs) and what reduced them (augmentation, enhancement).

## 8. Challenges and limitations
Data volume in 3 days, imbalanced regions, phone-only capture, near-duplicate video frames (handled by not pairing frames from the same video), unknown hidden-benchmark format (pipeline supports pairs, gallery/probe, and crop-only modes).

## 9. Ethics
Consent, purpose limitation, deletion, fairness analysis, no public figures' images in test data, risks of face recognition misuse.

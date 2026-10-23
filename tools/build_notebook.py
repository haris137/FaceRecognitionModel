"""Generates notebooks/FaceRec_Pakistan_Colab.ipynb."""
import json
import os

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").splitlines(True)})


md("""
# TechFest Face Recognition Challenge — Pakistani Context
**Pipeline:** RetinaFace/SCRFD detection → 5-point alignment & crop → quality gate + low-light enhancement → AdaFace IR-101 (fine-tuned) embeddings → cosine matching with a validation-calibrated threshold.

Runtime → Change runtime type → **T4 GPU**. Run the cells top to bottom.

Folder on Google Drive (create it once):
```
MyDrive/NemoFR/
   facerec_pk.zip          <- this project (upload once)
   data/raw/               <- one folder per consenting participant (P001/, P002/ ...) + identities.csv
```
""")

md("## 1. Setup")
code("""
from google.colab import drive
drive.mount('/content/drive')

DRIVE = '/content/drive/MyDrive/NemoFR'
RAW   = f'{DRIVE}/data/raw'
WORK  = '/content/work'                 # fast local disk
PROC  = f'{WORK}/processed'
RUNS  = f'{DRIVE}/runs'                 # checkpoints + results persist on Drive
import os; os.makedirs(WORK, exist_ok=True); os.makedirs(RUNS, exist_ok=True)

!unzip -oq {DRIVE}/facerec_pk.zip -d /content/
%cd /content/facerec_pk
!pip install -q -r requirements.txt
!git clone -q --depth 1 https://github.com/mk-minchul/AdaFace.git /content/AdaFace || true
""")
code("""
# AdaFace IR-101 (WebFace12M) + InsightFace buffalo_l detector
!python scripts/download_weights.py --name ir_101_webface12m --out {DRIVE}/weights
PRETRAINED = f'{DRIVE}/weights/adaface_ir_101_webface12m.ckpt'
ADAFACE = '/content/AdaFace'
import torch; print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE - switch runtime to GPU')
""")

md("""
## 2. Bring in the data
**A. Volunteers via Google Form** (recommended): export the responses Sheet as CSV to Drive and run the cell below.
Skip it if you already copied folders into `data/raw/` yourself.

**B. halfFace** (South Asian, masked + unmasked, smartphone photos): download from Mendeley Data (`pk44mkx9vm`), upload the zip to Drive, set the path of the full-face set below. Used for training only.
""")
code("""
# A. Google Form import (edit column names with --col_* flags if your form differs)
# !python scripts/import_google_form.py --responses {DRIVE}/form_responses.csv --raw {RAW}

# B. halfFace
# !unzip -oq {DRIVE}/halfface.zip -d /content/halfface
# !python scripts/import_halfface.py --src "/content/halfface/<Dataset 1 folder>" --raw {RAW}
!ls {RAW} | head; !ls {RAW} | wc -l
""")

md("## 3. Detect, align, quality-check and split (identity-disjoint)")
code("""
!python scripts/prepare_data.py --raw {RAW} --out {PROC} --test_frac 0.2 --val_frac 0.1
!zip -qr {RUNS}/processed.zip {PROC}      # backup so you can skip this step after a disconnect
from IPython.display import Markdown, display
display(Markdown(open(f'{PROC}/prepare_report.md').read()))
""")
code("""
# After a Colab disconnect, restore instead of re-running step 3:
# !mkdir -p {WORK} && unzip -oq {RUNS}/processed.zip -d /
""")
code("""
# Look at a few aligned crops and what the quality gate thinks of them
import pandas as pd, cv2, matplotlib.pyplot as plt
df = pd.read_csv(f'{PROC}/images.csv')
s = df.sample(min(16, len(df)), random_state=0)
fig, axs = plt.subplots(2, 8, figsize=(16, 4.6))
for ax, (_, r) in zip(axs.flat, s.iterrows()):
    ax.imshow(cv2.cvtColor(cv2.imread(f"{PROC}/{r.path_aligned}"), cv2.COLOR_BGR2RGB)); ax.axis('off')
    ax.set_title(f"{r.identity}\\nq={r.quality_score:.2f}{'' if r.usable else ' REJ'}", fontsize=8)
plt.tight_layout(); plt.show()
""")

md("## 4. Fine-tune AdaFace on Pakistani faces\nModel selection uses validation identities only. ~2-4 min/epoch on a T4 for ~3k images.")
code("""
!python scripts/train.py --data {PROC} --adaface_repo {ADAFACE} --ckpt {PRETRAINED} \\
        --out {RUNS}/ft --epochs 12 --batch 64 --unfreeze_blocks 6 --distill 1.0
FINETUNED = f'{RUNS}/ft/best.pt'
""")
code("""
import json, matplotlib.pyplot as plt
h = json.load(open(f'{RUNS}/ft/train_history.json'))['history']
plt.plot([x['epoch'] for x in h], [x['EER'] for x in h], 'o-'); plt.xlabel('epoch'); plt.ylabel('val EER'); plt.title('Validation EER (epoch 0 = pretrained)'); plt.grid(alpha=.3); plt.show()
""")

md("## 5. Evaluate: ablation, model comparison, groups, identification")
code("""
!python scripts/evaluate.py --data {PROC} --adaface_repo {ADAFACE} --pretrained {PRETRAINED} \\
        --finetuned {FINETUNED} --with_arcface --out {RUNS}/results
from IPython.display import Markdown, display
display(Markdown(open(f'{RUNS}/results/results_summary.md').read()))
""")
code("""
from IPython.display import Image as IPImage
for f in ['roc_ablation.png', 'score_distribution.png', 'det_curve.png']:
    display(IPImage(f'{RUNS}/results/{f}', width=650))
""")

md("""
## 6. Gallery of known people
Put 1-5 clear photos per person in `MyDrive/NemoFR/gallery_photos/<Name>/`. For a quick demo you can
enroll some of your own **test** volunteers (they were never trained on).
""")
code("""
!python scripts/enroll_gallery.py --adaface_repo {ADAFACE} --ckpt {FINETUNED} \\
        --threshold_file {RUNS}/results/threshold.json --people {DRIVE}/gallery_photos \\
        --gallery {RUNS}/gallery.npz --fresh
""")

md("## 7. Image or video in → recognised, cropped faces out")
code("""
INPUT = f'{DRIVE}/samples/test.jpg'      # or a .mp4 / folder
!python scripts/run_inference.py --adaface_repo {ADAFACE} --ckpt {FINETUNED} \\
        --threshold_file {RUNS}/results/threshold.json --gallery {RUNS}/gallery.npz \\
        --input "{INPUT}" --out {RUNS}/outputs
!ls {RUNS}/outputs
""")

md("## 8. Interactive demo (public link printed below)")
code("""
!python app.py --adaface_repo {ADAFACE} --ckpt {FINETUNED} \\
        --threshold_file {RUNS}/results/threshold.json --gallery {RUNS}/gallery.npz
""")

md("""
## 9. Hidden benchmark day (23 Oct)
Pick the mode that matches what the TechFest team gives you:
""")
code("""
# pairs CSV (img1,img2[,label]):
# !python scripts/benchmark.py pairs --pairs /content/bench/pairs.csv --root /content/bench \\
#     --adaface_repo {ADAFACE} --ckpt {FINETUNED} --threshold_file {RUNS}/results/threshold.json --out {RUNS}/bench_preds.csv

# gallery + probes folders:
# !python scripts/benchmark.py identify --gallery_dir /content/bench/gallery --probe_dir /content/bench/probes \\
#     --adaface_repo {ADAFACE} --ckpt {FINETUNED} --threshold_file {RUNS}/results/threshold.json --out {RUNS}/bench_preds.csv

# just detect + crop every face:
# !python scripts/benchmark.py crops --probe_dir /content/bench/images --adaface_repo {ADAFACE} --ckpt {FINETUNED} --out {RUNS}/bench_crops
""")

nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                    "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                    "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks", "FaceRec_Pakistan_Colab.ipynb")
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)

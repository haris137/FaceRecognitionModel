"""Copy the halfFace (Mendeley Data pk44mkx9vm) full-face set into the raw/ layout.

  python scripts/import_halfface.py --src /content/halfface/<folder with P_xxx sub-folders> --raw /content/drive/MyDrive/fr_data/raw

Each P_xxx folder becomes raw/HF_xxx with source=halfface. Gender is read from the
file-name suffix (..m / ..f) when present. halfFace identities are used for TRAINING
only (prepare_data.py keeps them out of val/test), and the dataset must be cited.
"""
import argparse
import os
import re
import shutil

import pandas as pd

IMG = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--raw", required=True)
    args = ap.parse_args()
    os.makedirs(args.raw, exist_ok=True)
    rows = []
    for root, dirs, files in os.walk(args.src):
        base = os.path.basename(root)
        m = re.match(r"^[Pp]_?(\d+)$", base)
        imgs = [f for f in files if os.path.splitext(f)[1].lower() in IMG]
        if not m or not imgs:
            continue
        ident = f"HF_{int(m.group(1)):03d}"
        dst = os.path.join(args.raw, ident)
        os.makedirs(dst, exist_ok=True)
        genders = set()
        for f in imgs:
            shutil.copy2(os.path.join(root, f), os.path.join(dst, f))
            g = re.search(r"([mf])(?:_|\.|$)", os.path.splitext(f)[0].lower()[-3:] + ".")
            if g:
                genders.add(g.group(1))
        gender = {"m": "male", "f": "female"}[genders.pop()] if len(genders) == 1 else "unknown"
        rows.append({"identity": ident, "gender": gender, "age_group": "unknown", "region": "unknown",
                     "skin_tone": "unknown", "source": "halfface", "consent_id": "dataset-license"})
    ids_path = os.path.join(args.raw, "identities.csv")
    new = pd.DataFrame(rows)
    if os.path.exists(ids_path):
        old = pd.read_csv(ids_path, dtype=str)
        new = pd.concat([old[~old["identity"].isin(new["identity"])], new], ignore_index=True)
    new.to_csv(ids_path, index=False)
    print(f"imported {len(rows)} halfFace identities into {args.raw}")


if __name__ == "__main__":
    main()

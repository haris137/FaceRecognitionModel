"""Run the pipeline on an external / hidden benchmark in whichever format it arrives.

Mode 1 - verification pairs (CSV with columns img1,img2[,label]):
  python scripts/benchmark.py pairs --pairs pairs.csv --root bench/ --ckpt ... --threshold_file ... --out preds.csv
  -> preds.csv with similarity and same_person; if `label` exists, prints all metrics.

Mode 2 - gallery + probes (gallery/<person>/*.jpg, probes/*.jpg or probes/<person>/*.jpg):
  python scripts/benchmark.py identify --gallery_dir bench/gallery --probe_dir bench/probes --ckpt ... --out preds.csv
  -> one row per probe: predicted identity, similarity, top-5. If probes sit in person folders, prints rank-1.

Mode 3 - folder of images, just detect + crop + embed:
  python scripts/benchmark.py crops --probe_dir bench/ --ckpt ... --out crops_out/
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from common import add_pipeline_args, pipeline_from_args

from facerec.io_utils import is_image, read_image, save_image
from facerec.metrics import verification_metrics


def embed_file(pipe, path, cache):
    if path in cache:
        return cache[path]
    img = read_image(path)
    res = pipe.extract(img, largest_only=True) if img is not None else []
    cache[path] = res[0] if res else None
    return cache[path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["pairs", "identify", "crops"])
    add_pipeline_args(ap)
    ap.add_argument("--pairs")
    ap.add_argument("--root", default="")
    ap.add_argument("--gallery_dir")
    ap.add_argument("--probe_dir")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pipe = pipeline_from_args(args)
    pipe.cfg.reject_low_quality = False   # benchmark: always answer
    cache = {}

    if args.mode == "pairs":
        df = pd.read_csv(args.pairs)
        c1, c2 = df.columns[0], df.columns[1]
        sims = []
        for a, b in tqdm(zip(df[c1], df[c2]), total=len(df)):
            fa = embed_file(pipe, os.path.join(args.root, str(a)), cache)
            fb = embed_file(pipe, os.path.join(args.root, str(b)), cache)
            sims.append(float(fa.embedding @ fb.embedding) if fa is not None and fb is not None else -1.0)
        df["similarity"] = sims
        df["same_person"] = (df["similarity"] >= pipe.cfg.threshold).astype(int)
        df["threshold"] = pipe.cfg.threshold
        df.to_csv(args.out, index=False)
        print(f"{(df.similarity == -1).sum()} pairs had a missing face. Saved {args.out}")
        if "label" in df.columns:
            print(json.dumps(verification_metrics(df.similarity.values, df.label.values, pipe.cfg.threshold), indent=1))

    elif args.mode == "identify":
        for person in sorted(os.listdir(args.gallery_dir)):
            d = os.path.join(args.gallery_dir, person)
            if os.path.isdir(d):
                pipe.enroll(person, [os.path.join(d, f) for f in sorted(os.listdir(d)) if is_image(f)])
        probes = []
        for root, _, files in os.walk(args.probe_dir):
            for f in sorted(files):
                if is_image(f):
                    p = os.path.join(root, f)
                    truth = os.path.basename(root) if os.path.abspath(root) != os.path.abspath(args.probe_dir) else None
                    probes.append((p, truth))
        rows = []
        for p, truth in tqdm(probes):
            r = embed_file(pipe, p, cache)
            top = pipe.gallery.search(r.embedding, k=5) if r is not None else []
            pred = top[0][0] if top and top[0][1] >= pipe.cfg.threshold else "Unknown"
            rows.append({"probe": os.path.relpath(p, args.probe_dir), "predicted": pred,
                         "similarity": top[0][1] if top else None, "top5": json.dumps(top), "truth": truth})
        out = pd.DataFrame(rows)
        out.to_csv(args.out, index=False)
        if out["truth"].notna().any():
            k = out[out.truth.notna()]
            r1 = np.mean([json.loads(t)[0][0] == tr if json.loads(t) else False for t, tr in zip(k.top5, k.truth)])
            print(f"rank-1 (ignoring threshold) = {r1:.4f}; accuracy with threshold = {(k.predicted == k.truth).mean():.4f}")
        print(f"saved {args.out}")

    else:
        os.makedirs(args.out, exist_ok=True)
        rows = []
        for f in tqdm(sorted(os.listdir(args.probe_dir))):
            if not is_image(f):
                continue
            img = read_image(os.path.join(args.probe_dir, f))
            for i, r in enumerate(pipe.extract(img) if img is not None else []):
                save_image(os.path.join(args.out, f"{os.path.splitext(f)[0]}_face{i}.jpg"), r.crop)
                rows.append({"file": f, "face": i, **r.summary()})
        pd.DataFrame(rows).to_csv(os.path.join(args.out, "faces.csv"), index=False)


if __name__ == "__main__":
    main()

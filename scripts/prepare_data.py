"""Turn raw collected media into aligned face crops + metadata + identity-disjoint splits.

Input layout (one folder per consenting participant, anonymised IDs):
    raw/
      identities.csv           identity,gender,age_group,region,skin_tone,source,consent_id
      image_tags.csv           (optional) identity,file,tags   e.g. P001,img3.jpg,"hijab|lowlight"
      P001/  selfie1.jpg  cctv__lowres-cap.jpg  walk.mp4 ...
      P002/  ...
Tags can also be put in the file name after a double underscore: "photo__mask-glasses.jpg".

Output:
    processed/aligned/<id>/<name>.jpg   112x112 aligned crop (ArcFace template)
    processed/naive/<id>/<name>.jpg     112x112 plain bbox crop (for the alignment ablation)
    processed/images.csv                one row per face with quality metrics and tags
    processed/identities.csv            identity attributes
    processed/splits.json               {"train": [...], "val": [...], "test": [...]}  (no identity overlap)
    processed/prepare_report.md         counts and failure rates for the report
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facerec.config import QualityConfig  # noqa: E402
from facerec.detect_align import FaceDetector, largest_face, align_face, naive_crop, bbox_min_side  # noqa: E402
from facerec.io_utils import read_image, iter_video_frames, is_image, is_video, save_image, parse_filename_tags  # noqa: E402
from facerec import quality as Q  # noqa: E402


def load_tag_table(raw):
    p = os.path.join(raw, "image_tags.csv")
    if not os.path.exists(p):
        return {}
    t = pd.read_csv(p, dtype=str).fillna("")
    return {(r["identity"], r["file"]): {x.strip().lower() for x in r["tags"].replace(",", "|").split("|") if x.strip()}
            for _, r in t.iterrows()}


def split_identities(ids_df, val_frac, test_frac, train_only_sources, seed):
    rng = np.random.default_rng(seed)
    forced = ids_df[ids_df["source"].isin(train_only_sources)]["identity"].tolist()
    pool = ids_df[~ids_df["identity"].isin(forced)]
    train, val, test = list(forced), [], []
    # stratify by gender so every split has both genders
    for _, grp in pool.groupby("gender"):
        idl = grp["identity"].tolist()
        rng.shuffle(idl)
        n_test = int(round(len(idl) * test_frac))
        n_val = int(round(len(idl) * val_frac))
        test += idl[:n_test]
        val += idl[n_test:n_test + n_val]
        train += idl[n_test + n_val:]
    return {"train": sorted(train), "val": sorted(val), "test": sorted(test)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video_every", type=int, default=8, help="keep every n-th video frame")
    ap.add_argument("--video_max", type=int, default=12, help="max frames kept per video")
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--train_only_sources", default="halfface,public",
                    help="sources that never go to val/test (benchmark = consented volunteers)")
    ap.add_argument("--min_images", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resplit_only", action="store_true", help="skip face extraction, only redo splits")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    id_dirs = sorted(d for d in os.listdir(args.raw) if os.path.isdir(os.path.join(args.raw, d)) and not d.startswith("."))

    # identity table
    ids_path = os.path.join(args.raw, "identities.csv")
    ids_df = pd.read_csv(ids_path, dtype=str).fillna("unknown") if os.path.exists(ids_path) else pd.DataFrame({"identity": []})
    ids_df.columns = [c.strip().lower() for c in ids_df.columns]
    missing = [d for d in id_dirs if d not in set(ids_df["identity"])]
    if missing:
        ids_df = pd.concat([ids_df, pd.DataFrame({"identity": missing})], ignore_index=True)
    for c in ["gender", "age_group", "region", "skin_tone", "source"]:
        if c not in ids_df.columns:
            ids_df[c] = "unknown"
        ids_df[c] = ids_df[c].fillna("unknown").astype(str).str.strip().str.lower()
    ids_df.loc[(ids_df["source"] == "unknown") & ids_df["identity"].str.startswith("HF_"), "source"] = "halfface"
    ids_df.loc[ids_df["source"] == "unknown", "source"] = "volunteer"
    ids_df = ids_df[ids_df["identity"].isin(id_dirs)]

    csv_path = os.path.join(args.out, "images.csv")
    if not args.resplit_only:
        det = FaceDetector()
        qcfg = QualityConfig()
        tag_table = load_tag_table(args.raw)
        rows, failures = [], []
        for ident in tqdm(id_dirs, desc="identities"):
            folder = os.path.join(args.raw, ident)
            for fn in sorted(os.listdir(folder)):
                path = os.path.join(folder, fn)
                tags = parse_filename_tags(fn) | tag_table.get((ident, fn), set())
                if is_image(path):
                    img = read_image(path)
                    items = [(None, img)] if img is not None else []
                    media = "image"
                elif is_video(path):
                    items = [(fi, fr) for fi, _, fr in iter_video_frames(path, args.video_every, args.video_max)]
                    media = "video"
                else:
                    continue
                if not items:
                    failures.append((ident, fn, "unreadable"))
                for fi, img in items:
                    dets = det.detect(img)
                    d = largest_face(dets)
                    if d is None:
                        failures.append((ident, fn if fi is None else f"{fn}#{fi}", "no face"))
                        continue
                    stem = os.path.splitext(fn)[0].split("__")[0] + ("" if fi is None else f"_f{fi:05d}")
                    al = align_face(img, d["kps"])
                    nv = naive_crop(img, d["bbox"])
                    m = Q.measure(al, bbox_min_side(d["bbox"]), d["score"])
                    qa = Q.assess(m, qcfg)
                    p_al = os.path.join("aligned", ident, stem + ".jpg")
                    p_nv = os.path.join("naive", ident, stem + ".jpg")
                    save_image(os.path.join(args.out, p_al), al)
                    save_image(os.path.join(args.out, p_nv), nv)
                    rows.append({
                        "identity": ident, "path_aligned": p_al, "path_naive": p_nv,
                        "source_file": f"{media}:{fn}", "media": media, "frame_idx": -1 if fi is None else fi,
                        "tags": "|".join(sorted(tags)), "n_faces_in_image": len(dets),
                        "det_score": round(d["score"], 4), "face_px": round(m["face_px"], 1),
                        "blur": round(m["blur"], 2), "brightness": round(m["brightness"], 2),
                        "contrast": round(m["contrast"], 2), "quality_score": qa["score"],
                        "usable": qa["usable"], "retried_detection": bool(d.get("retried", False)),
                        **{f"flag_{k}": v for k, v in qa["flags"].items()},
                    })
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        pd.DataFrame(failures, columns=["identity", "file", "reason"]).to_csv(os.path.join(args.out, "failures.csv"), index=False)
    else:
        df = pd.read_csv(csv_path, dtype={"identity": str})
        failures = pd.read_csv(os.path.join(args.out, "failures.csv")).values.tolist() if os.path.exists(os.path.join(args.out, "failures.csv")) else []

    counts = df.groupby("identity").size()
    eligible = set(counts[counts >= args.min_images].index)
    ids_df = ids_df[ids_df["identity"].isin(eligible)].reset_index(drop=True)
    ids_df.to_csv(os.path.join(args.out, "identities.csv"), index=False)
    splits = split_identities(ids_df, args.val_frac, args.test_frac,
                              [s.strip() for s in args.train_only_sources.split(",") if s.strip()], args.seed)
    assert not (set(splits["train"]) & set(splits["test"])) and not (set(splits["val"]) & set(splits["test"]))
    with open(os.path.join(args.out, "splits.json"), "w") as f:
        json.dump(splits, f, indent=1)

    # report
    lines = ["# Data preparation report", "",
             f"- identities with >= {args.min_images} faces: **{len(ids_df)}**",
             f"- face crops: **{len(df)}**  (from images: {int((df.media == 'image').sum())}, video frames: {int((df.media == 'video').sum())})",
             f"- detection failures: **{len(failures)}**",
             f"- rejected by quality gate (too small / low confidence): **{int((~df['usable'].astype(bool)).sum())}**", "",
             "## Split sizes (identity-disjoint)", "", "| split | identities | images |", "|---|---|---|"]
    for k, v in splits.items():
        lines.append(f"| {k} | {len(v)} | {int(df['identity'].isin(v).sum())} |")
    for attr in ["gender", "age_group", "region", "skin_tone", "source"]:
        vc = ids_df[attr].value_counts()
        lines += ["", f"## Identities by {attr}", "", "| value | count |", "|---|---|"] + [f"| {i} | {c} |" for i, c in vc.items()]
    tag_counts = pd.Series([t for s in df["tags"].fillna("") for t in str(s).split("|") if t]).value_counts()
    lines += ["", "## Images by condition tag", "", "| tag | images |", "|---|---|"] + [f"| {i} | {c} |" for i, c in tag_counts.items()]
    with open(os.path.join(args.out, "prepare_report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines[:12]))


if __name__ == "__main__":
    main()

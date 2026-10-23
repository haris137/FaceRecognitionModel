"""Full evaluation: ablation, model comparison, subgroup fairness, identification.

Protocol
  * thresholds are calibrated on the VALIDATION identities (EER point and FAR=1%),
    then applied unchanged to the TEST identities (no identity overlap with train/val);
  * every row of the ablation adds exactly one pipeline stage;
  * the final pipeline is broken down by gender, age, region, skin tone and
    capture condition (head covering, glasses, mask, low light, CCTV/low-res ...).

Outputs (in --out): ablation.csv/.md, subgroups.csv/.md, identification.md,
roc_ablation.png, score_distribution.png, det_curve.png, threshold.json, results_summary.md
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facerec.config import QualityConfig  # noqa: E402
from facerec.data import load_metadata, load_splits, subset, build_pairs, pair_groups  # noqa: E402
from facerec.metrics import verification_metrics, roc_points, threshold_at_far, identification_metrics  # noqa: E402
from facerec import quality as Q  # noqa: E402


def load_inputs(root, df, crop, enhance):
    col = "path_aligned" if crop == "aligned" else "path_naive"
    imgs = []
    qcfg = QualityConfig()
    for p in df[col]:
        im = cv2.imread(os.path.join(root, p))
        if enhance:
            im = Q.enhance(im, Q.measure(im), qcfg)
        imgs.append(im)
    return imgs


def fmt(x):
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--adaface_repo", required=True)
    ap.add_argument("--pretrained", required=True, help="official AdaFace checkpoint")
    ap.add_argument("--finetuned", default=None, help="best.pt from train.py")
    ap.add_argument("--arch", default="ir_101")
    ap.add_argument("--with_arcface", action="store_true", help="also compare InsightFace ArcFace R50")
    ap.add_argument("--operating_point", default="eer", choices=["eer", "far1"],
                    help="threshold rule calibrated on validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import torch
    from facerec.embedder import AdaFaceEmbedder, load_backbone
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = load_metadata(args.data)
    splits = load_splits(args.data)
    va, te = subset(df, splits, "val"), subset(df, splits, "test")
    pv, pt = build_pairs(va, seed=args.seed + 1), build_pairs(te, seed=args.seed + 2)
    print(f"val: {va.identity.nunique()} ids, {len(pv)} pairs | test: {te.identity.nunique()} ids, {len(pt)} pairs")

    models = {"pre": AdaFaceEmbedder(args.adaface_repo, args.pretrained, arch=args.arch, device=device)}
    if args.finetuned:
        models["ft"] = AdaFaceEmbedder(None, None, device=device,
                                       model=load_backbone(args.adaface_repo, args.arch, args.finetuned, device))
    if args.with_arcface:
        from facerec.embedder import ArcFaceEmbedder
        models["arcface"] = ArcFaceEmbedder()

    # each row adds one stage on top of the previous one
    configs = [
        ("A0  pretrained AdaFace, bbox crop (no alignment)", "pre", "naive", False, False),
        ("A1  + 5-point alignment", "pre", "aligned", False, False),
        ("A2  + quality enhancement (CLAHE/gamma)", "pre", "aligned", True, False),
        ("A3  + flip test-time augmentation", "pre", "aligned", True, True),
    ]
    if "ft" in models:
        configs.append(("A4  + fine-tuned on Pakistani data (FULL PIPELINE)", "ft", "aligned", True, True))
    if "arcface" in models:
        configs.append(("ref ArcFace R50 (InsightFace) with same pipeline", "arcface", "aligned", True, True))

    cache_inputs, rows, rocs, final = {}, [], {}, None
    for name, mk, crop, enh, tta in configs:
        key = (crop, enh)
        if key not in cache_inputs:
            cache_inputs[key] = (load_inputs(args.data, va, crop, enh), load_inputs(args.data, te, crop, enh))
        iv, it = cache_inputs[key]
        ev, _ = models[mk].embed(iv, flip_tta=tta)
        et, nt = models[mk].embed(it, flip_tta=tta)
        sv = np.sum(ev[pv[:, 0]] * ev[pv[:, 1]], 1)
        st = np.sum(et[pt[:, 0]] * et[pt[:, 1]], 1)
        mv = verification_metrics(sv, pv[:, 2])
        thr = mv["EER_threshold"] if args.operating_point == "eer" else threshold_at_far(sv, pv[:, 2], 0.01)
        mt = verification_metrics(st, pt[:, 2], threshold=thr)
        rows.append({"config": name, "val_EER": mv["EER"], **{k: mt[k] for k in
                     ("threshold", "accuracy", "precision", "recall", "f1", "FAR", "FRR", "EER", "AUC",
                      "TAR@FAR=1e-2", "TAR@FAR=1e-3")}})
        rocs[name] = roc_points(st, pt[:, 2])
        print(f"{name:55s} test EER={mt['EER']:.4f} acc={mt['accuracy']:.4f} F1={mt['f1']:.4f}")
        if mk != "arcface":
            final = dict(name=name, et=et, st=st, thr=thr, sv=sv, model=mk)

    abl = pd.DataFrame(rows)
    abl.to_csv(os.path.join(args.out, "ablation.csv"), index=False)
    cols = ["config", "accuracy", "precision", "recall", "f1", "FAR", "FRR", "EER", "AUC", "TAR@FAR=1e-2", "threshold"]
    with open(os.path.join(args.out, "ablation.md"), "w") as f:
        f.write(f"# Ablation on test set ({te.identity.nunique()} unseen identities, {len(pt)} pairs)\n\n"
                f"Threshold calibrated on validation ({args.operating_point}).\n\n" + md_table(abl, cols) + "\n")

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (fpr, tpr, _) in rocs.items():
        ax.plot(fpr, tpr, label=name.split("  ")[0] + "  " + name.split("  ", 1)[-1][:38])
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1)
    ax.set_xlabel("False Accept Rate (log)")
    ax.set_ylabel("True Accept Rate")
    ax.set_title("ROC on held-out test identities")
    ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "roc_ablation.png"), dpi=160)
    plt.close(fig)

    fpr, tpr, _ = rocs[final["name"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, 1 - tpr)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("FAR"); ax.set_ylabel("FRR"); ax.set_title("DET curve - full pipeline"); ax.grid(alpha=.3, which="both")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "det_curve.png"), dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    st, lab = final["st"], pt[:, 2]
    ax.hist(st[lab == 1], bins=60, alpha=.6, label="genuine (same person)", density=True)
    ax.hist(st[lab == 0], bins=60, alpha=.6, label="impostor (different people)", density=True)
    ax.axvline(final["thr"], color="k", ls="--", label=f"threshold {final['thr']:.3f}")
    ax.set_xlabel("cosine similarity"); ax.legend(); ax.set_title("Score distribution - full pipeline")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "score_distribution.png"), dpi=160); plt.close(fig)

    # ---- subgroup analysis on the final pipeline ----
    srows = []
    for attr, val, mask in pair_groups(te, pt):
        if mask.sum() < 20:
            continue
        m = verification_metrics(final["st"][mask], pt[mask, 2], threshold=final["thr"])
        if "EER" not in m:
            continue
        srows.append({"attribute": attr, "group": val, "pairs": int(mask.sum()), "genuine": m["n_genuine"],
                      "impostor": m["n_impostor"], "accuracy": m["accuracy"], "FAR": m["FAR"], "FRR": m["FRR"],
                      "EER": m["EER"], "TAR@FAR=1e-2": m["TAR@FAR=1e-2"]})
    sg = pd.DataFrame(srows)
    sg.to_csv(os.path.join(args.out, "subgroups.csv"), index=False)
    with open(os.path.join(args.out, "subgroups.md"), "w") as f:
        f.write(f"# Performance across groups - {final['name'].strip()}\n\nGlobal threshold {final['thr']:.4f}. "
                "Groups with < 20 pairs omitted.\n\n")
        if len(sg):
            f.write(md_table(sg, list(sg.columns)) + "\n")

    # ---- identification (gallery / probe) ----
    et = final["et"]
    rng = np.random.default_rng(args.seed)
    q = te["quality_score"].values
    gal_idx, probe_idx = [], []
    for ident, idx in te.groupby("identity").indices.items():
        idx = np.asarray(idx)[np.argsort(-q[idx])]   # best-quality photos enrolled, like a real system
        n_g = 1 if len(idx) < 4 else 2
        gal_idx += list(idx[:n_g]); probe_idx += list(idx[n_g:])
    gal_idx, probe_idx = np.array(gal_idx), np.array(probe_idx)
    ids = te["identity"].values
    uniq = np.unique(ids)
    unknown_ids = set(rng.choice(uniq, size=max(1, len(uniq) // 4), replace=False))
    def templates(sel_ids):
        names = [i for i in uniq if i in sel_ids]
        T = np.stack([et[gal_idx[ids[gal_idx] == n]].mean(0) for n in names])
        return T / np.linalg.norm(T, axis=1, keepdims=True), np.array(names)
    Tc, nc = templates(set(uniq))
    closed = identification_metrics(Tc, nc, et[probe_idx], ids[probe_idx], final["thr"])
    To, no = templates(set(uniq) - unknown_ids)
    openm = identification_metrics(To, no, et[probe_idx], ids[probe_idx], final["thr"],
                                   unknown_mask=np.isin(ids[probe_idx], list(unknown_ids)))
    with open(os.path.join(args.out, "identification.md"), "w") as f:
        f.write("# Identification (1:N) - full pipeline\n\n"
                f"Closed set (all {len(uniq)} test people enrolled with their 1-2 best photos):\n\n"
                f"- Rank-1: **{closed.get('rank1', 0):.4f}**  Rank-5: {closed.get('rank5', 0):.4f}\n\n"
                f"Open set ({len(unknown_ids)} people NOT enrolled; must be answered 'Unknown'):\n\n"
                f"- Detection & identification rate at threshold: **{openm.get('DIR@thr', 0):.4f}**\n"
                f"- False positive identification rate (unknown accepted as someone): **{openm.get('FPIR@thr', 0):.4f}**\n")

    # ---- threshold for deployment ----
    thr_eer = verification_metrics(final["sv"], pv[:, 2])["EER_threshold"]
    thr_far1 = threshold_at_far(final["sv"], pv[:, 2], 0.01)
    with open(os.path.join(args.out, "threshold.json"), "w") as f:
        json.dump({"threshold": float(final["thr"]), "rule": args.operating_point, "eer_threshold": thr_eer,
                   "far1_threshold": thr_far1, "model": final["model"], "config": final["name"]}, f, indent=1)

    with open(os.path.join(args.out, "results_summary.md"), "w") as f:
        for p in ("ablation.md", "identification.md", "subgroups.md"):
            f.write(open(os.path.join(args.out, p)).read() + "\n\n")
    print(f"\nSaved results to {args.out}. Deployment threshold = {final['thr']:.4f}")


if __name__ == "__main__":
    main()

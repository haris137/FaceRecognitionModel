"""Verification / identification metrics: accuracy, precision, recall, F1, ROC, FAR, FRR, EER, TAR@FAR."""
import numpy as np
from sklearn.metrics import roc_curve, auc


def eer_point(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i]), (fpr, tpr, thr)


def tar_at_far(fpr, tpr, far):
    ok = fpr <= far
    return float(tpr[ok].max()) if ok.any() else 0.0


def threshold_at_far(scores, labels, far=0.01):
    """Smallest threshold whose false-accept rate does not exceed `far`."""
    imp = np.sort(np.asarray(scores)[np.asarray(labels) == 0])[::-1]
    if len(imp) == 0:
        return 0.5
    k = int(np.floor(far * len(imp)))
    return float(imp[min(k, len(imp) - 1)] + 1e-6)


def verification_metrics(scores, labels, threshold=None):
    """All metrics required by the brief. If `threshold` is None the EER threshold is used
    (only do that on validation data; on test pass the validation threshold)."""
    scores, labels = np.asarray(scores, float), np.asarray(labels, int)
    out = {"n_pairs": int(len(labels)), "n_genuine": int(labels.sum()), "n_impostor": int((1 - labels).sum())}
    if labels.min() == labels.max():
        return out
    eer, eer_thr, (fpr, tpr, _) = eer_point(scores, labels)
    thr = eer_thr if threshold is None else float(threshold)
    pred = scores >= thr
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    tn = int(np.sum(~pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    out.update({
        "threshold": round(thr, 4),
        "accuracy": (tp + tn) / len(labels),
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "FAR": fp / (fp + tn) if fp + tn else 0.0,
        "FRR": fn / (fn + tp) if fn + tp else 0.0,
        "EER": eer,
        "EER_threshold": round(eer_thr, 4),
        "AUC": float(auc(fpr, tpr)),
        "TAR@FAR=1e-2": tar_at_far(fpr, tpr, 1e-2),
        "TAR@FAR=1e-3": tar_at_far(fpr, tpr, 1e-3),
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
    })
    return out


def roc_points(scores, labels):
    fpr, tpr, thr = roc_curve(labels, scores)
    return fpr, tpr, thr


def identification_metrics(gal_emb, gal_ids, probe_emb, probe_ids, threshold, unknown_mask=None):
    """Closed-set rank-1/5 + open-set DIR/FPIR at a threshold.

    unknown_mask: boolean per probe, True when the probe identity is NOT in the gallery.
    """
    gal_ids = np.asarray(gal_ids)
    probe_ids = np.asarray(probe_ids)
    sims = probe_emb @ gal_emb.T
    order = np.argsort(-sims, axis=1)
    if unknown_mask is None:
        unknown_mask = ~np.isin(probe_ids, gal_ids)
    known = ~unknown_mask
    out = {"n_gallery_ids": int(len(np.unique(gal_ids))), "n_probes_known": int(known.sum()),
           "n_probes_unknown": int(unknown_mask.sum())}
    if known.any():
        top_ids = gal_ids[order[known]]
        out["rank1"] = float(np.mean(top_ids[:, 0] == probe_ids[known]))
        out["rank5"] = float(np.mean([p in row[:5] for p, row in zip(probe_ids[known], top_ids)]))
        best = sims[known, order[known][:, 0]]
        out["DIR@thr"] = float(np.mean((top_ids[:, 0] == probe_ids[known]) & (best >= threshold)))
    if unknown_mask.any():
        best_u = sims[unknown_mask].max(axis=1)
        out["FPIR@thr"] = float(np.mean(best_u >= threshold))
    return out

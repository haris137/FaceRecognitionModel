"""Stage 3: image quality assessment, filtering and enhancement.

Measured on the aligned 112x112 crop:
  * sharpness  -> variance of the Laplacian (low = blurry / motion blur)
  * brightness -> mean grey level (low = poor lighting, high = overexposed)
  * contrast   -> grey-level std
  * face size  -> shorter side of the detector box in the ORIGINAL image
  * det score  -> detector confidence (low = partial / heavily occluded face)
Hard failures (too small, low confidence) are rejected; soft problems
(dark, low contrast) are repaired with CLAHE + gamma before embedding.
"""
import cv2
import numpy as np

from .config import QualityConfig


def measure(face112, bbox_min_px=None, det_score=None):
    gray = cv2.cvtColor(face112, cv2.COLOR_BGR2GRAY)
    # ignore the outer border (often background / replicated padding)
    core = gray[16:100, 16:96]
    return {
        "blur": float(cv2.Laplacian(core, cv2.CV_64F).var()),
        "brightness": float(core.mean()),
        "contrast": float(core.std()),
        "face_px": float(bbox_min_px) if bbox_min_px is not None else 112.0,
        "det_score": float(det_score) if det_score is not None else 1.0,
    }


def assess(m, cfg: QualityConfig = QualityConfig()):
    flags = {
        "too_small": m["face_px"] < cfg.min_face_px,
        "low_confidence": m["det_score"] < cfg.min_det_score,
        "blurry": m["blur"] < cfg.blur_var_min,
        "dark": m["brightness"] < cfg.dark_mean_max,
        "overexposed": m["brightness"] > cfg.bright_mean_min,
        "low_contrast": m["contrast"] < cfg.low_contrast_std,
    }
    s_blur = np.clip(np.log1p(m["blur"]) / np.log1p(400.0), 0, 1)
    s_size = np.clip(m["face_px"] / 80.0, 0, 1)
    s_light = 1.0 - 0.6 * np.clip(abs(m["brightness"] - 125.0) / 125.0, 0, 1)
    s_det = np.clip(m["det_score"], 0, 1)
    score = float((s_blur ** 0.5) * (s_size ** 0.5) * s_light * s_det)
    usable = not (flags["too_small"] or flags["low_confidence"])
    return {"flags": flags, "score": round(score, 4), "usable": bool(usable)}


def _gamma(img, g):
    lut = np.clip(((np.arange(256) / 255.0) ** g) * 255.0, 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def enhance(face112, m=None, cfg: QualityConfig = QualityConfig()):
    """Repair lighting problems; leave good images untouched."""
    if m is None:
        m = measure(face112)
    out = face112
    dark = m["brightness"] < cfg.dark_mean_max
    over = m["brightness"] > cfg.bright_mean_min
    flat = m["contrast"] < cfg.low_contrast_std
    if dark or flat or over:
        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(l)
        out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    if dark:
        mean = max(cv2.cvtColor(out, cv2.COLOR_BGR2GRAY).mean(), 1.0) / 255.0
        g = np.clip(np.log(0.45) / np.log(mean), 0.4, 1.0)   # brighten toward mid-grey
        out = _gamma(out, g)
    elif over:
        out = _gamma(out, 1.4)
    return out

"""Face Embedding Challenge - submission script (wired to our FaceRec pipeline).

    python make_submission.py <group> <data_dir>
e.g. practice:  python make_submission.py group05 practice
     lab day:   python make_submission.py group05 lab_pack     -> upload group05.npz
<data_dir> must contain train/ and test/.

embed() = SCRFD detect -> 5-point align -> quality enhancement -> AdaFace IR-101
(fine-tuned best.pt) -> flip-TTA -> 512-d L2-normalised vector.
Optional flags (all have Colab defaults):
    --ckpt PATH          best.pt (or the original AdaFace .ckpt)
    --adaface_repo PATH  cloned AdaFace code
    --key CSV            practice_key.csv -> prints top-1 / top-5 self-check
"""
import argparse, os, sys, numpy as np
from PIL import Image

# make the facerec package importable wherever this file is run from
for _p in ("/content/facerec_pk", os.path.dirname(os.path.abspath(__file__)),
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")):
    if os.path.isdir(os.path.join(_p, "facerec")) and _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_CKPT = "/content/drive/MyDrive/NemoFR/runs/ft/best.pt"
DEFAULT_ADAFACE = "/content/AdaFace"
EMB_DIM = 512
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

_PIPE = None
STATS = {"ok": 0, "retry": 0, "fallback": 0}     # how each image was embedded
FALLBACK_FILES = []


def get_pipe(ckpt=None, adaface_repo=None, arch="ir_101"):
    """Build the pipeline once (loads weights + detector)."""
    global _PIPE
    if _PIPE is None:
        from facerec.config import PipelineConfig
        from facerec.pipeline import build_pipeline
        cfg = PipelineConfig(quality=True, flip_tta=True)
        _PIPE = build_pipeline(adaface_repo or DEFAULT_ADAFACE, ckpt or DEFAULT_CKPT,
                               arch, None, None, cfg)
        _PIPE.cfg.reject_low_quality = False      # benchmark: always answer
    return _PIPE


def _unit(v):
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 and np.isfinite(n) else None


def _try_extract(pipe, bgr):
    """Largest face -> embedding, or None if no face was found."""
    try:
        res = pipe.extract(bgr, largest_only=True)
    except Exception:
        return None
    if not res:
        return None
    e = _unit(res[0].embedding)
    return e if e is not None and np.isfinite(e).all() else None


def _last_resort(rgb):
    """Only used if NO face can be found at all: tiny grey thumbnail as a
    512-d vector, so the file keeps a constant D with no NaNs."""
    g = np.asarray(Image.fromarray(rgb).convert("L").resize((32, 16)), dtype=np.float32).ravel()
    g -= g.mean()
    return _unit(g) if np.linalg.norm(g) > 0 else np.ones(EMB_DIM, np.float32) / np.sqrt(EMB_DIM)


def embed(img: Image.Image) -> np.ndarray:
    """Return a 1-D float vector (512-d, L2-normalised) from OUR model."""
    import cv2
    pipe = get_pipe(*_CFG)
    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)          # pipeline expects BGR (cv2.imread)

    e = _try_extract(pipe, bgr)
    if e is not None:
        STATS["ok"] += 1
        return e

    # --- retries for hard images (tiny face, face at the edge, huge photo) ---
    h, w = bgr.shape[:2]
    variants = []
    pad = int(0.25 * max(h, w))                          # face touching the border
    variants.append(cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_REPLICATE))
    if max(h, w) > 1600:                                 # very large photo
        s = 1600 / max(h, w)
        variants.append(cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA))
    if max(h, w) < 640:                                  # small / CCTV-like photo
        variants.append(cv2.resize(bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC))
    variants.append(cv2.convertScaleAbs(bgr, alpha=1.6, beta=25))   # very dark photo
    for v in variants:
        e = _try_extract(pipe, v)
        if e is not None:
            STATS["retry"] += 1
            return e

    STATS["fallback"] += 1
    return _last_resort(rgb)


_CFG = (None, None)


def run(folder):
    ids = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXT))
    embs = []
    for i in ids:
        before = STATS["fallback"]
        embs.append(np.asarray(embed(Image.open(f"{folder}/{i}").convert("RGB")), dtype=np.float32))
        if STATS["fallback"] > before:
            FALLBACK_FILES.append(f"{folder}/{i}")
    return np.array(ids), np.stack(embs)


def self_check(tr_ids, tr, te_ids, te, key_csv):
    """top-1 / top-5 against practice_key.csv (train file name = person + .jpg)."""
    import csv
    key = {r["test_id"]: r["person"] for r in csv.DictReader(open(key_csv))}
    tr_n = tr / np.linalg.norm(tr, axis=1, keepdims=True)
    te_n = te / np.linalg.norm(te, axis=1, keepdims=True)
    sims = te_n @ tr_n.T
    top1 = top5 = 0
    for k, tid in enumerate(te_ids):
        order = np.argsort(-sims[k])
        names = [os.path.splitext(tr_ids[j])[0] for j in order]
        truth = key[tid]
        top1 += names[0] == truth
        top5 += truth in names[:5]
        print(f"  {tid} -> {names[0]:28s} sim={sims[k][order[0]]:.3f}  "
              f"{'OK' if names[0] == truth else 'WRONG (true: ' + truth + ')'}")
    n = len(te_ids)
    print(f"practice self-check: top-1 = {top1}/{n} ({top1 / n:.0%}), top-5 = {top5}/{n} ({top5 / n:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("group")
    ap.add_argument("data")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--adaface_repo", default=DEFAULT_ADAFACE)
    ap.add_argument("--arch", default="ir_101")
    ap.add_argument("--key", default=None, help="practice_key.csv for a top-1/top-5 self-check")
    a = ap.parse_args()
    _CFG = (a.ckpt, a.adaface_repo)
    get_pipe(a.ckpt, a.adaface_repo, a.arch)

    tr_ids, tr = run(f"{a.data}/train"); te_ids, te = run(f"{a.data}/test")

    # sanity checks required by the submission format
    assert tr.shape[1] == te.shape[1], "train/test embedding sizes differ"
    assert np.isfinite(tr).all() and np.isfinite(te).all(), "NaN/inf in embeddings"
    np.savez(f"{a.group}.npz", train_ids=tr_ids, train_emb=tr, test_ids=te_ids, test_emb=te)
    print("saved", f"{a.group}.npz", tr.shape, te.shape)
    print(f"faces: {STATS['ok']} direct, {STATS['retry']} via retry, {STATS['fallback']} NO FACE FOUND (fallback vector)")
    for f in FALLBACK_FILES:
        print("  no face found in:", f)

    key = a.key or os.path.join(os.path.dirname(os.path.abspath(a.data)), "practice_key.csv")
    if os.path.basename(os.path.abspath(a.data)) == "practice" and os.path.exists(key):
        self_check(tr_ids, tr, te_ids, te, key)
    elif a.key:
        self_check(tr_ids, tr, te_ids, te, a.key)

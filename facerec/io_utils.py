"""Image / video reading helpers (EXIF-safe, unicode-path-safe)."""
import os
import cv2
import numpy as np
from PIL import Image, ImageOps

from .config import IMG_EXT, VID_EXT


def is_image(path):
    return os.path.splitext(path)[1].lower() in IMG_EXT


def is_video(path):
    return os.path.splitext(path)[1].lower() in VID_EXT


def read_image(path, max_side=1600):
    """Read an image as BGR uint8, applying the phone's EXIF rotation.

    Phone photos are often stored sideways with an EXIF 'orientation' flag;
    ignoring it makes the detector miss faces, so we always transpose first.
    """
    if not os.path.isfile(path):
        return None
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            img = np.asarray(im)[:, :, ::-1].copy()
    except Exception:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            return None
    return limit_size(img, max_side)


def limit_size(img, max_side=1600):
    h, w = img.shape[:2]
    s = max(h, w)
    if max_side and s > max_side:
        r = max_side / s
        img = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_AREA)
    return img


def iter_video_frames(path, every_n=5, max_frames=None, max_side=1280):
    """Yield (frame_index, timestamp_sec, BGR frame) for every n-th frame."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    idx, kept = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every_n == 0:
            yield idx, idx / fps, limit_size(frame, max_side)
            kept += 1
            if max_frames and kept >= max_frames:
                break
        idx += 1
    cap.release()


def save_image(path, img, quality=95):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        buf.tofile(path)
    return ok


def parse_filename_tags(filename):
    """'selfie03__mask-glasses-lowlight.jpg' -> {'mask','glasses','lowlight'}."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    if "__" not in stem:
        return set()
    return {t.strip().lower() for t in stem.split("__", 1)[1].split("-") if t.strip()}

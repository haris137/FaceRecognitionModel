"""Stage 1+2: face detection (RetinaFace/SCRFD from InsightFace) and 5-point alignment."""
import cv2
import numpy as np

# ArcFace / AdaFace canonical 5-point template for a 112x112 crop
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


def onnx_providers():
    import onnxruntime as ort
    ort.set_default_logger_severity(3)   # silence harmless shape warnings
    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]


class FaceDetector:
    """RetinaFace-family detector (InsightFace buffalo_l: det_10g = SCRFD-10GF).

    Returns boxes, 5 landmarks (eyes, nose, mouth corners) and confidence.
    A second pass on a padded + upscaled image rescues faces in tight crops
    or tiny / CCTV-style images that the first pass misses.
    """

    def __init__(self, det_size=640, det_thresh=0.5, model_pack="buffalo_l", root="~/.insightface"):
        from insightface.app import FaceAnalysis
        providers = onnx_providers()
        self.app = FaceAnalysis(name=model_pack, root=root, allowed_modules=["detection"], providers=providers)
        ctx = 0 if "CUDAExecutionProvider" in providers else -1
        self.app.prepare(ctx_id=ctx, det_size=(det_size, det_size), det_thresh=det_thresh)

    def _run(self, img):
        out = []
        for f in self.app.get(img):
            out.append({"bbox": f.bbox.astype(np.float32),
                        "kps": f.kps.astype(np.float32),
                        "score": float(f.det_score)})
        return out

    def detect(self, img, retry=True):
        dets = self._run(img)
        if dets or not retry:
            return dets
        h, w = img.shape[:2]
        scale = max(1.0, 480.0 / max(h, w))
        big = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        pad = int(0.35 * max(big.shape[:2]))
        big = cv2.copyMakeBorder(big, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        dets = self._run(big)
        for d in dets:
            d["bbox"] = (d["bbox"] - pad) / scale
            d["kps"] = (d["kps"] - pad) / scale
            d["retried"] = True
        return dets


def largest_face(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))


def bbox_min_side(bbox):
    return float(min(bbox[2] - bbox[0], bbox[3] - bbox[1]))


def align_face(img, kps, size=112):
    """Similarity transform of the 5 landmarks onto the ArcFace template.

    Removes in-plane rotation, scale and translation so that eyes/nose/mouth
    always land on the same pixels -- the embedding network was trained on
    exactly this geometry.
    """
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), dst, method=cv2.LMEDS)
    if M is None:  # degenerate landmarks
        M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, dtype=np.float32), dst)
    return cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)


def naive_crop(img, bbox, size=112, margin=0.08):
    """Ablation baseline: square bbox crop + resize, no landmark alignment."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + 2 * margin)
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = max(0, -x1, -y1, x2 - w, y2 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x1, y1, x2, y2 = x1 + pad, y1 + pad, x2 + pad, y2 + pad
    crop = img[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA if crop.shape[0] > size else cv2.INTER_CUBIC)

"""End-to-end pipeline + gallery matching for images and videos.

image/video -> detect -> align & crop -> quality gate / enhance -> embed -> match
"""
import json
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import PipelineConfig, QualityConfig
from .detect_align import align_face, naive_crop, bbox_min_side, largest_face
from . import quality as Q
from .io_utils import read_image, iter_video_frames


@dataclass
class FaceResult:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    crop: np.ndarray            # 112x112 BGR crop returned to the user (aligned, before enhancement)
    model_input: np.ndarray     # what was actually fed to the network
    quality: dict
    embedding: np.ndarray = None
    norm: float = 0.0
    identity: str = "Unknown"
    score: float = 0.0
    top_k: list = field(default_factory=list)
    frame_idx: int = None
    time_sec: float = None
    track_id: int = None

    def summary(self):
        return {
            "bbox": [round(float(v), 1) for v in self.bbox],
            "det_score": round(self.det_score, 3),
            "identity": self.identity,
            "similarity": round(float(self.score), 4),
            "top_k": [(n, round(float(s), 4)) for n, s in self.top_k],
            "quality_score": self.quality["score"],
            "quality_flags": [k for k, v in self.quality["flags"].items() if v],
            "frame_idx": self.frame_idx, "time_sec": None if self.time_sec is None else round(self.time_sec, 2),
            "track_id": self.track_id,
        }


class Gallery:
    """Enrolled identities. Each identity = one quality-weighted mean template."""

    def __init__(self):
        self.names, self.templates, self.counts = [], np.zeros((0, 512), np.float32), []

    def enroll(self, name, embs, weights=None):
        embs = np.asarray(embs, np.float32)
        if len(embs) == 0:
            return False
        w = np.ones(len(embs)) if weights is None else np.asarray(weights, np.float64) + 1e-3
        t = (embs * w[:, None]).sum(0)
        t /= np.linalg.norm(t) + 1e-9
        if name in self.names:
            i = self.names.index(name)
            self.templates[i] = t
            self.counts[i] = len(embs)
        else:
            self.names.append(name)
            self.templates = np.vstack([self.templates, t[None]])
            self.counts.append(len(embs))
        return True

    def remove(self, name):
        if name in self.names:
            i = self.names.index(name)
            self.names.pop(i)
            self.counts.pop(i)
            self.templates = np.delete(self.templates, i, axis=0)

    def search(self, emb, k=3):
        if not self.names:
            return []
        sims = self.templates @ emb
        order = np.argsort(-sims)[:k]
        return [(self.names[i], float(sims[i])) for i in order]

    def save(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        np.savez(path, names=np.array(self.names), templates=self.templates, counts=np.array(self.counts))

    @classmethod
    def load(cls, path):
        g = cls()
        if path and os.path.exists(path):
            d = np.load(path, allow_pickle=False)
            g.names = [str(n) for n in d["names"]]
            g.templates = d["templates"].astype(np.float32)
            g.counts = [int(c) for c in d["counts"]]
        return g

    def __len__(self):
        return len(self.names)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class FaceRecognitionPipeline:
    def __init__(self, detector, embedder, cfg: PipelineConfig = None, qcfg: QualityConfig = None, gallery=None):
        self.detector = detector
        self.embedder = embedder
        self.cfg = cfg or PipelineConfig()
        self.qcfg = qcfg or QualityConfig()
        self.gallery = gallery or Gallery()

    # ---------- core ----------
    def prepare_face(self, img, det):
        crop = align_face(img, det["kps"]) if self.cfg.align else naive_crop(img, det["bbox"])
        m = Q.measure(crop, bbox_min_side(det["bbox"]), det["score"])
        qa = Q.assess(m, self.qcfg)
        qa["metrics"] = {k: round(v, 2) for k, v in m.items()}
        model_input = Q.enhance(crop, m, self.qcfg) if self.cfg.quality else crop
        return crop, model_input, qa

    def extract(self, img, largest_only=False):
        if img is None:
            return []
        dets = self.detector.detect(img)
        if largest_only and dets:
            dets = [largest_face(dets)]
        results = []
        for d in dets:
            crop, mi, qa = self.prepare_face(img, d)
            results.append(FaceResult(d["bbox"], d["kps"], d["score"], crop, mi, qa))
        if results:
            embs, norms = self.embedder.embed([r.model_input for r in results], flip_tta=self.cfg.flip_tta)
            for r, e, n in zip(results, embs, norms):
                r.embedding, r.norm = e, float(n)
        return results

    def _match(self, r):
        if self.cfg.quality and self.cfg.reject_low_quality and not r.quality["usable"]:
            r.identity, r.score = "Rejected (low quality)", 0.0
            return r
        r.top_k = self.gallery.search(r.embedding, k=3)
        if r.top_k and r.top_k[0][1] >= self.cfg.threshold:
            r.identity, r.score = r.top_k[0]
        else:
            r.identity = "Unknown"
            r.score = r.top_k[0][1] if r.top_k else 0.0
        return r

    # ---------- public API ----------
    def recognize_image(self, img):
        if isinstance(img, str):
            img = read_image(img)
        return [self._match(r) for r in self.extract(img)]

    def verify(self, img1, img2):
        """1:1 verification on the largest face of each image."""
        f1 = self.extract(read_image(img1) if isinstance(img1, str) else img1, largest_only=True)
        f2 = self.extract(read_image(img2) if isinstance(img2, str) else img2, largest_only=True)
        if not f1 or not f2:
            return {"same_person": None, "similarity": None, "reason": "no face detected"}
        s = float(f1[0].embedding @ f2[0].embedding)
        return {"same_person": bool(s >= self.cfg.threshold), "similarity": round(s, 4),
                "threshold": self.cfg.threshold, "faces": (f1[0], f2[0])}

    def enroll(self, name, images):
        embs, w = [], []
        for im in images:
            im = read_image(im) if isinstance(im, str) else im
            if im is None:
                continue
            res = self.extract(im, largest_only=True)
            if res and res[0].quality["usable"]:
                embs.append(res[0].embedding)
                w.append(res[0].quality["score"] * res[0].norm)
        ok = self.gallery.enroll(name, embs, w)
        return {"name": name, "images_used": len(embs), "enrolled": ok}

    def recognize_video(self, path, every_n=5, max_frames=400, iou_thr=0.3, emb_thr=0.45, max_gap=6):
        """Detect per sampled frame, link faces into tracks, decide identity per track.

        A track's identity uses the quality-weighted mean embedding of all its
        faces, which is far more reliable than any single blurry frame.
        """
        tracks, frames_out = [], []
        for step, (fi, ts, frame) in enumerate(iter_video_frames(path, every_n, max_frames)):
            faces = self.extract(frame)
            for r in faces:
                r.frame_idx, r.time_sec = fi, ts
                best, best_s = None, 0.0
                for t in tracks:
                    if step - t["last_step"] > max_gap:
                        continue
                    s_iou = _iou(t["last_bbox"], r.bbox)
                    s_emb = float(t["mean"] @ r.embedding / (np.linalg.norm(t["mean"]) + 1e-9))
                    s = s_iou + s_emb if (s_iou >= iou_thr or s_emb >= emb_thr) else 0.0
                    if s > best_s:
                        best, best_s = t, s
                w = r.quality["score"] * r.norm + 1e-3
                if best is None:
                    best = {"id": len(tracks), "faces": [], "sum": np.zeros(512, np.float32), "wsum": 0.0}
                    tracks.append(best)
                best["faces"].append(r)
                best["sum"] += w * r.embedding
                best["wsum"] += w
                best["mean"] = best["sum"] / best["wsum"]
                best["last_bbox"], best["last_step"] = r.bbox, step
                r.track_id = best["id"]
            frames_out.append((fi, frame, faces))

        summaries = []
        for t in tracks:
            usable = [f for f in t["faces"] if f.quality["usable"]] or t["faces"]
            emb = t["mean"] / (np.linalg.norm(t["mean"]) + 1e-9)
            top = self.gallery.search(emb, k=3)
            ident = top[0][0] if top and top[0][1] >= self.cfg.threshold else "Unknown"
            best_face = max(usable, key=lambda f: f.quality["score"])
            for f in t["faces"]:
                f.identity = ident
                f.score = top[0][1] if top else 0.0
            summaries.append({"track_id": t["id"], "identity": ident,
                              "similarity": round(top[0][1], 4) if top else None,
                              "top_k": [(n, round(s, 4)) for n, s in top],
                              "n_faces": len(t["faces"]),
                              "first_sec": round(t["faces"][0].time_sec, 2),
                              "last_sec": round(t["faces"][-1].time_sec, 2),
                              "best_crop": best_face.crop})
        return summaries, frames_out

    # ---------- drawing ----------
    @staticmethod
    def annotate(img, results):
        out = img.copy()
        for r in results:
            x1, y1, x2, y2 = [int(v) for v in r.bbox]
            known = r.identity not in ("Unknown",) and not r.identity.startswith("Rejected")
            color = (60, 180, 60) if known else ((0, 140, 255) if r.identity == "Unknown" else (60, 60, 220))
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{r.identity} {r.score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
            cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out


def build_pipeline(adaface_repo, ckpt, arch="ir_101", threshold_file=None, gallery_path=None,
                   cfg: PipelineConfig = None, device=None):
    """Convenience factory used by the app, CLI and notebook."""
    from .detect_align import FaceDetector
    from .embedder import AdaFaceEmbedder
    cfg = cfg or PipelineConfig()
    if threshold_file and os.path.exists(threshold_file):
        with open(threshold_file) as f:
            cfg.threshold = float(json.load(f)["threshold"])
    det = FaceDetector(det_size=cfg.det_size, det_thresh=cfg.det_thresh)
    emb = AdaFaceEmbedder(adaface_repo, ckpt, arch=arch, device=device)
    return FaceRecognitionPipeline(det, emb, cfg, QualityConfig(), Gallery.load(gallery_path))

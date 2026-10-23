"""Training-time augmentation for aligned 112x112 BGR faces.

Pure OpenCV/NumPy (no albumentations) so it does not break when Colab
upgrades libraries. Simulates the real-world Pakistani conditions in the brief:
low-quality mobile / CCTV capture (blur, downscale, JPEG, noise, grey),
poor or harsh lighting, and occlusions (mask, dupatta/hijab/cap band, sunglasses).
"""
import cv2
import numpy as np


class FaceAugmenter:
    def __init__(self, p_flip=0.5, p_blur=0.3, p_downscale=0.3, p_jpeg=0.35, p_noise=0.2,
                 p_light=0.5, p_gray=0.1, p_mask=0.15, p_headwear=0.15, p_sunglasses=0.05,
                 p_affine=0.4, seed=None):
        self.p = dict(flip=p_flip, blur=p_blur, down=p_downscale, jpeg=p_jpeg, noise=p_noise,
                      light=p_light, gray=p_gray, mask=p_mask, head=p_headwear, sun=p_sunglasses,
                      affine=p_affine)
        self.rng = np.random.default_rng(seed)

    def _r(self, k):
        return self.rng.random() < self.p[k]

    def __call__(self, img):
        img = img.copy()
        r = self.rng
        if self._r("flip"):
            img = img[:, ::-1].copy()
        if self._r("affine"):
            ang = r.uniform(-8, 8)
            sc = r.uniform(0.94, 1.06)
            M = cv2.getRotationMatrix2D((56 + r.uniform(-3, 3), 56 + r.uniform(-3, 3)), ang, sc)
            img = cv2.warpAffine(img, M, (112, 112), borderMode=cv2.BORDER_REPLICATE)
        if self._r("mask"):
            img = synthetic_mask(img, r)
        if self._r("head"):
            img = synthetic_headwear(img, r)
        if self._r("sun"):
            img = synthetic_sunglasses(img, r)
        if self._r("light"):
            alpha = r.uniform(0.55, 1.35)          # contrast
            beta = r.uniform(-45, 35)              # brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
            if r.random() < 0.3:                   # colour cast (tube light / sodium lamp)
                cast = r.uniform(0.85, 1.15, size=3)
                img = np.clip(img.astype(np.float32) * cast, 0, 255).astype(np.uint8)
        if self._r("down"):
            s = int(r.integers(14, 48))            # CCTV-like tiny faces
            small = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (112, 112), interpolation=r.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC]))
        if self._r("blur"):
            if r.random() < 0.5:
                k = int(r.choice([3, 5, 7]))
                img = cv2.GaussianBlur(img, (k, k), 0)
            else:
                k = int(r.integers(5, 12))
                kern = np.zeros((k, k), np.float32)
                kern[k // 2, :] = 1.0 / k
                M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), r.uniform(0, 180), 1.0)
                kern = cv2.warpAffine(kern, M, (k, k))
                kern /= max(kern.sum(), 1e-6)
                img = cv2.filter2D(img, -1, kern)
        if self._r("noise"):
            img = np.clip(img.astype(np.float32) + r.normal(0, r.uniform(3, 12), img.shape), 0, 255).astype(np.uint8)
        if self._r("jpeg"):
            q = int(r.integers(15, 60))
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if self._r("gray"):
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return img


def _rand_color(r, palette=None):
    palette = palette or [(235, 235, 235), (200, 160, 90), (40, 40, 40), (120, 170, 110),
                          (190, 190, 230), (90, 60, 160), (60, 90, 200), (150, 150, 150)]
    c = np.array(palette[int(r.integers(len(palette)))], np.float32)
    return tuple(int(v) for v in np.clip(c + r.normal(0, 12, 3), 0, 255))


def synthetic_mask(img, r):
    """Surgical/cloth mask over nose-mouth-chin of an aligned face."""
    top = int(r.integers(62, 72))
    pts = np.array([[14, top + 6], [34, top - 2], [56, top - 6], [78, top - 2], [98, top + 6],
                    [96, 96], [80, 110], [56, 116], [32, 110], [16, 96]], np.int32)
    out = img.copy()
    cv2.fillPoly(out, [pts], _rand_color(r))
    if r.random() < 0.5:  # pleats
        for y in (top + 14, top + 24, top + 34):
            cv2.line(out, (20, y), (92, y), (0, 0, 0), 1)
    return cv2.addWeighted(out, 0.95, img, 0.05, 0)


def synthetic_headwear(img, r):
    """Band across the forehead and/or sides: dupatta, hijab, cap, turban."""
    out = img.copy()
    color = _rand_color(r)
    kind = r.integers(3)
    if kind == 0:                         # cap / turban: forehead band
        h = int(r.integers(16, 34))
        cv2.rectangle(out, (0, 0), (111, h), color, -1)
    elif kind == 1:                       # hijab: forehead + both sides
        h = int(r.integers(14, 28))
        w = int(r.integers(8, 20))
        cv2.rectangle(out, (0, 0), (111, h), color, -1)
        cv2.rectangle(out, (0, 0), (w, 111), color, -1)
        cv2.rectangle(out, (111 - w, 0), (111, 111), color, -1)
    else:                                 # loose dupatta: one side
        w = int(r.integers(10, 26))
        x0 = 0 if r.random() < 0.5 else 111 - w
        pts = np.array([[x0, 0], [x0 + w, 0], [x0 + w + int(r.integers(-6, 6)), 111], [x0, 111]], np.int32)
        cv2.fillPoly(out, [pts], color)
    return out


def synthetic_sunglasses(img, r):
    out = img.copy()
    for cx in (38, 74):
        cv2.ellipse(out, (cx, 52), (int(r.integers(12, 16)), int(r.integers(8, 11))), 0, 0, 360, (20, 20, 20), -1)
    cv2.line(out, (50, 50), (62, 50), (20, 20, 20), 2)
    return out

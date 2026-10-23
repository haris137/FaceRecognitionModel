"""Central configuration for the Pakistani-context face recognition pipeline."""
from dataclasses import dataclass, asdict

# Pretrained AdaFace checkpoints (official repo: github.com/mk-minchul/AdaFace).
# R100 trained on WebFace12M is the strongest public AdaFace model.
ADAFACE_WEIGHTS = {
    "ir_101_webface12m": {"arch": "ir_101", "gdrive_id": "1dswnavflETcnAuplZj1IOKKP0eM8ITgT"},
    "ir_101_webface4m":  {"arch": "ir_101", "gdrive_id": "18jQkqB0avFqWa0Pas52g54xNshUOQJpQ"},
    "ir_50_webface4m":   {"arch": "ir_50",  "gdrive_id": "1BmDRrhPsHSbXcWZoYFPJg2KJn1sd3QpN"},
    "ir_50_ms1mv2":      {"arch": "ir_50",  "gdrive_id": "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI"},
}

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}
VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".3gp", ".webm", ".m4v"}

# Condition tags (from file names "photo__mask-glasses.jpg" or image_tags.csv)
# grouped into the categories the brief asks us to analyse.
TAG_GROUPS = {
    "head_covering": {"hijab", "dupatta", "turban", "cap", "scarf", "chador", "niqab"},
    "glasses": {"glasses", "sunglasses", "specs"},
    "mask": {"mask", "facemask"},
    "facial_hair": {"beard", "moustache", "mustache", "stubble"},
    "low_light": {"lowlight", "dark", "night", "backlight"},
    "low_quality_capture": {"cctv", "lowres", "blur", "blurry", "webcam", "screenshot"},
    "outdoor": {"outdoor", "sunlight"},
}


@dataclass
class QualityConfig:
    min_face_px: int = 24          # shorter bbox side below this -> reject (cannot be recognised reliably)
    min_det_score: float = 0.55    # detector confidence below this -> reject
    blur_var_min: float = 35.0     # Laplacian variance on the 112x112 crop; below -> "blurry"
    dark_mean_max: float = 70.0    # mean grey level below -> "dark", gets CLAHE + gamma
    bright_mean_min: float = 195.0 # above -> "overexposed", gets gamma > 1
    low_contrast_std: float = 28.0 # std below -> "low contrast", gets CLAHE


@dataclass
class PipelineConfig:
    det_size: int = 640
    det_thresh: float = 0.5
    align: bool = True             # 5-point similarity alignment (vs. plain bbox crop)
    quality: bool = True           # quality gate + low-light enhancement
    flip_tta: bool = True          # average embedding of image and its mirror
    threshold: float = 0.30        # cosine threshold; overwritten by the calibrated value
    reject_low_quality: bool = True

    def to_dict(self):
        return asdict(self)

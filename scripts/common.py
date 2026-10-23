"""Shared CLI arguments for scripts that load the full pipeline."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def add_pipeline_args(ap):
    ap.add_argument("--adaface_repo", default="/content/AdaFace")
    ap.add_argument("--ckpt", required=True, help="best.pt from train.py or the official AdaFace .ckpt")
    ap.add_argument("--arch", default="ir_101")
    ap.add_argument("--threshold_file", default=None, help="threshold.json from evaluate.py")
    ap.add_argument("--threshold", type=float, default=None, help="override threshold")
    ap.add_argument("--gallery", default=None, help="gallery .npz from enroll_gallery.py")
    ap.add_argument("--no_quality", action="store_true")
    ap.add_argument("--no_tta", action="store_true")
    return ap


def pipeline_from_args(args):
    from facerec.config import PipelineConfig
    from facerec.pipeline import build_pipeline
    cfg = PipelineConfig(quality=not args.no_quality, flip_tta=not args.no_tta)
    pipe = build_pipeline(args.adaface_repo, args.ckpt, args.arch, args.threshold_file, args.gallery, cfg)
    if args.threshold is not None:
        pipe.cfg.threshold = args.threshold
    return pipe

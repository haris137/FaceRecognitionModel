"""Download pretrained AdaFace weights (official Google Drive links) and the InsightFace detector pack.

  python scripts/download_weights.py --name ir_101_webface12m --out /content/weights
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facerec.config import ADAFACE_WEIGHTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="ir_101_webface12m", choices=list(ADAFACE_WEIGHTS))
    ap.add_argument("--out", default="weights")
    args = ap.parse_args()
    import gdown
    os.makedirs(args.out, exist_ok=True)
    dst = os.path.join(args.out, f"adaface_{args.name}.ckpt")
    if not os.path.exists(dst):
        gdown.download(id=ADAFACE_WEIGHTS[args.name]["gdrive_id"], output=dst, quiet=False)
    print("AdaFace weights:", dst, f"({os.path.getsize(dst) / 1e6:.0f} MB)")
    # triggers the one-time download of buffalo_l (detector + ArcFace) to ~/.insightface
    from facerec.detect_align import FaceDetector
    FaceDetector()
    print("InsightFace buffalo_l ready")


if __name__ == "__main__":
    main()

"""Image or video in -> detected, recognised and cropped faces out.

  python scripts/run_inference.py --ckpt runs/ft/best.pt --threshold_file results/threshold.json \
         --gallery gallery/gallery.npz --input some_photo.jpg --out outputs/
Writes: <out>/<name>_annotated.jpg, <out>/<name>_crops/*.jpg, <out>/<name>.json
For a video, one crop per tracked person (their best-quality frame) plus annotated key frames.
"""
import argparse
import json
import os

from common import add_pipeline_args, pipeline_from_args

from facerec.io_utils import is_video, read_image, save_image
from facerec.pipeline import FaceRecognitionPipeline


def safe(s):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in s)


def main():
    ap = add_pipeline_args(argparse.ArgumentParser())
    ap.add_argument("--input", required=True, help="image, video, or folder of them")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--every_n", type=int, default=5)
    args = ap.parse_args()
    pipe = pipeline_from_args(args)
    os.makedirs(args.out, exist_ok=True)
    inputs = [os.path.join(args.input, f) for f in sorted(os.listdir(args.input))] if os.path.isdir(args.input) else [args.input]
    for path in inputs:
        name = os.path.splitext(os.path.basename(path))[0]
        crop_dir = os.path.join(args.out, f"{name}_crops")
        if is_video(path):
            tracks, frames = pipe.recognize_video(path, every_n=args.every_n)
            for t in tracks:
                save_image(os.path.join(crop_dir, f"track{t['track_id']:02d}_{safe(t['identity'])}.jpg"), t.pop("best_crop"))
            for fi, frame, faces in frames[:: max(1, len(frames) // 8)]:
                save_image(os.path.join(args.out, f"{name}_frame{fi:05d}.jpg"), FaceRecognitionPipeline.annotate(frame, faces))
            report = {"input": path, "type": "video", "tracks": tracks}
        else:
            img = read_image(path)
            if img is None:
                print("cannot read", path)
                continue
            res = pipe.recognize_image(img)
            for i, r in enumerate(res):
                save_image(os.path.join(crop_dir, f"face{i:02d}_{safe(r.identity)}.jpg"), r.crop)
            save_image(os.path.join(args.out, f"{name}_annotated.jpg"), FaceRecognitionPipeline.annotate(img, res))
            report = {"input": path, "type": "image", "faces": [r.summary() for r in res]}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(report, f, indent=1, default=str)
        n = len(report.get("faces", report.get("tracks", [])))
        print(f"{path}: {n} face(s) -> {crop_dir}")


if __name__ == "__main__":
    main()

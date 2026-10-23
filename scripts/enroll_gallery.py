"""Build the gallery of known people.

  python scripts/enroll_gallery.py --ckpt runs/ft/best.pt --people gallery_photos/ --gallery gallery/gallery.npz
where gallery_photos/<Person Name>/*.jpg (1-5 clear photos per person works best).
"""
import argparse
import os

from common import add_pipeline_args, pipeline_from_args

from facerec.io_utils import is_image


def main():
    ap = add_pipeline_args(argparse.ArgumentParser())
    ap.add_argument("--people", required=True, help="folder with one sub-folder of photos per person")
    ap.add_argument("--fresh", action="store_true", help="start an empty gallery instead of updating")
    args = ap.parse_args()
    if args.fresh:
        gallery_out, args.gallery = args.gallery, None
    else:
        gallery_out = args.gallery
    pipe = pipeline_from_args(args)
    for person in sorted(os.listdir(args.people)):
        d = os.path.join(args.people, person)
        if not os.path.isdir(d):
            continue
        files = [os.path.join(d, f) for f in sorted(os.listdir(d)) if is_image(f)]
        r = pipe.enroll(person, files)
        print(f"{person:30s} {r['images_used']}/{len(files)} photos used")
    pipe.gallery.save(gallery_out)
    print(f"gallery with {len(pipe.gallery)} people saved to {gallery_out}")


if __name__ == "__main__":
    main()

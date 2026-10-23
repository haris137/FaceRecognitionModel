"""Fine-tune AdaFace on the collected Pakistani-face data.

Strategy (small data -> protect what the model already knows):
  * start from AdaFace IR-101 pretrained on WebFace12M;
  * freeze the early layers, train only the last `--unfreeze_blocks` residual
    blocks + the output (embedding) layer, BatchNorm of frozen parts kept in eval;
  * new AdaFace margin head over our identities, initialised from the class-mean
    embeddings of the pretrained model (converges in a few epochs);
  * embedding-preservation (distillation) loss to a frozen copy of the original
    model, so fine-tuning adapts to Pakistani faces without forgetting general skill;
  * heavy augmentation simulating CCTV/mobile quality, lighting and occlusions;
  * model selection on the identity-disjoint VALIDATION split (EER), never test.
"""
import argparse
import copy
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facerec.augment import FaceAugmenter  # noqa: E402
from facerec.data import load_metadata, load_splits, subset, build_pairs  # noqa: E402
from facerec.embedder import load_backbone, import_adaface, faces_to_tensor, AdaFaceEmbedder  # noqa: E402
from facerec.metrics import verification_metrics  # noqa: E402


class FaceDataset(Dataset):
    def __init__(self, root, paths, labels, augment=None):
        self.root, self.paths, self.labels, self.aug = root, list(paths), list(labels), augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(os.path.join(self.root, self.paths[i]))
        if self.aug is not None:
            # each worker gets its own RNG stream
            if self.aug.rng is None or not hasattr(self, "_seeded"):
                info = torch.utils.data.get_worker_info()
                self.aug.rng = np.random.default_rng((info.id if info else 0) + int(time.time() * 1000) % 100000)
                self._seeded = True
            img = self.aug(img)
        x = faces_to_tensor([img])[0]
        return x, self.labels[i]


def load_crops(root, paths):
    return [cv2.imread(os.path.join(root, p)) for p in paths]


def evaluate_val(model, val_imgs, pairs, device):
    emb = AdaFaceEmbedder(None, None, device=device, model=model)
    e, _ = emb.embed(val_imgs, flip_tta=False)
    s = np.sum(e[pairs[:, 0]] * e[pairs[:, 1]], axis=1)
    return verification_metrics(s, pairs[:, 2])


def set_trainable(model, n_blocks):
    for p in model.parameters():
        p.requires_grad = False
    blocks = list(model.body)[-n_blocks:] if n_blocks > 0 else []
    for m in blocks + [model.output_layer]:
        for p in m.parameters():
            p.requires_grad = True
    return blocks


def train_mode(model, trainable_blocks):
    model.eval()                      # frozen BN layers keep their running stats
    for m in trainable_blocks + [model.output_layer]:
        m.train()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed dir from prepare_data.py")
    ap.add_argument("--adaface_repo", required=True)
    ap.add_argument("--ckpt", required=True, help="pretrained AdaFace checkpoint")
    ap.add_argument("--arch", default="ir_101")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr_backbone", type=float, default=3e-5)
    ap.add_argument("--lr_head", type=float, default=1e-3)
    ap.add_argument("--unfreeze_blocks", type=int, default=6, help="last N residual blocks to train (IR-101 has 49)")
    ap.add_argument("--distill", type=float, default=1.0, help="weight of embedding-preservation loss (0 = off)")
    ap.add_argument("--margin", type=float, default=0.4)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--include_rejected", action="store_true", help="also train on quality-rejected crops")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    _, head_mod = import_adaface(args.adaface_repo)

    df = load_metadata(args.data)
    splits = load_splits(args.data)
    tr = subset(df, splits, "train", usable_only=not args.include_rejected)
    va = subset(df, splits, "val", usable_only=False)
    classes = sorted(tr["identity"].unique())
    cmap = {c: i for i, c in enumerate(classes)}
    labels = tr["identity"].map(cmap).values
    print(f"train: {len(tr)} images / {len(classes)} ids | val: {len(va)} images / {va.identity.nunique()} ids | device={device}")

    model = load_backbone(args.adaface_repo, args.arch, args.ckpt, device)
    teacher = copy.deepcopy(model).eval() if args.distill > 0 else None
    if teacher is not None:
        for p in teacher.parameters():
            p.requires_grad = False

    val_imgs = load_crops(args.data, va["path_aligned"])
    val_pairs = build_pairs(va, seed=123)
    base = evaluate_val(model, val_imgs, val_pairs, device)
    print(f"[epoch 0 | pretrained] val EER={base['EER']:.4f}  TAR@FAR1%={base['TAR@FAR=1e-2']:.4f}")
    history = [{"epoch": 0, **{k: base[k] for k in ("EER", "TAR@FAR=1e-2", "AUC")}}]

    # head initialised from class means of the pretrained embeddings
    head = head_mod.AdaFace(embedding_size=512, classnum=len(classes), m=args.margin, h=0.333, s=64., t_alpha=0.01).to(device)
    tr_imgs = load_crops(args.data, tr["path_aligned"])
    e, _ = AdaFaceEmbedder(None, None, device=device, model=model).embed(tr_imgs, flip_tta=False)
    means = np.stack([e[labels == k].mean(0) for k in range(len(classes))])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    head.kernel.data = torch.from_numpy(means.T.copy()).float().to(device)
    del tr_imgs

    blocks = set_trainable(model, args.unfreeze_blocks)
    params_b = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": params_b, "lr": args.lr_backbone},
                             {"params": head.parameters(), "lr": args.lr_head}], weight_decay=5e-4)
    ds = FaceDataset(args.data, tr["path_aligned"], labels, FaceAugmenter())
    # class-balanced sampling: people with 40 photos should not drown people with 8
    w = 1.0 / np.bincount(labels)[labels]
    sampler = torch.utils.data.WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(labels), replacement=True)
    dl = DataLoader(ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, drop_last=True,
                    pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    total = args.epochs * max(1, len(dl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_backbone, args.lr_head], total_steps=total, pct_start=0.15)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    ce = nn.CrossEntropyLoss()

    best = {"EER": base["EER"], "TAR@FAR=1e-2": base["TAR@FAR=1e-2"], "epoch": 0}
    torch.save({"backbone": model.state_dict(), "arch": args.arch, "epoch": 0, "val": base}, os.path.join(args.out, "best.pt"))
    bad = 0
    for ep in range(1, args.epochs + 1):
        train_mode(model, blocks)
        head.train()
        t0, run_ce, run_d, n = time.time(), 0.0, 0.0, 0
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device == "cuda"):
                f, nrm = model(x)
                if teacher is not None:
                    with torch.no_grad():
                        ft, _ = teacher(x)
            f, nrm = f.float(), nrm.float()
            logits = head(f, nrm, y)
            loss_ce = ce(logits, y)
            loss_d = (1 - (f * ft.float()).sum(1)).mean() if teacher is not None else torch.zeros((), device=device)
            loss = loss_ce + args.distill * loss_d
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_ce += loss_ce.item()
            run_d += loss_d.item()
            n += 1
        model.eval()
        m = evaluate_val(model, val_imgs, val_pairs, device)
        history.append({"epoch": ep, "ce": run_ce / max(n, 1), "distill": run_d / max(n, 1),
                        **{k: m[k] for k in ("EER", "TAR@FAR=1e-2", "AUC")}})
        improved = (m["EER"] < best["EER"] - 1e-4) or (abs(m["EER"] - best["EER"]) <= 1e-4 and m["TAR@FAR=1e-2"] > best["TAR@FAR=1e-2"])
        print(f"[epoch {ep}] ce={run_ce / max(n, 1):.3f} distill={run_d / max(n, 1):.4f} "
              f"val EER={m['EER']:.4f} TAR@FAR1%={m['TAR@FAR=1e-2']:.4f} ({time.time() - t0:.0f}s){'  *best*' if improved else ''}")
        if improved:
            best = {"EER": m["EER"], "TAR@FAR=1e-2": m["TAR@FAR=1e-2"], "epoch": ep}
            torch.save({"backbone": model.state_dict(), "arch": args.arch, "epoch": ep, "val": m}, os.path.join(args.out, "best.pt"))
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print("early stopping")
                break

    with open(os.path.join(args.out, "train_history.json"), "w") as f:
        json.dump({"args": vars(args), "best": best, "pretrained_val": {k: base[k] for k in ("EER", "TAR@FAR=1e-2", "AUC")},
                   "history": history, "classes": len(classes)}, f, indent=1)
    if best["epoch"] == 0:
        print("WARNING: fine-tuning did not beat the pretrained model on validation; best.pt = pretrained weights.")
    print(f"best epoch {best['epoch']}: val EER {best['EER']:.4f} (pretrained {base['EER']:.4f})")


if __name__ == "__main__":
    main()

"""Stage 4: embedding extraction.

AdaFaceEmbedder  - main model (PyTorch, fine-tunable). Input: aligned 112x112 BGR.
ArcFaceEmbedder  - InsightFace buffalo_l w600k_r50 (ONNX) used as a model-choice baseline.
Both return L2-normalised 512-d embeddings plus a per-face "norm" which is a
quality proxy (AdaFace's feature norm correlates with image quality).
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F


def import_adaface(repo_dir):
    repo_dir = os.path.abspath(os.path.expanduser(repo_dir))
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    import net  # noqa
    import head  # noqa
    return net, head


def load_backbone(repo_dir, arch, ckpt_path, device="cpu"):
    """Load either an official AdaFace .ckpt or a checkpoint saved by our train.py."""
    net, _ = import_adaface(repo_dir)
    model = net.build_model(arch)
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "backbone" in ck:
            sd = ck["backbone"]
        else:
            sd = ck.get("state_dict", ck)
            stripped = {k[6:]: v for k, v in sd.items() if k.startswith("model.")}
            sd = stripped or sd
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model.to(device).eval()


def faces_to_tensor(faces_bgr):
    """List of 112x112x3 uint8 BGR -> float tensor N,3,112,112 normalised to [-1,1] (AdaFace convention)."""
    arr = np.stack(faces_bgr).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr.transpose(0, 3, 1, 2).copy())


class AdaFaceEmbedder:
    name = "adaface"

    def __init__(self, repo_dir, ckpt_path, arch="ir_101", device=None, batch_size=64, model=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model if model is not None else load_backbone(repo_dir, arch, ckpt_path, self.device)
        self.model.eval()
        self.batch_size = batch_size

    @torch.no_grad()
    def embed(self, faces_bgr, flip_tta=True):
        if len(faces_bgr) == 0:
            return np.zeros((0, 512), np.float32), np.zeros((0,), np.float32)
        embs, norms = [], []
        for i in range(0, len(faces_bgr), self.batch_size):
            x = faces_to_tensor(faces_bgr[i:i + self.batch_size]).to(self.device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                f, n = self.model(x)
                if flip_tta:
                    f2, n2 = self.model(torch.flip(x, dims=[3]))
            f, n = f.float(), n.float()
            if flip_tta:
                f2, n2 = f2.float(), n2.float()
                # norm-weighted fusion: the sharper view counts more
                f = f * n + f2 * n2
                n = (n + n2) / 2
            embs.append(F.normalize(f, dim=1).cpu().numpy())
            norms.append(n.view(-1).cpu().numpy())
        return np.concatenate(embs).astype(np.float32), np.concatenate(norms).astype(np.float32)


class ArcFaceEmbedder:
    """InsightFace ArcFace R50 (WebFace600K) via ONNX, for the model-choice comparison."""
    name = "arcface_r50"

    def __init__(self, model_path="~/.insightface/models/buffalo_l/w600k_r50.onnx"):
        import insightface
        from .detect_align import onnx_providers
        providers = onnx_providers()
        self.model = insightface.model_zoo.get_model(os.path.expanduser(model_path), providers=providers)
        self.model.prepare(ctx_id=0 if "CUDAExecutionProvider" in providers else -1)

    def _feat(self, faces):
        out = []
        for i in range(0, len(faces), 32):
            chunk = list(faces[i:i + 32])
            try:
                out.append(np.asarray(self.model.get_feat(chunk)).reshape(len(chunk), -1))
            except Exception:  # model exported with fixed batch size 1
                out.append(np.concatenate([np.asarray(self.model.get_feat([c])).reshape(1, -1) for c in chunk]))
        return np.concatenate(out)

    def embed(self, faces_bgr, flip_tta=True):
        if len(faces_bgr) == 0:
            return np.zeros((0, 512), np.float32), np.zeros((0,), np.float32)
        f = self._feat(faces_bgr)
        if flip_tta:
            f = f + self._feat([np.ascontiguousarray(x[:, ::-1]) for x in faces_bgr])
        n = np.linalg.norm(f, axis=1)
        return (f / n[:, None]).astype(np.float32), n.astype(np.float32)

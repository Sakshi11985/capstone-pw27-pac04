"""Frozen image encoders. Each takes a float32 batch [N, 3, 224, 224] (already
normalised by clips.to_model_input) and returns L2-normalised features [N, D].

* clip_vitb16   : OpenAI CLIP ViT-B/16 via open_clip (main encoder, D = 512)
* dinov2_vitb14 : DINOv2 ViT-B/14 via torch.hub (ablation, D = 768)
* dummy         : fixed random projection of pooled pixels, CPU only, for tests
"""
from __future__ import annotations

import numpy as np

from .clips import CLIP_MEAN, CLIP_STD

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _l2(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class DummyEncoder:
    """Deterministic stand-in so the pipeline can be tested without a GPU."""
    dim = 64
    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, seed: int = 0):
        self.proj = np.random.default_rng(seed).standard_normal((3 * 8 * 8, self.dim)).astype(np.float32)

    def encode(self, batch: np.ndarray) -> np.ndarray:
        n = batch.shape[0]
        pooled = batch.reshape(n, 3, 8, 28, 8, 28).mean(axis=(3, 5)).reshape(n, -1)
        return _l2(pooled @ self.proj)


class _TorchEncoder:
    def __init__(self, model, device: str, fp16: bool, dim: int):
        import torch
        self.torch = torch
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        if self.device != device:
            print(f"[encoder] CUDA not available, falling back to CPU (slow)")
        self.fp16 = fp16 and self.device.startswith("cuda")
        self.model = model.to(self.device).eval()
        if self.fp16:
            self.model = self.model.half()
        self.dim = dim

    def _forward(self, x):
        raise NotImplementedError

    def encode(self, batch: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.from_numpy(batch).to(self.device)
            x = x.half() if self.fp16 else x.float()
            f = self._forward(x).float()
            f = f / f.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return f.cpu().numpy()


class ClipEncoder(_TorchEncoder):
    mean, std = CLIP_MEAN, CLIP_STD

    def __init__(self, device="cuda", fp16=True):
        import open_clip
        model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        super().__init__(model, device, fp16, dim=512)

    def _forward(self, x):
        return self.model.encode_image(x)


class DinoV2Encoder(_TorchEncoder):
    mean, std = IMAGENET_MEAN, IMAGENET_STD

    def __init__(self, device="cuda", fp16=True):
        import torch
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        super().__init__(model, device, fp16, dim=768)

    def _forward(self, x):
        return self.model(x)


def build_encoder(name: str, device: str = "cuda", fp16: bool = True):
    if name == "clip_vitb16":
        return ClipEncoder(device, fp16)
    if name == "dinov2_vitb14":
        return DinoV2Encoder(device, fp16)
    if name == "dummy":
        return DummyEncoder()
    raise ValueError(f"unknown encoder {name}")

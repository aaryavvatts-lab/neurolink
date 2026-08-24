"""Vision-side latents: Meta's DINOv2 (self-supervised ViT), optionally DINOv3."""
from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def embed_images(paths: list[str], model_name: str, image_size: int = 224,
                 device: str | None = None, batch_size: int = 16
                 ) -> dict[str, np.ndarray]:
    """Return CLS, mean-patch and their concatenation for each image.

    The stimuli are grayscale; they are replicated to 3 channels because the
    ViT expects RGB, which is the standard way these models are applied to
    single-channel images.
    """
    device = device or pick_device()
    proc = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    # Override the processor's default resize so we control the input resolution.
    proc.size = {"height": image_size, "width": image_size}
    if hasattr(proc, "crop_size"):
        proc.crop_size = {"height": image_size, "width": image_size}

    cls_out, mean_out = [], []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch_size]]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        out = model(pixel_values=px).last_hidden_state          # (B, 1+P, D)
        cls_out.append(out[:, 0].float().cpu().numpy())
        mean_out.append(out[:, 1:].mean(dim=1).float().cpu().numpy())

    cls = np.concatenate(cls_out).astype(np.float32)
    mean = np.concatenate(mean_out).astype(np.float32)
    return {"cls": cls, "mean_patch": mean,
            "concat": np.concatenate([cls, mean], axis=1).astype(np.float32)}

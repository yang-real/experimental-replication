"""Shared helpers for feature-map visualization and quantitative analysis."""

from __future__ import annotations

import glob
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def collect_pairs(data_root: str, domain: str) -> list[tuple[str, str]]:
    img_dir = os.path.join(data_root, "Val", domain, "images_png")
    mask_dir = os.path.join(data_root, "Val", domain, "masks_png")
    pairs: list[tuple[str, str]] = []

    for img_path in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        stem = Path(img_path).stem
        candidates = [
            stem,
            stem.replace("_RGB", ""),
            stem + "_label",
            stem.replace("_RGB", "") + "_label",
        ]
        for name in candidates:
            match = None
            for ext in [".png", ".tif", ".tiff"]:
                candidate = os.path.join(mask_dir, name + ext)
                if os.path.exists(candidate):
                    match = candidate
                    break
            if match:
                pairs.append((img_path, match))
                break
    return pairs


def sample_pairs(
    data_root: str,
    domain: str,
    count: int,
    seed: int = 42,
) -> list[tuple[str, str]]:
    pairs = collect_pairs(data_root, domain)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    return pairs[:count]


def load_rgb_mask(
    img_path: str,
    mask_path: str,
    cfg: dict,
    size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    image = Image.open(img_path).convert("RGB").resize(
        (size, size), Image.Resampling.BILINEAR
    )
    mask = Image.open(mask_path).resize(
        (size, size), Image.Resampling.NEAREST
    )

    rgb = np.asarray(image).astype(np.float32) / 255.0
    rgb = (rgb - np.asarray(cfg["rgb_mean"], dtype=np.float32)) / np.asarray(
        cfg["rgb_std"], dtype=np.float32
    )
    rgb_tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()

    mask_array = np.asarray(mask).astype(np.int64)
    mask_array = np.clip(mask_array, 0, 7) - 1
    mask_array[mask_array == -1] = cfg["ignore_index"]
    mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).long()
    return rgb_tensor, mask_tensor, np.asarray(image)


def normalize_map(value: torch.Tensor) -> np.ndarray:
    value = value.detach().float().cpu()
    low = torch.quantile(value.flatten(), 0.02)
    high = torch.quantile(value.flatten(), 0.98)
    value = ((value - low) / (high - low + 1e-8)).clamp(0, 1)
    return value.numpy()


def feature_pca(feature: torch.Tensor) -> np.ndarray:
    if feature.ndim == 4:
        feature = feature[0]
    feature = feature.detach().float()
    channels, height, width = feature.shape
    pixels = feature.permute(1, 2, 0).reshape(height * width, channels)
    pixels = pixels - pixels.mean(dim=0, keepdim=True)
    u, s, _ = torch.linalg.svd(pixels, full_matrices=False)
    image = (u[:, :3] * s[:3]).reshape(height, width, 3).cpu().numpy()
    for index in range(3):
        image[..., index] = normalize_map(torch.from_numpy(image[..., index]))
    return image


def region_stats(
    feature: torch.Tensor,
    boundary: torch.Tensor,
    interior: torch.Tensor,
) -> dict[str, float] | None:
    if feature.ndim == 4:
        feature = feature[0]
    feature = feature.detach().float()
    channels, height, width = feature.shape

    boundary_small = F.interpolate(
        boundary.float().unsqueeze(1),
        size=(height, width),
        mode="nearest",
    )[0, 0] > 0.5
    interior_small = F.interpolate(
        interior.float().unsqueeze(1),
        size=(height, width),
        mode="nearest",
    )[0, 0] > 0.5

    flat = feature.permute(1, 2, 0).reshape(height * width, channels)
    boundary_features = flat[boundary_small.reshape(-1)]
    interior_features = flat[interior_small.reshape(-1)]
    if boundary_features.numel() == 0 or interior_features.numel() == 0:
        return None

    boundary_mean = boundary_features.mean(dim=0)
    interior_mean = interior_features.mean(dim=0)
    return {
        "boundary_mean_norm": boundary_features.norm(dim=1).mean().item(),
        "interior_mean_norm": interior_features.norm(dim=1).mean().item(),
        "boundary_interior_cosine": F.cosine_similarity(
            boundary_mean,
            interior_mean,
            dim=0,
        ).item(),
    }


def load_model(
    cfg: dict,
    checkpoint: str,
    device: torch.device,
):
    from core.model import SegmentationModel
    from core.utils import load_checkpoint

    cfg["encoder_kwargs"]["pretrained"] = False
    model = SegmentationModel(cfg).to(device)
    load_checkpoint(checkpoint, model, device=device, strict=False)
    model.eval()
    return model

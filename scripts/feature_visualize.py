#!/usr/bin/env python3
"""Create F1-F4 PCA visualizations for one checkpoint and dataset domain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.boundary_analysis import _compute_boundary_mask
from core.config_loader import load_config
from train import _resolve_weights

from feature_utils import feature_pca, load_model, load_rgb_mask, region_stats, sample_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--domain", choices=["Urban", "Rural"], default="Urban")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _resolve_weights(load_config(args.config))
    cfg["data_root"] = args.data_root
    model = load_model(cfg, args.checkpoint, device)

    pair = sample_pairs(args.data_root, args.domain, 1, args.seed)[0]
    rgb, mask, display = load_rgb_mask(pair[0], pair[1], cfg)
    rgb = rgb.to(device)
    mask = mask.to(device)

    with torch.inference_mode(), autocast(enabled=cfg.get("use_amp", True)):
        features = model.rgb_encoder(rgb)

    boundary = _compute_boundary_mask(mask, cfg["ignore_index"], radius=2)
    interior = (mask != cfg["ignore_index"]) & (~boundary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 5, figsize=(20, 5))
    axes[0].imshow(display)
    axes[0].set_title("Input")
    axes[0].axis("off")

    stats = {}
    for index, feature in enumerate(features, start=1):
        axes[index].imshow(feature_pca(feature))
        axes[index].set_title(f"F{index} PCA")
        axes[index].axis("off")
        stats[f"F{index}"] = region_stats(feature, boundary, interior)

    figure.suptitle(f"{args.name} - {args.domain}", fontsize=16)
    figure.tight_layout()
    image_path = output_dir / f"{args.name}_{args.domain.lower()}_features.png"
    figure.savefig(image_path, dpi=250, bbox_inches="tight")
    plt.close(figure)

    metrics = {
        "model": args.name,
        "domain": args.domain,
        "image": pair[0],
        "mask": pair[1],
        "features": stats,
    }
    metrics_path = output_dir / f"{args.name}_{args.domain.lower()}_statistics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", image_path)
    print("Saved:", metrics_path)


if __name__ == "__main__":
    main()

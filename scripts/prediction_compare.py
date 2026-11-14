#!/usr/bin/env python3
"""Generate prediction/error comparison sheets for two checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from feature_utils import load_model, load_rgb_mask, sample_pairs


PALETTE = np.asarray(
    [
        [128, 64, 128],
        [0, 0, 255],
        [0, 255, 255],
        [0, 128, 255],
        [166, 120, 50],
        [0, 160, 0],
        [255, 180, 0],
    ],
    dtype=np.uint8,
)


def colorize(mask: torch.Tensor, ignore_index: int) -> np.ndarray:
    values = mask.detach().cpu().numpy().astype(np.int64)
    output = np.zeros((*values.shape, 3), dtype=np.uint8)
    valid = values != ignore_index
    output[valid] = PALETTE[values[valid]]
    output[~valid] = 30
    return output


def error_map(prediction: np.ndarray, target: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    correct = prediction == target
    output = np.zeros((*correct.shape, 3), dtype=np.uint8)
    output[~valid] = 20
    output[valid & correct] = [70, 170, 70]
    output[valid & ~correct] = [220, 50, 50]
    return output


def ab_map(pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray, ignore_index: int) -> np.ndarray:
    valid = target != ignore_index
    a_ok = pred_a == target
    b_ok = pred_b == target
    output = np.zeros((*target.shape, 3), dtype=np.uint8)
    output[~valid] = 20
    output[valid & a_ok & b_ok] = [70, 170, 70]
    output[valid & a_ok & ~b_ok] = [40, 100, 255]
    output[valid & ~a_ok & b_ok] = [255, 150, 30]
    output[valid & ~a_ok & ~b_ok] = [220, 50, 50]
    return output


def parse_checkpoints(values: list[str]) -> dict[str, str]:
    checkpoints = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected name=path, got {value!r}")
        checkpoints[name] = path
    if len(checkpoints) != 2:
        raise ValueError("Exactly two checkpoints are required")
    return checkpoints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore-index", type=int, default=255)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    names = list(checkpoints)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    from core.config_loader import load_config
    from train import _resolve_weights

    cfg = _resolve_weights(load_config(args.config))
    cfg["data_root"] = args.data_root

    models = {
        name: load_model(cfg, path, device)
        for name, path in checkpoints.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for domain in ["Urban", "Rural"]:
        pairs = sample_pairs(
            args.data_root,
            domain,
            count=args.samples_per_domain,
            seed=args.seed,
        )
        figure, axes = plt.subplots(
            len(pairs),
            5,
            figsize=(23, 4.3 * len(pairs)),
            squeeze=False,
        )

        for row_index, (img_path, mask_path) in enumerate(pairs):
            rgb, mask, display = load_rgb_mask(img_path, mask_path, cfg)
            rgb = rgb.to(device)
            mask = mask.to(device)

            predictions = {}
            for name, model in models.items():
                with torch.inference_mode(), autocast(enabled=cfg.get("use_amp", True)):
                    outputs = model(rgb)
                    logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

                if logits.shape[-2:] != mask.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=mask.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                predictions[name] = logits.argmax(dim=1)

            target = mask[0].detach().cpu().numpy()
            pred_a = predictions[names[0]][0].detach().cpu().numpy()
            pred_b = predictions[names[1]][0].detach().cpu().numpy()

            valid = target != args.ignore_index
            valid_count = max(int(valid.sum()), 1)
            a_ok = (pred_a == target) & valid
            b_ok = (pred_b == target) & valid

            rows.append(
                {
                    "domain": domain,
                    "image": img_path,
                    "checkpoint_a": names[0],
                    "checkpoint_b": names[1],
                    "accuracy_a": float(a_ok.sum() / valid_count),
                    "accuracy_b": float(b_ok.sum() / valid_count),
                    "disagreement": float(
                        ((pred_a != pred_b) & valid).sum() / valid_count
                    ),
                    "a_only_correct": float((a_ok & (~b_ok)).sum() / valid_count),
                    "b_only_correct": float((b_ok & (~a_ok)).sum() / valid_count),
                    "both_wrong": float(
                        (valid & (~a_ok) & (~b_ok)).sum() / valid_count
                    ),
                }
            )

            images = [
                display,
                colorize(mask[0], args.ignore_index),
                colorize(predictions[names[0]][0], args.ignore_index),
                colorize(predictions[names[1]][0], args.ignore_index),
                ab_map(pred_a, pred_b, target, args.ignore_index),
            ]
            titles = [
                "Input",
                "Ground Truth",
                names[0],
                names[1],
                "A/B Comparison",
            ]

            for column, (image, title) in enumerate(zip(images, titles)):
                axes[row_index, column].imshow(image)
                axes[row_index, column].axis("off")
                if row_index == 0:
                    axes[row_index, column].set_title(title, fontsize=13)

        figure.suptitle(f"Prediction comparison - {domain}", fontsize=18)
        figure.tight_layout()
        figure.savefig(
            output_dir / f"prediction_comparison_{domain.lower()}.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(figure)

    with (output_dir / "prediction_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "prediction_metrics.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))
    print("Saved:", output_dir)


if __name__ == "__main__":
    main()

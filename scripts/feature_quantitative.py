#!/usr/bin/env python3
"""Compute Boundary/Interior feature similarity for one or more checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.boundary_analysis import _compute_boundary_mask
from core.config_loader import load_config
from train import _resolve_weights

from feature_utils import load_model, load_rgb_mask, region_stats, sample_pairs


class PairDataset(torch.utils.data.Dataset):
    def __init__(self, pairs, cfg):
        self.pairs = pairs
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        rgb, mask, _ = load_rgb_mask(pair[0], pair[1], self.cfg)
        return rgb[0], mask[0]


def parse_checkpoints(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        name, separator, path = value.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"Expected name=path, got {value!r}")
        result[name] = path
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoints = parse_checkpoints(args.checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = _resolve_weights(load_config(args.config))
    cfg["data_root"] = args.data_root

    samples = {
        domain.lower(): sample_pairs(
            args.data_root,
            domain,
            args.samples_per_domain,
            args.seed,
        )
        for domain in ["Urban", "Rural"]
    }
    results: dict[str, dict] = {}

    for model_name, checkpoint in checkpoints.items():
        model = load_model(cfg, checkpoint, device)
        results[model_name] = {}

        for domain_name, pairs in samples.items():
            loader = DataLoader(
                PairDataset(pairs, cfg),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device.type == "cuda",
                persistent_workers=args.num_workers > 0,
            )
            values = {
                f"F{index}": {"cosine": [], "boundary_norm": [], "interior_norm": []}
                for index in range(1, 5)
            }

            with torch.inference_mode():
                for rgb, mask in loader:
                    rgb = rgb.to(device, non_blocking=True)
                    mask = mask.to(device, non_blocking=True)
                    with autocast(enabled=cfg.get("use_amp", True)):
                        features = model.rgb_encoder(rgb)

                    boundary = _compute_boundary_mask(
                        mask,
                        cfg["ignore_index"],
                        radius=2,
                    )
                    interior = (mask != cfg["ignore_index"]) & (~boundary)

                    for index, feature in enumerate(features, start=1):
                        stats = region_stats(feature, boundary, interior)
                        if stats is not None:
                            values[f"F{index}"]["cosine"].append(
                                stats["boundary_interior_cosine"]
                            )

            domain_result = {}
            for key, value in values.items():
                cosine = np.asarray(value["cosine"], dtype=np.float64)
                domain_result[key] = {
                    "n": int(len(cosine)),
                    "cosine_mean": float(cosine.mean()),
                    "cosine_std": float(cosine.std()),
                }
            results[model_name][domain_name] = domain_result

        del model
        torch.cuda.empty_cache()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Boundary and Interior mIoU for a Mamba-Segmentation checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.boundary_analysis import _compute_boundary_mask
from core.config_loader import load_config
from core.dataset import LovedaDataset
from core.model import SegmentationModel
from core.utils import SegmentationEvaluator, load_checkpoint
from train import _resolve_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--radius", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _resolve_weights(load_config(args.config))
    cfg["data_root"] = args.data_root
    cfg["encoder_kwargs"]["pretrained"] = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SegmentationModel(cfg).to(device)
    load_checkpoint(args.checkpoint, model, device=device, strict=False)
    model.eval()

    img_dirs = [os.path.join(args.data_root, p) for p in cfg["val_img_dirs"]]
    mask_dirs = [os.path.join(args.data_root, p) for p in cfg["val_mask_dirs"]]
    dataset = LovedaDataset(
        img_dirs=img_dirs,
        mask_dirs=mask_dirs,
        crop_size=cfg["crop_size"],
        is_train=False,
        rgb_mean=cfg["rgb_mean"],
        rgb_std=cfg["rgb_std"],
        ignore_index=cfg["ignore_index"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    boundary_eval = SegmentationEvaluator(cfg["num_classes"], device)
    interior_eval = SegmentationEvaluator(cfg["num_classes"], device)
    boundary_eval.reset()
    interior_eval.reset()

    with torch.inference_mode():
        for batch in loader:
            rgb = batch["rgb"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            with autocast(enabled=cfg.get("use_amp", True)):
                outputs = model(rgb)
                logits = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            if logits.shape[-2:] != mask.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=mask.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            pred = logits.argmax(dim=1)
            boundary = _compute_boundary_mask(
                mask,
                ignore_index=cfg["ignore_index"],
                radius=args.radius,
            )
            interior = (mask != cfg["ignore_index"]) & (~boundary)

            boundary_target = mask.clone()
            boundary_target[~boundary] = cfg["ignore_index"]
            boundary_eval.update(pred, boundary_target)

            interior_target = mask.clone()
            interior_target[~interior] = cfg["ignore_index"]
            interior_eval.update(pred, interior_target)

    boundary_metrics = boundary_eval.get_metrics()
    interior_metrics = interior_eval.get_metrics()
    result = {
        "checkpoint": args.checkpoint,
        "boundary_radius": args.radius,
        "samples": len(dataset),
        "boundary_mIoU": boundary_metrics["mIoU"] * 100,
        "boundary_mF1": boundary_metrics["mF1"] * 100,
        "boundary_OA": boundary_metrics["OA"] * 100,
        "interior_mIoU": interior_metrics["mIoU"] * 100,
        "interior_mF1": interior_metrics["mF1"] * 100,
        "interior_OA": interior_metrics["OA"] * 100,
        "gap": (interior_metrics["mIoU"] - boundary_metrics["mIoU"]) * 100,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

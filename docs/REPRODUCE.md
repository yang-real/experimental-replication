# Reproduction Guide

## 1. Get the official code

```bash
git clone https://github.com/Dineth14/Mamba-Segmentation.git
cd Mamba-Segmentation
git reset --hard 1139642b8dae2ab416dc50995409c1814b9951c4
```

Apply `patches/upstream_compatibility.patch`:

```bash
git apply ../patches/upstream_compatibility.patch
```

The patch adds:

- FP16-compatible Boundary Loss kernel handling;
- correct LoveDA validation paths in `eval_domain.py`;
- class-name propagation during evaluation;
- deterministic `TRAIN_SEED` initialization in `train.py`.

Copy the analysis scripts into the official checkout:

```bash
mkdir -p reproduction_scripts
cp ../scripts/*.py reproduction_scripts/
```

## 2. Prepare data

Set the LoveDA dataset location:

```bash
export LOVEDA_ROOT=/path/to/LoveDA
```

The expected layout is:

```text
LoveDA/
├── Train/
│   ├── Urban/images_png/
│   ├── Urban/masks_png/
│   ├── Rural/images_png/
│   └── Rural/masks_png/
└── Val/
    ├── Urban/images_png/
    ├── Urban/masks_png/
    ├── Rural/images_png/
    └── Rural/masks_png/
```

If the dataset uses an extra outer `Train/Train` or `Val/Val` layout, adjust
`configs/base.yaml` accordingly.

## 3. Prepare checkpoints

Official checkpoints are available at:

https://huggingface.co/dineth18/Mamba-Segmentation

Do not commit `.pth` files to this repository. The detailed model paths and
metrics are listed in `reports/`.

## 4. Reproduce official-checkpoint evaluation

```bash
export NO_ALBUMENTATIONS_UPDATE=1

python -u eval_domain.py \
  --config configs/vmamba.yaml \
  --ckpt /path/to/vssmsmall_best.pth \
  --domain all \
  --split val \
  --batch 4 \
  --profile 0 \
  --out_json results/vmamba_small_all.json
```

Use `--domain urban` or `--domain rural` for domain-specific evaluation.

## 5. Reproduce the Boundary Loss ablation

The official model is not the ablation baseline because it uses a different
training budget. Train both arms from the same ImageNet initialization:

```bash
export TRAIN_SEED=42
export PYTHONHASHSEED=42
export LOVEDA_ROOT=/path/to/LoveDA
export NO_ALBUMENTATIONS_UPDATE=1
```

Full TriBraid loss:

```bash
python -u train.py --config configs/vmamba.yaml \
  data_root="$LOVEDA_ROOT" \
  output_dir=/path/to/output/ablation15k_full_s42 \
  max_iters=15000 \
  val_interval=7500 \
  log_interval=250 \
  boundary_warmup_iters=2000 \
  boundary_weight=0.5 \
  batch_size=4
```

No Boundary Loss:

```bash
python -u train.py --config configs/vmamba.yaml \
  data_root="$LOVEDA_ROOT" \
  output_dir=/path/to/output/ablation15k_noboundary_s42 \
  max_iters=15000 \
  val_interval=7500 \
  log_interval=250 \
  boundary_warmup_iters=2000 \
  boundary_weight=0 \
  batch_size=4
```

Only `boundary_weight` and the output directory differ.

## 6. Analyze checkpoints

```bash
python reproduction_scripts/boundary_eval.py \
  --config configs/vmamba.yaml \
  --checkpoint /path/to/best.pth \
  --data-root "$LOVEDA_ROOT" \
  --output results/boundary_metrics.json
```

```bash
python reproduction_scripts/feature_visualize.py \
  --config configs/vmamba.yaml \
  --checkpoint /path/to/best.pth \
  --data-root "$LOVEDA_ROOT" \
  --output-dir figures/features \
  --name full_loss
```

```bash
python reproduction_scripts/feature_quantitative.py \
  --config configs/vmamba.yaml \
  --checkpoint /path/to/best.pth \
  --data-root "$LOVEDA_ROOT" \
  --output results/feature_quantitative.json \
  --samples-per-domain 50
```

## 7. Scope

The reported 15k ablation is exploratory and uses one seed. It is not a
replacement for the paper's 50k multi-seed training protocol.

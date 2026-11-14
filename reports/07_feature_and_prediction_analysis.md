# Feature and Prediction Analysis

## Scope

This analysis uses only forward inference. No additional models were trained.

## Cross-backbone feature visualization

All five checkpoints use the same images:

- Urban: `4076.png`
- Rural: `3378.png`

The extracted features are F1-F4 at strides 4, 8, 16 and 32. Each feature map
is projected to three principal components for visualization.

Models:

- VMamba-Small
- MambaVision-Base
- Spatial-Mamba-Base
- CNN / DeepLabv3 / ResNet-50
- Published transformer-labelled checkpoint

The transformer-labelled configuration actually uses a ResNet-18 CNN encoder,
as documented in `reports/04_transformer_unetformer_resnet18.md`.

## VMamba A/B prediction comparison

The two VMamba-Small checkpoints are:

- Full TriBraid Loss
- No Boundary Loss

Three fixed random samples per domain are shown.

| Domain | Image | Full accuracy | No-boundary accuracy | Difference |
|---|---|---:|---:|---:|
| Urban | 4076 | 80.78% | 77.52% | +3.26 |
| Urban | 3911 | 76.78% | 76.91% | -0.13 |
| Urban | 3663 | 86.28% | 87.26% | -0.98 |
| Rural | 3378 | 11.76% | 63.48% | -51.72 |
| Rural | 2635 | 70.01% | 71.52% | -1.51 |
| Rural | 2578 | 73.42% | 46.46% | +26.97 |

Difference is full-loss accuracy minus no-boundary accuracy. Values are in
percentage points.

The multi-sample results show substantial image-level variability:

- the full model has clear wins on some images;
- the no-boundary model has clear wins on others;
- one Rural image is a severe failure case for the full model.

On the complete validation set, however, the full model has the higher mean
score:

- Full mIoU: 54.30
- No-boundary mIoU: 53.84

Therefore, a single qualitative image must not be used as evidence for the
average effect. The multi-image panel and full validation metrics should be
reported together.

## Colour legend

The A/B comparison panel uses:

- green: both models correct
- blue: only full TriBraid correct
- orange: only no-boundary correct
- red: both models wrong
- black: ignored pixels

## Limitations

- One random seed for the ablation.
- 15,000 training iterations rather than the paper's 50,000.
- Feature PCA colours are relative coordinates and should not be interpreted as
  semantic class colours.
- The cross-backbone transformer entry is the published ResNet-18
  configuration, not a true UNetFormer transformer.

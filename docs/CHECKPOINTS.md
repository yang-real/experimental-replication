# Checkpoint Manifest

Checkpoints are not stored in this GitHub repository. The official segmentation
weights are published at:

https://huggingface.co/dineth18/Mamba-Segmentation

## Used in this reproduction

| Family | Experiment | Official path |
|---|---|---|
| VMamba-Small | All→All | `Comparison_Experiments/Vmamb_small_512/checkpoints/best.pth` |
| VMamba-Small | Urban→Rural | `Comparison_Experiments/vmamba_small_urbantrain_512/checkpoints/best.pth` |
| VMamba-Small | Rural→Urban | `Comparison_Experiments/vmamba_small_ruraltrain_512/checkpoints/best.pth` |
| CNN | All→All | `Comparison_Experiments/cnn_deeplabv3p_r50_512/checkpoints/best.pth` |
| CNN | Urban→Rural | `Comparison_Experiments/cnn_deeplabv3p_resnet50_urbantrain_512/checkpoints/best.pth` |
| CNN | Rural→Urban | `Comparison_Experiments/cnn_deeplabv3p_resnet50_ruraltrain_512/checkpoints/best.pth` |
| MambaVision-Base | All→All | `Comparison_Experiments/mambavision_base_512/checkpoints/best.pth` |
| MambaVision-Base | Urban→Rural | `Comparison_Experiments/mambavision_base_urbantrain_512/checkpoints/best.pth` |
| MambaVision-Base | Rural→Urban | `Comparison_Experiments/mambavision_base_ruraltrain_512/checkpoints/best.pth` |
| Transformer-labeled | All→All | `Comparison_Experiments/transformer_unetformer_r18_512/checkpoints/best.pth` |
| Transformer-labeled | Urban→Rural | `Comparison_Experiments/transformerunetformer_resnet18_urbantrain_512/checkpoints/best.pth` |
| Transformer-labeled | Rural→Urban | `Comparison_Experiments/transformerunetformer_resnet18_ruraltrain_512/checkpoints/best.pth` |
| Spatial-Mamba-Base | All→All | `Comparison_Experiments/spatialmamba_base_512/checkpoints/best.pth` |
| Spatial-Mamba-Base | Urban→Rural | `Comparison_Experiments/spatialmamba_base_urbantrain_512/checkpoints/best.pth` |
| Spatial-Mamba-Base | Rural→Urban | `Comparison_Experiments/spatialmamba_base_ruraltrain_512/checkpoints/best.pth` |

## Locally trained ablation

The VMamba-Small ablation checkpoints were produced by this study and are not
distributed in the GitHub repository. Their metrics and training logs are stored
under `evidence/ablation_15k/`.

## Adding hashes

For every external checkpoint used in a reported result, record:

```text
sha256sum best.pth
```

The hash, file size, best iteration, configuration and evaluation value should
be stored together.

# Environment

The experiments were evaluated with:

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 24 GB |
| Python | 3.10.8 |
| PyTorch | 2.1.2+cu118 |
| torchvision | 0.16.2+cu118 |
| CUDA toolkit | 11.8 |
| timm | 0.4.12 |
| mamba_ssm | 1.1.1 |
| Cython | 0.29.37 |
| setuptools | 69.5.1 |

## Base environment

```bash
python -m venv --system-site-packages /root/autodl-tmp/venvs/base-repro
source /root/autodl-tmp/venvs/base-repro/bin/activate
python -m pip install "setuptools==69.5.1"
python -m pip install -r requirements.txt
```

## MambaVision environment

MambaVision's package metadata requests `mamba-ssm==2.2.4`. In this reproduction,
the published checkpoints ran successfully with `mamba-ssm==1.1.1`, so the
package was installed without forcing its dependency set:

```bash
python -m venv --system-site-packages /root/autodl-tmp/venvs/mambavision
source /root/autodl-tmp/venvs/mambavision/bin/activate
python -m pip install "setuptools==69.5.1"
python -m pip install --no-deps -e MambaVision/MambaVision
python -m pip install -r requirements-mambavision.txt
```

## Spatial-Mamba CUDA extensions

Two custom extensions are required:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8

cd spatial-mamba/Spatial-Mamba/kernels/selective_scan
python -m pip install . --no-build-isolation

cd ../dwconv2d
python -m pip install . --no-build-isolation
```

The PyTorch 2.1 compatibility adjustment for `dwconv2d` removes
`at::NoGradGuard`, which is no longer available in this installation. Inference
is executed inside `torch.inference_mode()`.

The selective-scan build is limited to SM89 for the RTX 4090.

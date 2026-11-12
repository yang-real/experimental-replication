# Spatial-Mamba PyTorch 2.1 compatibility changes

These changes are applied inside the external Spatial-Mamba checkout. They are
not part of the official Mamba-Segmentation repository.

## 1. selective_scan: compile only SM89

File:

`spatial-mamba/Spatial-Mamba/kernels/selective_scan/setup.py`

Replace the hard-coded SM70/SM80/SM90 architecture block with:

```python
cc_flag.append("-gencode")
cc_flag.append("arch=compute_89,code=sm_89")
```

This is sufficient for the RTX 4090 and avoids compiling unused architectures.

## 2. dwconv2d: remove obsolete NoGradGuard

File:

`spatial-mamba/Spatial-Mamba/kernels/dwconv2d/depthwise_fwd/launch.cu`

Remove lines containing:

```cpp
at::NoGradGuard
```

The official model evaluation runs inside `torch.inference_mode()`, so these
C++ guards are not required for inference. PyTorch 2.1.2 does not expose the
older `at::NoGradGuard` name used by this kernel.

## 3. Build

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

Before importing either extension directly, import PyTorch:

```bash
python -c "import torch; import selective_scan_cuda_oflex_rh; print('OK')"
```

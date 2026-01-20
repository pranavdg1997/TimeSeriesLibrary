# AutoGluon GPU Setup Guide

## Problem
AutoGluon TimeSeries is not detecting your NVIDIA GPU.

## Solution

I've made changes to the `universal_ts` library to:
1. **Auto-detect GPU availability** when creating an AutoGluon model
2. **Automatically configure** AutoGluon to use GPU if available
3. **Provide clear feedback** about GPU status

## Changes Made

### Modified: `universal_ts/backends/autogluon_backend.py`

Added:
- `num_gpus` parameter to `AutoGluonBackend.__init__()`
- `_detect_gpu()` method that checks PyTorch and MXNet for GPU availability
- Automatic GPU configuration when initializing TimeSeriesPredictor
- Status messages showing whether GPU will be used

## How to Use

### Option 1: Auto-detect GPU (Recommended)
```python
from universal_ts import UniversalForecaster

# GPU will be auto-detected
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=10
)
# Output: [autogluon] Use 1 GPU(s)
# OR:     [autogluon] No GPU detected. Using CPU only.
```

### Option 2: Explicitly specify GPU count
```python
# Force use of 1 GPU
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=10,
    num_gpus=1
)

# Force CPU-only (useful for debugging)
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=10,
    num_gpus=0
)
```

## Diagnostic Steps

### Step 1: Run GPU Diagnostic
```python
%run check_gpu.py
```

This will check:
- PyTorch CUDA support
- MXNet GPU support
- NVIDIA drivers (nvidia-smi)
- Provide installation recommendations

### Step 2: Test AutoGluon with GPU
```python
%run test_autogluon_gpu.py
```

This will:
- Run the GPU diagnostic
- Create a test AutoGluon model
- Show whether GPU is being used
- Verify forecasting works

## Common Issues & Fixes

### Issue 1: PyTorch doesn't have CUDA support
**Symptom:** `check_gpu.py` shows "CUDA not available in PyTorch"

**Fix:** Install CUDA-enabled PyTorch
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Check your CUDA version first:
```bash
nvidia-smi  # Look for "CUDA Version: X.X"
```

### Issue 2: MXNet doesn't have GPU support
**Symptom:** `check_gpu.py` shows "MXNet cannot access GPU"

**Fix:** Install GPU-enabled MXNet
```bash
pip uninstall mxnet
pip install mxnet-cu112  # For CUDA 11.2
# Check https://mxnet.apache.org for your CUDA version
```

### Issue 3: NVIDIA drivers not installed
**Symptom:** `nvidia-smi` command not found

**Fix:** Install NVIDIA drivers from https://www.nvidia.com/Download/index.aspx

### Issue 4: CUDA version mismatch
**Symptom:** PyTorch installed but CUDA not available

**Fix:** Ensure PyTorch CUDA version matches your system CUDA version
```bash
# Check system CUDA
nvidia-smi

# Reinstall PyTorch with matching CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<VERSION>
```

### Issue 5: RTX 50-series (Blackwell) Compatibility
**Symptom:** `UserWarning: NVIDIA GeForce RTX 5080 with CUDA capability sm_120 is not compatible with the current PyTorch installation.`

**Reason:** The RTX 50-series uses a new architecture (`sm_120`) that requires CUDA 12.x and PyTorch 2.x compiled specifically for it. Older PyTorch builds (like those for CUDA 11.8) will not work.

**Fix:** Update to the latest PyTorch with CUDA 12.4 support (or newer):
```bash
# Uninstall old torch version
pip uninstall torch torchvision torchaudio

# Install latest stable or nightly with CUDA 12.4 support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

---

## After Installing GPU Packages

1. **Restart Jupyter kernel** (Kernel → Restart Kernel)
2. **Re-run diagnostic:** `%run check_gpu.py`
3. **Test AutoGluon:** `%run test_autogluon_gpu.py`

## Verification

When GPU is properly configured, you should see:
```
[autogluon] Use 1 GPU(s)
```

When creating your model. AutoGluon will then use GPU for training deep learning models like DeepAR, Transformer, etc.

## Notes

- Not all AutoGluon models use GPU (e.g., statistical models like ETS, ARIMA run on CPU)
- GPU acceleration is most beneficial for deep learning models and large datasets
- You can check which models were trained by looking at the leaderboard: `model.backend.predictor.leaderboard()`

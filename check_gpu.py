"""
GPU Detection Diagnostic Script for AutoGluon TimeSeries
Run this to diagnose why GPU is not being detected
"""

import sys
print("=" * 60)
print("GPU DETECTION DIAGNOSTIC")
print("=" * 60)

# 1. Check PyTorch and CUDA
print("\n1. PyTorch CUDA Support:")
try:
    import torch
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("   [WARNING] CUDA not available in PyTorch!")
        print("   -> You may have CPU-only PyTorch installed")
except ImportError:
    print("   [ERROR] PyTorch not installed")

# 2. Check MXNet
print("\n2. MXNet GPU Support:")
try:
    import mxnet as mx
    print(f"   MXNet version: {mx.__version__}")
    try:
        gpu_count = mx.context.num_gpus()
        print(f"   GPU count: {gpu_count}")
        if gpu_count > 0:
            print(f"   [SUCCESS] MXNet can access GPU")
        else:
            print("   [WARNING] MXNet cannot access GPU")
    except:
        print("   [WARNING] Could not query GPU from MXNet")
except ImportError:
    print("   [ERROR] MXNet not installed")

# 3. Check AutoGluon
print("\n3. AutoGluon TimeSeries:")
try:
    import autogluon.timeseries as ats
    print(f"   AutoGluon TimeSeries version: {ats.__version__}")
    print("   [SUCCESS] AutoGluon TimeSeries installed")
except ImportError:
    print("   [ERROR] AutoGluon TimeSeries not installed")

# 4. Check NVIDIA drivers
print("\n4. NVIDIA Driver Check:")
import subprocess
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        print("   [SUCCESS] nvidia-smi found")
        # Extract GPU info
        lines = result.stdout.split('\n')
        for line in lines:
            if 'NVIDIA' in line or 'CUDA Version' in line:
                print(f"   {line.strip()}")
    else:
        print("   [ERROR] nvidia-smi failed")
except FileNotFoundError:
    print("   [ERROR] nvidia-smi not found - NVIDIA drivers may not be installed")
except Exception as e:
    print(f"   [ERROR] Error running nvidia-smi: {e}")

# 5. Recommendations
print("\n" + "=" * 60)
print("RECOMMENDATIONS:")
print("=" * 60)

try:
    import torch
    if not torch.cuda.is_available():
        print("\n[INFO] Install CUDA-enabled PyTorch:")
        print("   Visit: https://pytorch.org/get-started/locally/")
        print("   Example for CUDA 11.8:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\n   Example for CUDA 12.1:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
except:
    pass

try:
    import mxnet as mx
    if mx.context.num_gpus() == 0:
        print("\n[INFO] Install GPU-enabled MXNet:")
        print("   pip uninstall mxnet")
        print("   pip install mxnet-cu112  # For CUDA 11.2")
        print("   # Check https://mxnet.apache.org/versions/1.9.1/get_started for your CUDA version")
except:
    pass

print("\n[INFO] After installing GPU packages:")
print("   1. Restart your Jupyter kernel")
print("   2. Re-run this diagnostic script")
print("   3. Test AutoGluon with GPU")
print("=" * 60)

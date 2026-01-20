$ErrorActionPreference = "Stop"

$CondaExe = "C:\ProgramData\miniconda3\Scripts\conda.exe"

Write-Host "Creating new conda environment 'ts_gpu'..."
& $CondaExe create -n ts_gpu python=3.12 -y

Write-Host "Activating environment..."
# Note: We will use 'conda run' for subsequent commands.

Write-Host "Installing universal-ts and all backends (standard versions)..."
& $CondaExe run -n ts_gpu pip install -e ".[all]"

Write-Host "Uninstalling standard PyTorch (incompatible with RTX 5080)..."
& $CondaExe run -n ts_gpu pip uninstall -y torch torchvision torchaudio

Write-Host "Installing PyTorch Nightly with CUDA 12.9 support..."
# Attempting to install from nightly channel for CUDA 12.8 (fallback from 12.9)
& $CondaExe run -n ts_gpu pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

Write-Host "Verifying GPU detection..."
& $CondaExe run -n ts_gpu python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Arch List: {torch.cuda.get_arch_list()}')"

Write-Host "Setup complete! Activate the environment with: conda activate ts_gpu"

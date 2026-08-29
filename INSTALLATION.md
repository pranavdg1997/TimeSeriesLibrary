# Universal Time Series Library - Installation & Setup Guide

## Overview

The Universal Time Series Library is a unified Python library for time series forecasting that supports multiple backends (Prophet, AutoGluon, sktime, Darts) with GPU acceleration.

## Quick Start

### Option 1: Basic Installation (CPU only)
```bash
# Note: This library is not yet published to PyPI
# Install from source instead (see Development Setup below)
```

### Option 2: With Specific Backends (After PyPI Release)
```bash
# Prophet only
pip install universal-ts[prophet]

# AutoGluon (includes GPU support)
pip install universal-ts[autogluon]

# sktime
pip install universal-ts[sktime]

# Darts (includes deep learning models)
pip install universal-ts[darts]

# All backends
pip install universal-ts[all]
```

## Development Setup

### Prerequisites
- Python 3.8+
- Git
- **Note: This library is not yet published to PyPI. Install from source.**

### Option 1: Using Conda (Recommended for GPU support)

#### Step 1: Create Environment
```bash
# Create new conda environment
conda create -n universal-ts python=3.11 -y
conda activate universal-ts

# Or use the pre-configured GPU environment if available
conda activate ts_gpu
```

#### Step 2: Clone and Install
```bash
# Clone the repository
git clone https://github.com/yourusername/universal-ts.git
cd universal-ts

# Install in development mode (recommended)
pip install -e ".[dev]"

# Install with GPU support (recommended)
pip install -e ".[dev,all]"
```

#### Step 3: GPU Environment Setup (Optional)
If you have an NVIDIA GPU and want GPU acceleration:

```bash
# Run the GPU setup script
./scripts/setup_gpu_env.ps1  # PowerShell
# or
bash scripts/setup_gpu_env.sh  # Linux/Mac
```

### Option 2: Using pip (Quick setup)

```bash
# Clone repository
git clone https://github.com/yourusername/universal-ts.git
cd universal-ts

# Install in development mode
pip install -e ".[dev]"

# For GPU support, install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Option 3: Using Docker

#### Step 1: Create Dockerfile
Create `Dockerfile` in the repository root:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CUDA (for GPU support)
# CUDA Dockerfile example - see GPU section below

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY pyproject.toml .
RUN pip install -e ".[dev,all]"

# Copy source code
COPY . .

# Run tests
CMD ["pytest", "tests/"]
```

#### Step 2: Build and Run
```bash
# CPU-only version
docker build -t universal-ts .
docker run -it universal-ts

# GPU version (requires nvidia-docker)
docker build -f Dockerfile.gpu -t universal-ts-gpu .
docker run --gpus all -it universal-ts-gpu
```

### Option 4: Using uv (Modern Python Package Manager)

#### Step 1: Install uv
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or with pip
pip install uv
```

#### Step 2: Create Project Environment
```bash
# Clone repository
git clone https://github.com/yourusername/universal-ts.git
cd universal-ts

# Create virtual environment
uv venv

# Activate
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install in development mode
uv pip install -e ".[dev,all]"
```

## GPU Setup

### Prerequisites for GPU Support
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.8+ or 12.x
- NVIDIA drivers matching CUDA version

### Automatic GPU Setup

The library provides a script to automatically configure GPU environment:

#### Windows (PowerShell)
```powershell
.\scripts\setup_gpu_env.ps1
```

#### Linux/Mac (Bash)
```bash
bash scripts/setup_gpu_env.sh
```

### Manual GPU Setup

#### Step 1: Check System
```bash
# Check NVIDIA drivers
nvidia-smi

# Check CUDA version
nvidia-smi | grep "CUDA Version"
```

#### Step 2: Install PyTorch with CUDA
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4 (recommended for RTX 50-series)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

#### Step 3: Verify GPU Setup
```python
# Run GPU diagnostic
python scripts/check_gpu.py

# Or in Python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
```

### RTX 50-Series (Blackwell) Support

For RTX 5080/5090 GPUs:

```bash
# These cards require CUDA 12.x and latest PyTorch
pip uninstall torch torchvision torchaudio

# Install with CUDA 12.4+ support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Or use PyTorch nightly for latest support
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

## Environment Files

### requirements.txt (Traditional)
Create `requirements.txt`:
```
# Core dependencies
pandas>=1.3.0
numpy>=1.20.0
holidays>=0.20
matplotlib>=3.3.0

# Development dependencies
pytest>=7.0
pytest-cov>=3.0
black>=22.0
flake8>=4.0
isort>=5.0

# Backend dependencies (uncomment as needed)
# prophet>=1.1
# autogluon.timeseries>=0.8.0
# sktime>=0.20.0
# darts>=0.24.0
# torch>=1.13.0
```

### requirements-gpu.txt
Create `requirements-gpu.txt`:
```
# CPU requirements
-r requirements.txt

# GPU-specific
torch>=1.13.0+cu118 --find-links https://download.pytorch.org/whl/torch_stable.html
# or
torch>=1.13.0+cu121 --find-links https://download.pytorch.org/whl/torch_stable.html
```

### environment.yml (Conda)
Create `environment.yml`:
```yaml
name: universal-ts
channels:
  - conda-forge
  - pytorch
  - defaults
dependencies:
  - python=3.11
  - pip
  - numpy>=1.20.0
  - pandas>=1.3.0
  - matplotlib>=3.3.0
  - pytorch::pytorch>=1.13.0
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip:
    - universal-ts[all,dev]
    - prophet>=1.1
    - autogluon.timeseries>=0.8.0
    - sktime>=0.20.0
    - darts>=0.24.0
    - pytest>=7.0
    - pytest-cov>=3.0
    - black>=22.0
    - flake8>=4.0
    - isort>=5.0
```

## Testing Setup

### Run All Tests
```bash
# From project root
pytest tests/

# With coverage
pytest tests/ --cov=universal_ts --cov-report=html

# With verbose output
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Core functionality tests
pytest tests/universal_ts/test_core.py

# Backend tests
pytest tests/universal_ts/backends/

# Integration tests
pytest tests/test_integration.py

# GPU-specific tests
pytest tests/ -k gpu
```

### Development Workflow
```bash
# 1. Install pre-commit hooks (if configured)
pre-commit install

# 2. Run linting
black universal_ts/ tests/
isort universal_ts/ tests/
flake8 universal_ts/ tests/

# 3. Run tests
pytest tests/

# 4. Check coverage
pytest tests/ --cov=universal_ts --cov-report=term-missing
```

## IDE Setup

### VS Code
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true
    }
}
```

### PyCharm
1. Open the project directory
2. File → Settings → Project → Python Interpreter
3. Add new interpreter → Existing environment
4. Select `.venv/bin/python` or conda environment
5. Mark `tests/` as tests directory

## Troubleshooting

### Common Installation Issues

#### Issue: "pip install universal-ts[all]" fails
```bash
# Install backends individually
pip install universal-ts[prophet]
pip install universal-ts[autogluon]
pip install universal-ts[sktime]
pip install universal-ts[darts]
```

#### Issue: PyTorch CUDA not available
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<VERSION>

# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Prophet installation fails on Windows
```bash
# Install Visual C++ Build Tools first
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then install Prophet
pip install prophet
```

### Runtime Issues

#### Issue: GPU not detected
1. Run diagnostic: `python scripts/check_gpu.py`
2. Verify drivers: `nvidia-smi`
3. Check PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`

#### Issue: Tests fail with import errors
```bash
# Install in development mode
pip install -e .

# Install test dependencies
pip install -e ".[dev]"
```

## Verification

### Test Installation
```python
from universal_ts import UniversalForecaster
import pandas as pd

# Test with all available backends
backends = ['prophet', 'autogluon', 'sktime', 'darts']

for backend in backends:
    try:
        model = UniversalForecaster(backend=backend)
        print(f"✓ {backend} backend available")
    except Exception as e:
        print(f"✗ {backend} backend not available: {e}")
```

### Test GPU Functionality
```python
from universal_ts import UniversalForecaster
import pandas as pd

# Test GPU with AutoGluon
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=10,
    verbosity=2  # Shows GPU status
)

# Create sample data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100),
    'y': range(100)
})

# Fit and check for GPU usage message
model.fit(df, time_limit=30)
```

## Next Steps

After successful installation:

1. **Try the examples**: Check the `examples/` directory for quickstart scripts
2. **Read the training guide**: See `training_guide.md` for model selection guidance
3. **Run the tests**: Verify your installation with `pytest tests/`
4. **Check GPU setup**: Run `python scripts/check_gpu.py` if using NVIDIA GPU

## Attribution & Licensing

This library is built upon excellent open-source time series forecasting libraries. We extend our sincere gratitude to their developers and communities:

### Core Backend Libraries

**[Prophet](https://github.com/facebook/prophet)**
- Copyright: Facebook (Meta)
- License: MIT License
- Used for: Business-friendly forecasting with holidays, trend analysis

**[AutoGluon-TimeSeries](https://github.com/autogluon/autogluon)**
- Copyright: Amazon Web Services
- License: Apache License 2.0
- Used for: Automated machine learning, ensemble methods, deep learning models

**[sktime](https://github.com/sktime/sktime)**
- Copyright: sktime developers
- License: BSD 3-Clause License
- Used for: Statistical forecasting, traditional time series models

**[Darts](https://github.com/unit8co/darts)**
- Copyright: Unit8
- License: Apache License 2.0
- Used for: Deep learning models, advanced forecasting architectures

### Supporting Libraries

**[PyTorch](https://github.com/pytorch/pytorch)**
- Copyright: PyTorch developers and various contributors
- License: BSD-style license
- Used for: GPU acceleration, deep learning backend

**[pandas](https://github.com/pandas-dev/pandas)**
- Copyright: PyData Development Team
- License: BSD 3-Clause License
- Used for: Data manipulation and time series handling

**[NumPy](https://github.com/numpy/numpy)**
- Copyright: NumPy developers
- License: BSD License
- Used for: Numerical computations, array operations

**[holidays](https://github.com/dr-prodigy/python-holidays)**
- License: MIT License
- Used for: Holiday calendar features and date handling

## Support

- **Documentation**: See `README.md` and `training_guide.md`
- **GPU Issues**: See `GPU_SETUP_GUIDE.md`
- **Examples**: See `examples/` directory
- **Issues**: Report bugs on GitHub Issues

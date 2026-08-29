# Universal Time Series Forecasting Library

A unified Python library for time series forecasting with a **Prophet-like interface** supporting multiple backends (Prophet, AutoGluon, sktime, Darts).

## Features

- 🎯 **Unified API**: Prophet-like interface across all backends
- 📊 **Multiple Backends**: Prophet, AutoGluon, sktime, Darts
- ⚡ **GPU Acceleration**: Built-in GPU support with automatic architecture compatibility checks (including RTX 50-series/Blackwell).
- 🔄 **Panel Data Support**: Forecast multiple series simultaneously
- 📅 **Holiday Features**: Built-in US/UK holidays (extensible)
- 📈 **Comprehensive Metrics**: MAE, RMSE, MAPE, sMAPE, MASE, coverage
- 🧪 **Well-Tested**: Extensive unit and integration tests

## 📚 Documentation

- [**Training Guide**](training_guide.md): Learn about time series types, terminology, and model selection.
- [**GPU Setup Guide**](GPU_SETUP_GUIDE.md): Detailed instructions on configuring your system for GPU acceleration.

## Installation

**⚠️ Important: This library is not yet published to PyPI. Please install from source using the instructions below.**

### Development Installation (Recommended)
```bash
# Clone the repository
git clone https://github.com/yourusername/universal-ts.git
cd universal-ts

# Create and activate conda environment (recommended)
conda create -n universal-ts python=3.11 -y
conda activate universal-ts

# Install in development mode
pip install -e ".[dev,all]"

# Or install specific backends only
pip install -e ".[dev]" prophet sktime darts
pip install autogluon.timeseries
```

### Future PyPI Installation (When Published)
```bash
# Prophet
pip install universal-ts[prophet]

# AutoGluon
pip install universal-ts[autogluon]

# sktime
pip install universal-ts[sktime]

# Darts (includes deep learning models)
pip install universal-ts[darts]

# All backends
pip install universal-ts[all]
```

#### GPU Environment Setup (RTX 50-series/Universal)
If you have a modern NVIDIA GPU (like the RTX 5080), run our setup script to create a compatible `ts_gpu` environment:
```powershell
.\scripts\setup_gpu_env.ps1
```

## Quick Start

### Single Series Forecasting

```python
import pandas as pd
from universal_ts import UniversalForecaster

# Prepare your data
df = pd.DataFrame({
    'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
    'y': range(100)  # Your target values
})

# Create and fit model
model = UniversalForecaster(backend='prophet')
model.fit(df)

# Generate forecasts
forecast = model.predict(horizon=30)
print(forecast.head())
```

### Panel Data Forecasting

```python
# Data with multiple series
df = pd.DataFrame({
    'store_id': ['A', 'A', 'A', 'B', 'B', 'B'],
    'ds': pd.date_range('2020-01-01', periods=3).tolist() * 2,
    'y': [10, 11, 12, 20, 21, 22]
})

# Fit with group identifier
model = UniversalForecaster(backend='autogluon', prediction_length=10)
model.fit(df, group_id='store_id')

# Forecast for all series
forecast = model.predict(horizon=10)
```

## Supported Backends

| Backend | Panel Data | Covariates | Probabilistic | Models | GPU Support |
|---------|-----------|------------|---------------|---------|-------------|
| **Prophet** | ✅ (via separate models) | ✅ | ✅ | Prophet | No |
| **AutoGluon** | ✅ (native) | ✅ | ✅ | Ensemble (WeightedEnsemble, DeepAR, etc.) | ✅ Yes |
| **sktime** | ✅ (via separate models) | ✅ | Partial | Naive, AutoETS, AutoARIMA | No |
| **Darts** | ✅ (via separate models) | ✅ | Partial | TiDE, N-BEATS, ARIMA, ExponentialSmoothing | ✅ Yes (DL models) |

## API Reference

### UniversalForecaster

```python
UniversalForecaster(
    backend='prophet',           # Backend to use
    freq=None,                   # Time series frequency
    country_holidays=None,       # Holiday countries
    **backend_kwargs             # Backend-specific args
)
```

**Methods:**
- `fit(df, group_id=None, **kwargs)`: Fit the model
- `predict(horizon, df_future=None, **kwargs)`: Generate forecasts
- `add_regressor(name, **kwargs)`: Add a covariate
- `add_seasonality(name, period, fourier_order, **kwargs)`: Add custom seasonality
- `get_model_info()`: Get model metadata

## Examples

See the `examples/` directory for complete examples:
- `quickstart_prophet.py`: Single series with Prophet
- `quickstart_autogluon.py`: Single series with AutoGluon
- `quickstart_autogluon_panel.py`: Panel data with AutoGluon (Fastest SOTA)
- `quickstart_darts.py`: Deep Learning (TiDE) vs Statistical (ARIMA) with GPU support
- `quickstart_sktime.py`: Statistical forecasting with sktime

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/universal-ts.git
cd universal-ts

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/
```

## 🙏 Acknowledgments & Attribution

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

### Library Integration Philosophy

This library follows the **wrapper/adapter pattern**, providing a unified interface while preserving the unique strengths of each backend:

- **No reinvention**: We leverage existing, battle-tested implementations
- **Best practices**: Each backend maintains its own optimization and parameter tuning
- **Choice**: Users can select the best tool for their specific use case
- **Compatibility**: Existing code for these libraries remains directly usable

### License Compliance

This library is released under the **MIT License** and is compatible with all included backend licenses:

- ✅ MIT License (Prophet, holidays, this library)
- ✅ Apache 2.0 (AutoGluon, Darts)
- ✅ BSD 3-Clause (sktime, pandas, NumPy)

All attribution and license notices are preserved in our documentation and code comments.

## License

MIT License - see LICENSE file for details.

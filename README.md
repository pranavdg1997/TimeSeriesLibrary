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

### Base Installation
```bash
pip install universal-ts
```

### With Specific Backends
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
.\setup_gpu_env.ps1
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

## License

MIT License - see LICENSE file for details.

"""
Universal Time Series Forecasting Library

A unified interface for multiple time series forecasting backends
with Prophet-like API.
"""

from .core import UniversalForecaster
from .metrics import mae, mse, rmse, mape, smape, mase, coverage, evaluate
from .features import HolidayFeatureGenerator

__version__ = "0.1.0"

__all__ = [
    "UniversalForecaster",
    "mae",
    "mse",
    "rmse",
    "mape",
    "smape",
    "mase",
    "coverage",
    "evaluate",
    "HolidayFeatureGenerator",
]

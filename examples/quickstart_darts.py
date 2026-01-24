"""Quickstart example using Darts backend."""

import pandas as pd
import numpy as np
import warnings
from universal_ts import UniversalForecaster, evaluate

# Suppress some common warnings from Darts/PyTorch
warnings.filterwarnings("ignore")

# Load Air Passengers dataset
from darts.datasets import AirPassengersDataset
print("Loading Air Passengers dataset...")
series = AirPassengersDataset().load()
df = series.to_dataframe().reset_index()
df.columns = ['ds', 'y']

# Split into train and test
train = df.iloc[:-30]
test = df.iloc[-30:]

print(f"Data shape: {df.shape}")
print(f"Train size: {len(train)}, Test size: {len(test)}")

# 1. Using a classic model (CPU)
print("\n--- Testing Darts ARIMA (CPU) ---")
model_arima = UniversalForecaster(
    backend='darts',
    model='arima'
)
model_arima.fit(train)
forecast_arima = model_arima.predict(horizon=30)
print(f"ARIMA Forecast shape: {forecast_arima.shape}")

# 2. Using a Deep Learning model (GPU accelerated if available)
print("\n--- Testing Darts TiDE (GPU if available) ---")
# DL models like 'tide' and 'nbeats' will auto-detect compatible GPUs
model_tide = UniversalForecaster(
    backend='darts',
    model='tide',
    input_chunk_length=30,  # Required for DL models
    output_chunk_length=30, # Required for DL models
    n_epochs=5              # Small number for quick demo
)

print("Fitting TiDE model...")
model_tide.fit(train)

print("Generating TiDE forecasts...")
forecast_tide = model_tide.predict(horizon=30)

# Evaluate TiDE
results = evaluate(
    test,
    forecast_tide,
    metrics=['mae', 'rmse', 'mape']
)

print("\nTiDE Evaluation Results:")
print(results)

# Show model info
print("\nModel Info:")
info = model_tide.get_model_info()
for key, value in info.items():
    if key != 'backend_info':
        print(f"  {key}: {value}")

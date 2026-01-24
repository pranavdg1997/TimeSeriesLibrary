"""Quickstart example using sktime backend."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Load Airline passengers dataset
from sktime.datasets import load_airline
print("Loading Airline passengers dataset...")
y = load_airline()
# Convert to Month End for sktime compatibility
df = pd.DataFrame({'ds': y.index.to_timestamp().to_period('M').to_timestamp('M'), 'y': y.values})

print("Data shape:", df.shape)

# Split into train and test
train = df.iloc[:-30]
test = df.iloc[-30:]

print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

# Create and fit model with sktime (AutoETS)
# sktime models are CPU-based.
model = UniversalForecaster(
    backend='sktime',
    model='auto_ets'
)

print("\nFitting sktime AutoETS model...")
model.fit(train)

# Generate forecasts
print("Generating forecasts...")
forecast = model.predict(horizon=30)

print("\nForecast shape:", forecast.shape)
print("\nFirst few forecast rows:")
print(forecast.head())

# Evaluate
results = evaluate(
    test,
    forecast,
    metrics=['mae', 'rmse', 'mape']
)

print("\nEvaluation Results:")
print(results)

# Show model info
print("\nModel Info:")
for key, value in model.get_model_info().items():
    if key != 'backend_info':
        print(f"  {key}: {value}")

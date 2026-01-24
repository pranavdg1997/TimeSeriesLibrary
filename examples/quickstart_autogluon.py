"""Quickstart example using AutoGluon backend."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Load Shampoo Sales dataset
from sktime.datasets import load_shampoo_sales
print("Loading Shampoo Sales dataset...")
y = load_shampoo_sales()
df = pd.DataFrame({'ds': y.index.to_timestamp(), 'y': y.values})

print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Split into train and test
# Split into train and test
test_size = 5
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

# Create and fit model with AutoGluon
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=len(test)
)

print("\nFitting model...")
model.fit(train)

# Generate forecasts
print("Generating forecasts...")
forecast = model.predict(horizon=len(test))

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
info = model.get_model_info()
for key, value in info.items():
    if key != 'backend_info':
        print(f"  {key}: {value}")

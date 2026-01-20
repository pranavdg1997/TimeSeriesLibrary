"""Quickstart example using Prophet backend."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Create synthetic daily data
np.random.seed(42)
dates = pd.date_range('2020-01-01', periods=365, freq='D')
trend = np.arange(365) * 0.1
seasonality = 10 * np.sin(2 * np.pi * np.arange(365) / 7)  # Weekly seasonality
noise = np.random.normal(0, 2, 365)
values = 100 + trend + seasonality + noise

df = pd.DataFrame({
    'ds': dates,
    'y': values
})

print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Split into train and test
train = df[df['ds'] < '2020-11-01']
test = df[df['ds'] >= '2020-11-01']

print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

# Create and fit model with US holidays
# Note: Holiday features are automatically added to future dates
# during predict() if you don't provide a df_future.
model = UniversalForecaster(
    backend='prophet',
    country_holidays=['US']
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

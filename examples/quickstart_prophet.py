"""Quickstart example using Prophet backend."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Load Peyton Manning dataset (standard Prophet example)
print("Downloading Peyton Manning dataset...")
data_url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_wp_log_peyton_manning.csv'
df = pd.read_csv(data_url)
df['ds'] = pd.to_datetime(df['ds'])

print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Split into train and test
# Split into train and test
test_size = 365
train = df.iloc[:-test_size]
test = df.iloc[-test_size:]

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

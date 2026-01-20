"""Quickstart example using AutoGluon backend with panel data."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Create synthetic panel data (multiple stores)
np.random.seed(42)
n_stores = 3
n_days = 180
data = []

for store_id in range(n_stores):
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Each store has different trend and base level
    base = 100 + store_id * 50
    trend = np.arange(n_days) * (0.2 + store_id * 0.1)
    seasonality = 15 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    noise = np.random.normal(0, 5, n_days)
    values = base + trend + seasonality + noise
    
    store_df = pd.DataFrame({
        'store_id': f'store_{store_id}',
        'ds': dates,
        'y': values
    })
    data.append(store_df)

df = pd.concat(data, ignore_index=True)

print("Panel data shape:", df.shape)
print(f"Number of stores: {df['store_id'].nunique()}")
print("\nFirst few rows:")
print(df.head())

# Split into train and test
train = df[df['ds'] < '2020-05-01']
test = df[df['ds'] >= '2020-05-01']

print(f"\nTrain size: {len(train)}, Test size: {len(test)}")

# Create and fit AutoGluon model
# GPU is automatically detected and used if available.
# You can also explicitly set num_gpus=1 or num_gpus=0 to force GPU/CPU.
model = UniversalForecaster(
    backend='autogluon',
    prediction_length=30,
    eval_metric='MASE',
    verbosity=2,
    num_gpus=None  # Set to 1 if you want to force GPU usage
)

print("\nFitting AutoGluon model (this may take a minute)...")
model.fit(train, group_id='store_id', time_limit=60)

# Generate forecasts
print("\nGenerating forecasts...")
forecast = model.predict(horizon=30)

print("\nForecast shape:", forecast.shape)
print("\nSample forecasts per store:")
for store in df['store_id'].unique():
    store_forecast = forecast[forecast['store_id'] == store]
    print(f"\n{store}: {len(store_forecast)} forecasts")
    print(store_forecast.head(3))

# Evaluate per store and overall
results = evaluate(
    test,
    forecast,
    metrics=['mae', 'rmse', 'mape'],
    group_id_col='store_id'
)

print("\nEvaluation Results (per store + overall):")
print(results)

# Show model info
print("\nModel Info:")
info = model.get_model_info()
for key, value in info.items():
    if key not in ['backend_info']:
        print(f"  {key}: {value}")

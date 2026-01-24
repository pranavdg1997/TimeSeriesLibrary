"""Quickstart example using AutoGluon backend with panel data."""

import pandas as pd
import numpy as np
from universal_ts import UniversalForecaster, evaluate

# Load M4 Daily Dataset (Subset)
print("Downloading M4 Daily dataset (subset)...")
data_url = "https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily_subset/train.csv"
df = pd.read_csv(data_url)
# The dataset has columns: item_id, timestamp, target
df = df.rename(columns={'item_id': 'store_id', 'timestamp': 'ds', 'target': 'y'})
# Filter to just a few items for quicker demonstration
selected_ids = df['store_id'].unique()[:3]
df = df[df['store_id'].isin(selected_ids)].copy()
df['ds'] = pd.to_datetime(df['ds'])

print("Panel data shape:", df.shape)
print(f"Number of stores: {df['store_id'].nunique()}")
print("\nFirst few rows:")
print(df.head())

# Split into train and test
# Split into train and test
prediction_length = 30
train = df.groupby('store_id').apply(lambda x: x.iloc[:-prediction_length]).reset_index(drop=True)
test = df.groupby('store_id').apply(lambda x: x.iloc[-prediction_length:]).reset_index(drop=True)

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

"""
Test AutoGluon GPU Detection
Run this to verify GPU is being detected and used by AutoGluon
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("AUTOGLUON GPU TEST")
print("=" * 60)

# First run the GPU diagnostic
import os
print("\nRunning GPU diagnostic...\n")
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
check_gpu_path = os.path.join(root_dir, 'check_gpu.py')
with open(check_gpu_path, 'r', encoding='utf-8') as f:
    exec(f.read())

print("\n" + "=" * 60)
print("TESTING AUTOGLUON WITH UNIVERSAL_TS")
print("=" * 60)

try:
    from universal_ts import UniversalForecaster
    
    # Create simple test data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum() + 100
    
    df = pd.DataFrame({
        'ds': dates,
        'y': values
    })
    
    print("\nCreating AutoGluon model...")
    # The backend will auto-detect GPU and print status
    model = UniversalForecaster(
        backend='autogluon',
        prediction_length=10,
        verbosity=2  # Show more details
    )
    
    print("\nFitting model (this may take a moment)...")
    model.fit(df, time_limit=60)  # 60 second time limit
    
    print("\nGenerating forecast...")
    forecast = model.predict(horizon=10)
    
    print(f"\n[SUCCESS] Forecast generated without errors. Shape: {forecast.shape}")
    print("\nModel info:")
    info = model.get_model_info()
    for key, value in info.items():
        if key != 'backend_info' and key != 'leaderboard':
            print(f"  {key}: {value}")
    
    # Check if GPU was actually used
    if hasattr(model.backend, 'num_gpus'):
        if model.backend.num_gpus > 0:
            print(f"\n[INFO] Model configured to use {model.backend.num_gpus} GPU(s)")
        else:
            print("\n[WARNING] Model is using CPU only")
            print("   Run check_gpu.py for diagnostics")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)

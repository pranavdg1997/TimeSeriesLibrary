"""Integration test for single series forecasting."""

import pytest
import pandas as pd
import numpy as np


def create_synthetic_series(n_periods=100, freq='D', trend=0.1, noise=0.1):
    """Create synthetic time series data."""
    dates = pd.date_range('2020-01-01', periods=n_periods, freq=freq)
    trend_component = np.arange(n_periods) * trend
    noise_component = np.random.normal(0, noise, n_periods)
    values = trend_component + noise_component
    
    return pd.DataFrame({
        'ds': dates,
        'y': values
    })


class TestSingleSeriesForecasting:
    """Integration tests for single series forecasting."""
    
    def test_prophet_single_series(self):
        """Test Prophet backend with single series."""
        pytest.importorskip("prophet")
        from universal_ts import UniversalForecaster
        
        # Create synthetic data
        df = create_synthetic_series(n_periods=100)
        
        # Fit and predict
        model = UniversalForecaster(backend='prophet')
        model.fit(df)
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
        assert 'yhat_lower' in forecast.columns
        assert 'yhat_upper' in forecast.columns
    
    def test_autogluon_single_series(self):
        """Test AutoGluon backend with single series."""
        pytest.importorskip("autogluon.timeseries")
        from universal_ts import UniversalForecaster
        
        # Create synthetic data
        df = create_synthetic_series(n_periods=50)
        
        # Fit and predict
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=10,
            verbosity=0
        )
        model.fit(df, time_limit=30)
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
    
    def test_sktime_single_series(self):
        """Test sktime backend with single series."""
        pytest.importorskip("sktime")
        from universal_ts import UniversalForecaster
        
        # Create synthetic data
        df = create_synthetic_series(n_periods=50)
        
        # Fit and predict
        model = UniversalForecaster(backend='sktime', model='naive')
        model.fit(df)
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
    
    def test_darts_single_series(self):
        """Test Darts backend with single series."""
        pytest.importorskip("darts")
        from universal_ts import UniversalForecaster
        
        # Create synthetic data
        df = create_synthetic_series(n_periods=50)
        
        # Fit and predict
        model = UniversalForecaster(backend='darts', model='naive_seasonal')
        model.fit(df)
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 10
        assert 'ds' in forecast.columns
        assert 'yhat' in forecast.columns
    
    def test_with_holidays(self):
        """Test forecasting with holiday features."""
        pytest.importorskip("prophet")
        from universal_ts import UniversalForecaster
        
        # Create synthetic data
        df = create_synthetic_series(n_periods=365, freq='D')
        
        # Fit with holidays
        model = UniversalForecaster(
            backend='prophet',
            country_holidays=['US']
        )
        model.fit(df)
        forecast = model.predict(horizon=30)
        
        # Assertions
        assert len(forecast) == 30
        assert 'yhat' in forecast.columns

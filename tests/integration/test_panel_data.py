"""Integration test for panel data forecasting."""

import pytest
import pandas as pd
import numpy as np


def create_synthetic_panel(n_series=3, n_periods=50, freq='D'):
    """Create synthetic panel data."""
    data = []
    
    for i in range(n_series):
        dates = pd.date_range('2020-01-01', periods=n_periods, freq=freq)
        trend = np.arange(n_periods) * (0.1 + i * 0.05)
        noise = np.random.normal(0, 0.1, n_periods)
        values = trend + noise + i * 10  # Offset each series
        
        series_df = pd.DataFrame({
            'group_id': f'series_{i}',
            'ds': dates,
            'y': values
        })
        data.append(series_df)
    
    return pd.concat(data, ignore_index=True)


class TestPanelDataForecasting:
    """Integration tests for panel data forecasting."""
    
    def test_prophet_panel_data(self):
        """Test Prophet backend with panel data."""
        pytest.importorskip("prophet")
        from universal_ts import UniversalForecaster
        
        # Create synthetic panel data
        df = create_synthetic_panel(n_series=3, n_periods=50)
        
        # Fit and predict
        model = UniversalForecaster(backend='prophet')
        model.fit(df, group_id='group_id')
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 30  # 10 periods * 3 series
        assert 'group_id' in forecast.columns
        assert set(forecast['group_id'].unique()) == {'series_0', 'series_1', 'series_2'}
    
    def test_autogluon_panel_data(self):
        """Test AutoGluon backend with panel data."""
        pytest.importorskip("autogluon.timeseries")
        from universal_ts import UniversalForecaster
        
        # Create synthetic panel data
        df = create_synthetic_panel(n_series=2, n_periods=40)
        
        # Fit and predict
        model = UniversalForecaster(
            backend='autogluon',
            prediction_length=10,
            verbosity=0
        )
        model.fit(df, group_id='group_id', time_limit=30)
        forecast = model.predict(horizon=10)
        
        # Assertions
        assert len(forecast) == 20  # 10 periods * 2 series
        assert 'group_id' in forecast.columns
    
    def test_evaluation_panel_data(self):
        """Test evaluation with panel data."""
        pytest.importorskip("prophet")
        from universal_ts import UniversalForecaster, evaluate
        
        # Create synthetic panel data
        df = create_synthetic_panel(n_series=2, n_periods=60)
        
        # Split train/test
        train = df[df['ds'] < '2020-02-20']
        test = df[df['ds'] >= '2020-02-20']
        
        # Fit and predict
        model = UniversalForecaster(backend='prophet')
        model.fit(train, group_id='group_id')
        forecast = model.predict(horizon=len(test) // 2)
        
        # Evaluate
        results = evaluate(
            test,
            forecast,
            metrics=['mae', 'rmse'],
            group_id_col='group_id'
        )
        
        # Assertions
        assert 'group_id' in results.columns
        assert 'OVERALL' in results['group_id'].values
        assert 'mae' in results.columns
        assert 'rmse' in results.columns

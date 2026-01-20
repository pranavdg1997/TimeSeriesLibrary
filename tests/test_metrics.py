"""Tests for metrics module."""

import pytest
import pandas as pd
import numpy as np
from universal_ts.metrics import (
    mae, mse, rmse, mape, smape, mase, coverage, evaluate
)


class TestBasicMetrics:
    """Tests for basic metric functions."""
    
    def test_mae(self):
        """Test Mean Absolute Error."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        result = mae(y_true, y_pred)
        expected = np.mean(np.abs(y_true - y_pred))
        assert np.isclose(result, expected)
        assert np.isclose(result, 0.14)
    
    def test_mse(self):
        """Test Mean Squared Error."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])
        result = mse(y_true, y_pred)
        assert result == 0.0
    
    def test_rmse(self):
        """Test Root Mean Squared Error."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([2, 3, 4, 5, 6])
        result = rmse(y_true, y_pred)
        assert result == 1.0
    
    def test_mape(self):
        """Test Mean Absolute Percentage Error."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = mape(y_true, y_pred)
        # (10/100 + 10/200 + 10/300) / 3 * 100 = 5.0
        assert np.isclose(result, 5.0, atol=0.1)
    
    def test_smape(self):
        """Test Symmetric Mean Absolute Percentage Error."""
        y_true = np.array([100, 200, 300])
        y_pred = np.array([110, 190, 310])
        result = smape(y_true, y_pred)
        assert result > 0 and result < 100
    
    def test_mase(self):
        """Test Mean Absolute Scaled Error."""
        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y_true = np.array([11, 12, 13])
        y_pred = np.array([11.5, 12.5, 13.5])
        result = mase(y_true, y_pred, y_train, seasonality=1)
        assert result > 0
    
    def test_coverage(self):
        """Test coverage metric."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_lower = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        y_upper = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        result = coverage(y_true, y_lower, y_upper)
        assert result == 1.0  # All values within intervals


class TestEvaluate:
    """Tests for evaluate function."""
    
    def test_single_series_evaluation(self):
        """Test evaluation for single series."""
        ground_truth = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': range(10)
        })
        forecast = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'yhat': [x + 0.1 for x in range(10)]
        })
        
        result = evaluate(ground_truth, forecast, metrics=['mae', 'rmse'])
        
        assert 'mae' in result.columns
        assert 'rmse' in result.columns
        assert len(result) == 1
        assert result['mae'].iloc[0] == 0.1
    
    def test_panel_data_evaluation(self):
        """Test evaluation for panel data."""
        ground_truth = pd.DataFrame({
            'group_id': ['A'] * 5 + ['B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5).tolist() * 2,
            'y': list(range(5)) + list(range(5, 10))
        })
        forecast = pd.DataFrame({
            'group_id': ['A'] * 5 + ['B'] * 5,
            'ds': pd.date_range('2020-01-01', periods=5).tolist() * 2,
            'yhat': [x + 0.1 for x in range(5)] + [x + 0.2 for x in range(5, 10)]
        })
        
        result = evaluate(
            ground_truth,
            forecast,
            metrics=['mae'],
            group_id_col='group_id'
        )
        
        assert len(result) == 3  # 2 groups + overall
        assert 'group_id' in result.columns
        assert set(result['group_id']) == {'A', 'B', 'OVERALL'}
    
    def test_with_prediction_intervals(self):
        """Test evaluation with prediction intervals."""
        ground_truth = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=5),
            'y': [1, 2, 3, 4, 5]
        })
        forecast = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=5),
            'yhat': [1.1, 2.1, 3.1, 4.1, 5.1],
            'yhat_lower': [0.5, 1.5, 2.5, 3.5, 4.5],
            'yhat_upper': [1.5, 2.5, 3.5, 4.5, 5.5]
        })
        
        result = evaluate(ground_truth, forecast, metrics=['mae', 'coverage'])
        
        assert 'coverage' in result.columns
        assert result['coverage'].iloc[0] == 1.0

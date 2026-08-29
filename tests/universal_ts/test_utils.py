"""Tests for core utilities module."""

import pytest
import pandas as pd
import numpy as np
from universal_ts.utils import (
    validate_dataframe,
    validate_datetime_column,
    validate_monotonic_time,
    infer_frequency,
    sort_by_time,
    handle_missing_values,
    prepare_dataframe,
)
from universal_ts.exceptions import DataValidationError


class TestValidateDataframe:
    """Tests for validate_dataframe function."""
    
    def test_valid_dataframe(self):
        """Test with valid dataframe."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': range(10)
        })
        # Should not raise
        validate_dataframe(df)
    
    def test_missing_ds_column(self):
        """Test with missing ds column."""
        df = pd.DataFrame({'y': range(10)})
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df)
    
    def test_missing_y_column(self):
        """Test with missing y column."""
        df = pd.DataFrame({'ds': pd.date_range('2020-01-01', periods=10)})
        with pytest.raises(DataValidationError, match="Missing required columns"):
            validate_dataframe(df)
    
    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="DataFrame is empty"):
            validate_dataframe(df)
    
    def test_non_numeric_target(self):
        """Test with non-numeric target column."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': ['a'] * 10
        })
        with pytest.raises(DataValidationError, match="must be numeric"):
            validate_dataframe(df)


class TestValidateDatetimeColumn:
    """Tests for validate_datetime_column function."""
    
    def test_already_datetime(self):
        """Test with datetime column."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': range(10)
        })
        result = validate_datetime_column(df)
        assert pd.api.types.is_datetime64_any_dtype(result['ds'])
    
    def test_string_dates_coerce(self):
        """Test coercing string dates."""
        df = pd.DataFrame({
            'ds': ['2020-01-01', '2020-01-02', '2020-01-03'],
            'y': [1, 2, 3]
        })
        result = validate_datetime_column(df, coerce=True)
        assert pd.api.types.is_datetime64_any_dtype(result['ds'])
    
    def test_invalid_dates_no_coerce(self):
        """Test with invalid dates and no coercion."""
        df = pd.DataFrame({
            'ds': [1, 2, 3],
            'y': [1, 2, 3]
        })
        with pytest.raises(DataValidationError, match="must be datetime"):
            validate_datetime_column(df, coerce=False)


class TestValidateMonotonicTime:
    """Tests for validate_monotonic_time function."""
    
    def test_monotonic_increasing(self):
        """Test with monotonic increasing time."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10),
            'y': range(10)
        })
        # Should not raise
        validate_monotonic_time(df)
    
    def test_non_monotonic(self):
        """Test with non-monotonic time."""
        df = pd.DataFrame({
            'ds': pd.to_datetime(['2020-01-03', '2020-01-01', '2020-01-02']),
            'y': [1, 2, 3]
        })
        with pytest.raises(DataValidationError, match="not monotonically increasing"):
            validate_monotonic_time(df)
    
    def test_panel_data_monotonic(self):
        """Test panel data with monotonic time per group."""
        df = pd.DataFrame({
            'group_id': ['A', 'A', 'A', 'B', 'B', 'B'],
            'ds': pd.to_datetime([
                '2020-01-01', '2020-01-02', '2020-01-03',
                '2020-01-01', '2020-01-02', '2020-01-03'
            ]),
            'y': [1, 2, 3, 4, 5, 6]
        })
        # Should not raise
        validate_monotonic_time(df, group_id_col='group_id')


class TestInferFrequency:
    """Tests for infer_frequency function."""
    
    def test_daily_frequency(self):
        """Test inferring daily frequency."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10, freq='D'),
            'y': range(10)
        })
        freq = infer_frequency(df)
        assert freq == 'D'
    
    def test_hourly_frequency(self):
        """Test inferring hourly frequency."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10, freq='h'),
            'y': range(10)
        })
        freq = infer_frequency(df)
        assert freq == 'h'


class TestHandleMissingValues:
    """Tests for handle_missing_values function."""
    
    def test_drop_method(self):
        """Test drop method."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=5),
            'y': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        result = handle_missing_values(df, method='drop')
        assert len(result) == 3
        assert not result['y'].isna().any()
    
    def test_ffill_method(self):
        """Test forward fill method."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=5),
            'y': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        result = handle_missing_values(df, method='ffill')
        assert not result['y'].isna().any()
        assert result['y'].tolist() == [1.0, 1.0, 3.0, 3.0, 5.0]
    
    def test_zero_method(self):
        """Test zero fill method."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=5),
            'y': [1.0, np.nan, 3.0, np.nan, 5.0]
        })
        result = handle_missing_values(df, method='zero')
        assert not result['y'].isna().any()
        assert result['y'].tolist() == [1.0, 0.0, 3.0, 0.0, 5.0]


class TestPrepareDataframe:
    """Tests for prepare_dataframe function."""
    
    def test_full_preparation(self):
        """Test full dataframe preparation."""
        df = pd.DataFrame({
            'ds': ['2020-01-03', '2020-01-01', '2020-01-02'],
            'y': [3.0, 1.0, 2.0]
        })
        result, freq = prepare_dataframe(df)
        
        # Check sorted
        assert result['ds'].tolist() == pd.to_datetime([
            '2020-01-01', '2020-01-02', '2020-01-03'
        ]).tolist()
        
        # Check datetime conversion
        assert pd.api.types.is_datetime64_any_dtype(result['ds'])
        
        # Check frequency inference
        assert freq == 'D'

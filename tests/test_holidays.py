"""Tests for holiday features module."""

import pytest
import pandas as pd
from universal_ts.features import HolidayFeatureGenerator


class TestHolidayFeatureGenerator:
    """Tests for HolidayFeatureGenerator class."""
    
    def test_initialization(self):
        """Test basic initialization."""
        generator = HolidayFeatureGenerator(countries=["US", "UK"])
        assert generator.countries == ["US", "UK"]
    
    def test_add_holiday_features_us(self):
        """Test adding US holiday features."""
        df = pd.DataFrame({
            'ds': pd.to_datetime([
                '2020-01-01',  # New Year's Day
                '2020-01-02',  # Not a holiday
                '2020-07-04',  # Independence Day
                '2020-12-25',  # Christmas
            ])
        })
        
        generator = HolidayFeatureGenerator(countries=["US"])
        result = generator.add_holiday_features(df)
        
        assert 'is_holiday' in result.columns
        assert 'is_holiday_US' in result.columns
        assert result['is_holiday_US'].tolist() == [1, 0, 1, 1]
        assert result['is_holiday'].tolist() == [1, 0, 1, 1]
    
    def test_add_holiday_features_multiple_countries(self):
        """Test adding holiday features for multiple countries."""
        df = pd.DataFrame({
            'ds': pd.to_datetime([
                '2020-01-01',  # New Year in both
                '2020-07-04',  # US Independence Day
                '2020-12-26',  # UK Boxing Day
            ])
        })
        
        generator = HolidayFeatureGenerator(countries=["US", "UK"])
        result = generator.add_holiday_features(df)
        
        assert 'is_holiday_US' in result.columns
        assert 'is_holiday_UK' in result.columns
        assert 'is_holiday' in result.columns
        
        # New Year is holiday in both
        assert result.loc[0, 'is_holiday'] == 1
        # July 4 is US only
        assert result.loc[1, 'is_holiday_US'] == 1
        assert result.loc[1, 'is_holiday_UK'] == 0
        # Boxing Day is UK only
        assert result.loc[2, 'is_holiday_US'] == 0
        assert result.loc[2, 'is_holiday_UK'] == 1
    
    def test_no_countries(self):
        """Test with no countries specified."""
        df = pd.DataFrame({
            'ds': pd.date_range('2020-01-01', periods=10)
        })
        
        generator = HolidayFeatureGenerator(countries=[])
        result = generator.add_holiday_features(df)
        
        # Should return unchanged dataframe
        assert len(result.columns) == len(df.columns)
    
    def test_get_holiday_dates(self):
        """Test getting holiday dates in a range."""
        generator = HolidayFeatureGenerator(countries=["US"])
        
        holidays_df = generator.get_holiday_dates(
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31'),
            country="US"
        )
        
        assert 'date' in holidays_df.columns
        assert 'country' in holidays_df.columns
        assert 'holiday_name' in holidays_df.columns
        assert len(holidays_df) > 0
        
        # Check for known holidays
        dates = holidays_df['date'].dt.date.tolist()
        assert pd.Timestamp('2020-01-01').date() in dates
        assert pd.Timestamp('2020-07-04').date() in dates
        assert pd.Timestamp('2020-12-25').date() in dates

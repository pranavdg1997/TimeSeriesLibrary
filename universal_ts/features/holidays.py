"""Holiday feature generation using python-holidays library."""

from __future__ import annotations
from typing import List, Optional, Dict
import pandas as pd
import holidays as pyholidays


class HolidayFeatureGenerator:
    """
    Generate holiday features for time series forecasting.
    
    Uses the python-holidays library to create binary indicator
    columns for holidays in specified countries.
    
    Parameters
    ----------
    countries : list of str, optional
        List of country codes (ISO 3166-1 alpha-2), e.g., ["US", "UK"]
        If None, no holiday features will be generated.
    years : list of int or range, optional
        Years to include in holiday calendars. If None, will be inferred
        from the data when add_holiday_features is called.
    """
    
    def __init__(
        self,
        countries: Optional[List[str]] = None,
        years: Optional[range] = None
    ):
        self.countries = countries or []
        self.years = years
        self.holiday_calendars: Dict[str, pyholidays.HolidayBase] = {}
        
        # Initialize holiday calendars if years provided
        if self.years:
            self._initialize_calendars()
    
    def _initialize_calendars(self) -> None:
        """Initialize holiday calendars for specified countries and years."""
        for country in self.countries:
            try:
                self.holiday_calendars[country] = pyholidays.country_holidays(
                    country,
                    years=self.years
                )
            except (KeyError, AttributeError) as e:
                raise ValueError(
                    f"Country code '{country}' not supported by python-holidays. "
                    f"Error: {e}"
                )
    
    def add_holiday_features(
        self,
        df: pd.DataFrame,
        time_col: str = "ds"
    ) -> pd.DataFrame:
        """
        Add holiday indicator columns to DataFrame.
        
        Creates the following columns:
        - is_holiday: 1 if date is a holiday in ANY specified country, 0 otherwise
        - is_holiday_{COUNTRY}: 1 if date is a holiday in that country, 0 otherwise
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with datetime column
        time_col : str, default="ds"
            Name of the datetime column
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added holiday feature columns
        """
        if not self.countries:
            return df
        
        df = df.copy()
        
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        # Initialize calendars if not already done
        if not self.holiday_calendars:
            # Infer years from data
            min_year = df[time_col].dt.year.min()
            max_year = df[time_col].dt.year.max()
            self.years = range(min_year, max_year + 1)
            self._initialize_calendars()
        
        # Add per-country holiday indicators
        holiday_columns = []
        for country, calendar in self.holiday_calendars.items():
            col_name = f"is_holiday_{country}"
            df[col_name] = df[time_col].dt.date.apply(lambda x: int(x in calendar))
            holiday_columns.append(col_name)
        
        # Add overall holiday indicator (any country)
        if holiday_columns:
            df["is_holiday"] = df[holiday_columns].max(axis=1)
        
        return df
    
    def get_holiday_dates(
        self,
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        country: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get all holiday dates in a date range.
        
        Parameters
        ----------
        start_date : pd.Timestamp
            Start of date range
        end_date : pd.Timestamp
            End of date range
        country : str, optional
            Specific country code. If None, returns holidays from all countries.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: date, country, holiday_name
        """
        if not self.holiday_calendars:
            # Initialize calendars for the date range
            min_year = start_date.year
            max_year = end_date.year
            self.years = range(min_year, max_year + 1)
            self._initialize_calendars()
        
        results = []
        
        countries_to_check = [country] if country else self.countries
        
        for country_code in countries_to_check:
            if country_code not in self.holiday_calendars:
                continue
                
            calendar = self.holiday_calendars[country_code]
            
            # Get all holidays in date range
            date_range = pd.date_range(start_date, end_date, freq="D")
            for date in date_range:
                date_obj = date.date()
                if date_obj in calendar:
                    results.append({
                        "date": date,
                        "country": country_code,
                        "holiday_name": calendar.get(date_obj)
                    })
        
        return pd.DataFrame(results)
    
    def get_supported_countries(self) -> List[str]:
        """
        Get list of country codes supported by python-holidays.
        
        Returns
        -------
        list of str
            List of supported country codes
        """
        return list(pyholidays.list_supported_countries().keys())

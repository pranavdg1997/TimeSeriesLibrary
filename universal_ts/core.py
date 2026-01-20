"""Core UniversalForecaster class with Prophet-like API."""

from __future__ import annotations
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import warnings

from .base import BaseBackendModel
from .exceptions import (
    BackendNotFoundError,
    BackendNotInstalledError,
    FitNotCalledError,
    UnsupportedOperationError,
)
from .utils import prepare_dataframe
from .features import HolidayFeatureGenerator


# Backend registry - lazy loaded
BACKEND_REGISTRY: Dict[str, str] = {
    "prophet": "universal_ts.backends.prophet_backend.ProphetBackend",
    "autogluon": "universal_ts.backends.autogluon_backend.AutoGluonBackend",
    "sktime": "universal_ts.backends.sktime_backend.SktimeBackend",
    "darts": "universal_ts.backends.darts_backend.DartsBackend",
}


def _load_backend(backend_name: str) -> type:
    """
    Lazy load a backend class.
    
    Parameters
    ----------
    backend_name : str
        Name of the backend to load
        
    Returns
    -------
    type
        Backend class
        
    Raises
    ------
    BackendNotFoundError
        If backend name is not recognized
    BackendNotInstalledError
        If backend dependencies are not installed
    """
    if backend_name not in BACKEND_REGISTRY:
        raise BackendNotFoundError(
            f"Backend '{backend_name}' not found. "
            f"Available backends: {list(BACKEND_REGISTRY.keys())}"
        )
    
    module_path = BACKEND_REGISTRY[backend_name]
    module_name, class_name = module_path.rsplit(".", 1)
    
    try:
        import importlib
        module = importlib.import_module(module_name)
        backend_class = getattr(module, class_name)
        return backend_class
    except ImportError as e:
        raise BackendNotInstalledError(
            f"Backend '{backend_name}' requires additional dependencies. "
            f"Install with: pip install universal-ts[{backend_name}]\n"
            f"Error: {e}"
        )


class UniversalForecaster:
    """
    Universal time series forecaster with multiple backend support.
    
    Provides a Prophet-like interface for time series forecasting with
    support for multiple backends (Prophet, AutoGluon, sktime, Darts).
    
    Parameters
    ----------
    backend : str, default="prophet"
        Backend to use for forecasting. Options:
        - "prophet": Facebook Prophet
        - "autogluon": AutoGluon TimeSeries
        - "sktime": sktime forecasting
        - "darts": Darts forecasting
    freq : str, optional
        Frequency of the time series (pandas offset alias).
        If None, will attempt to infer from data.
    country_holidays : list of str, optional
        Country codes for holiday features, e.g., ["US", "UK"].
        Uses python-holidays library.
    **backend_kwargs
        Additional arguments passed to the backend model
        
    Examples
    --------
    >>> import pandas as pd
    >>> from universal_ts import UniversalForecaster
    >>> 
    >>> # Create sample data
    >>> df = pd.DataFrame({
    ...     'ds': pd.date_range('2020-01-01', periods=100, freq='D'),
    ...     'y': range(100)
    ... })
    >>> 
    >>> # Fit and predict
    >>> model = UniversalForecaster(backend='prophet')
    >>> model.fit(df)
    >>> forecast = model.predict(horizon=10)
    """
    
    def __init__(
        self,
        backend: str = "prophet",
        freq: Optional[str] = None,
        country_holidays: Optional[List[str]] = None,
        **backend_kwargs
    ):
        self.backend_name = backend
        self.freq = freq
        self.country_holidays = country_holidays
        self.backend_kwargs = backend_kwargs
        
        # Initialize backend
        backend_class = _load_backend(backend)
        self.backend: BaseBackendModel = backend_class(**backend_kwargs)
        
        # Initialize holiday generator
        self.holiday_generator = None
        if country_holidays:
            self.holiday_generator = HolidayFeatureGenerator(countries=country_holidays)
        
        # State
        self.is_fitted = False
        self.group_id_col = None
        self.covariates = []
        self.regressors = []
        self.seasonalities = []
        
    def fit(
        self,
        df: pd.DataFrame,
        group_id: Optional[str] = None,
        **kwargs
    ) -> "UniversalForecaster":
        """
        Fit the forecaster to historical data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with required columns:
            - 'ds': datetime column (will be converted if not datetime)
            - 'y': target values (numeric)
            - Optional: group_id column for panel data
            - Optional: additional columns for covariates/regressors
        group_id : str, optional
            Name of the column containing group/series identifiers
            for panel data (multiple time series)
        **kwargs
            Additional arguments passed to the backend's fit method
            
        Returns
        -------
        self
            Fitted forecaster instance
            
        Examples
        --------
        >>> # Single series
        >>> model.fit(df)
        >>> 
        >>> # Panel data
        >>> model.fit(df, group_id='store_id')
        """
        # Prepare and validate data
        df_prepared, inferred_freq = prepare_dataframe(
            df,
            time_col="ds",
            target_col="y",
            group_id_col=group_id,
            freq=self.freq
        )
        
        # Use inferred frequency if not provided
        if self.freq is None and inferred_freq:
            self.freq = inferred_freq
            warnings.warn(
                f"Inferred frequency: {inferred_freq}. "
                "Specify freq parameter to override.",
                UserWarning
            )
        
        # Add holiday features if configured
        if self.holiday_generator:
            df_prepared = self.holiday_generator.add_holiday_features(df_prepared)
            # Add holiday columns to covariates list
            holiday_cols = [col for col in df_prepared.columns if col.startswith("is_holiday")]
            self.covariates.extend(holiday_cols)
        
        # Store group_id column name
        self.group_id_col = group_id
        
        # Check if backend supports panel data
        if group_id and not self.backend.supports_panel_data():
            warnings.warn(
                f"Backend '{self.backend_name}' does not natively support panel data. "
                "Will fit separate models for each series.",
                UserWarning
            )
        
        # Fit the backend
        self.backend.fit(
            df_prepared,
            group_id_col=group_id,
            covariates=self.covariates if self.covariates else None,
            **kwargs
        )
        
        self.is_fitted = True
        return self
    
    def predict(
        self,
        horizon: int,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts for the specified horizon.
        
        Parameters
        ----------
        horizon : int
            Number of time steps to forecast into the future
        df_future : pd.DataFrame, optional
            Future values of covariates/regressors. Must contain:
            - 'ds': future timestamps
            - Optional: group_id column if panel data
            - Required: any covariate columns used during fit
        **kwargs
            Additional arguments passed to the backend's predict method
            
        Returns
        -------
        pd.DataFrame
            Forecasts with columns:
            - 'ds': forecast timestamps
            - 'yhat': point forecasts
            - Optional: 'group_id' if panel data
            - Optional: 'yhat_lower', 'yhat_upper' for prediction intervals
            - Optional: quantile columns if probabilistic forecasts
            
        Raises
        ------
        FitNotCalledError
            If predict is called before fit
            
        Examples
        --------
        >>> # Simple forecast
        >>> forecast = model.predict(horizon=30)
        >>> 
        >>> # With future covariates
        >>> df_future = pd.DataFrame({
        ...     'ds': pd.date_range('2020-04-11', periods=30, freq='D'),
        ...     'temperature': [20, 21, 22, ...]
        ... })
        >>> forecast = model.predict(horizon=30, df_future=df_future)
        """
        if not self.is_fitted:
            raise FitNotCalledError(
                "Model has not been fitted. Call fit() before predict()."
            )
        
        # Auto-generate future dataframe if needed for holiday features or other covariates
        if df_future is None and (self.holiday_generator or self.covariates):
            last_date = None
            group_ids = None
            
            # Try to get last date and group IDs from backend
            if hasattr(self.backend, 'models') and self.backend.models:
                # Prophet-style models
                first_model = list(self.backend.models.values())[0]
                if hasattr(first_model, 'history') and 'ds' in first_model.history.columns:
                    last_date = first_model.history['ds'].max()
                group_ids = list(self.backend.models.keys()) if len(self.backend.models) > 1 else None
            elif hasattr(self.backend, 'train_data'):
                # AutoGluon-style backends
                last_date = self.backend.train_data['timestamp'].max()
                if 'item_id' in self.backend.train_data.columns:
                    group_ids = self.backend.train_data['item_id'].unique().tolist()
            
            if last_date is not None:
                # Generate future dates
                if self.freq:
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(1, unit=self.freq[0] if self.freq else 'D'),
                        periods=horizon,
                        freq=self.freq
                    )
                else:
                    # Default to daily if no info
                    future_dates = pd.date_range(
                        start=last_date + pd.Timedelta(days=1),
                        periods=horizon,
                        freq='D'
                    )
                
                # Create future dataframe
                if group_ids and len(group_ids) > 1:
                    # Panel data: repeat dates for each group
                    dfs = []
                    group_col = self.backend.group_id_col or 'item_id'
                    for g in group_ids:
                        dfs.append(pd.DataFrame({
                            group_col: [g] * len(future_dates),
                            'ds': future_dates
                        }))
                    df_future = pd.concat(dfs, ignore_index=True)
                else:
                    # Single series
                    df_future = pd.DataFrame({'ds': future_dates})
        
        # Add holiday features to future dataframe if configured
        if df_future is not None and self.holiday_generator:
            df_future = self.holiday_generator.add_holiday_features(df_future)
        
        # Generate predictions
        forecast = self.backend.predict(
            horizon=horizon,
            df_future=df_future,
            **kwargs
        )
        
        return forecast
    
    def add_regressor(
        self,
        name: str,
        prior_scale: Optional[float] = None,
        standardize: bool = True,
        mode: Optional[str] = None
    ) -> "UniversalForecaster":
        """
        Add a regressor (covariate) to the model.
        
        This method should be called before fit(). The regressor column
        must be present in the training data passed to fit().
        
        Parameters
        ----------
        name : str
            Name of the regressor column in the DataFrame
        prior_scale : float, optional
            Prior scale for the regressor (Prophet-specific)
        standardize : bool, default=True
            Whether to standardize the regressor (Prophet-specific)
        mode : str, optional
            'additive' or 'multiplicative' (Prophet-specific)
            
        Returns
        -------
        self
            Forecaster instance for method chaining
            
        Examples
        --------
        >>> model = UniversalForecaster(backend='prophet')
        >>> model.add_regressor('temperature')
        >>> model.add_regressor('promotion', mode='additive')
        >>> model.fit(df)
        """
        if self.is_fitted:
            warnings.warn(
                "Adding regressor after fit() has no effect. "
                "Call add_regressor() before fit().",
                UserWarning
            )
        
        regressor_config = {
            "name": name,
            "prior_scale": prior_scale,
            "standardize": standardize,
            "mode": mode,
        }
        self.regressors.append(regressor_config)
        
        # Add to covariates list
        if name not in self.covariates:
            self.covariates.append(name)
        
        return self
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ) -> "UniversalForecaster":
        """
        Add a custom seasonality component to the model.
        
        This method should be called before fit().
        Note: This is primarily for Prophet backend. Other backends
        may not support custom seasonalities.
        
        Parameters
        ----------
        name : str
            Name of the seasonality component
        period : float
            Period of the seasonality in days
        fourier_order : int
            Number of Fourier terms to use
        prior_scale : float, optional
            Prior scale for the seasonality
        mode : str, optional
            'additive' or 'multiplicative'
            
        Returns
        -------
        self
            Forecaster instance for method chaining
            
        Examples
        --------
        >>> model = UniversalForecaster(backend='prophet')
        >>> model.add_seasonality('monthly', period=30.5, fourier_order=5)
        >>> model.fit(df)
        """
        if self.is_fitted:
            warnings.warn(
                "Adding seasonality after fit() has no effect. "
                "Call add_seasonality() before fit().",
                UserWarning
            )
        
        if self.backend_name not in ["prophet"]:
            warnings.warn(
                f"Backend '{self.backend_name}' may not support custom seasonalities. "
                "This configuration may be ignored.",
                UserWarning
            )
        
        seasonality_config = {
            "name": name,
            "period": period,
            "fourier_order": fourier_order,
            "prior_scale": prior_scale,
            "mode": mode,
        }
        self.seasonalities.append(seasonality_config)
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the forecaster and backend.
        
        Returns
        -------
        dict
            Model metadata including backend info, configuration, and capabilities
        """
        info = {
            "backend": self.backend_name,
            "is_fitted": self.is_fitted,
            "freq": self.freq,
            "group_id_col": self.group_id_col,
            "country_holidays": self.country_holidays,
            "covariates": self.covariates,
            "regressors": self.regressors,
            "seasonalities": self.seasonalities,
        }
        
        # Add backend-specific info
        if self.is_fitted:
            info["backend_info"] = self.backend.get_model_info()
        
        return info

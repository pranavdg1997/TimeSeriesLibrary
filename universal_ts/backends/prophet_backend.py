"""Prophet backend implementation."""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd
import warnings

from ..base import BaseBackendModel
from ..exceptions import BackendNotInstalledError, UnsupportedOperationError

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


class ProphetBackend(BaseBackendModel):
    """
    Facebook Prophet backend for time series forecasting.
    
    Wraps Prophet's API to conform to the universal_ts interface.
    
    Parameters
    ----------
    **kwargs
        Arguments passed to Prophet constructor (e.g., seasonality_mode,
        changepoint_prior_scale, etc.)
    """
    
    def __init__(self, **kwargs):
        if not PROPHET_AVAILABLE:
            raise BackendNotInstalledError(
                "Prophet is not installed. Install with: pip install universal-ts[prophet]"
            )
        
        super().__init__(**kwargs)
        self.prophet_kwargs = kwargs
        self.models: Dict[str, Prophet] = {}  # For panel data: group_id -> model
        self.is_panel = False
        
    def fit(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit Prophet model(s) to training data.
        
        For panel data, fits separate Prophet models for each series.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds' and 'y' columns
        group_id_col : str, optional
            Column name for group identifiers (panel data)
        covariates : list of str, optional
            Names of regressor columns
        **kwargs
            Additional Prophet fit parameters
        """
        self.group_id_col = group_id_col
        self.is_panel = group_id_col is not None
        self.covariates = covariates or []
        
        if self.is_panel:
            # Fit separate model for each group
            for group_id, group_df in df.groupby(group_id_col):
                model = self._create_prophet_model()
                
                # Add regressors
                for cov in self.covariates:
                    if cov in group_df.columns:
                        model.add_regressor(cov)
                
                # Prepare data for Prophet
                prophet_df = group_df[['ds', 'y'] + self.covariates].copy()
                
                # Fit model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(prophet_df, **kwargs)
                
                self.models[str(group_id)] = model
        else:
            # Single series
            model = self._create_prophet_model()
            
            # Add regressors
            for cov in self.covariates:
                if cov in df.columns:
                    model.add_regressor(cov)
            
            # Prepare data for Prophet
            prophet_df = df[['ds', 'y'] + self.covariates].copy()
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(prophet_df, **kwargs)
            
            self.models['_single_'] = model
        
        self.is_fitted = True
        
        # Infer frequency from first model
        first_model = list(self.models.values())[0]
        if hasattr(first_model, 'history'):
            self.freq = pd.infer_freq(first_model.history['ds'])
    
    def _create_prophet_model(self) -> Prophet:
        """Create a new Prophet model instance."""
        return Prophet(**self.prophet_kwargs)
    
    def predict(
        self,
        horizon: int,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts using Prophet.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        df_future : pd.DataFrame, optional
            Future dataframe with regressors
        **kwargs
            Additional Prophet predict parameters
            
        Returns
        -------
        pd.DataFrame
            Forecasts with columns: ds, yhat, yhat_lower, yhat_upper, [group_id]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_forecasts = []
        
        if self.is_panel:
            # Predict for each group
            for group_id, model in self.models.items():
                # Create future dataframe
                if df_future is not None:
                    # Filter future data for this group
                    future_df = df_future[df_future[self.group_id_col] == group_id].copy()
                    if future_df.empty:
                        # No future data for this group, create default
                        future_df = model.make_future_dataframe(periods=horizon, include_history=False)
                    else:
                        # Ensure we have the right columns
                        future_df = future_df[['ds'] + self.covariates].copy()
                else:
                    future_df = model.make_future_dataframe(periods=horizon, include_history=False)
                
                # Generate forecast
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    forecast = model.predict(future_df, **kwargs)
                
                # Extract relevant columns
                forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                forecast[self.group_id_col] = group_id
                all_forecasts.append(forecast)
            
            result = pd.concat(all_forecasts, ignore_index=True)
        else:
            # Single series
            model = self.models['_single_']
            
            # Create future dataframe
            if df_future is not None:
                future_df = df_future[['ds'] + self.covariates].copy()
            else:
                future_df = model.make_future_dataframe(periods=horizon, include_history=False)
            
            # Generate forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                forecast = model.predict(future_df, **kwargs)
            
            # Extract relevant columns
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        
        return result
    
    def supports_panel_data(self) -> bool:
        """Prophet supports panel data via separate models per series."""
        return True
    
    def supports_covariates(self) -> bool:
        """Prophet supports additional regressors."""
        return True
    
    def supports_probabilistic(self) -> bool:
        """Prophet provides prediction intervals."""
        return True
    
    def add_regressor(
        self,
        name: str,
        prior_scale: Optional[float] = None,
        standardize: bool = True,
        mode: Optional[str] = None
    ) -> None:
        """
        Add a regressor to Prophet models.
        
        Note: This must be called before fit().
        """
        if self.is_fitted:
            raise ValueError("Cannot add regressor after fitting. Call before fit().")
        
        # Store for later use during fit
        if name not in self.covariates:
            self.covariates.append(name)
    
    def add_seasonality(
        self,
        name: str,
        period: float,
        fourier_order: int,
        prior_scale: Optional[float] = None,
        mode: Optional[str] = None
    ) -> None:
        """
        Add custom seasonality to Prophet models.
        
        Note: This must be called before fit().
        """
        if self.is_fitted:
            raise ValueError("Cannot add seasonality after fitting. Call before fit().")
        
        # This would need to be stored and applied during model creation
        # For now, users should pass seasonality params to Prophet constructor
        warnings.warn(
            "add_seasonality should be configured via Prophet constructor kwargs. "
            "Pass seasonality parameters when creating UniversalForecaster.",
            UserWarning
        )

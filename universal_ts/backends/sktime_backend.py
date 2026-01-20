"""sktime backend implementation."""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import warnings

from ..base import BaseBackendModel
from ..exceptions import BackendNotInstalledError

try:
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.ets import AutoETS
    from sktime.forecasting.arima import AutoARIMA
    SKTIME_AVAILABLE = True
except ImportError:
    SKTIME_AVAILABLE = False


class SktimeBackend(BaseBackendModel):
    """
    sktime backend for time series forecasting.
    
    Provides access to sktime's forecasting models via a simple registry.
    
    Parameters
    ----------
    model : str, default="naive"
        Model to use. Options: "naive", "auto_ets", "auto_arima"
    model_kwargs : dict, optional
        Arguments passed to the sktime model constructor
    **kwargs
        Additional backend configuration
    """
    
    # Model registry
    MODEL_REGISTRY = {
        "naive": NaiveForecaster if SKTIME_AVAILABLE else None,
        "auto_ets": AutoETS if SKTIME_AVAILABLE else None,
        "auto_arima": AutoARIMA if SKTIME_AVAILABLE else None,
    }
    
    def __init__(
        self,
        model: str = "naive",
        model_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        if not SKTIME_AVAILABLE:
            raise BackendNotInstalledError(
                "sktime is not installed. Install with: pip install universal-ts[sktime]"
            )
        
        super().__init__(**kwargs)
        self.model_name = model
        self.model_kwargs = model_kwargs or {}
        self.models: Dict[str, Any] = {}  # For panel data: group_id -> model
        self.is_panel = False
        
        # Validate model name
        if model not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )
    
    def fit(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit sktime forecaster(s).
        
        For panel data, fits separate models for each series.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds' and 'y' columns
        group_id_col : str, optional
            Column name for group identifiers
        covariates : list of str, optional
            Names of exogenous variable columns
        **kwargs
            Additional fit parameters
        """
        self.group_id_col = group_id_col
        self.is_panel = group_id_col is not None
        self.covariates = covariates or []
        
        if self.is_panel:
            # Fit separate model for each group
            for group_id, group_df in df.groupby(group_id_col):
                model = self._create_model()
                
                # Prepare data
                y = group_df.set_index('ds')['y']
                X = group_df.set_index('ds')[self.covariates] if self.covariates else None
                
                # Fit model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if X is not None:
                        model.fit(y, X=X, **kwargs)
                    else:
                        model.fit(y, **kwargs)
                
                self.models[str(group_id)] = model
        else:
            # Single series
            model = self._create_model()
            
            # Prepare data
            y = df.set_index('ds')['y']
            X = df.set_index('ds')[self.covariates] if self.covariates else None
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if X is not None:
                    model.fit(y, X=X, **kwargs)
                else:
                    model.fit(y, **kwargs)
            
            self.models['_single_'] = model
        
        self.is_fitted = True
        
        # Infer frequency
        self.freq = pd.infer_freq(df['ds'])
    
    def _create_model(self) -> Any:
        """Create a new sktime model instance."""
        model_class = self.MODEL_REGISTRY[self.model_name]
        return model_class(**self.model_kwargs)
    
    def predict(
        self,
        horizon: int,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts using sktime.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        df_future : pd.DataFrame, optional
            Future exogenous variables
        **kwargs
            Additional predict parameters
            
        Returns
        -------
        pd.DataFrame
            Forecasts with columns: ds, yhat, [group_id]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        all_forecasts = []
        
        # Create forecasting horizon
        fh = ForecastingHorizon(range(1, horizon + 1), is_relative=True)
        
        if self.is_panel:
            # Predict for each group
            for group_id, model in self.models.items():
                # Prepare future exogenous variables
                X_future = None
                if df_future is not None and self.covariates:
                    group_future = df_future[df_future[self.group_id_col] == group_id]
                    if not group_future.empty:
                        X_future = group_future.set_index('ds')[self.covariates]
                
                # Generate forecast
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if X_future is not None:
                        y_pred = model.predict(fh=fh, X=X_future, **kwargs)
                    else:
                        y_pred = model.predict(fh=fh, **kwargs)
                
                # Convert to DataFrame
                forecast_df = pd.DataFrame({
                    'ds': y_pred.index,
                    'yhat': y_pred.values,
                    self.group_id_col: group_id
                })
                all_forecasts.append(forecast_df)
            
            result = pd.concat(all_forecasts, ignore_index=True)
        else:
            # Single series
            model = self.models['_single_']
            
            # Prepare future exogenous variables
            X_future = None
            if df_future is not None and self.covariates:
                X_future = df_future.set_index('ds')[self.covariates]
            
            # Generate forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if X_future is not None:
                    y_pred = model.predict(fh=fh, X=X_future, **kwargs)
                else:
                    y_pred = model.predict(fh=fh, **kwargs)
            
            # Convert to DataFrame
            result = pd.DataFrame({
                'ds': y_pred.index,
                'yhat': y_pred.values
            })
        
        return result
    
    def supports_panel_data(self) -> bool:
        """sktime supports panel data via separate models."""
        return True
    
    def supports_covariates(self) -> bool:
        """sktime supports exogenous variables."""
        return True
    
    def supports_probabilistic(self) -> bool:
        """Some sktime models support probabilistic forecasts."""
        # This depends on the specific model
        return False  # Conservative default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get sktime model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "num_models": len(self.models),
        })
        return info

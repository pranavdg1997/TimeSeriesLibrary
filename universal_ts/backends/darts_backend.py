"""Darts backend implementation."""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd
import warnings

from ..base import BaseBackendModel
from ..exceptions import BackendNotInstalledError

try:
    from darts import TimeSeries
    from darts.models import (
        NaiveSeasonal, ARIMA, ExponentialSmoothing,
        TiDEModel, NBEATSModel
    )
    DARTS_AVAILABLE = True
except ImportError:
    DARTS_AVAILABLE = False

from ..utils import detect_gpu


class DartsBackend(BaseBackendModel):
    """
    Darts backend for time series forecasting.
    
    Provides access to Darts forecasting models via a simple registry.
    
    Parameters
    ----------
    model : str, default="naive_seasonal"
        Model to use. Options: "naive_seasonal", "arima", "exponential_smoothing"
    model_kwargs : dict, optional
        Arguments passed to the Darts model constructor
    **kwargs
        Additional backend configuration
    """
    
    # Model registry
    MODEL_REGISTRY = {
        "naive_seasonal": NaiveSeasonal if DARTS_AVAILABLE else None,
        "arima": ARIMA if DARTS_AVAILABLE else None,
        "exponential_smoothing": ExponentialSmoothing if DARTS_AVAILABLE else None,
        "tide": TiDEModel if DARTS_AVAILABLE else None,
        "nbeats": NBEATSModel if DARTS_AVAILABLE else None,
    }
    
    # DL models that support GPU
    DL_MODELS = ["tide", "nbeats"]
    
    def __init__(
        self,
        model: str = "naive_seasonal",
        model_kwargs: Optional[Dict[str, Any]] = None,
        num_gpus: Optional[int] = None,
        **kwargs
    ):
        if not DARTS_AVAILABLE:
            raise BackendNotInstalledError(
                "Darts is not installed. Install with: pip install universal-ts[darts]"
            )
        
        super().__init__(**kwargs)
        self.model_name = model
        self.model_kwargs = model_kwargs or {}
        
        # Merge extra kwargs into model_kwargs
        self.model_kwargs.update(kwargs)
        self.models: Dict[str, Any] = {}  # For panel data: group_id -> model
        self.is_panel = False
        
        # Validate model name
        if model not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model '{model}'. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )
            
        # GPU configuration for DL models
        if model in self.DL_MODELS:
            if num_gpus is None:
                num_gpus = detect_gpu()
            self.num_gpus = num_gpus
            
            if self.num_gpus > 0:
                print(f"[darts] Using GPU for model '{model}'")
                # Add Lightning trainer kwargs for GPU
                if 'pl_trainer_kwargs' not in self.model_kwargs:
                    self.model_kwargs['pl_trainer_kwargs'] = {}
                self.model_kwargs['pl_trainer_kwargs'].update({
                    "accelerator": "gpu",
                    "devices": self.num_gpus
                })
            else:
                if num_gpus is not None and num_gpus > 0:
                     print(f"[darts] GPU requested but none detected or compatible. Falling back to CPU for '{model}'.")
                else:
                     print(f"[darts] Using CPU for model '{model}'")
                
                # Explicitly disable GPU to avoid auto-detection by Lightning
                if 'pl_trainer_kwargs' not in self.model_kwargs:
                    self.model_kwargs['pl_trainer_kwargs'] = {}
                self.model_kwargs['pl_trainer_kwargs'].update({
                    "accelerator": "cpu"
                })
        else:
            self.num_gpus = 0
    
    def fit(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit Darts model(s).
        
        For panel data, fits separate models for each series.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds' and 'y' columns
        group_id_col : str, optional
            Column name for group identifiers
        covariates : list of str, optional
            Names of covariate columns (past or future covariates)
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
                
                # Convert to Darts TimeSeries
                series = self._convert_to_timeseries(group_df)
                
                # Prepare covariates if any
                covariates_ts = None
                if self.covariates:
                    covariates_ts = self._convert_covariates_to_timeseries(group_df)
                
                # Fit model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if covariates_ts is not None:
                        # Try to pass covariates (not all models support them)
                        try:
                            model.fit(series, future_covariates=covariates_ts, **kwargs)
                        except TypeError:
                            # Model doesn't support covariates
                            model.fit(series, **kwargs)
                    else:
                        model.fit(series, **kwargs)
                
                self.models[str(group_id)] = model
        else:
            # Single series
            model = self._create_model()
            
            # Convert to Darts TimeSeries
            series = self._convert_to_timeseries(df)
            
            # Prepare covariates if any
            covariates_ts = None
            if self.covariates:
                covariates_ts = self._convert_covariates_to_timeseries(df)
            
            # Fit model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if covariates_ts is not None:
                    try:
                        model.fit(series, future_covariates=covariates_ts, **kwargs)
                    except TypeError:
                        model.fit(series, **kwargs)
                else:
                    model.fit(series, **kwargs)
            
            self.models['_single_'] = model
        
        self.is_fitted = True
        
        # Infer frequency
        self.freq = pd.infer_freq(df['ds'])
    
    def _create_model(self) -> Any:
        """Create a new Darts model instance."""
        model_class = self.MODEL_REGISTRY[self.model_name]
        return model_class(**self.model_kwargs)
    
    def _convert_to_timeseries(self, df: pd.DataFrame) -> TimeSeries:
        """Convert DataFrame to Darts TimeSeries."""
        return TimeSeries.from_dataframe(
            df[['ds', 'y']],
            time_col='ds',
            value_cols='y'
        )
    
    def _convert_covariates_to_timeseries(self, df: pd.DataFrame) -> TimeSeries:
        """Convert covariate columns to Darts TimeSeries."""
        if not self.covariates:
            return None
        
        return TimeSeries.from_dataframe(
            df[['ds'] + self.covariates],
            time_col='ds',
            value_cols=self.covariates
        )
    
    def predict(
        self,
        horizon: int,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts using Darts.
        
        Parameters
        ----------
        horizon : int
            Number of periods to forecast
        df_future : pd.DataFrame, optional
            Future covariates
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
        
        if self.is_panel:
            # Predict for each group
            for group_id, model in self.models.items():
                # Prepare future covariates
                future_covariates = None
                if df_future is not None and self.covariates:
                    group_future = df_future[df_future[self.group_id_col] == group_id]
                    if not group_future.empty:
                        future_covariates = self._convert_covariates_to_timeseries(group_future)
                
                # Generate forecast
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if future_covariates is not None:
                        try:
                            forecast = model.predict(n=horizon, future_covariates=future_covariates, **kwargs)
                        except TypeError:
                            forecast = model.predict(n=horizon, **kwargs)
                    else:
                        forecast = model.predict(n=horizon, **kwargs)
                
                # Convert to DataFrame
                forecast_df = forecast.to_dataframe().reset_index()
                forecast_df.columns = ['ds', 'yhat']
                forecast_df[self.group_id_col] = group_id
                all_forecasts.append(forecast_df)
            
            result = pd.concat(all_forecasts, ignore_index=True)
        else:
            # Single series
            model = self.models['_single_']
            
            # Prepare future covariates
            future_covariates = None
            if df_future is not None and self.covariates:
                future_covariates = self._convert_covariates_to_timeseries(df_future)
            
            # Generate forecast
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if future_covariates is not None:
                    try:
                        forecast = model.predict(n=horizon, future_covariates=future_covariates, **kwargs)
                    except TypeError:
                        forecast = model.predict(n=horizon, **kwargs)
                else:
                    forecast = model.predict(n=horizon, **kwargs)
            
            # Convert to DataFrame
            result = forecast.to_dataframe().reset_index()
            result.columns = ['ds', 'yhat']
        
        return result
    
    def supports_panel_data(self) -> bool:
        """Darts supports panel data via separate models."""
        return True
    
    def supports_covariates(self) -> bool:
        """Some Darts models support covariates."""
        return True
    
    def supports_probabilistic(self) -> bool:
        """Some Darts models support probabilistic forecasts."""
        return False  # Conservative default
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Darts model information."""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "model_kwargs": self.model_kwargs,
            "num_models": len(self.models),
        })
        return info

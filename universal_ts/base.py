"""Base abstract class for all forecasting backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any
import pandas as pd


class BaseBackendModel(ABC):
    """
    Abstract base class for all forecasting backend implementations.
    
    All backend models must implement the fit and predict methods,
    and declare their capabilities via property methods.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the backend model.
        
        Parameters
        ----------
        **kwargs
            Backend-specific configuration parameters
        """
        self.is_fitted = False
        self.freq = None
        self.group_id_col = None
        self.target_col = "y"
        self.time_col = "ds"
        
    @abstractmethod
    def fit(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit the forecasting model to training data.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data with at minimum:
            - 'ds': datetime column
            - 'y': target values
            - Optional: group_id column for panel data
            - Optional: covariate columns
        group_id_col : str, optional
            Name of the column containing group/series identifiers
            for panel data
        covariates : list of str, optional
            Names of covariate columns to use as regressors
        **kwargs
            Additional backend-specific parameters
        """
        pass
    
    @abstractmethod
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
            Future values of covariates/regressors. Must contain
            'ds' column and any required covariate columns.
        **kwargs
            Additional backend-specific parameters
            
        Returns
        -------
        pd.DataFrame
            Forecasts with columns:
            - 'ds': forecast timestamps
            - 'yhat': point forecasts
            - Optional: 'group_id' if panel data
            - Optional: 'yhat_lower', 'yhat_upper' for prediction intervals
            - Optional: quantile columns if probabilistic forecasts
        """
        pass
    
    @abstractmethod
    def supports_panel_data(self) -> bool:
        """
        Whether this backend natively supports panel/multi-series data.
        
        Returns
        -------
        bool
            True if backend can handle multiple series in one fit call
        """
        pass
    
    @abstractmethod
    def supports_covariates(self) -> bool:
        """
        Whether this backend supports exogenous covariates/regressors.
        
        Returns
        -------
        bool
            True if backend can use additional features
        """
        pass
    
    @abstractmethod
    def supports_probabilistic(self) -> bool:
        """
        Whether this backend supports probabilistic forecasts.
        
        Returns
        -------
        bool
            True if backend can generate prediction intervals/quantiles
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns
        -------
        dict
            Model metadata and configuration
        """
        return {
            "is_fitted": self.is_fitted,
            "freq": self.freq,
            "group_id_col": self.group_id_col,
            "supports_panel": self.supports_panel_data(),
            "supports_covariates": self.supports_covariates(),
            "supports_probabilistic": self.supports_probabilistic(),
        }

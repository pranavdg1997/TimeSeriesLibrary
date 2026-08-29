"""AutoGluon TimeSeries backend implementation."""

from __future__ import annotations
from typing import Optional, List, Dict, Any
import pandas as pd
import warnings

from ..base import BaseBackendModel
from ..exceptions import BackendNotInstalledError
from ..utils import detect_gpu

try:
    from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False


class AutoGluonBackend(BaseBackendModel):
    """
    AutoGluon TimeSeries backend for time series forecasting.

    Supports panel data natively via AutoGluon's TimeSeriesDataFrame.

    Parameters
    ----------
    prediction_length : int, optional
        Forecast horizon. If not provided, will be set during fit.
    eval_metric : str, default="WQL"
        Evaluation metric for model selection
    num_gpus : int, optional
        Number of GPUs to use for training. If None, will auto-detect.
        Set to 0 to force CPU-only training.
    **kwargs
        Additional arguments passed to TimeSeriesPredictor
    """

    def __init__(
        self,
        prediction_length: Optional[int] = None,
        eval_metric: str = "WQL",
        num_gpus: Optional[int] = None,
        **kwargs
    ):
        if not AUTOGLUON_AVAILABLE:
            raise BackendNotInstalledError(
                "AutoGluon is not installed. "
                "Install with: pip install universal-ts[autogluon]"
            )

        super().__init__(**kwargs)
        self.prediction_length = prediction_length
        self.eval_metric = eval_metric

        # Store kwargs
        self.ag_kwargs = kwargs.copy()

        # GPU configuration
        if num_gpus is None:
            # Auto-detect GPU
            num_gpus = detect_gpu()
        self.num_gpus = num_gpus

        if self.num_gpus > 0:
            print(f"[autogluon] Use {self.num_gpus} GPU(s)")
        else:
            print("[autogluon] No GPU detected. Using CPU only.")

        self.predictor: Optional[TimeSeriesPredictor] = None

    def fit(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str] = None,
        covariates: Optional[List[str]] = None,
        **kwargs
    ) -> None:
        """
        Fit AutoGluon TimeSeriesPredictor.

        Parameters
        ----------
        df : pd.DataFrame
            Training data with 'ds' and 'y' columns
        group_id_col : str, optional
            Column name for item/series identifiers
        covariates : list of str, optional
            Names of known covariates columns
        **kwargs
            Additional fit parameters (e.g., time_limit, presets)
        """
        self.group_id_col = group_id_col
        self.covariates = covariates or []

        # Convert to AutoGluon TimeSeriesDataFrame format
        ts_df = self._convert_to_timeseries_dataframe(df, group_id_col)

        # Infer prediction length if not provided
        if self.prediction_length is None:
            # Use a reasonable default (10% of data length or 10, whichever is larger)
            if group_id_col:
                # Use first series for inference
                first_series = df[df[group_id_col] == df[group_id_col].iloc[0]]
                series_length = len(first_series)
            else:
                series_length = len(df)

            self.prediction_length = max(10, int(series_length * 0.1))
            warnings.warn(
                f"prediction_length not specified. Using {self.prediction_length}. "
                "Set prediction_length parameter to override.",
                UserWarning
            )

        # Create predictor
        self.predictor = TimeSeriesPredictor(
            target="y",
            prediction_length=self.prediction_length,
            eval_metric=self.eval_metric,
            known_covariates_names=self.covariates if self.covariates else None,
            **self.ag_kwargs
        )

        # Fit predictor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.predictor.fit(
                train_data=ts_df,
                **kwargs
            )

        self.is_fitted = True
        self.train_data = ts_df  # Store training data for prediction

        # Infer frequency
        self.freq = ts_df.freq

    def _convert_to_timeseries_dataframe(
        self,
        df: pd.DataFrame,
        group_id_col: Optional[str]
    ) -> TimeSeriesDataFrame:
        """
        Convert standard DataFrame to AutoGluon TimeSeriesDataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with 'ds', 'y', and optional group_id and covariates
        group_id_col : str, optional
            Name of the group ID column

        Returns
        -------
        TimeSeriesDataFrame
            AutoGluon format time series data
        """
        # Set target column
        self.target_col = "y"

        # Prepare columns
        columns_to_keep = ['ds', self.target_col]
        if self.covariates:
            columns_to_keep.extend(self.covariates)
        if group_id_col:
            columns_to_keep.insert(0, group_id_col)

        df_ag = df[columns_to_keep].copy()

        # Rename columns to AutoGluon convention
        rename_dict = {'ds': 'timestamp'}

        if group_id_col:
            rename_dict[group_id_col] = 'item_id'
        else:
            # Add a default item_id if none provided (required by AutoGluon)
            df_ag['item_id'] = 'item_0'

        df_ag = df_ag.rename(columns=rename_dict)

        # Convert to TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df_ag,
            id_column='item_id',
            timestamp_column='timestamp'
        )

        return ts_df

    def predict(
        self,
        horizon: int,
        df_future: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate forecasts using AutoGluon.

        Parameters
        ----------
        horizon : int
            Number of periods to forecast (must match prediction_length)
        df_future : pd.DataFrame, optional
            Future known covariates
        **kwargs
            Additional predict parameters (e.g., model, quantile_levels)

        Returns
        -------
        pd.DataFrame
            Forecasts with columns: ds, yhat, [group_id], [quantiles]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        if horizon != self.prediction_length:
            warnings.warn(
                f"AutoGluon was trained with prediction_length={self.prediction_length}, "
                f"but horizon={horizon} was requested. Using prediction_length.",
                UserWarning
            )

        # Prepare known covariates if provided
        known_covariates = None
        if df_future is not None and self.covariates:
            known_covariates = self._convert_to_timeseries_dataframe(
                df_future,
                self.group_id_col
            )

        # Generate predictions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictions = self.predictor.predict(
                data=self.train_data,
                known_covariates=known_covariates,
                **kwargs
            )

        # Convert predictions to standard format
        result = self._convert_predictions_to_dataframe(predictions)

        return result

    def _convert_predictions_to_dataframe(
        self,
        predictions: TimeSeriesDataFrame
    ) -> pd.DataFrame:
        """
        Convert AutoGluon predictions to standard DataFrame format.

        Parameters
        ----------
        predictions : TimeSeriesDataFrame
            AutoGluon predictions

        Returns
        -------
        pd.DataFrame
            Standard format with ds, yhat, [group_id], [quantiles]
        """
        # Reset index to get timestamp and item_id as columns
        df = predictions.reset_index()

        # Rename columns
        rename_dict = {'timestamp': 'ds', 'mean': 'yhat'}
        if 'item_id' in df.columns and self.group_id_col:
            rename_dict['item_id'] = self.group_id_col

        df = df.rename(columns=rename_dict)

        # Add prediction intervals if available
        if '0.1' in predictions.columns and '0.9' in predictions.columns:
            df['yhat_lower'] = predictions['0.1'].values
            df['yhat_upper'] = predictions['0.9'].values

        return df

    def supports_panel_data(self) -> bool:
        """AutoGluon natively supports panel data."""
        return True

    def supports_covariates(self) -> bool:
        """AutoGluon supports known covariates."""
        return True

    def supports_probabilistic(self) -> bool:
        """AutoGluon provides quantile forecasts."""
        return True

    def get_model_info(self) -> Dict[str, Any]:
        """Get AutoGluon model information."""
        info = super().get_model_info()

        if self.is_fitted and self.predictor:
            info.update({
                "prediction_length": self.prediction_length,
                "eval_metric": self.eval_metric,
                "leaderboard": self.predictor.leaderboard().to_dict()
                if hasattr(self.predictor, 'leaderboard') else None,
            })

        return info

"""Evaluation metrics for time series forecasting."""

from __future__ import annotations
from typing import Optional, List, Dict, Union
import numpy as np
import pandas as pd


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MAE value
    """
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        MSE value
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        RMSE value
    """
    return np.sqrt(mse(y_true, y_pred))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    epsilon : float, default=1e-10
        Small value to avoid division by zero
        
    Returns
    -------
    float
        MAPE value (as percentage, 0-100)
    """
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    epsilon : float, default=1e-10
        Small value to avoid division by zero
        
    Returns
    -------
    float
        sMAPE value (as percentage, 0-200)
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2 + epsilon
    return np.mean(numerator / denominator) * 100


def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonality: int = 1
) -> float:
    """
    Mean Absolute Scaled Error.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Predicted values
    y_train : np.ndarray
        Training data for computing the scaling factor
    seasonality : int, default=1
        Seasonal period for naive forecast baseline
        
    Returns
    -------
    float
        MASE value
    """
    # Compute MAE of predictions
    mae_pred = mae(y_true, y_pred)
    
    # Compute MAE of naive seasonal forecast on training data
    naive_forecast = y_train[:-seasonality] if seasonality > 0 else y_train[:-1]
    naive_actual = y_train[seasonality:] if seasonality > 0 else y_train[1:]
    mae_naive = mae(naive_actual, naive_forecast)
    
    # Avoid division by zero
    if mae_naive == 0:
        return np.inf if mae_pred > 0 else 0.0
    
    return mae_pred / mae_naive


def coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> float:
    """
    Coverage: proportion of true values within prediction intervals.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_lower : np.ndarray
        Lower bounds of prediction intervals
    y_upper : np.ndarray
        Upper bounds of prediction intervals
        
    Returns
    -------
    float
        Coverage proportion (0-1)
    """
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    return np.mean(within_interval)


# Metric registry
METRIC_FUNCTIONS = {
    "mae": mae,
    "mse": mse,
    "rmse": rmse,
    "mape": mape,
    "smape": smape,
    "mase": mase,
    "coverage": coverage,
}


def evaluate(
    ground_truth: pd.DataFrame,
    forecast: pd.DataFrame,
    metrics: List[str] = None,
    group_id_col: Optional[str] = None,
    time_col: str = "ds",
    target_col: str = "y",
    pred_col: str = "yhat",
    train_data: Optional[pd.DataFrame] = None,
    seasonality: int = 1
) -> pd.DataFrame:
    """
    Evaluate forecasts against ground truth.
    
    Parameters
    ----------
    ground_truth : pd.DataFrame
        Ground truth data with columns: ds, y, [group_id]
    forecast : pd.DataFrame
        Forecast data with columns: ds, yhat, [group_id], [yhat_lower, yhat_upper]
    metrics : list of str, optional
        Metrics to compute. Default: ["mae", "rmse", "mape"]
    group_id_col : str, optional
        Name of group ID column for panel data
    time_col : str, default="ds"
        Name of time column
    target_col : str, default="y"
        Name of target column in ground_truth
    pred_col : str, default="yhat"
        Name of prediction column in forecast
    train_data : pd.DataFrame, optional
        Training data required for MASE computation
    seasonality : int, default=1
        Seasonal period for MASE
        
    Returns
    -------
    pd.DataFrame
        Evaluation results with metrics per series (if panel) and overall
    """
    if metrics is None:
        metrics = ["mae", "rmse", "mape"]
    
    # Validate metrics
    invalid_metrics = set(metrics) - set(METRIC_FUNCTIONS.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unknown metrics: {invalid_metrics}. "
            f"Available: {list(METRIC_FUNCTIONS.keys())}"
        )
    
    # Merge ground truth and forecast
    merge_cols = [time_col]
    if group_id_col:
        merge_cols.append(group_id_col)
    
    merged = pd.merge(
        ground_truth[[*merge_cols, target_col]],
        forecast[[*merge_cols, pred_col] + 
                 ([col for col in ["yhat_lower", "yhat_upper"] if col in forecast.columns])],
        on=merge_cols,
        how="inner"
    )
    
    if merged.empty:
        raise ValueError("No matching timestamps between ground truth and forecast")
    
    results = []
    
    if group_id_col:
        # Compute metrics per series
        for group_id, group_df in merged.groupby(group_id_col):
            group_metrics = _compute_metrics(
                group_df[target_col].values,
                group_df[pred_col].values,
                metrics,
                group_df.get("yhat_lower"),
                group_df.get("yhat_upper"),
                train_data[train_data[group_id_col] == group_id][target_col].values 
                if train_data is not None and group_id_col in train_data.columns else None,
                seasonality
            )
            group_metrics[group_id_col] = group_id
            results.append(group_metrics)
        
        # Compute overall metrics
        overall_metrics = _compute_metrics(
            merged[target_col].values,
            merged[pred_col].values,
            metrics,
            merged.get("yhat_lower"),
            merged.get("yhat_upper"),
            train_data[target_col].values if train_data is not None else None,
            seasonality
        )
        overall_metrics[group_id_col] = "OVERALL"
        results.append(overall_metrics)
    else:
        # Single series
        overall_metrics = _compute_metrics(
            merged[target_col].values,
            merged[pred_col].values,
            metrics,
            merged.get("yhat_lower"),
            merged.get("yhat_upper"),
            train_data[target_col].values if train_data is not None else None,
            seasonality
        )
        results.append(overall_metrics)
    
    return pd.DataFrame(results)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: List[str],
    y_lower: Optional[pd.Series],
    y_upper: Optional[pd.Series],
    y_train: Optional[np.ndarray],
    seasonality: int
) -> Dict[str, float]:
    """Helper to compute metrics for a single series."""
    results = {}
    
    for metric_name in metrics:
        if metric_name == "mase":
            if y_train is None:
                results[metric_name] = np.nan
            else:
                results[metric_name] = mase(y_true, y_pred, y_train, seasonality)
        elif metric_name == "coverage":
            if y_lower is None or y_upper is None:
                results[metric_name] = np.nan
            else:
                results[metric_name] = coverage(
                    y_true,
                    y_lower.values,
                    y_upper.values
                )
        else:
            metric_func = METRIC_FUNCTIONS[metric_name]
            results[metric_name] = metric_func(y_true, y_pred)
    
    return results

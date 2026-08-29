"""Utility functions for data validation and preprocessing."""

from __future__ import annotations
from typing import Optional, List
import pandas as pd
import numpy as np
from .exceptions import DataValidationError


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    time_col: str = "ds",
    target_col: str = "y"
) -> None:
    """
    Validate that DataFrame has required structure for forecasting.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list of str, optional
        Additional required columns beyond ds and y
    time_col : str, default="ds"
        Name of the time column
    target_col : str, default="y"
        Name of the target column
        
    Raises
    ------
    DataValidationError
        If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected pandas DataFrame, got {type(df)}")
    
    if df.empty:
        raise DataValidationError("DataFrame is empty")
    
    # Check for required columns
    required = [time_col, target_col]
    if required_columns:
        required.extend(required_columns)
    
    missing = set(required) - set(df.columns)
    if missing:
        raise DataValidationError(
            f"Missing required columns: {missing}. "
            f"DataFrame must have at minimum '{time_col}' and '{target_col}' columns."
        )
    
    # Validate target column is numeric
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        raise DataValidationError(
            f"Target column '{target_col}' must be numeric, got {df[target_col].dtype}"
        )


def validate_datetime_column(
    df: pd.DataFrame,
    time_col: str = "ds",
    coerce: bool = True
) -> pd.DataFrame:
    """
    Validate and optionally coerce time column to datetime.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time column
    time_col : str, default="ds"
        Name of the time column
    coerce : bool, default=True
        If True, attempt to convert to datetime
        
    Returns
    -------
    pd.DataFrame
        DataFrame with datetime time column
        
    Raises
    ------
    DataValidationError
        If time column cannot be converted to datetime
    """
    df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        if coerce:
            try:
                df[time_col] = pd.to_datetime(df[time_col])
            except Exception as e:
                raise DataValidationError(
                    f"Could not convert '{time_col}' to datetime: {e}"
                )
        else:
            raise DataValidationError(
                f"Column '{time_col}' must be datetime, got {df[time_col].dtype}"
            )
    
    return df


def validate_monotonic_time(
    df: pd.DataFrame,
    time_col: str = "ds",
    group_id_col: Optional[str] = None,
    strict: bool = False
) -> None:
    """
    Validate that time index is monotonically increasing within each series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate
    time_col : str, default="ds"
        Name of the time column
    group_id_col : str, optional
        Name of the group ID column for panel data
    strict : bool, default=False
        If True, require strictly increasing (no duplicates)
        
    Raises
    ------
    DataValidationError
        If time is not monotonic
    """
    if group_id_col:
        # Check each group separately
        for group_id, group_df in df.groupby(group_id_col):
            _check_monotonic(group_df[time_col], group_id, strict)
    else:
        _check_monotonic(df[time_col], None, strict)


def _check_monotonic(
    time_series: pd.Series,
    group_id: Optional[str],
    strict: bool
) -> None:
    """Helper to check if a time series is monotonic."""
    is_monotonic = time_series.is_monotonic_increasing
    
    if strict:
        # Check for duplicates
        has_duplicates = time_series.duplicated().any()
        if has_duplicates or not is_monotonic:
            group_str = f" in group '{group_id}'" if group_id else ""
            raise DataValidationError(
                f"Time column is not strictly increasing{group_str}. "
                "Each timestamp must be unique and in ascending order."
            )
    elif not is_monotonic:
        group_str = f" in group '{group_id}'" if group_id else ""
        raise DataValidationError(
            f"Time column is not monotonically increasing{group_str}"
        )


def infer_frequency(
    df: pd.DataFrame,
    time_col: str = "ds",
    group_id_col: Optional[str] = None
) -> Optional[str]:
    """
    Infer the frequency of the time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time column
    time_col : str, default="ds"
        Name of the time column
    group_id_col : str, optional
        Name of the group ID column. If provided, infer from first group.
        
    Returns
    -------
    str or None
        Inferred frequency string (pandas offset alias) or None if cannot infer
    """
    if group_id_col:
        # Use first group for inference
        first_group = df[group_id_col].iloc[0]
        subset = df[df[group_id_col] == first_group]
    else:
        subset = df
    
    try:
        freq = pd.infer_freq(subset[time_col])
        return freq
    except (ValueError, TypeError):
        return None


def sort_by_time(
    df: pd.DataFrame,
    time_col: str = "ds",
    group_id_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Sort DataFrame by time (and optionally group ID).
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to sort
    time_col : str, default="ds"
        Name of the time column
    group_id_col : str, optional
        Name of the group ID column
        
    Returns
    -------
    pd.DataFrame
        Sorted DataFrame
    """
    sort_cols = [group_id_col, time_col] if group_id_col else [time_col]
    return df.sort_values(sort_cols).reset_index(drop=True)


def handle_missing_values(
    df: pd.DataFrame,
    target_col: str = "y",
    method: str = "drop",
    fill_value: Optional[float] = None
) -> pd.DataFrame:
    """
    Handle missing values in the target column.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with potential missing values
    target_col : str, default="y"
        Name of the target column
    method : str, default="drop"
        Method to handle missing values:
        - "drop": remove rows with missing targets
        - "ffill": forward fill
        - "bfill": backward fill
        - "zero": fill with zeros
        - "value": fill with specified value
    fill_value : float, optional
        Value to use when method="value"
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    df = df.copy()
    
    if method == "drop":
        df = df.dropna(subset=[target_col])
    elif method == "ffill":
        df[target_col] = df[target_col].ffill()
    elif method == "bfill":
        df[target_col] = df[target_col].bfill()
    elif method == "zero":
        df[target_col] = df[target_col].fillna(0)
    elif method == "value":
        if fill_value is None:
            raise ValueError("fill_value must be provided when method='value'")
        df[target_col] = df[target_col].fillna(fill_value)
    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: drop, ffill, bfill, zero, value"
        )
    
    return df


def prepare_dataframe(
    df: pd.DataFrame,
    time_col: str = "ds",
    target_col: str = "y",
    group_id_col: Optional[str] = None,
    freq: Optional[str] = None,
    handle_missing: str = "drop"
) -> tuple[pd.DataFrame, Optional[str]]:
    """
    Prepare DataFrame for forecasting: validate, sort, handle missing values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    time_col : str, default="ds"
        Name of the time column
    target_col : str, default="y"
        Name of the target column
    group_id_col : str, optional
        Name of the group ID column
    freq : str, optional
        Frequency string. If None, will attempt to infer.
    handle_missing : str, default="drop"
        How to handle missing values in target
        
    Returns
    -------
    pd.DataFrame
        Prepared DataFrame
    str or None
        Inferred or provided frequency
    """
    # Validate basic structure
    validate_dataframe(df, time_col=time_col, target_col=target_col)
    
    # Convert time column to datetime
    df = validate_datetime_column(df, time_col=time_col)
    
    # Sort by time
    df = sort_by_time(df, time_col=time_col, group_id_col=group_id_col)
    
    # Validate monotonic time
    validate_monotonic_time(df, time_col=time_col, group_id_col=group_id_col)
    
    # Handle missing values
    df = handle_missing_values(df, target_col=target_col, method=handle_missing)
    
    # Infer frequency if not provided
    if freq is None:
        freq = infer_frequency(df, time_col=time_col, group_id_col=group_id_col)
    
    return df, freq


def detect_gpu() -> int:
    """
    Detect available compatible GPUs for forecasting backends.
    
    Checks for PyTorch CUDA availability and compatibility with the 
    installed PyTorch version (to avoid 'no kernel image' errors on 
    newer GPUs like RTX 50-series).
    
    Returns
    -------
    int
        Number of available and compatible GPUs.
    """
    gpu_count = 0
    
    # Try PyTorch first (most common)
    try:
        import torch
        import warnings
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            
            # Check for compatibility (e.g. Blackwell sm_120)
            if gpu_count > 0:
                try:
                    capability = torch.cuda.get_device_capability(0)
                    major, minor = capability
                    arch = f"sm_{major}{minor}"
                    supported_arches = torch.cuda.get_arch_list()
                    
                    if arch not in supported_arches:
                        device_name = torch.cuda.get_device_name(0)
                        warnings.warn(
                            f"GPU {device_name} with capability {arch} is not compatible "
                            f"with current PyTorch installation (supported arches: {supported_arches}). "
                            "Auto-disabling GPU to avoid runtime errors.",
                            UserWarning
                        )
                        return 0
                except Exception:
                    # If check fails, assume it's okay but be cautious
                    pass
            
            return gpu_count
    except ImportError:
        pass
    
    # Try MXNet
    try:
        import mxnet as mx
        gpu_count = mx.context.num_gpus()
        return gpu_count
    except:
        pass
    
    return 0

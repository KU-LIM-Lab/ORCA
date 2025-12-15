"""Preprocessing tools for data preprocessing pipeline.

These tools are pure functions that operate on DataFrames.
They do not handle Redis I/O or database queries - that's handled by the agent.
All tools take a DataFrame as input and return a processed DataFrame.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter


def clean_nulls_tool(
    df: pd.DataFrame,
    null_ratio: float = 0.95
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Drop columns with high percentage of null values.
    
    Args:
        df: Input DataFrame
        null_ratio: Threshold ratio (0.0-1.0). Columns with >null_ratio nulls are dropped.
                   Example: 0.95 means keep columns with at least 5% non-null values.
    
    Returns:
        Tuple of (cleaned DataFrame, list of dropped column names)
    """
    if df is None or df.empty:
        return df, []
    
    # Clamp ratio to valid range
    ratio = min(max(null_ratio, 0.0), 1.0)
    thresh = max(int(len(df) * ratio), 1)
    
    cleaned_df = df.dropna(axis=1, thresh=thresh)
    dropped_cols = [col for col in df.columns if col not in cleaned_df.columns]
    
    return cleaned_df, dropped_cols


def encode_categorical_tool(
    df: pd.DataFrame,
    threshold: int = 20
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Perform one-hot encoding on low-cardinality categorical columns.
    
    Args:
        df: Input DataFrame
        threshold: Maximum cardinality for columns to be one-hot encoded
    
    Returns:
        Tuple of (encoded DataFrame, list of encoded column names)
    """
    if df is None or df.empty:
        return df, []
    
    threshold = max(int(threshold), 1)
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    
    # Filter by cardinality
    low_card_cols = [
        col for col in categorical_cols
        if df[col].nunique(dropna=True) <= threshold
    ]
    
    if low_card_cols:
        encoded_df = pd.get_dummies(df, columns=low_card_cols, dummy_na=False)
    else:
        encoded_df = df.copy()
    
    return encoded_df, low_card_cols


def detect_schema_tool(
    df: pd.DataFrame,
    high_cardinality_threshold: int = 50
) -> Dict[str, Any]:
    """
    Detect data types and calculate cardinality for each variable.
    
    Args:
        df: Input DataFrame
        high_cardinality_threshold: Threshold for high cardinality detection
    
    Returns:
        Schema dictionary with variable information
    """
    if df is None or df.empty:
        return {
            "variables": {},
            "mixed_data_types": False,
            "high_cardinality_vars": [],
            "statistics": {
                "n_continuous": 0,
                "n_categorical": 0,
                "n_binary": 0
            }
        }
    
    if df.columns.duplicated().any():
        def _make_unique_columns(cols):
            counts = Counter()
            new_cols = []
            for c in cols:
                counts[c] += 1
                new_cols.append(c if counts[c] == 1 else f"{c}__{counts[c]-1}")
            return new_cols
        df = df.copy()
        df.columns = _make_unique_columns(df.columns)
    
    variables = {}
    n_continuous = 0
    n_categorical = 0
    n_binary = 0
    high_cardinality_vars = []
    
    for col in df.columns:
        var_info = {
            "missing_ratio": float(df[col].isna().sum() / len(df)),
            "cardinality": None,
            "unique_values": None  # For HITL confirmation
        }
        
        # Get unique values (excluding NaN)
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        # Detect data type
        # Check bool type first (pandas bool is considered numeric, but should be treated as categorical)
        if pd.api.types.is_bool_dtype(df[col]):
            # Boolean column - always binary categorical
            var_info["data_type"] = "Binary"
            var_info["cardinality"] = 2
            var_info["unique_values"] = sorted([bool(v) for v in unique_vals])
            n_binary += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column
            # Float types are always continuous (even with few unique values)
            if pd.api.types.is_float_dtype(df[col]):
                var_info["data_type"] = "Continuous"
                n_continuous += 1
            # Integer types: check if truly binary (0/1 or boolean-like)
            elif pd.api.types.is_integer_dtype(df[col]):
                if n_unique == 2:
                    # Check if values are 0/1 or boolean-like
                    unique_list = sorted(unique_vals.tolist())
                    is_binary = (
                        unique_list == [0, 1] or
                        unique_list == [0.0, 1.0] or
                        (len(unique_list) == 2 and all(v in [0, 1, True, False] for v in unique_list))
                    )
                    if is_binary:
                        var_info["data_type"] = "Binary"
                        var_info["cardinality"] = 2
                        var_info["unique_values"] = unique_list
                        n_binary += 1
                    else:
                        # Integer with 2 unique values but not 0/1 -> likely continuous with small sample
                        var_info["data_type"] = "Continuous"
                        n_continuous += 1
                elif n_unique <= 10:
                    # If range is small (e.g., 1-5) and sequential, might be ordinal
                    unique_list = sorted(unique_vals.tolist())
                    min_val, max_val = unique_list[0], unique_list[-1]
                    range_size = max_val - min_val + 1
                    
                    # If range is small and values are sequential, likely ordinal
                    if range_size <= 10 and range_size == n_unique:
                        var_info["data_type"] = "Ordinal"
                        var_info["cardinality"] = n_unique
                        var_info["unique_values"] = unique_list
                        n_categorical += 1
                    else:
                        # Likely continuous (count data, IDs, etc.) with few samples
                        var_info["data_type"] = "Continuous"
                        n_continuous += 1
                else:
                    # High cardinality integer -> continuous
                    var_info["data_type"] = "Continuous"
                    n_continuous += 1
            else:
                # Other numeric types (shouldn't happen often) -> continuous
                var_info["data_type"] = "Continuous"
                n_continuous += 1
        else:
            # Categorical column (object, category, string)
            var_info["cardinality"] = n_unique
            
            if n_unique == 2:
                # Binary categorical
                var_info["data_type"] = "Binary"
                var_info["unique_values"] = sorted([str(v) for v in unique_vals])
                n_binary += 1
            elif n_unique <= 10:
                # Low cardinality - store unique values for HITL confirmation
                var_info["unique_values"] = sorted([str(v) for v in unique_vals])
                # Heuristic for ordinal: check if values look like ordered categories
                # (e.g., "low", "medium", "high" or numeric strings)
                var_info["data_type"] = "Nominal"  # Default to nominal, can be updated by user in HITL
                n_categorical += 1
            else:
                # High cardinality categorical
                var_info["data_type"] = "Nominal"
                n_categorical += 1
            
            # Check for high cardinality
            if var_info["cardinality"] and var_info["cardinality"] > high_cardinality_threshold:
                high_cardinality_vars.append(col)
        
        variables[col] = var_info
    
    # Detect mixed data types
    has_continuous = n_continuous > 0
    has_categorical = (n_categorical + n_binary) > 0
    mixed_data_types = has_continuous and has_categorical
    
    return {
        "variables": variables,
        "mixed_data_types": mixed_data_types,
        "high_cardinality_vars": high_cardinality_vars,
        "statistics": {
            "n_continuous": n_continuous,
            "n_categorical": n_categorical,
            "n_binary": n_binary
        }
    }


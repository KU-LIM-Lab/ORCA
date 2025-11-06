"""Preprocessing tools for data preprocessing pipeline.

These tools are pure functions that operate on DataFrames.
They do not handle Redis I/O or database queries - that's handled by the agent.
All tools take a DataFrame as input and return a processed DataFrame.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


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
        if pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column
            if n_unique == 2:
                # Binary numeric
                var_info["data_type"] = "Binary"
                var_info["cardinality"] = 2
                var_info["unique_values"] = sorted(unique_vals.tolist())  # For HITL
                n_binary += 1
            elif n_unique <= 10:
                # Low cardinality numeric - could be ordinal or binary
                # Heuristic: if values are integers in a small range, treat as ordinal
                if pd.api.types.is_integer_dtype(df[col]):
                    var_info["data_type"] = "Ordinal"
                    var_info["cardinality"] = n_unique
                    var_info["unique_values"] = sorted(unique_vals.tolist())  # For HITL
                    n_categorical += 1
                else:
                    var_info["data_type"] = "Continuous"
                    n_continuous += 1
            else:
                # Continuous numeric
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


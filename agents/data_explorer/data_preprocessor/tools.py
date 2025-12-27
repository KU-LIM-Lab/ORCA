"""Preprocessing tools for data preprocessing pipeline.

These tools are pure functions that operate on DataFrames.
They do not handle Redis I/O or database queries - that's handled by the agent.
All tools take a DataFrame as input and return a processed DataFrame.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np
from collections import Counter
import re

def coerce_df_to_numeric(
    df: pd.DataFrame,
    dropna: bool = True,
    datetime_unit: str = "s",
    verbose: bool = True,
    # NEW:
    drop_text_cols: bool = True,
    text_unique_ratio_thresh: float = 0.5,   # high-cardinality text
    text_avg_len_thresh: int = 30,           # long strings
    text_avg_tokens_thresh: float = 3.0,     # "sentence-like"
    treat_id_like_as_text: bool = True,      # uuid-ish / long ids
) -> pd.DataFrame:
    TEXTY_NAME_PAT = re.compile(r"(desc|description|memo|note|comment|content|text|message)", re.I)

    df_out = df.copy()

    to_drop = []

    for col in df_out.columns:
        s = df_out[col]

        # 1) Boolean → int
        if pd.api.types.is_bool_dtype(s):
            df_out[col] = s.astype(int)
            if verbose: print(f"[bool -> int] {col}")
            continue

        # 2) Datetime → timestamp
        if pd.api.types.is_datetime64_any_dtype(s):
            factor = 1e9 if datetime_unit == "s" else 1e6
            df_out[col] = s.astype("int64") / factor
            if verbose: print(f"[datetime -> ts] {col}")
            continue

        # 3) Categorical → codes
        if pd.api.types.is_categorical_dtype(s):
            df_out[col] = s.cat.codes
            if verbose: print(f"[category -> codes] {col}")
            continue

        # 4) Object/string handling
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            s_str = s.astype("string")

            # try numeric first (same idea as before)
            s_clean = (
                s_str
                .str.replace(",", "", regex=False)
                .str.replace("₩", "", regex=False)
                .str.strip()
            )
            numeric = pd.to_numeric(s_clean, errors="coerce")

            if numeric.notna().mean() > 0.8:
                df_out[col] = numeric
                if verbose: print(f"[object -> numeric] {col}")
                continue

            # Heuristics: decide "free-text" vs "category"
            non_na = s_clean.dropna()
            n = len(non_na)

            # if empty column, just coerce
            if n == 0:
                df_out[col] = pd.to_numeric(s_clean, errors="coerce")
                if verbose: print(f"[empty object -> numeric coerce] {col}")
                continue

            nunique = non_na.nunique(dropna=True)
            unique_ratio = nunique / n

            avg_len = non_na.str.len().mean()
            avg_tokens = non_na.str.split().map(len).mean()

            # uuid/id-like heuristic (optional)
            id_like = False
            if treat_id_like_as_text:
                # many long tokens with dashes or hex-ish patterns
                sample = non_na.head(200)
                hexish = sample.str.fullmatch(r"[0-9a-fA-F-]{16,}").mean()
                id_like = (hexish > 0.5) or (avg_len >= 24 and unique_ratio > 0.8)

            name_texty = bool(TEXTY_NAME_PAT.search(col))

            is_text = (
                name_texty
                or id_like
                or unique_ratio >= text_unique_ratio_thresh
                or avg_len >= text_avg_len_thresh
                or avg_tokens >= text_avg_tokens_thresh
            )

            if drop_text_cols and is_text:
                to_drop.append(col)
                if verbose:
                    print(f"[drop text-like col] {col} "
                          f"(unique_ratio={unique_ratio:.2f}, avg_len={avg_len:.1f}, avg_tokens={avg_tokens:.1f})")
                continue

            # otherwise treat as categorical
            df_out[col] = pd.Categorical(s_clean).codes
            if verbose:
                print(f"[object -> categorical codes] {col} "
                      f"(unique_ratio={unique_ratio:.2f}, avg_len={avg_len:.1f}, avg_tokens={avg_tokens:.1f})")
            continue

        # 5) Everything else → force numeric
        df_out[col] = pd.to_numeric(s, errors="coerce")
        if verbose: print(f"[forced numeric] {col}")

    if to_drop:
        df_out = df_out.drop(columns=to_drop)

    if dropna:
        before = len(df_out)
        if verbose:
            print(df_out.head(3))
            print(f"[Data row]:{len(df_out)}")
            # Show which columns contain NaNs and how many before dropping rows.
            na_counts = df_out.isna().sum()
            na_cols = na_counts[na_counts > 0].sort_values(ascending=False)
            if not na_cols.empty:
                print("[NaN columns]\n" + na_cols.to_string())
            else:
                print("[NaN columns] None")
        df_out = df_out.dropna()
        if verbose: print(f"Dropped {before - len(df_out)} rows due to NaN")

    # Cap rows to 1000 with random sampling to keep downstream methods tractable.
    if len(df_out) > 1000:
        df_out = df_out.sample(n=1000, random_state=42)
        if verbose:
            print("[Row cap] Sampled 2000 rows")

    return df_out

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
    
    # cleaned_df = df.dropna(axis=1, thresh=thresh)
    cleaned_df = coerce_df_to_numeric(df)
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
            "unique_values": None  
        }
        
        # Get unique values (excluding NaN)
        unique_vals = df[col].dropna().unique()
        n_unique = len(unique_vals)
        
        # Detect data type
        # Check bool type first
        if pd.api.types.is_bool_dtype(df[col]):
            # Boolean column
            var_info["data_type"] = "Binary"
            var_info["cardinality"] = 2
            var_info["unique_values"] = sorted([bool(v) for v in unique_vals])
            n_binary += 1
        elif pd.api.types.is_numeric_dtype(df[col]):
            # Numeric column
            # Float types
            if pd.api.types.is_float_dtype(df[col]):
                var_info["data_type"] = "Continuous"
                n_continuous += 1
            # Integer types
            elif pd.api.types.is_integer_dtype(df[col]):
                unique_list = sorted(unique_vals.tolist())

                if n_unique == 1:
                    var_info["data_type"] = "Ordinal"
                    var_info["cardinality"] = 1
                    var_info["unique_values"] = unique_list
                    n_categorical += 1

                # (1) 0/1 -> Binary
                elif n_unique == 2:
                    is_binary = set(unique_list).issubset({0, 1, True, False, 0.0, 1.0})
                    if is_binary:
                        var_info["data_type"] = "Binary"
                        var_info["cardinality"] = 2
                        var_info["unique_values"] = unique_list
                        n_binary += 1
                    else:
                        var_info["data_type"] = "Nominal"
                        var_info["cardinality"] = 2
                        var_info["unique_values"] = unique_list
                        n_categorical += 1

                else:
                    # (2) low/mid cardinality
                    ORDINAL_MAX_UNIQUE = 30  
                    LOW_MAX_UNIQUE = 10

                    min_val, max_val = unique_list[0], unique_list[-1]
                    range_size = max_val - min_val + 1
                    is_sequential = (range_size == n_unique)

                    if n_unique <= LOW_MAX_UNIQUE:
                        if is_sequential:
                            var_info["data_type"] = "Ordinal"
                        else:
                            var_info["data_type"] = "Nominal"
                        var_info["cardinality"] = n_unique
                        var_info["unique_values"] = unique_list
                        n_categorical += 1

                    elif n_unique <= ORDINAL_MAX_UNIQUE and is_sequential:
                        var_info["data_type"] = "Ordinal"
                        var_info["cardinality"] = n_unique
                        var_info["unique_values"] = unique_list
                        n_categorical += 1

                    else:
                        var_info["data_type"] = "Continuous"
                        n_continuous += 1
                        
        else:
            # Other data types
            var_info["cardinality"] = n_unique
            
            if n_unique == 2:
                # Binary
                var_info["data_type"] = "Binary"
                var_info["unique_values"] = sorted([str(v) for v in unique_vals])
                n_binary += 1
            elif n_unique <= 10:
                var_info["unique_values"] = sorted([str(v) for v in unique_vals])
                var_info["data_type"] = "Nominal" 
                n_categorical += 1
            else:
                var_info["data_type"] = "Nominal"
                n_categorical += 1
            
            # High cardinality check
            if var_info["cardinality"] and var_info["cardinality"] > high_cardinality_threshold:
                high_cardinality_vars.append(col)
        
        variables[col] = var_info
    
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

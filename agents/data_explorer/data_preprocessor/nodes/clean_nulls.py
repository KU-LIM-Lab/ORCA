from __future__ import annotations

from typing import Dict
import pandas as pd
from utils.redis_df import load_df_parquet, save_df_parquet

def coerce_df_to_numeric(
    df: pd.DataFrame,
    dropna: bool = True,
    datetime_unit: str = "s",   # "s" or "ms"
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Convert a DataFrame into a fully numeric DataFrame
    suitable for statsmodels / DoWhy / sklearn.

    Rules:
    - bool        -> int (0/1)
    - category    -> codes
    - object      -> try numeric, otherwise categorical codes
    - datetime    -> unix timestamp
    - remaining   -> pd.to_numeric(coerce)

    Parameters
    ----------
    dropna : bool
        Whether to drop rows with NaN after conversion
    datetime_unit : {"s", "ms"}
        Unit for datetime conversion
    """

    df_out = df.copy()

    for col in df_out.columns:
        s = df_out[col]

        # 1) Boolean → int
        if pd.api.types.is_bool_dtype(s):
            df_out[col] = s.astype(int)
            if verbose:
                print(f"[bool -> int] {col}")
            continue

        # 2) Datetime → timestamp
        if pd.api.types.is_datetime64_any_dtype(s):
            factor = 1e9 if datetime_unit == "s" else 1e6
            df_out[col] = s.astype("int64") / factor
            if verbose:
                print(f"[datetime -> ts] {col}")
            continue

        # 3) Categorical → codes
        if pd.api.types.is_categorical_dtype(s):
            df_out[col] = s.cat.codes
            if verbose:
                print(f"[category -> codes] {col}")
            continue

        # 4) Object → try numeric → fallback to categorical
        if pd.api.types.is_object_dtype(s):
            # strip common noise
            s_clean = (
                s.astype(str)
                 .str.replace(",", "", regex=False)
                 .str.replace("₩", "", regex=False)
                 .str.strip()
            )
            numeric = pd.to_numeric(s_clean, errors="coerce")

            # if "mostly numeric", treat as numeric
            if numeric.notna().mean() > 0.8:
                df_out[col] = numeric
                if verbose:
                    print(f"[object -> numeric] {col}")
            else:
                df_out[col] = pd.Categorical(s).codes
                if verbose:
                    print(f"[object -> categorical codes] {col}")
            continue

        # 5) Everything else → force numeric
        df_out[col] = pd.to_numeric(s, errors="coerce")
        if verbose:
            print(f"[forced numeric] {col}")

    if dropna:
        before = len(df_out)
        df_out = df_out.dropna()
        if verbose:
            print(f"Dropped {before - len(df_out)} rows due to NaN")

    return df_out


def clean_nulls_node(state: Dict) -> Dict:
    """Drop high-null columns from the cached DataFrame and persist the update."""

    redis_key = state.get("df_redis_key")
    if not redis_key:
        state.setdefault("warnings", []).append("clean_nulls: missing df_redis_key; run fetch step first.")
        return state

    force_refresh = bool(state.get("force_refresh"))
    if state.get("_done_clean_nulls") and not force_refresh:
        return state

    try:
        df = load_df_parquet(redis_key)
    except Exception as exc:
        state.setdefault("warnings", []).append(f"clean_nulls: failed to load cached dataframe: {exc}")
        return state

    if df is None or df.empty:
        state.setdefault("warnings", []).append("clean_nulls: cached dataframe is empty; skipping.")
        return state

    try:
        ratio = state.get("clean_nulls_ratio", 0.95)
        ratio = 0.0 if ratio is None else float(ratio)
        ratio = min(max(ratio, 0.0), 1.0)
        thresh = max(int(len(df) * ratio), 1)

        cleaned_df = coerce_df_to_numeric(df)
        dropped_cols = [col for col in df.columns if col not in cleaned_df.columns]

        # Persist updated DataFrame under a derived Redis key to keep history of steps
        processed_key = f"{redis_key}:clean_nulls"
        save_df_parquet(processed_key, cleaned_df)

        state["df_redis_key"] = processed_key
        state["df_shape"] = tuple(cleaned_df.shape)
        state["columns"] = list(cleaned_df.columns)
        state["dropped_null_columns"] = dropped_cols
        state["df_cached"] = True
        state["df_raw"] = None
        state["_done_clean_nulls"] = True

        print(tuple(cleaned_df.shape))
    except Exception as exc:
        state.setdefault("warnings", []).append(f"clean_nulls failed: {exc}")

    return state

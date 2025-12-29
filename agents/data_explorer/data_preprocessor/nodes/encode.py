from __future__ import annotations

from typing import Dict, List

import pandas as pd

from utils.redis_df import load_df_parquet, save_df_parquet


def _get_categorical_columns(df: pd.DataFrame) -> List[str]:
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return categorical


def encode_node(state: Dict) -> Dict:
    """Perform low-cardinality one-hot encoding on the cached DataFrame and persist it."""

    redis_key = state.get("df_redis_key")
    if not redis_key:
        state.setdefault("warnings", []).append("encode: missing df_redis_key; ensure fetch step persisted data.")
        return state

    force_refresh = bool(state.get("force_refresh"))
    if state.get("_done_encode") and not force_refresh:
        return state

    try:
        df = load_df_parquet(redis_key)
    except Exception as exc:
        state.setdefault("warnings", []).append(f"encode: failed to load cached dataframe: {exc}")
        return state

    if df is None or df.empty:
        state.setdefault("warnings", []).append("encode: cached dataframe is empty; skipping one-hot encoding.")
        return state

    try:
        threshold = state.get("one_hot_threshold", 20)
        threshold = int(threshold) if threshold is not None else 20
        threshold = max(threshold, 1)

        categorical_cols = _get_categorical_columns(df)
        low_card_cols = [col for col in categorical_cols if df[col].nunique(dropna=True) <= threshold]

        if low_card_cols:
            encoded_df = pd.get_dummies(df, columns=low_card_cols, dummy_na=False)
        else:
            encoded_df = df

        processed_key = f"{redis_key}:encode"
        save_df_parquet(processed_key, encoded_df)

        state["df_redis_key"] = processed_key
        state["df_shape"] = tuple(encoded_df.shape)
        state["columns"] = list(encoded_df.columns)
        state["encoded_columns"] = low_card_cols
        state["df_cached"] = True
        state["df_raw"] = None
        state["_done_encode"] = True
    except Exception as exc:
        state.setdefault("warnings", []).append(f"encode failed: {exc}")

    return state


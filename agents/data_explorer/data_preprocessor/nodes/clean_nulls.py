from __future__ import annotations

from typing import Dict

from utils.redis_df import load_df_parquet, save_df_parquet


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

        cleaned_df = df.dropna(axis=1, thresh=thresh)
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

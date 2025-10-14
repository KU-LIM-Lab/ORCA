from typing import Dict
import pandas as pd


def scale_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    if df is None:
        return state
    try:
        method = (state.get("scaling") or "none").lower()
        if method == "standard":
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            for c in num_cols:
                mu = df[c].mean()
                sigma = df[c].std() or 1.0
                df[c] = (df[c] - mu) / sigma
        elif method == "minmax":
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            for c in num_cols:
                mn = df[c].min()
                mx = df[c].max()
                rng = (mx - mn) or 1.0
                df[c] = (df[c] - mn) / rng
        # Only store df_raw if not in fetch_only mode to avoid msgpack serialization errors
        if not state.get("fetch_only", False):
            state["df_raw"] = df
        state["_done_scale"] = True
    except Exception as e:
        state.setdefault("warnings", []).append(f"scale failed: {e}")
    return state



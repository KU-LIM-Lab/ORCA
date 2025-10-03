from typing import Dict
import pandas as pd


def impute_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    if df is None:
        return state
    try:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna("__MISSING__")
        state["df_raw"] = df
        state["_done_impute"] = True
    except Exception as e:
        state.setdefault("warnings", []).append(f"impute failed: {e}")
    return state



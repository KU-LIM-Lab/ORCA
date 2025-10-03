from typing import Dict
import pandas as pd


def encode_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    if df is None:
        return state
    try:
        # Simple one-hot encoding for object dtype with low cardinality
        cat_cols = [c for c in df.columns if df[c].dtype == object]
        low_card = [c for c in cat_cols if df[c].nunique(dropna=True) <= (state.get("one_hot_threshold") or 20)]
        if low_card:
            df = pd.get_dummies(df, columns=low_card, dummy_na=False)
        state["df_raw"] = df
        state["_done_encode"] = True
    except Exception as e:
        state.setdefault("warnings", []).append(f"encode failed: {e}")
    return state



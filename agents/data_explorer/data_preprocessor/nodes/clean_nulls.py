from typing import Dict


def clean_nulls_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    if df is None:
        return state
    try:
        # Simple heuristic defaults; can be parameterized later
        thresh_col = int(0.95 * len(df))  # drop columns with >95% nulls
        df = df.dropna(axis=1, thresh=thresh_col)
        state["df_raw"] = df
        state["_done_clean_nulls"] = True
    except Exception as e:
        state.setdefault("warnings", []).append(f"clean_nulls failed: {e}")
    return state



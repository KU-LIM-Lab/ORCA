from typing import Dict


def report_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    report = {
        "rows": getattr(df, "shape", (0, 0))[0] if df is not None else 0,
        "cols": getattr(df, "shape", (0, 0))[1] if df is not None else 0,
        "warnings": state.get("warnings", [])
    }
    
    # Only store df_preprocessed if not in fetch_only mode to avoid msgpack serialization errors
    if not state.get("fetch_only", False):
        state["df_preprocessed"] = df
    else:
        state["df_preprocessed"] = None
    
    state["preprocess_report"] = report
    return state



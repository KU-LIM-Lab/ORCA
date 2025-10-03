from typing import Dict


def report_node(state: Dict) -> Dict:
    df = state.get("df_raw")
    report = {
        "rows": getattr(df, "shape", (0, 0))[0] if df is not None else 0,
        "cols": getattr(df, "shape", (0, 0))[1] if df is not None else 0,
        "warnings": state.get("warnings", [])
    }
    state["df_preprocessed"] = df
    state["preprocess_report"] = report
    return state



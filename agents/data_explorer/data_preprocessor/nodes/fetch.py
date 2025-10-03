from typing import Dict
from utils.database import Database
import pandas as pd

def fetch_node(state: Dict) -> Dict:
    """Fetch df_raw by executing final_sql if df_raw is not provided."""
    if state.get("df_raw") is None:
        final_sql = state.get("final_sql")
        if not final_sql:
            state.setdefault("warnings", []).append("No df_raw or final_sql provided; fetch skipped.")
            return state
        
        try:
            # DEBUG: expose final_sql and db_id to help identify which database and query are being executed
            final_sql = state.get("final_sql")
            db_id = state.get("db_id")
            print(f"[FETCH] Running final_sql: {final_sql}")
            print(f"[FETCH] Using db_id: {db_id}")

            db = Database()
            rows, columns = db.run_query(sql=final_sql, db_id=db_id)
            df = pd.DataFrame(rows, columns=columns)

            # Check if dataframe is valid
            if df is None:
                error_msg = "Query returned None - no data found"
                print(f"[FETCH] ERROR: {error_msg}")
                state.setdefault("warnings", []).append(error_msg)
                state["df_raw"] = None
                return state
            if hasattr(df, 'empty') and df.empty:
                error_msg = "Query returned empty dataframe"
                print(f"[FETCH] ERROR: {error_msg}")
                state.setdefault("warnings", []).append(error_msg)
                state["df_raw"] = None
                return state
            if not hasattr(df, 'shape'):
                error_msg = "Query result is not a valid dataframe"
                print(f"[FETCH] ERROR: {error_msg}")
                state.setdefault("warnings", []).append(error_msg)
                state["df_raw"] = None
                return state

            state["df_raw"] = df
            print(f"[FETCH] Successfully loaded dataframe with shape: {df.shape}")

        except Exception as e:
            error_msg = f"Failed to fetch data: {str(e)}"
            print(f"[FETCH] EXCEPTION: {error_msg}")
            state.setdefault("warnings", []).append(error_msg)
            # Do not raise: instead set df_raw to None and return state so caller can inspect warnings
            state["df_raw"] = None
            return state
    
    return state



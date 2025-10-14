from typing import Dict
from utils.database import Database
from utils.redis_client import redis_client
import pandas as pd
import io

def fetch_node(state: Dict) -> Dict:
    """Fetch/coerce to DataFrame; optionally persist to Redis; support fetch_only mode."""
    fetch_only = bool(state.get("fetch_only"))
    print(f"[FETCH] fetch_only flag: {fetch_only}")
    print(f"[FETCH] persist_to_redis: {state.get('persist_to_redis', False)}")
    local_df = None
    # If df_raw exists but is not a DataFrame, coerce using optional columns
    if state.get("df_raw") is not None and not hasattr(state.get("df_raw"), "shape"):
        try:
            columns = state.get("columns")
            local_df = pd.DataFrame(state.get("df_raw"), columns=columns if columns else None)
            # print(f"[FETCH] df_raw to DataFrame with shape: {local_df.shape}")
            # Avoid storing DataFrame in state when fetch_only to prevent msgpack errors
            if not fetch_only:
                state["df_raw"] = local_df
            else:
                state["df_raw"] = None
        except Exception as e:
            print(f"[FETCH] ERROR converting df_raw to DataFrame: {e}")

    if (state.get("df_raw") is None and local_df is None) or (
        state.get("df_raw") is not None and not hasattr(state.get("df_raw"), "shape")
    ):
        final_sql = state.get("final_sql")
        if not final_sql:
            state.setdefault("warnings", []).append("No df_raw or final_sql provided; fetch skipped.")
            return state
        
        try:
            # DEBUG: expose final_sql and db_id to help identify which database and query are being executed
            final_sql = state.get("final_sql")
            db_id = state.get("db_id")
            # print(f"[FETCH] Running final_sql: {final_sql}")
            # print(f"[FETCH] Using db_id: {db_id}")

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

            local_df = df
            state["columns"] = columns
            state["df_shape"] = getattr(df, "shape", None)
            if not fetch_only:
                state["df_raw"] = df
            else:
                state["df_raw"] = None
            print(f"[FETCH] Successfully loaded dataframe with shape: {df.shape}")

        except Exception as e:
            error_msg = f"Failed to fetch data: {str(e)}"
            print(f"[FETCH] EXCEPTION: {error_msg}")
            state.setdefault("warnings", []).append(error_msg)
            # Do not raise: instead set df_raw to None and return state so caller can inspect warnings
            state["df_raw"] = None
            return state
    
    # Persist to Redis if requested
    try:
        df = local_df if local_df is not None else state.get("df_raw")
        print(f"[FETCH] About to persist to Redis. df is None: {df is None}, has to_parquet: {hasattr(df, 'to_parquet') if df is not None else False}")
        if state.get("persist_to_redis") and df is not None and hasattr(df, "to_parquet"):
            sql = state.get("final_sql", "")
            db_id = state.get("db_id", "default")
            key = f"{db_id}:df:{abs(hash(sql))}"
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            redis_client.set(key, buf.getvalue())
            state["df_redis_key"] = key
            print(f"[FETCH] Stored DataFrame in Redis. Key: {key}")
            # If fetch_only, do not keep full df in state but save Redis key
            if state.get("fetch_only"):
                state["df_raw"] = None
                print(f"[FETCH] Cleared df_raw from state due to fetch_only=True")
            else:
                print(f"[FETCH] Keeping df_raw in state due to fetch_only=False")
        else:
            print(f"[FETCH] Not persisting to Redis. persist_to_redis: {state.get('persist_to_redis')}, df is None: {df is None}")
    except Exception as e:
        print(f"[FETCH] Redis persistence failed: {e}")
        state.setdefault("warnings", []).append(f"Persist to Redis failed: {e}")

    return state



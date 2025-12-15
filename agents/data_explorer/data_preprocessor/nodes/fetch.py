from typing import Dict
import pandas as pd
from utils.database import Database
from utils.redis_df import save_df_parquet, load_df_parquet

database = Database()


def fetch_node(state: Dict) -> Dict:
    """Fetch raw data and persist it in Redis to avoid storing heavy frames in state."""
    
    db_id = state.get("db_id")
    if not db_id:
        raise ValueError("Missing 'db_id' in state")

    final_sql = state.get("final_sql") or state.get("final_sql_query")
    if not final_sql:
        raise ValueError("No SQL query found in state. Provide 'final_sql' before fetch.")

    session_id = state.get("session_id", "default_session")
    redis_key = f"{db_id}:raw_df:{session_id}"
    force_refresh = bool(state.get("force_refresh")) # Redis에 저장이 되어있는지와는 별개로 sql로 다시 load하고 싶다면 True로 설정

    df = None
    loaded_from_cache = False

    if not force_refresh and redis_key:
        try:
            df = load_df_parquet(redis_key)
            loaded_from_cache = df is not None
        except Exception as e:
            state.setdefault("warnings", []).append(f"Failed to load cached dataframe: {e}")
            df = None

    if df is None:
        try:
            rows, columns = database.run_query(sql=final_sql, db_id=db_id)
            df = pd.DataFrame(rows, columns=columns)
        except Exception as e:
            raise RuntimeError(f"Failed to execute SQL query: {e}") from e

        if df is None or df.empty:
            state.setdefault("warnings", []).append("Query returned no rows.")

        try:
            save_df_parquet(redis_key, df)
        except Exception as e:
            state.setdefault("warnings", []).append(f"Failed to cache dataframe in Redis: {e}")
            redis_key = None

    state["df_redis_key"] = redis_key
    state["df_shape"] = tuple(df.shape) if df is not None else None
    state["columns"] = list(df.columns) if df is not None else None
    state["df_cached"] = loaded_from_cache and redis_key is not None

    print(tuple(df.shape))
    print(list(df.columns))

    return state


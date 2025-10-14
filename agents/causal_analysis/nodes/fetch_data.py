# causal_analysis/nodes/fetch_data.py

import pandas as pd
import json, re
from typing import Dict
from utils.database import Database 
from utils.redis_df import save_df_parquet
from utils.llm import call_llm
from prompts.causal_analysis_prompts import fix_sql_prompt 
from langchain_core.language_models.chat_models import BaseChatModel

database = Database()

def prepare_state(state: dict) -> dict:
    if state["variable_info"]:
        state["parsed_query"] = state["variable_info"]
    return state

def build_fetch_data_node(llm: BaseChatModel):
    def node(state: Dict) -> Dict:
        if "parsed_query" not in state:
            state = prepare_state(state)
        
        sql_query = state["sql_query"]
        schema_str = state["table_schema_str"]

        if not sql_query:
            raise ValueError("No SQL query found in state. Please run generate_sql_query_node first.")

        db_id = state["db_id"]
        if not db_id:
            raise ValueError("Missing 'db_id' in state")


        graph_nodes = state["causal_graph"]["nodes"]
        expression_dict = state["expression_dict"]
        expected_columns_base = [v.split(".")[-1] for v in graph_nodes]        
        
        expected_columns_base = [var.split('.')[-1] for var in graph_nodes]
             
        def run_and_validate_query(sql, db_id, expected_columns_base):
            rows, columns = database.run_query(sql=sql, db_id=db_id)
            df = pd.DataFrame(rows, columns=columns)

            missing = [col for col in expected_columns_base if col not in df.columns]
            if missing:
                raise ValueError(f"Missing expected columns in SQL result: {missing}")
            return df

        try:
            df = run_and_validate_query(sql_query, db_id, expected_columns_base)
        except Exception as e:
            last_error = str(e)
            for _ in range(3): # Retry up to 3 times
                revised_response = call_llm(
                    prompt=fix_sql_prompt,
                    # parser=fix_sql_parser,
                    variables={
                        "original_sql": sql_query,
                        "error_message": last_error,
                        "graph_nodes": graph_nodes,
                        "expression_dict": json.dumps(expression_dict, indent=2),
                        "table_schemas": schema_str,
                    },
                    llm=llm
                )
                # Extract SQL query from the response
                sql_match = re.search(r"```sql\s*(.*?)```", revised_response, re.DOTALL | re.IGNORECASE)
                if sql_match:
                    revised_query = sql_match.group(1).strip()
                else:
                    revised_query = revised_response.strip()
                
                try:
                    df = run_and_validate_query(revised_query, db_id, expected_columns_base)
                    state["sql_query"] = revised_query
                    break
                except Exception as second_error:
                    last_error = str(second_error)
            else:
                raise RuntimeError(f"SQL retry also failed: {last_error}")

        # Persist DataFrame to Redis and store only key in state to avoid serialization issues
        try:
            db_id = state.get("db_id", "default")
            session_id = state.get("session_id", "default_session")
            key = f"{db_id}:causal_analysis_df:{session_id}"
            save_df_parquet(key, df)
            state["df_redis_key"] = key
            state["df_shape"] = tuple(df.shape)
            state["columns"] = list(df.columns)
        except Exception:
            # Fallback: do not store raw df in state
            pass
        return state
    return node
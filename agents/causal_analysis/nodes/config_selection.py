# nodes/config_selection.py

from typing import Dict, List, Optional
import pandas as pd
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import RunnableLambda

from agents.causal_analysis.state import Strategy
from utils.llm import call_llm
from utils.redis_df import load_df_parquet

from prompts.causal_analysis_prompts import (
    causal_strategy_prompt,
    strategy_output_parser
)

def clean_var_names(vars: List[str]) -> List[str]:
    return [v.split(".")[-1] if isinstance(v, str) and "." in v else v for v in vars]

def build_config_selection_node(llm: BaseChatModel) -> RunnableLambda:
    """
    Selects a causal strategy (task, identification method, estimator, refuter) 
    using an LLM based on user question, variables, and data sample.
    """
    
    def _load_dataframe_from_state(state: Dict) -> Optional[pd.DataFrame]:
        # Load from df_redis_key
        redis_key = state.get("df_redis_key")
        if redis_key:
            try:
                df = load_df_parquet(redis_key)
                if df is not None:
                    # Cache in state for future use
                    state["df_preprocessed"] = df
                    return df
            except Exception as e:
                print(f"⚠️ Failed to load DataFrame from Redis key {redis_key}: {e}")
        
        return None

    def invoke(state: Dict) -> Dict:
        df = _load_dataframe_from_state(state)
        parsed_vars = state.get("parsed_query") or {}
        question = state.get("input", "")

        # Check if df is valid DataFrame
        if df is not None and hasattr(df, 'head'):
            df_sample = df.head(3).to_csv(index=False)
        else:
            df_sample = ""

        # Extract variables safely
        treatment_raw = parsed_vars.get("treatment")
        outcome_raw = parsed_vars.get("outcome")
        
        if not treatment_raw or not outcome_raw:
            raise ValueError(f"Missing required variables: treatment={treatment_raw}, outcome={outcome_raw}. Available keys: {list(parsed_vars.keys())}")
        
        treatment = clean_var_names([treatment_raw])[0]
        outcome = clean_var_names([outcome_raw])[0]
        confounders = clean_var_names(parsed_vars.get("confounders", []))
        mediators = clean_var_names(parsed_vars.get("mediators", []))
        ivs = clean_var_names(parsed_vars.get("instrumental_variables", []))
        colliders = clean_var_names(parsed_vars.get("colliders", []))

        # Data type information - check if columns exist in DataFrame
        if df is not None and hasattr(df, 'columns'):
            if outcome not in df.columns:
                raise ValueError(f"Outcome variable '{outcome}' not found in DataFrame columns: {list(df.columns)}")
            if treatment not in df.columns:
                raise ValueError(f"Treatment variable '{treatment}' not found in DataFrame columns: {list(df.columns)}")
            
            unique_outcome_values = df[outcome].dropna().unique()
            outcome_type = "binary" if len(unique_outcome_values) == 2 else "continuous"
            treatment_type = str(df[treatment].dtype)
        else:
            # Fallback values when DataFrame is not available
            outcome_type = "continuous"
            treatment_type = "unknown"

        prompt_input = {
            "question": question,
            "treatment": treatment,
            "outcome": outcome,
            "confounders": confounders,
            "mediators": mediators,
            "instrumental_variables": ivs,
            "colliders": colliders,
            "df_sample": df_sample,
            "treatment_type": treatment_type,
            "outcome_type": outcome_type
        }

        result = call_llm(
            prompt=causal_strategy_prompt,
            parser=strategy_output_parser,
            variables=prompt_input,
            llm=llm
        )

        state["strategy"] = Strategy(
            task=result.causal_task,
            identification_method=result.identification_strategy,
            estimator=result.estimation_method,
            refuter=result.refutation_methods[0] if result.refutation_methods else None
        )
        return state

    return RunnableLambda(invoke)
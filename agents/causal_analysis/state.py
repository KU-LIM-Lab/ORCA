# agents/causal_analysis/state.py

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, ConfigDict
import pandas as pd
from dowhy import CausalModel
from dowhy.causal_estimator import CausalEstimate

class Strategy(BaseModel):
    task: str
    identification_method: str
    estimator: str
    refuter: Optional[str] = None

class CausalAnalysisState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Input from data_explorer (required)
    df_preprocessed: Optional[pd.DataFrame] = None
    
    # Causal variables (can be provided or parsed)
    treatment_variable: Optional[str] = None
    outcome_variable: Optional[str] = None
    confounders: Optional[List[str]] = None
    
    # Optional inputs
    input: Optional[str] = None  # Natural language question
    causal_graph: Optional[Dict[str, Any]] = None  # Pre-defined causal graph
    
    # Parsed information (if input is provided)
    parsed_query: Optional[Dict[str, Any]] = None 
    variable_info: Optional[Dict[str, Any]] = None  
    expression_dict: Optional[Dict[str, str]] = None 
    
    # Analysis configuration
    strategy: Optional[Strategy] = None  
    
    # DoWhy analysis results
    causal_model: Optional[CausalModel] = None  # dowhy model
    causal_estimand: Optional[Any] = None  
    causal_estimate: Optional[CausalEstimate] = None  
    causal_effect_ate: Optional[float] = None  # ATE (Average Treatment Effect)
    causal_effect_ci: Optional[Any] = None 
    refutation_result: Optional[str] = None 
    
    # Final output
    final_answer: Optional[str] = None
    error: Optional[str] = None 
    
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)
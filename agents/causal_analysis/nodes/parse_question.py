# causal_analysis/nodes/parse_question.py
import json
from typing import Callable, Dict, List, Optional
import networkx as nx
import pandas as pd

from utils.llm import call_llm
from utils.data_prep.metadata import generate_table_markdown
from utils.redis_df import load_df_parquet
from prompts.causal_analysis_prompts import (
    identify_treatment_outcome_prompt,
    identify_treatment_outcome_parser
)

from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel

from utils.redis_client import redis_client

def build_parse_question_node(llm: BaseChatModel) -> Callable:
    """
    Build parse_question node that:
    1. Identifies Treatment and Outcome variables (LLM or from state)
    2. Identifies variable roles from causal graph (Confounders, Mediators, Colliders, Instruments)
    3. Updates parsed_query with all identified information
    """
    
    def _load_dataframe_from_state(state: Dict) -> Optional[pd.DataFrame]:
        """
        Load DataFrame from state, trying df_preprocessed first, then Redis.
        Similar to CausalDiscoveryAgent._load_dataframe_from_state
        """
        df = state.get("df_preprocessed")
        if isinstance(df, pd.DataFrame):
            return df
        
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
        
        return None
    
    def _identify_treatment_outcome(state: Dict) -> Dict[str, str]:
        """
        Identify treatment and outcome variables.
        Returns dict with 'treatment' and 'outcome' keys.
        """
        # Check if already provided in state
        if state.get("treatment_variable") and state.get("outcome_variable"):
            return {
                "treatment": state["treatment_variable"],
                "outcome": state["outcome_variable"]
            }
        
        # Use LLM to identify from question and data
        question = state.get("input", "")
        df = _load_dataframe_from_state(state)
        
        # Prepare data sample
        df_sample = ""
        if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
            df_sample = df.head(3).to_csv(index=False)
        
        # Prepare table schema
        tables = ""
        db_id = state.get("db_id")
        if db_id:
            table_keys = redis_client.keys(f"{db_id}:metadata:*")
            table_markdowns = []
            for key in table_keys:
                if key == f"{db_id}:metadata:table_names":
                    continue
                table_name = key.split(":")[2]
                raw = redis_client.get(key)
                if not raw:
                    continue
                try:
                    metadata = json.loads(raw)
                    schema = metadata.get("schema", {})
                    markdown = generate_table_markdown({table_name: schema})
                    table_markdowns.append(markdown)
                except Exception as e:
                    print(f"⚠️ Error parsing metadata for key {key}: {e}")
            tables = "\n\n".join(table_markdowns)
        
        # Get available columns
        columns = []
        if df is not None and isinstance(df, pd.DataFrame):
            columns = list(df.columns)
        
        # Call LLM
        result = call_llm(
            prompt=identify_treatment_outcome_prompt,
            parser=identify_treatment_outcome_parser,
            variables={
                "question": question,
                "df_sample": df_sample,
                "tables": tables,
                "columns": ", ".join(columns) if columns else "Not available"
            },
            llm=llm
        )
        
        return {
            "treatment": result.treatment,
            "outcome": result.outcome
        }
    
    def _build_graph_from_state(state: Dict) -> Optional[nx.DiGraph]:
        """
        Build NetworkX DiGraph from causal_graph in state.
        Handles different graph formats.
        """
        causal_graph = state.get("causal_graph")
        if not causal_graph:
            return None
        
        G = nx.DiGraph()
        
        # Try to get nx_graph if already built
        if "nx_graph" in causal_graph and causal_graph["nx_graph"] is not None:
            return causal_graph["nx_graph"]
        
        # Extract nodes
        nodes = []
        if "nodes" in causal_graph:
            nodes = causal_graph["nodes"]
        elif "graph" in causal_graph and "variables" in causal_graph["graph"]:
            nodes = causal_graph["graph"]["variables"]
        elif "variables" in causal_graph:
            nodes = causal_graph["variables"]
        
        if not nodes:
            return None
        
        G.add_nodes_from(nodes)
        
        # Extract edges
        edges = []
        if "edges" in causal_graph:
            edges = causal_graph["edges"]
        elif "graph" in causal_graph and "edges" in causal_graph["graph"]:
            edges = causal_graph["graph"]["edges"]
        
        # Add edges to graph
        for edge in edges:
            if isinstance(edge, dict):
                from_node = str(edge.get("from", ""))
                to_node = str(edge.get("to", ""))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                from_node = str(edge[0])
                to_node = str(edge[1])
            else:
                continue
            
            if from_node in nodes and to_node in nodes:
                G.add_edge(from_node, to_node)
        
        return G
    

    def _identify_roles_from_graph(treatment: str, outcome: str, graph: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Identifies variable roles based on the Causal Graph (DAG) topology.
        
        Mediators: Nodes on the causal path T -> ... -> M -> ... -> O.
        Confounders: Common causes of T and O (Backdoor paths), excluding mediators.
        Instruments: Ancestors of T that affect O ONLY through T (Unconfoundedness + Exclusion).
        Colliders/Excluded: Nodes that must NOT be controlled (Colliders, Descendants of Outcome, etc.).
        
        """
        if not graph or treatment not in graph or outcome not in graph:
            return {"confounders": [], "mediators": [], "instruments": [], "colliders": []}

        nodes = set(graph.nodes())
        
        # Precompute Ancestors and Descendants
        try:
            anc_T = nx.ancestors(graph, treatment)
            desc_T = nx.descendants(graph, treatment)
            anc_O = nx.ancestors(graph, outcome)
            desc_O = nx.descendants(graph, outcome)
        except nx.NetworkXError:
            # Check for cycles if logic fails
            return {"confounders": [], "mediators": [], "instruments": [], "colliders": []}

        # Role Identification Logic 

        # A. Mediators (Frontdoor Path)
        # Definition: Descendants of T AND Ancestors of O
        # Logic: T -> M -> O
        mediators_set = (desc_T & anc_O) - {treatment, outcome}

        # B. Confounders (Backdoor Path)
        # Definition: Common Ancestors of T and O
        # Exclusion: Must NOT be a mediator (to avoid blocking causal path).
        # Safety: Must NOT be a descendant of T (Post-treatment).
        raw_confounders = (anc_T & anc_O) - {treatment, outcome}
        confounders_set = {
            c for c in raw_confounders 
            if c not in mediators_set 
            and c not in desc_T  # Explicitly exclude post-treatment
        }

        # C. Instruments (IV)
        # Definition: Affects T, but affects O ONLY via T.
        # Logic: Ancestor of T, BUT NOT Ancestor of O (direct path to O implies confounding or direct effect).
        # Exclusion: Must not be a confounder or mediator.
        raw_instruments = anc_T - anc_O
        instruments_set = {
            z for z in raw_instruments
            if z not in confounders_set
            and z not in mediators_set
            and z not in desc_T  # Should be pre-treatment
            and z != outcome
        }

        # D. Colliders & Do-Not-Control (The "Others")
        # Variables that, if conditioned on, might induce bias.
        # Includes:
        #   1. True Colliders (T -> C <- O)
        #   2. Descendants of Outcome (O -> D)
        #   3. Descendants of Colliders
        #   4. Irrelevant variables (disconnected or downstream only)
        
        # Specifically mark Outcome Descendants as strictly forbidden
        outcome_descendants = desc_O
        
        classified_nodes = confounders_set | mediators_set | instruments_set | {treatment, outcome}
        
        # All remaining nodes fall into 'colliders/excluded'
        # This safely wraps: True Colliders, Outcome Descendants, Irrelevant Disconnected Nodes
        others_set = nodes - classified_nodes
        
        # Final 'colliders' list is the union of unclassified nodes and outcome descendants
        colliders_set = others_set | outcome_descendants

        return {
            "confounders": sorted(list(confounders_set)),
            "mediators": sorted(list(mediators_set)),
            "instruments": sorted(list(instruments_set)),
            "colliders": sorted(list(colliders_set))
        }
    
    def _parse_question(state: Dict) -> Dict:
        if state.get("variable_info"):
            state["parsed_query"] = state["variable_info"]
            return state
        
        treatment_outcome = _identify_treatment_outcome(state)
        treatment = treatment_outcome["treatment"]
        outcome = treatment_outcome["outcome"]
        
        df_preprocessed = _load_dataframe_from_state(state)
        if df_preprocessed is None:
            raise ValueError("df_preprocessed is required. Data should be loaded before this node (either in state or via df_redis_key).")
        
        parsed = state.get("parsed_query") or {}
        parsed["treatment"] = treatment
        parsed["outcome"] = outcome
        
        # Step 5: build causal graph
        graph = _build_graph_from_state(state)
        if graph:
            roles = _identify_roles_from_graph(treatment, outcome, graph)
            
            # Merge with existing confounders/mediators/instruments
            existing_confounders = parsed.get("confounders", [])
            existing_mediators = parsed.get("mediators", [])
            existing_instruments = parsed.get("instrumental_variables", [])
            
            # Combine and deduplicate
            all_confounders = list(set(existing_confounders + roles["confounders"]))
            all_mediators = list(set(existing_mediators + roles["mediators"]))
            all_instruments = list(set(existing_instruments + roles["instruments"]))
            
            parsed["confounders"] = all_confounders
            parsed["mediators"] = all_mediators
            parsed["instrumental_variables"] = all_instruments
            parsed["colliders"] = roles["colliders"]
        
        state["parsed_query"] = parsed
        return state

    return RunnableLambda(_parse_question)
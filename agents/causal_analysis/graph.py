# agents/causal_analysis/graph.py
from langgraph.graph import StateGraph, END

from .state import CausalAnalysisState
from .nodes.parse_question import build_parse_question_node
from .nodes.config_selection import build_config_selection_node
from .nodes.dowhy_analysis import build_dowhy_analysis_node
from .nodes.generate_answer import build_generate_answer_node

def generate_causal_analysis_graph(llm):
    graph = StateGraph(CausalAnalysisState)

    # Add entry node for conditional routing
    graph.add_node("__entry__", lambda state: state)
    
    # Add nodes (removed data preprocessing nodes)
    graph.add_node("parse_question", build_parse_question_node(llm))
    graph.add_node("config_selection", build_config_selection_node(llm))
    graph.add_node("dowhy_analysis", build_dowhy_analysis_node())
    graph.add_node("generate_answer", build_generate_answer_node(llm))

    # Conditional routing: check if variables are already provided
    def route_entry(state):
        # If we have treatment and outcome variables, skip parsing
        if state.get("treatment_variable") and state.get("outcome_variable"):
            return "config_selection"
        # If we have input question, parse it first
        elif state.get("input"):
            return "parse_question"
        # Otherwise, error
        else:
            raise ValueError("Either treatment/outcome variables or input question must be provided")

    graph.set_entry_point("__entry__")
    graph.add_conditional_edges("__entry__", route_entry)

    # Flow: parse_question -> config_selection -> dowhy_analysis -> generate_answer
    graph.add_edge("parse_question", "config_selection")
    graph.add_edge("config_selection", "dowhy_analysis")
    graph.add_edge("dowhy_analysis", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
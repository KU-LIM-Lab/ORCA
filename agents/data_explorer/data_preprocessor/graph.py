from langgraph.graph import StateGraph, END
from .state import PreprocessState

# Node implementations are simple pass-through stubs for now
from .nodes.fetch import fetch_node
from .nodes.clean_nulls import clean_nulls_node
from .nodes.encode import encode_node


def generate_preprocess_graph(llm=None):
    graph = StateGraph(PreprocessState)

    graph.add_node("fetch_node", fetch_node)
    graph.add_node("clean_nulls_node", clean_nulls_node)
    graph.add_node("encode_node", encode_node)

    graph.set_entry_point("fetch_node")
    graph.add_edge("fetch_node", "clean_nulls_node")
    graph.add_edge("clean_nulls_node", "encode_node")
    graph.add_edge("encode_node", END)

    return graph.compile()



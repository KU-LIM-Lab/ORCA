from langgraph.graph import StateGraph, END
from .state import PreprocessState

# Node implementations are simple pass-through stubs for now
from .nodes.fetch import fetch_node
from .nodes.clean_nulls import clean_nulls_node
from .nodes.type_cast import type_cast_node
from .nodes.derive import derive_features_node
from .nodes.impute import impute_node
from .nodes.encode import encode_node
from .nodes.scale import scale_node
from .nodes.split import split_node
from .nodes.report import report_node


def generate_preprocess_graph(llm=None):
    graph = StateGraph(PreprocessState)

    graph.add_node("fetch_node", fetch_node)
    graph.add_node("clean_nulls_node", clean_nulls_node)
    graph.add_node("type_cast_node", type_cast_node)
    graph.add_node("derive_features_node", derive_features_node)
    graph.add_node("impute_node", impute_node)
    graph.add_node("encode_node", encode_node)
    graph.add_node("scale_node", scale_node)
    graph.add_node("split_node", split_node)
    graph.add_node("report_node", report_node)

    graph.set_entry_point("fetch_node")

    # Conditional routing chaining
    # If fetch_only flag is set, finish after fetch
    graph.add_conditional_edges("fetch_node", lambda s: "END" if s.get("fetch_only") else "NEXT", {
        "END": END,
        "NEXT": "clean_nulls_node",
    })
    graph.add_edge("clean_nulls_node", "type_cast_node")
    graph.add_edge("type_cast_node", "derive_features_node")
    graph.add_edge("derive_features_node", "impute_node")
    graph.add_edge("impute_node", "encode_node")
    graph.add_edge("encode_node", "scale_node")
    graph.add_edge("scale_node", "split_node")
    graph.add_edge("split_node", "report_node")
    graph.add_edge("report_node", END)

    return graph.compile()



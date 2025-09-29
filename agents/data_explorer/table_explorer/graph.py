from langgraph.graph import StateGraph, END

from .state import TableState
from .nodes.describe import describe_table_node
from .nodes.related import related_tables_node
from .nodes.recommend import recommend_analysis_node
from functools import partial


def generate_description_graph(llm=None):
    graph = StateGraph(TableState)

    graph.add_node("describe_table_node", partial(describe_table_node, llm=llm))
    graph.add_node("related_tables_node", related_tables_node)
    graph.add_node("recommend_analysis_node", partial(recommend_analysis_node, llm=llm))

    graph.set_entry_point("describe_table_node")
    graph.add_edge("describe_table_node", "related_tables_node")
    graph.add_edge("related_tables_node", "recommend_analysis_node")
    graph.add_edge("recommend_analysis_node", END)

    return graph.compile()
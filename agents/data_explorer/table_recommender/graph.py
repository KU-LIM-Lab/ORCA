from langgraph.graph import StateGraph, START, END
from .state import RecommendState
from .nodes.objective_summary import extract_objective_summary
from .nodes.table_recommend import recommend_tables
from .nodes.erd import generate_erd

from functools import partial

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def set_input_type(state):
    if state["input"].endswith((".pdf", ".docx", ".pptx")):
        state["input_type"] = "document"
    else:
        state["input_type"] = "text"
    return state

def route_input_type(state):
    return state["input_type"]

def parse_document(state):
    ext = Path(state['input']).suffix.lower()
    if ext == '.pdf':
        loader = PyPDFLoader(state['input'])
        docs = loader.load()
        state['parsed_text'] = "\n\n".join(doc.page_content for doc in docs)
    elif ext == '.docx':
        loader = Docx2txtLoader(state['input'])  # 추가
    else:
        raise ValueError(f"Error: unsupported file extension: {ext}")
    docs = loader.load()
    state['parsed_text'] = "\n\n".join(doc.page_content for doc in docs)
    return state

def generate_table_recommendation_graph(llm):
    graph = StateGraph(RecommendState)

    graph.add_node("set_input_node", set_input_type)
    graph.add_node("parse_document_node", parse_document)
    graph.add_node("extract_objective_summary_node", partial(extract_objective_summary, llm = llm))
    graph.add_node("recommend_tables_node", partial(recommend_tables, llm = llm))
    graph.add_node("generate_erd_node", generate_erd)

    graph.set_entry_point("set_input_node")

    graph.add_conditional_edges(
        "set_input_node",  
        route_input_type,   
        {
            "text": "extract_objective_summary_node",
            "document": "parse_document_node"
        }
    )

    graph.add_edge("parse_document_node", "extract_objective_summary_node")
    graph.add_edge("extract_objective_summary_node", "recommend_tables_node")
    graph.add_edge("recommend_tables_node", "generate_erd_node")
    graph.add_edge("generate_erd_node", END)

    return graph.compile()
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes.selector import selector_node
from .nodes.decomposer import decomposer_node
from .nodes.refiner import refiner_node
from .nodes.review import review_node
from .nodes.system import system_node
from functools import partial


def router(state: AgentState):
    return state['send_to']

def generate_text2sql_graph(llm):
    graph = StateGraph(AgentState)
    graph.set_entry_point('selector_node') 
    graph.add_node('selector_node', partial(selector_node, llm=llm))  
    graph.add_node('decomposer_node', partial(decomposer_node, llm = llm)) 
    graph.add_node('refiner_node', partial(refiner_node, llm = llm)) 
    graph.add_node('review_node', partial(review_node, llm = llm))
    graph.add_node('system_node', system_node)

    graph.add_conditional_edges('selector_node', router, {
        'decomposer_node': 'decomposer_node',
    })
    graph.add_conditional_edges('decomposer_node', router, {
        'refiner_node': 'refiner_node',
    })
    graph.add_conditional_edges('refiner_node', router, {
        'review_node': 'review_node',
        'refiner_node': 'refiner_node',
    })

    graph.add_conditional_edges('review_node', router, {
            'refiner_node': 'refiner_node',
            'system_node': 'system_node',
    })

    graph.add_conditional_edges('system_node', lambda s: END, {END: END})
    
    return graph.compile()
from typing import Any, Dict, Optional

from core.base import SubgraphAgent, AgentType
from .graph import generate_text2sql_graph


class Text2SQLGeneratorAgent(SubgraphAgent):
    """SubgraphAgent override for the text2sql generation pipeline.

    This subclass wires the existing compiled-graph factory into the
    SubgraphAgent lifecycle without requiring changes to the graph module.
    """

    def __init__(self, llm: Optional[Any] = None, name: str = "text2sql_generator", config: Optional[Dict[str, Any]] = None):
        # Pass a placeholder None graph; we override compile_graph to build it
        super().__init__(name=name, graph=None, config=config)
        self.agent_type = AgentType.SUBGRAPH
        self.llm = llm

    def get_required_state_keys(self):
        return ["db_id", "query","evidence"]

    def compile_graph(self) -> None:
        """Override to use existing compiled graph factory.

        The base class expects an object with `.compile()`, but our graph module
        already returns a compiled runnable via `generate_text2sql_graph`.
        We therefore set `compiled_graph` directly.
        """
        self.compiled_graph = generate_text2sql_graph(llm=self.llm)
        self.on_event("graph_compiled")

from typing import Any, Dict, Optional

from core.base import SubgraphAgent, AgentType
from .graph import generate_preprocess_graph


class DataPreprocessorAgent(SubgraphAgent):
    """SubgraphAgent for tabular data preprocessing pipeline."""

    def __init__(self, llm: Optional[Any] = None, name: str = "data_preprocessor", config: Optional[Dict[str, Any]] = None):
        super().__init__(name=name, graph=None, config=config)
        self.agent_type = AgentType.SUBGRAPH
        self.llm = llm

    def get_required_state_keys(self):
        # Either df_raw or final_sql must be provided; we enforce at runtime
        return ["db_id"]

    def compile_graph(self) -> None:
        self.compiled_graph = generate_preprocess_graph(llm=self.llm)
        self.on_event("graph_compiled")



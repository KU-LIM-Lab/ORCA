# agents/report_generation/agent.py
from typing import Optional, Dict, Any
from core.base import SpecialistAgent, AgentType
from core.state import AgentState
from monitoring.metrics.collector import MetricsCollector


class ReportGenerationAgent(SpecialistAgent):
    """Generates human-friendly reports for explorer, discovery, inference outputs."""

    def __init__(self, name: str = "report_generator", config: Optional[Dict[str, Any]] = None,
                 metrics_collector: Optional[MetricsCollector] = None):
        super().__init__(name, AgentType.SPECIALIST, config, metrics_collector)
        self.set_domain_expertise([
            "report_generation",
            "interpretation",
            "summarization"
        ])

    def _register_specialist_tools(self) -> None:
        # Register formatting tools
        self.register_tool("report_tools", self._report_tools, "Formatting utilities for reports")

    def step(self, state: AgentState) -> AgentState:
        try:
            substep = state.get("current_substep", "generate_report")
            if substep != "generate_report":
                state["error"] = f"Unknown substep: {substep}"
                return state

            sections: Dict[str, Any] = {}

            # Data Explorer: table explorer/recommender outputs
            try:
                from utils.prettify import (
                    print_final_output_recommender,
                    print_final_output_explorer,
                    print_final_output_sql,
                    print_final_output_causal,
                )
            except Exception:
                # If prettify not available, fallback to identity formatting
                def print_final_output_recommender(x):
                    return str(x)
                def print_final_output_explorer(x):
                    return str(x)
                def print_final_output_sql(x):
                    return str(x)
                def print_final_output_causal(x):
                    return str(x)

            # 1) Data Explorer section (build from existing state keys)
            explorer_payload: Dict[str, Any] = {}
            if state.get("schema_analysis"):
                explorer_payload["table_name"] = (state.get("selected_tables") or [""])[0]
                explorer_payload["table_analysis"] = state.get("schema_analysis", {})
                sections["data_explorer"] = print_final_output_explorer(explorer_payload)

            # 1-b) Table Recommender section (with ERD if present)
            recommender_output = state.get("final_output") or {}
            if recommender_output:
                sections["table_recommender"] = print_final_output_recommender(recommender_output)
                if recommender_output.get("erd_image_path"):
                    sections["erd_image_path"] = recommender_output.get("erd_image_path")

            # Optional SQL generation/execution section
            # Optional: if raw SQL present, surface minimal SQL section
            if state.get("sql_query") or state.get("df_raw") is not None:
                sections["text2sql"] = print_final_output_sql({
                    "sql": state.get("sql_query"),
                    "result": state.get("df_raw"),
                    "columns": None
                })

            # 2) Full pipeline causal report (discovery + inference)
            causal_summary_payload: Dict[str, Any] = {
                "parsed_query": {
                    "treatment": state.get("treatment_variable"),
                    "outcome": state.get("outcome_variable"),
                    "confounders": state.get("confounders"),
                },
                "sql_query": state.get("sql_query"),
                "df_raw": state.get("df_raw"),
                "strategy": {
                    "task": "causal_inference" if state.get("causal_inference_status") else "",
                    "identification_method": state.get("inference_method") or state.get("identification_strategy", ""),
                    "estimator": state.get("inference_method", ""),
                    "refuter": None,
                },
                "final_answer": state.get("graph_selection_reasoning") or "",
            }
            sections["causal_summary"] = print_final_output_causal(causal_summary_payload)

            # LLM-based final narrative (optional)
            narrative = None
            try:
                from utils.llm import call_llm
                prompt = self._build_pipeline_summary_prompt(state)
                narrative = call_llm(prompt)
            except Exception:
                narrative = None
            if narrative:
                sections["narrative_summary"] = narrative

            state["final_report"] = {
                "status": "completed",
                "sections": sections,
            }
            return state
        except Exception as e:
            state["error"] = f"Report generation failed: {str(e)}"
            return state

    def _report_tools(self, method: str, *args, **kwargs) -> Dict[str, Any]:
        return {"error": f"Unknown report tool: {method}"}

    def _build_pipeline_summary_prompt(self, state: AgentState) -> str:
        parts = []
        parts.append("Summarize the analysis pipeline results in concise bullet points.")
        if state.get("selected_graph"):
            parts.append("- Include key discovered causal relations.")
        if state.get("causal_estimates"):
            parts.append("- Include the main causal effect estimates and any caveats.")
        if state.get("erd_image_path"):
            parts.append(f"- ERD image saved at: {state.get('erd_image_path')}")
        return "\n".join(parts)



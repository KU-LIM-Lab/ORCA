# -*- coding: utf-8 -*-
# agents/report_generation/agent.py
from __future__ import annotations

from typing import Optional, Dict, Any
from core.base import SpecialistAgent, AgentType
from core.state import AgentState
from monitoring.metrics.collector import MetricsCollector

from agents.report_generation.tools import (
    build_table_exploration_section,
    build_table_recommendation_section,
    build_text2sql_section,
    build_causal_inference_section,
    build_causal_discovery_prompt,
)

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
        # 간단 라우팅 (필요 시 확장)
        self.register_tool("report_tools", self._report_tools, "Formatting utilities for reports")

    def step(self, state: AgentState) -> AgentState:
        try:
            substep = state.get("current_substep", "generate_report")
            if substep != "generate_report":
                state["error"] = f"Unknown substep: {substep}"
                return state

            sections: Dict[str, Any] = {}

            # 1) Table Exploration
            if state.get("schema_analysis") or state.get("table_analysis"):
                sections["1_table_exploration"] = build_table_exploration_section(state)

            # 2) Table Recommendation (final_output 활용)
            recommender_output = state.get("final_output") or {}
            if recommender_output:
                sections["2_table_recommendation"] = build_table_recommendation_section(recommender_output)

            # 3) Text2SQL
            if state.get("sql_query") is not None or state.get("df_raw") is not None:
                sections["3_text2sql"] = build_text2sql_section(
                    sql=state.get("sql_query"),
                    result=state.get("df_raw"),
                    columns=None,
                    llm_review=state.get("llm_review"),
                    error=state.get("error") if state.get("sql_error_mode") else None
                )

            # 4) Causal Discovery (LLM Narrative)
            discovery_md = None
            try:
                from utils.llm import call_llm
                prompt = build_causal_discovery_prompt(state)
                
                # Check prompt size and warn if too large
                prompt_length = len(prompt)
                if prompt_length > 50000:  # ~12.5k tokens
                    print(f"⚠️  [REPORT] Causal discovery prompt is very large ({prompt_length} chars), this may be slow")
                
                discovery_md = call_llm(prompt)  # LLM이 마크다운/불릿으로 서술
            except Exception as e:
                print(f"⚠️  [REPORT] Failed to generate causal discovery narrative: {e}")
                discovery_md = None
            if discovery_md:
                sections["4_causal_discovery"] = f"## 4) Causal Discovery Report\n\n{discovery_md}\n\n---"
            else:
                # LLM 실패 시, 최소한의 페일세이프(핵심 표만)
                fallback = []
                fallback.append("## 4) Causal Discovery Report (fallback)")
                if state.get("assumption_method_scores"):
                    from agents.report_generation.tools import matrix_to_md
                    fallback.append("### Assumption–Method Compatibility")
                    fallback.append(matrix_to_md(state.get("assumption_method_scores")))
                if state.get("algorithm_scores"):
                    scores = state.get("algorithm_scores") or {}
                    rows = [[k, round(v, 3)] for k, v in sorted(scores.items(), key=lambda x: -x[1])]
                    from agents.report_generation.tools import to_markdown_table
                    fallback.append("### Algorithm Scores")
                    fallback.append(to_markdown_table(rows, ["algorithm","score"]))
                fallback.append("\n---")
                sections["4_causal_discovery"] = "\n".join(fallback)

            # 5) Causal Inference (only if anything relevant exists)
            if state.get("causal_estimates") or state.get("inference_method") or state.get("treatment_variable"):
                sections["5_causal_inference"] = build_causal_inference_section(state)

            # 메타데이터 포함 최종 보고서
            # Optimize: limit execution_log to last 50 entries to avoid bloat
            execution_log = state.get("execution_log", [])
            state["final_report"] = {
                "status": "completed",
                "query": state.get("initial_query"),
                "total_steps": len(execution_log),
                "sections": sections,
                "execution_log": state.get("execution_log", []),
                "results": state.get("results", {}),
                # 편의: 전체 마크다운 합치기
                "markdown": "\n".join([
                    sections.get("1_table_exploration",""),
                    sections.get("2_table_recommendation",""),
                    sections.get("3_text2sql",""),
                    sections.get("4_causal_discovery",""),
                    sections.get("5_causal_inference",""),
                ]).strip()
            }
            
            # Mark pipeline as completed
            state["executor_completed"] = True
            print("[REPORT] ✅ Report generation completed - pipeline finished")
            
            return state

        except Exception as e:
            state["error"] = f"Report generation failed: {str(e)}"
            return state
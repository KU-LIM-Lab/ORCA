# -*- coding: utf-8 -*-
# agents/report_generation/tools.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import re
import json
import math
import pandas as pd

# =========================
# Formatting utilities
# =========================

def _h(name: str, level: int = 2) -> str:
    """마크다운 헤더 생성 (#, ##, ### ...)"""
    level = max(1, min(6, level))
    return f"{'#' * level} {name}".strip()

def _kv(key: str, val: Any) -> str:
    """key: value 1줄"""
    if val is None or val == "":
        val = "-"
    return f"- **{key}**: {val}"

def _hr() -> str:
    return "\n---\n"

def to_markdown_table(rows: List[List[Any]], header: Optional[List[str]] = None) -> str:
    """2D 리스트를 마크다운 표로 렌더링"""
    if not rows and not header:
        return "_[no data]_"
    out = []
    if header:
        out.append("| " + " | ".join(map(str, header)) + " |")
        out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        out.append("| " + " | ".join(map(lambda x: str(x) if x is not None else "", r)) + " |")
    return "\n".join(out)

def df_preview(df: Optional[pd.DataFrame], n: int = 10) -> str:
    """DataFrame 미리보기 → 마크다운 표"""
    if not isinstance(df, pd.DataFrame):
        return "_[no dataframe]_"
    head = df.head(n)
    return head.to_markdown(index=False)

def dict_to_md(d: Dict[str, Any]) -> str:
    """dict 1단계만 key-value bullet로"""
    lines = []
    for k, v in d.items():
        if isinstance(v, (dict, list)):
            v = json.dumps(v, ensure_ascii=False)
        lines.append(_kv(k, v))
    return "\n".join(lines)

def matrix_to_md(matrix: Dict[str, Dict[str, float]], round_ndigits: int = 3) -> str:
    """{row:{col:score}} 형태를 표로"""
    if not matrix:
        return "_[empty matrix]_"
    cols = sorted({c for r in matrix.values() for c in r.keys()})
    rows = []
    for rn in sorted(matrix.keys()):
        row_vals = [round(matrix[rn].get(c, float("nan")), round_ndigits) for c in cols]
        rows.append([rn] + row_vals)
    return to_markdown_table(rows, header=["Assumption/Method"] + cols)

def list_to_md(lst: Optional[List[Any]]) -> str:
    if not lst:
        return "- (none)"
    return "\n".join(f"- {x}" for x in lst)

def edges_to_mermaid(edges: List[Tuple[str, str, str]]) -> str:
    """
    엣지 리스트 → Mermaid 그래프 코드
    edges: [(src, dst, type)], type in {"directed","bidirected","undirected"}
    """
    if not edges:
        return "```mermaid\ngraph LR\n%% no edges\n```"
    lines = ["```mermaid", "graph LR"]
    for s, t, et in edges:
        if et == "bidirected":
            lines.append(f'  {s} <--> {t}')
        elif et == "undirected":
            lines.append(f'  {s} --- {t}')
        else:
            lines.append(f'  {s} --> {t}')
    lines.append("```")
    return "\n".join(lines)

# =========================
# 섹션 빌더: 1) Table Exploration
# =========================

def build_table_exploration_section(state: Dict[str, Any]) -> str:
    """
    state 키 기대:
      - selected_tables: List[str]
      - schema_analysis: Dict[str, Any]  (table_description, columns(list of dict), analysis_considerations)
    """
    lines = []
    table_name = (state.get("selected_tables") or [""])[0]
    # prefer schema_analysis, but accept table_analysis as fallback
    schema = state.get("schema_analysis") or state.get("table_analysis") or {}
    columns = schema.get("columns", []) if isinstance(schema, dict) else []

    lines.append(_h("1) Table Exploration Report", 2))
    lines.append(_kv("Table", f"`{table_name}`"))
    lines.append("")
    lines.append(_h("Table Description", 3))
    # Support both schema_analysis.table_description and any string fallback
    table_desc = None
    if isinstance(schema, dict):
        table_desc = schema.get("table_description")
    if table_desc is None and isinstance(schema, str):
        table_desc = schema
    lines.append((table_desc or "_[no description]_") + "\n")

    lines.append(_h("Columns", 3))
    if columns:
        col_rows = []
        for c in columns:
            col_rows.append([
                c.get("column_name",""),
                c.get("data_type",""),
                str(c.get("nullable","")),
                str(c.get("nulls","")),
                " / ".join(c.get("notes", []) if isinstance(c.get("notes"), list) else [c.get("notes","")]).strip(" / ")
            ])
        lines.append(to_markdown_table(col_rows, ["column","type","nullable","nulls","notes"]))
    else:
        lines.append("- (no columns)")
    lines.append("")

    lines.append(_h("Analysis Considerations", 3))
    if isinstance(schema, dict):
        lines.append(schema.get("analysis_considerations","_[none]_"))
    else:
        lines.append("_[none]_")
    lines.append(_hr())
    return "\n".join(lines)

# =========================
# 2) Table Recommendation
# =========================

def build_table_recommendation_section(final_output: Dict[str, Any]) -> str:
    """
    final_output 키 기대:
      - objective_summary: str
      - recommended_tables: List[ obj with .table, .important_columns ]
      - recommended_method: str (번호 목록 깨짐 방지 처리)
      - erd_image_path: str (경로)
    """
    lines = []
    lines.append(_h("2) Table Recommendation Report", 2))
    lines.append(_h("Objective Summary", 3))
    lines.append(final_output.get("objective_summary","_[no summary]_"))
    lines.append("")

    lines.append(_h("Recommended Tables (Important Columns)", 3))
    tables = final_output.get("recommended_tables", []) or []
    if tables:
        for i, t in enumerate(tables, 1):
            tname = getattr(t, "table", None) or t.get("table") if isinstance(t, dict) else None
            cols  = getattr(t, "important_columns", None) or t.get("important_columns") if isinstance(t, dict) else None
            col_text = ", ".join(cols or [])
            lines.append(f"{i}. `{tname}` — {col_text}")
    else:
        lines.append("- (no tables)")
    lines.append("")

    lines.append(_h("Recommended Analysis Method", 3))
    method_text = final_output.get("recommended_method", "_[no method]_")
    method_text = re.sub(r'(?<!^)(?<!\n)(\d+\.\s)', r'\n\1', method_text)
    lines.append(method_text)
    lines.append("")

    lines.append(_h("ERD Image Path", 3))
    lines.append(final_output.get("erd_image_path","_[no erd]_"))
    lines.append(_hr())
    return "\n".join(lines)

# =========================
# 3) Text2SQL
# =========================

def build_text2sql_section(sql: Optional[str], result: Any, columns: Optional[List[str]] = None, llm_review: Any = None, error: Optional[str] = None) -> str:
    """SQL/결과/오류/리뷰 섹션 생성 (결과는 DataFrame도 처리)"""
    lines = []
    lines.append(_h("3) Text2SQL Report", 2))

    # SQL
    lines.append(_h("Generated SQL", 3))
    lines.append(sql or "_[no sql]_")
    lines.append("")

    # Result
    lines.append(_h("Execution Result (preview)", 3))
    if isinstance(result, pd.DataFrame):
        # Only preview head to avoid memory issues
        lines.append(df_preview(result.head(10), n=10))
    elif isinstance(result, list):
        # 간단 리스트 렌더
        if columns:
            lines.append(to_markdown_table(result[:10], header=columns))
        else:
            if result[:10] and isinstance(result[0], (list, tuple)):
                lines.append(to_markdown_table(result[:10]))
            else:
                lines.append(list_to_md(result[:10]))
        if len(result) > 10:
            lines.append(f"\n> Too many rows ({len(result)}). showing top-10 only.")
    else:
        lines.append(f"`{str(result)}`")
    lines.append("")

    # Error
    lines.append(_h("Error", 3))
    lines.append(str(error) if error else "_[no error]_")
    lines.append("")

    # LLM Review
    if llm_review is not None:
        lines.append(_h("LLM Review", 3))
        if isinstance(llm_review, list):
            lines.extend(map(str, llm_review))
        else:
            lines.append(str(llm_review))
        lines.append("")
    lines.append(_hr())
    return "\n".join(lines)

# =========================
# 4) Causal Discovery  (LLM 요약 보고서용 프롬프트 생성)
# =========================

def build_causal_discovery_prompt(state: Dict[str, Any]) -> str:
    """
    4번 보고서(Discovery)는 LLM이 내러티브로 작성.
    이 함수는 state에서 근거를 끌어와 '프롬프트'를 생성한다.
    """
    # 핵심 키들 꺼내기
    data_assumptions = state.get("data_assumptions", {}) or {}
    assumption_method_scores = state.get("assumption_method_scores", {}) or {}
    algorithm_scores = state.get("algorithm_scores", {}) or {}
    selected_algorithms = state.get("selected_algorithms", []) or []

    data_profile = state.get("data_profile", {}) or {}
    algorithm_tiers = state.get("algorithm_tiers", {}) or {}
    tiering_reasoning = state.get("tiering_reasoning", "")
    algorithm_tiering_completed = state.get("algorithm_tiering_completed", False)

    algorithm_results = state.get("algorithm_results", {}) or {}
    candidate_graphs = state.get("candidate_graphs", []) or []
    intermediate_scores = state.get("intermediate_scores", {}) or {}
    pruned_candidates = state.get("pruned_candidates", []) or []
    pruning_log = state.get("pruning_log", []) or []
    scorecard = state.get("scorecard", {}) or {}
    top_candidates = state.get("top_candidates", []) or []
    consensus_pag = state.get("consensus_pag", {}) or {}
    synthesis_reasoning = state.get("synthesis_reasoning", "")
    ensemble_synthesis_completed = state.get("ensemble_synthesis_completed", False)

    selected_graph = state.get("selected_graph", {}) or {}
    graph_selection_reasoning = state.get("graph_selection_reasoning","")

    # 프롬프트 본문
    parts = []
    parts.append("You are a senior data scientist writing a human-friendly *Causal Discovery Report*.")
    parts.append("Summarize clearly with bullet points and short paragraphs. Do not invent facts beyond the provided JSON.")
    parts.append("\n[DATA PROFILE]")
    parts.append(json.dumps({"data_profile": data_profile}, ensure_ascii=False, indent=2))

    parts.append("\n[ASSUMPTION–METHOD COMPATIBILITY]")
    parts.append(json.dumps({
        "data_assumptions": data_assumptions,
        "assumption_method_scores": assumption_method_scores
    }, ensure_ascii=False, indent=2))

    parts.append("\n[ALGORITHM SCORES & TIERING]")
    parts.append(json.dumps({
        "algorithm_scores": algorithm_scores,
        "selected_algorithms": selected_algorithms,
        "algorithm_tiers": algorithm_tiers,
        "tiering_reasoning": tiering_reasoning,
        "algorithm_tiering_completed": algorithm_tiering_completed
    }, ensure_ascii=False, indent=2))

    parts.append("\n[EXECUTION & CANDIDATES]")
    parts.append(json.dumps({
        "algorithm_results": algorithm_results,
        "candidate_graphs_count": len(candidate_graphs),
        "intermediate_scores": intermediate_scores
    }, ensure_ascii=False, indent=2))

    parts.append("\n[PRUNING / EVALUATION / ENSEMBLE]")
    parts.append(json.dumps({
        "pruned_candidates_count": len(pruned_candidates),
        "pruning_log_tail": pruning_log[-3:],
        "scorecard": scorecard,
        "top_candidates_count": len(top_candidates),
        "consensus_pag": consensus_pag,
        "synthesis_reasoning": synthesis_reasoning,
        "ensemble_synthesis_completed": ensemble_synthesis_completed
    }, ensure_ascii=False, indent=2))

    parts.append("\n[FINAL GRAPH DECISION]")
    parts.append(json.dumps({
        "selected_graph": selected_graph,
        "graph_selection_reasoning": graph_selection_reasoning
    }, ensure_ascii=False, indent=2))

    parts.append("""
Write the report with these sections:
1) Overview
2) Data & Assumptions (what was validated; implications)
3) Method Selection (compatibility matrix → chosen algorithms)
4) Execution Summary (what ran; notable diagnostics)
5) Candidate Evaluation (pruning rationale, scorecard, uncertainty)
6) Consensus & Ensemble (if any; how consensus/PAG formed)
7) Final Graph (key edges; brief justification; cautions)
8) Next Steps (data to collect, tests to run, sensitivity checks)
Use concise bullets, include numbers where available, and keep it within ~400-600 words.
""")
    return "\n".join(parts)

# =========================
# 5) Causal Inference
# =========================

def build_causal_inference_section(state: Dict[str, Any]) -> str:
    """
    state 키 기대:
      - parsed_query: Dict[str, Any] (treatment/outcome/confounders)
      - sql_query: str
      - df_raw: DataFrame
      - inference_method or identification_strategy
      - estimator, refuter(optional)
      - causal_estimates: Dict[str, Any] (ATE 등)
      - final_answer: str (요약)
    """
    lines = []
    lines.append(_h("5) Causal Inference Report", 2))

    # Parsed Query
    parsed_query = {
        "treatment": state.get("treatment_variable"),
        "outcome": state.get("outcome_variable"),
        "confounders": state.get("confounders"),
    }
    lines.append(_h("Parsed Query", 3))
    lines.append(dict_to_md(parsed_query))
    lines.append("")

    # SQL
    lines.append(_h("Data Query", 3))
    lines.append(state.get("sql_query","_[no sql]_"))
    lines.append("")

    # Data Preview
    lines.append(_h("Data Preview", 3))
    lines.append(df_preview(state.get("df_raw"), n=10))
    lines.append("")

    # Strategy
    lines.append(_h("Strategy", 3))
    strategy = {
        "identification_method": state.get("inference_method") or state.get("identification_strategy"),
        "estimator": state.get("inference_method"),
        "refuter": state.get("refuter")
    }
    lines.append(dict_to_md(strategy))
    lines.append("")

    # Estimates
    lines.append(_h("Estimates", 3))
    estimates = state.get("causal_estimates", {}) or {}
    if estimates:
        # 표로 예쁘게
        rows = [[k, estimates[k]] for k in sorted(estimates.keys())]
        lines.append(to_markdown_table(rows, header=["metric","value"]))
    else:
        lines.append("- (no estimates)")
    lines.append("")

    # Final Answer
    lines.append(_h("Final Answer", 3))
    lines.append(state.get("final_answer") or state.get("graph_selection_reasoning","_[no answer]_"))
    lines.append(_hr())
    return "\n".join(lines)
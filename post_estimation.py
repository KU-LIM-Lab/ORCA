import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from agents.causal_analysis import generate_causal_analysis_graph
from agents.causal_analysis.state import Strategy
from utils.llm import get_llm
from utils.redis_df import save_df_parquet

def to_dot_from_graph_json(graph_json: dict) -> str:
    """
    graph_json 형식(variables + edges)을 DOT으로 변환.
    고립 노드도 variables 기반으로 모두 명시.
    """
    vars_ = graph_json["graph"].get("variables", [])
    edges = graph_json["graph"].get("edges", [])

    lines = ["digraph G {"]

    # 1) 모든 노드 선언 (고립 노드 포함)
    for v in vars_:
        lines.append(f'  "{v}";')

    # 2) 엣지 선언 (direction이 uncertain이어도 일단 -> 로 넣거나, 원하면 필터링 가능)
    for e in edges:
        u, v = e["from"], e["to"]
        etype = e.get("type", "->")
        if etype == "->":
            lines.append(f'  "{u}" -> "{v}";')
        else:
            # 필요시 o->, o-o 등을 DOT로 표현하는 규칙을 추가
            # 지금은 안전하게 ->로 fallback
            lines.append(f'  "{u}" -> "{v}";')

    lines.append("}")
    return "\n".join(lines)

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _resolve_run_dir(
    base_dir: Path,
    participant_id: str,
    condition: str,
    task_id: str,
    run_id: Optional[str],
) -> Path:
    task_dir = base_dir / participant_id / condition / task_id
    if run_id:
        run_dir = task_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir

    if not task_dir.exists():
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    candidates = [p for p in task_dir.iterdir() if p.is_dir()]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No run directories under: {task_dir}")
    raise ValueError(
        "Multiple run directories found. Provide --run-id to disambiguate: "
        + ", ".join(sorted(p.name for p in candidates))
    )

def _strategy_to_dict(strategy: Any) -> Optional[Dict[str, Any]]:
    if strategy is None:
        return None
    if isinstance(strategy, dict):
        return strategy
    if isinstance(strategy, Strategy):
        return strategy.model_dump()
    model_dump = getattr(strategy, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    return None

def _ensure_output_path(root_dir: Path, participant_id: str) -> Path:
    output_dir = root_dir / "code" / "ORCA" / "user_study" / "ate_result" / "orca"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{participant_id}_ate_result.json"

def main() -> None:
    parser = argparse.ArgumentParser(description="Replay causal effect estimation for user study runs.")
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--condition", default="orca")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--db-id", default="reef_db")
    parser.add_argument("--initial-query", required=True)
    parser.add_argument("--redis-key", default=None)
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[2]
    base_dir = root_dir / "code" / "ORCA" / "user_study" / "runs"

    run_dir = _resolve_run_dir(base_dir, args.participant_id, args.condition, args.task_id, args.run_id)
    artifacts_dir = run_dir / "artifacts"

    df_path = artifacts_dir / "step1_dataset.parquet"
    graph_path = artifacts_dir / "graph_final.json"
    if not df_path.exists():
        raise FileNotFoundError(f"Missing dataset artifact: {df_path}")
    if not graph_path.exists():
        raise FileNotFoundError(f"Missing graph artifact: {graph_path}")

    df = pd.read_parquet(df_path)
    graph_json = load_json(str(graph_path))
    dot_graph = to_dot_from_graph_json(graph_json)

    redis_key = args.redis_key or f"user_study:{args.participant_id}:{args.task_id}:df_preprocessed"
    save_df_parquet(redis_key, df)

    llm = get_llm(model="gpt-4o-mini", temperature=0.3, provider="openai")
    app = generate_causal_analysis_graph(llm=llm)

    state = {
        "input": args.initial_query,
        "initial_query": args.initial_query,
        "db_id": args.db_id,
        "df_redis_key": redis_key,
        "causal_graph": graph_json,
    }
    result_state = app.invoke(state)

    result = {
        "ate": result_state.get("causal_effect_ate"),
        "ci": result_state.get("causal_effect_ci"),
        "refutation_result": result_state.get("refutation_result"),
        "dot_graph": dot_graph,
        "parsed_query": result_state.get("parsed_query"),
        "strategy": _strategy_to_dict(result_state.get("strategy")),
    }
    output_path = _ensure_output_path(root_dir, args.participant_id)
    save_json(str(output_path), result)
    print("Saved:", output_path)

if __name__ == "__main__":
    main()

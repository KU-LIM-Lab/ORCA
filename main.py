import logging
import sys
from typing import Optional, Dict, Any
import argparse

import pandas as pd

from core.state import create_initial_state
from orchestration.graph import create_orchestration_graph
from utils.system_init import initialize_system
from utils.synthetic_data import generate_er_synthetic


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

def run_full_pipeline(
    query: str,
    db_id: str = "reef_db",
    planner_config: Optional[Dict[str, Any]] = None,
    executor_config: Optional[Dict[str, Any]] = None,
    use_synthetic_df: bool = True,
) -> Dict[str, Any]:
    # 1) Initialize system (db + metadata)
    init = initialize_system(db_id, "postgresql", {})
    if not (init and init.is_connected):
        raise RuntimeError("System initialization failed")

    # 2) Build orchestration graph
    graph = create_orchestration_graph(
        planner_config=planner_config,
        executor_config=executor_config,
        metrics_collector=None,
    )
    graph.compile()

    # 3) Create initial state
    state = create_initial_state(query, db_id)
    state["analysis_mode"] = "full_pipeline"
    if use_synthetic_df:
        # Store dataframe externally and pass only a key to keep state serializable for checkpointing
        df, _meta = generate_er_synthetic(n_nodes=5, edge_prob=0.3, n_samples=300, seed=123)
        try:
            from utils.redis_client import redis_client
            key = f"{db_id}:df_preprocessed"
            # Save as JSON to avoid binary; small demo
            redis_client.set(key, df.to_json(orient="split"))
            state["df_preprocessed_key"] = key
        except Exception:
            # Fallback: keep inline (may break checkpointing if large)
            state["df_preprocessed"] = df

    # Optional seeds to pass planner gating for data exploration
    state.setdefault("schema_info", {"tables": []})
    state.setdefault("table_metadata", {})

    # 4) Execute
    # Note: graph.execute() internally uses compiled_graph.invoke() (or stream() for interactive mode)
    # Similar to main_agent.py's app.invoke() pattern, but wrapped with planner+executor orchestration
    result_state = graph.execute(query, context=state)

    # 5) Minimal debug print
    logger.info("Pipeline completed. keys=%s", list(result_state.keys()))
    logger.info("Selected algorithms: %s", result_state.get("selected_algorithms"))
    logger.info("Selected graph edges: %d", len(result_state.get("selected_graph", {}).get("edges", [])))
    if result_state.get("final_report"):
        logger.info("Report sections: %s", list(result_state["final_report"].get("sections", {}).keys()))

    return {
        "success": not bool(result_state.get("error")),
        "state": result_state,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORCA Agent entrypoint")
    parser.add_argument("--query", help="Query to execute (if not provided, reads from terminal)")
    parser.add_argument("--db-id", default="reef_db", help="Database ID")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive (HITL) mode")
    parser.add_argument("--print-report", action="store_true", help="Print final report markdown")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"], help="Log level")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Get query from argument or terminal input
    query = args.query
    if not query:
        print("🤖 ORCA Agent: Enter your query (or 'exit' to quit)")
        query = input("🧑 Query: ").strip()
        if not query or query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            sys.exit(0)

    planner_cfg: Dict[str, Any] = {}
    executor_cfg: Dict[str, Any] = {"interactive": bool(args.interactive)}

    output = run_full_pipeline(
        query,
        db_id=args.db_id,
        planner_config=planner_cfg,
        executor_config=executor_cfg,
        use_synthetic_df=True,
    )
    if output["success"]:
        logger.info("Success")
        if args.print_report and output["state"].get("final_report"):
            fr = output["state"].get("final_report", {})
            print("\n=== Final Report (Markdown) ===")
            print(fr.get("markdown", ""))
    else:
        logger.error("Failed: %s", output["state"].get("error"))
        sys.exit(1)
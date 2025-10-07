import logging
from typing import Optional, Dict, Any

import pandas as pd

from core.state import create_initial_state
from orchestration.graph import create_orchestration_graph
from utils.system_init import initialize_system


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def make_debug_dataframe(n: int = 200) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, n)
    Y = 2.0 * X + rng.normal(0, 0.5, n)
    Z = 1.5 * Y + 0.8 * X + rng.normal(0, 0.3, n)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


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
        df = make_debug_dataframe(300)
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
    result_state = graph.execute(query, context=state)

    # 5) Minimal debug print
    logger.info("Pipeline completed. Keys in final state: %s", list(result_state.keys()))
    logger.info("Selected algorithms: %s", result_state.get("selected_algorithms"))
    logger.info("Selected graph edges: %d", len(result_state.get("selected_graph", {}).get("edges", [])))
    if result_state.get("final_report"):
        logger.info("Report sections: %s", list(result_state["final_report"].get("sections", {}).keys()))

    return {
        "success": not bool(result_state.get("error")),
        "state": result_state,
    }


if __name__ == "__main__":
    output = run_full_pipeline("Run full pipeline for debugging")
    if output["success"]:
        logger.info("Success")
    else:
        logger.error("Failed: %s", output["state"].get("error"))
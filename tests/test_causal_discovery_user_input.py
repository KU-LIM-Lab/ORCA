from __future__ import annotations

import sys
import pandas as pd

from agents.causal_discovery.agent import CausalDiscoveryAgent
from orchestration.executor.agent import ExecutorAgent
from utils.synthetic_data import generate_er_synthetic


def main() -> int:
    """Run causal discovery with a user-selected algorithm override and validate behavior.

    Returns 0 on success, non-zero on failure.
    """
    # 1) Generate a small synthetic dataset
    df, _meta = generate_er_synthetic(n_nodes=5, edge_prob=0.3, n_samples=300, seed=123)
    if not isinstance(df, pd.DataFrame):
        print("ERROR: Generated data is not a pandas DataFrame", file=sys.stderr)
        return 1

    # 2) Prepare initial state and agents
    state = {
        "df_preprocessed": df,
    }

    executor = ExecutorAgent()
    cd_agent = CausalDiscoveryAgent()

    # 3) Run causal discovery profiling and tiering
    state["current_substep"] = "data_profiling"
    state = cd_agent.step(state)
    if state.get("error"):
        print(f"ERROR during data_profiling: {state.get('error')}", file=sys.stderr)
        return 1
    if not state.get("data_profiling_completed"):
        print("ERROR: Data profiling did not complete", file=sys.stderr)
        return 1

    state["current_substep"] = "algorithm_tiering"
    state = cd_agent.step(state)
    if state.get("error"):
        print(f"ERROR during algorithm_tiering: {state.get('error')}", file=sys.stderr)
        return 1
    if not state.get("algorithm_tiering_completed"):
        print("ERROR: Algorithm tiering did not complete", file=sys.stderr)
        return 1

    # 4) Simulate HITL edit: user selects PC only
    executor._apply_edits(state, {"selected_algorithms": ["PC"]})
    if state.get("selected_algorithms") != ["PC"]:
        print("ERROR: selected_algorithms was not applied via _apply_edits", file=sys.stderr)
        return 1

    # 5) Run the algorithm portfolio; agent가 앞선 HITL 선택을 반영하는지 확인
    state["current_substep"] = "run_algorithms_portfolio"
    state = cd_agent.step(state)
    if state.get("error"):
        print(f"ERROR during run_algorithms_portfolio: {state.get('error')}", file=sys.stderr)
        return 1

    algo_results = state.get("algorithm_results", {}) or {}
    executed = set(algo_results.keys())

    # Expect only PC results when override is present
    if executed != {"PC"}:
        print(f"ERROR: Expected only PC to run, but executed: {executed}", file=sys.stderr)
        return 1

    print("SUCCESS: Causal discovery respected user-selected algorithm override (PC only).")
    return 0


if __name__ == "__main__":
    sys.exit(main())


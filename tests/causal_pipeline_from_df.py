# examples/causal_pipeline_from_df.py
"""
Run causal discovery → causal inference → report generation directly from an in-memory DataFrame,
skipping data exploration steps.
"""

from datetime import datetime
import pandas as pd

from orchestration.graph import create_orchestration_graph
from tests.synthetic_data import generate_er_synthetic


def main():
    # Use synthetic data generator
    df, meta = generate_er_synthetic(n_nodes=10, edge_prob=0.4, n_samples=500, seed=123)
    # Pick default treatment/outcome as first two variables
    variables = meta.get("variables", list(df.columns))
    treatment = variables[0] if variables else df.columns[0]
    outcome = variables[1] if len(variables) > 1 else df.columns[min(1, len(df.columns)-1)]

    graph = create_orchestration_graph(
        metrics_collector=None,
        orchestration_config={"interactive": False},
    )

    # Provide ground-truth DataFrame and skip early phases
    context = {
        "gt_df": df,
        "skip": ["table_selection", "table_retrieval", "data_preprocessing"],
        # Provide labels for inference
        "treatment": treatment,
        "outcome": outcome,
    }

    print("=== Run Causal Pipeline From In-Memory DataFrame ===")
    result = graph.execute("Run from df", context=context, session_id=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    print("Status:", result.get("execution_status"))
    print("Completed substeps:", len(result.get("completed_substeps", []) or []))
    print("Selected graph present:", bool(result.get("selected_graph")))
    print("Causal estimates present:", bool(result.get("causal_estimates")))
    print("Final report present:", bool(result.get("final_report")))


if __name__ == "__main__":
    main()



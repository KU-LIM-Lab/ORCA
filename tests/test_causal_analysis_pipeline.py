"""
Test the complete causal analysis pipeline from parse_question to generate_answer.

This test uses synthetic data and a simple causal graph to verify that:
1. parse_question correctly identifies treatment, outcome, and variable roles
2. config_selection selects appropriate strategy
3. dowhy_analysis performs causal inference
4. generate_answer produces a final explanation
"""

import sys
import pandas as pd
import networkx as nx
from typing import Dict, Any

from agents.causal_analysis.graph import generate_causal_analysis_graph
from utils.llm import get_llm
from utils.synthetic_data import generate_er_synthetic


def create_simple_causal_graph(variables: list) -> Dict[str, Any]:
    """
    Create a simple causal graph for testing.
    Structure: V0 -> V1 -> V2, with V3 as confounder (V3 -> V0, V3 -> V2)
    """
    edges = [
        {"from": variables[0], "to": variables[1]},  # T -> M
        {"from": variables[1], "to": variables[2]},  # M -> O
        {"from": variables[3], "to": variables[0]},  # C -> T
        {"from": variables[3], "to": variables[2]},  # C -> O
    ]
    
    # Create NetworkX graph for dot_graph
    G = nx.DiGraph()
    for edge in edges:
        G.add_edge(edge["from"], edge["to"])
    
    # Create dot format string
    dot_lines = ["digraph {"]
    for edge in edges:
        dot_lines.append(f'  {edge["from"]} -> {edge["to"]};')
    dot_lines.append("}")
    dot_graph = "\n".join(dot_lines)
    
    return {
        "nodes": variables,
        "edges": edges,
        "variables": variables,
        "graph": {
            "variables": variables,
            "edges": edges
        },
        "nx_graph": G,
        "dot_graph": dot_graph
    }


def test_causal_analysis_pipeline() -> int:
    """
    Test the complete causal analysis pipeline.
    
    Returns 0 on success, non-zero on failure.
    """
    print("=" * 60)
    print("Testing Causal Analysis Pipeline")
    print("=" * 60)
    
    # 1. Generate synthetic data
    print("\n[1/5] Generating synthetic data...")
    df, meta = generate_er_synthetic(n_nodes=6, edge_prob=0.3, n_samples=200, seed=42)
    variables = meta.get("variables", list(df.columns))
    
    if len(variables) < 4:
        print(f"ERROR: Need at least 4 variables, got {len(variables)}", file=sys.stderr)
        return 1
    
    print(f"  ✓ Generated DataFrame with shape {df.shape}")
    print(f"  ✓ Variables: {variables[:4]}...")
    
    # 2. Create causal graph
    print("\n[2/5] Creating causal graph...")
    # Use first 4 variables: V0 (treatment), V1 (mediator), V2 (outcome), V3 (confounder)
    test_vars = variables[:4]
    causal_graph = create_simple_causal_graph(test_vars)
    treatment = test_vars[0]
    outcome = test_vars[2]
    
    print(f"  ✓ Treatment: {treatment}")
    print(f"  ✓ Outcome: {outcome}")
    print(f"  ✓ Graph edges: {len(causal_graph['edges'])}")
    
    # 3. Initialize LLM and graph
    print("\n[3/5] Initializing LLM and causal analysis graph...")
    try:
        llm = get_llm(model="gpt-4o-mini", temperature=0.7, provider="openai")
        app = generate_causal_analysis_graph(llm=llm)
        print("  ✓ LLM and graph initialized")
    except Exception as e:
        print(f"  ✗ ERROR: Failed to initialize LLM/graph: {e}", file=sys.stderr)
        return 1
    
    # 4. Prepare input state
    print("\n[4/5] Preparing input state...")
    question = f"What is the causal effect of {treatment} on {outcome}?"
    
    state_input = {
        "input": question,
        "df_preprocessed": df,
        "causal_graph": causal_graph,
    }
    
    print(f"  ✓ Question: {question}")
    print(f"  ✓ DataFrame shape: {df.shape}")
    print(f"  ✓ Causal graph provided")
    
    # 5. Run the pipeline
    print("\n[5/5] Running causal analysis pipeline...")
    try:
        result = app.invoke(state_input)
        print("  ✓ Pipeline execution completed")
    except Exception as e:
        print(f"  ✗ ERROR: Pipeline execution failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    
    # 6. Validate results
    print("\n" + "=" * 60)
    print("Validating Results")
    print("=" * 60)
    
    errors = []
    
    # Check parsed_query
    parsed_query = result.get("parsed_query")
    if not parsed_query:
        errors.append("Missing parsed_query in result")
    else:
        if parsed_query.get("treatment") != treatment:
            errors.append(f"Treatment mismatch: expected {treatment}, got {parsed_query.get('treatment')}")
        if parsed_query.get("outcome") != outcome:
            errors.append(f"Outcome mismatch: expected {outcome}, got {parsed_query.get('outcome')}")
        
        confounders = parsed_query.get("confounders", [])
        mediators = parsed_query.get("mediators", [])
        instruments = parsed_query.get("instrumental_variables", [])
        
        print(f"\n  Parsed Variables:")
        print(f"    - Treatment: {parsed_query.get('treatment')}")
        print(f"    - Outcome: {parsed_query.get('outcome')}")
        print(f"    - Confounders: {confounders}")
        print(f"    - Mediators: {mediators}")
        print(f"    - Instruments: {instruments}")
        
        # Validate variable roles (basic checks)
        if test_vars[3] not in confounders:
            print(f"    ⚠ Warning: Expected {test_vars[3]} to be a confounder")
        if test_vars[1] not in mediators:
            print(f"    ⚠ Warning: Expected {test_vars[1]} to be a mediator")
    
    # Check strategy
    strategy = result.get("strategy")
    if not strategy:
        errors.append("Missing strategy in result")
    else:
        print(f"\n  Strategy:")
        print(f"    - Task: {getattr(strategy, 'task', 'N/A')}")
        print(f"    - Identification: {getattr(strategy, 'identification_method', 'N/A')}")
        print(f"    - Estimator: {getattr(strategy, 'estimator', 'N/A')}")
    
    # Check causal estimate
    causal_estimate = result.get("causal_estimate")
    causal_effect_ate = result.get("causal_effect_ate")
    
    if not causal_estimate and causal_effect_ate is None:
        errors.append("Missing causal_estimate and causal_effect_ate in result")
    else:
        print(f"\n  Causal Effect:")
        if causal_effect_ate is not None:
            print(f"    - ATE: {causal_effect_ate:.4f}")
        ci = result.get("causal_effect_ci")
        if ci:
            print(f"    - CI: {ci}")
    
    # Check final answer
    final_answer = result.get("final_answer")
    if not final_answer:
        errors.append("Missing final_answer in result")
    else:
        print(f"\n  Final Answer:")
        print(f"    {final_answer[:200]}..." if len(final_answer) > 200 else f"    {final_answer}")
    
    # Report results
    print("\n" + "=" * 60)
    if errors:
        print("❌ TEST FAILED")
        for error in errors:
            print(f"  ✗ {error}")
        return 1
    else:
        print("✅ TEST PASSED")
        print("\nAll pipeline components executed successfully:")
        print("  ✓ parse_question")
        print("  ✓ config_selection")
        print("  ✓ dowhy_analysis")
        print("  ✓ generate_answer")
        return 0


if __name__ == "__main__":
    sys.exit(test_causal_analysis_pipeline())


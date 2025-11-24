from __future__ import annotations

import json
import time
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from datetime import datetime

from cdt.metrics import SHD, SID


from utils.settings import CONFIG
from agents.causal_discovery.agent import CausalDiscoveryAgent
from agents.data_explorer.data_preprocessor.agent import DataPreprocessorAgent
from agents.causal_discovery.tools import (
    LiNGAMTool, ANMTool, PCTool, GESTool, FCITool, CAMTool
)
from utils.synthetic_data import generate_er_synthetic
import networkx as nx

"""
Comparison script for individual algorithms vs causal discovery agent.

This script:
1. Runs individual causal discovery algorithms directly (PC, GES, LiNGAM, ANM, CAM, FCI)
2. Runs the same algorithms through the causal discovery agent pipeline
3. Evaluates both approaches using SHD and SID metrics against ground truth
4. Generates a comprehensive comparison report
"""


def convert_graph_to_networkx(graph: Dict[str, Any]) -> nx.DiGraph:
    """Convert ORCA graph format to NetworkX DiGraph
    
    Args:
        graph: ORCA graph dictionary with "graph" key containing "variables" and "edges"
        
    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()
    
    if not graph or "error" in graph:
        return G
    
    # Extract variables and edges
    if "graph" in graph:
        variables = graph["graph"].get("variables", [])
        edges = graph["graph"].get("edges", [])
    elif "variables" in graph:
        variables = graph["variables"]
        edges = graph.get("edges", [])
    else:
        return G
    
    # Add nodes
    G.add_nodes_from(variables)
    
    # Add edges
    for edge in edges:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            from_node = edge["from"]
            to_node = edge["to"]
            if from_node in variables and to_node in variables:
                G.add_edge(from_node, to_node)
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            from_node = str(edge[0])
            to_node = str(edge[1])
            if from_node in variables and to_node in variables:
                G.add_edge(from_node, to_node)
    
    return G


def calculate_shd_cdt(true_graph: Dict[str, Any], pred_graph: Dict[str, Any]) -> int:
    """Calculate Structural Hamming Distance using CDT library
    
    Args:
        true_graph: Ground truth graph in ORCA format
        pred_graph: Predicted graph in ORCA format
        
    Returns:
        SHD value (integer)
    """
    
    true_nx = convert_graph_to_networkx(true_graph)
    pred_nx = convert_graph_to_networkx(pred_graph)
    
    shd_value = SHD(true_nx, pred_nx)
    return int(shd_value)



def calculate_sid_cdt(true_graph: Dict[str, Any], pred_graph: Dict[str, Any]) -> float:
    """Calculate Structural Intervention Distance using CDT library
    
    Args:
        true_graph: Ground truth graph in ORCA format
        pred_graph: Predicted graph in ORCA format
        
    Returns:
        SID value (float), or float('inf') if calculation fails
    """
    try:
        true_nx = convert_graph_to_networkx(true_graph)
        pred_nx = convert_graph_to_networkx(pred_graph)
        
        sid_value = SID(true_nx, pred_nx)
        return float(sid_value)
    except Exception as e:
        # CDT SID requires R package, if not available, return inf
        error_msg = str(e).lower()
        if "r package" in error_msg or "sid" in error_msg.lower():
            # Silently return inf if R package not available
            return float('inf')
        else:
            # Other errors, log and return inf
            print(f"Warning: SID calculation failed: {e}")
            return float('inf')


def extract_edges_from_graph(graph: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract edge list from graph structure"""
    if not graph or "error" in graph:
        return []
    
    edges_list = []
    if "graph" in graph and "edges" in graph["graph"]:
        edges_list = graph["graph"]["edges"]
    elif "edges" in graph:
        edges_list = graph["edges"]
    else:
        return []
    
    edges = []
    for edge in edges_list:
        if isinstance(edge, dict) and "from" in edge and "to" in edge:
            edges.append((edge["from"], edge["to"]))
        elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
            edges.append((str(edge[0]), str(edge[1])))
    
    return edges


def calculate_graph_metrics(predicted_edges: List[Tuple[str, str]], 
                          true_edges: List[Tuple[str, str]], 
                          all_variables: List[str]) -> Dict[str, float]:
    """Calculate precision, recall, F1, and structural Hamming distance"""
    
    # Convert to sets for easier comparison
    pred_set = set(predicted_edges)
    true_set = set(true_edges)
    
    # Calculate basic metrics
    true_positives = len(pred_set & true_set)
    false_positives = len(pred_set - true_set)
    false_negatives = len(true_set - pred_set)
    
    # Precision, Recall, F1
    precision = true_positives / len(pred_set) if pred_set else 0.0
    recall = true_positives / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Structural Hamming Distance (SHD)
    # SHD = |E_pred - E_true| + |E_true - E_pred|
    shd = len(pred_set - true_set) + len(true_set - pred_set)
    
    # Normalize SHD by maximum possible edges
    max_edges = len(all_variables) * (len(all_variables) - 1)
    normalized_shd = shd / max_edges if max_edges > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "shd": shd,
        "normalized_shd": normalized_shd,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_predicted": len(pred_set),
        "total_true": len(true_set)
    }


def evaluate_graph_with_sid(discovered_graph: Dict[str, Any], 
                            ground_truth: Dict[str, Any], 
                            variables: List[str]) -> Dict[str, Any]:
    """Evaluate discovered graph against ground truth with SHD and SID using CDT"""
    
    # Extract edges for basic metrics calculation
    pred_edges = extract_edges_from_graph(discovered_graph)
    true_edges = ground_truth.get("edges", [])
    
    # Convert true edges to tuples if needed
    if true_edges and isinstance(true_edges[0], dict):
        true_edges = [(e["from"], e["to"]) for e in true_edges if "from" in e and "to" in e]
    
    # Calculate basic metrics (precision, recall, F1)
    basic_metrics = calculate_graph_metrics(pred_edges, true_edges, variables)
    
    # Convert ground truth to ORCA graph format if needed
    true_graph = ground_truth
    if "graph" not in true_graph:
        true_graph = {
            "graph": {
                "variables": variables,
                "edges": [{"from": e[0], "to": e[1]} for e in true_edges] if (true_edges and isinstance(true_edges[0], tuple)) else true_edges
            }
        }
    
    # Ensure discovered graph has proper structure
    disc_graph = discovered_graph
    if "graph" not in disc_graph and "error" not in disc_graph:
        disc_graph = {
            "graph": {
                "variables": variables,
                "edges": [{"from": e[0], "to": e[1]} for e in pred_edges]
            }
        }
    
    # Calculate SHD using CDT
    shd_raw = calculate_shd_cdt(true_graph, disc_graph)
    
    # Normalize SHD by maximum possible edges
    max_edges = len(variables) * (len(variables) - 1)
    normalized_shd = shd_raw / max_edges if max_edges > 0 else 0.0
    
    # Calculate SID using CDT
    sid_value = calculate_sid_cdt(true_graph, disc_graph)
    
    # Normalize SID by number of nodes (if not infinite)
    normalized_sid = sid_value / len(variables) if (sid_value != float('inf') and len(variables) > 0) else 1.0
    
    return {
        "shd": shd_raw,
        "normalized_shd": normalized_shd,
        "sid": normalized_sid,
        "sid_raw": sid_value,
        "precision": basic_metrics["precision"],
        "recall": basic_metrics["recall"],
        "f1_score": basic_metrics["f1_score"],
        "true_positives": basic_metrics["true_positives"],
        "false_positives": basic_metrics["false_positives"],
        "false_negatives": basic_metrics["false_negatives"]
    }


def run_individual_algorithm(algorithm_name: str, df: pd.DataFrame, 
                            variable_schema: Optional[Dict[str, Any]] = None,
                            **kwargs) -> Dict[str, Any]:
    """Run individual algorithm directly using Tool classes
    
    Args:
        algorithm_name: Name of algorithm (PC, GES, LiNGAM, ANM, CAM, FCI)
        df: DataFrame to analyze
        variable_schema: Optional variable schema for mixed data algorithms
        **kwargs: Additional algorithm-specific parameters
        
    Returns:
        Dictionary with graph result and execution time
    """
    start_time = time.time()
    result = {"error": None, "graph": None, "execution_time": 0.0}
    
    try:
        if algorithm_name == "LiNGAM":
            graph_result = LiNGAMTool.direct_lingam(df)
        elif algorithm_name == "ANM":
            delta = kwargs.get("delta", 0.02)
            tau = kwargs.get("tau", 0.05)
            graph_result = ANMTool.anm_discovery(df, delta=delta, tau=tau)
        elif algorithm_name == "PC":
            alpha = kwargs.get("alpha", 0.05)
            indep_test = kwargs.get("indep_test", "fisherz")
            graph_result = PCTool.discover(df, alpha=alpha, indep_test=indep_test, 
                                         variable_schema=variable_schema)
        elif algorithm_name == "GES":
            score_func = kwargs.get("score", "bic-g")
            graph_result = GESTool.discover(df, score_func=score_func)
        elif algorithm_name == "CAM":
            graph_result = CAMTool.discover(df, **kwargs)
        elif algorithm_name == "FCI":
            alpha = kwargs.get("alpha", 0.05)
            indep_test = kwargs.get("indep_test", "fisherz")
            graph_result = FCITool.discover(df, alpha=alpha, indep_test=indep_test,
                                           variable_schema=variable_schema)
        else:
            result["error"] = f"Unknown algorithm: {algorithm_name}"
            return result
        
        execution_time = time.time() - start_time
        
        if "error" in graph_result:
            result["error"] = graph_result["error"]
        else:
            result["graph"] = graph_result
            result["execution_time"] = execution_time
            
    except Exception as e:
        result["error"] = str(e)
        result["execution_time"] = time.time() - start_time
    
    return result


def run_agent_pipeline(df: pd.DataFrame, ground_truth: Dict[str, Any],
                       algorithm_name: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run algorithm through causal discovery agent pipeline
    
    Args:
        df: DataFrame to analyze
        ground_truth: Ground truth graph for evaluation
        algorithm_name: Algorithm name (for filtering results)
        config: Optional agent configuration
        
    Returns:
        Dictionary with graph result, execution time, and agent scores
    """
    # Step 1: Preprocessing
    preprocessor = DataPreprocessorAgent()
    preprocessor.df = df.copy()
    preprocessor._data_fetched = True
    
    prep_state: Dict[str, Any] = {
        "df_preprocessed": df.copy(),
        "db_id": "test",
        "skip_one_hot_encoding": True
    }
    
    prep_state["current_substep"] = "schema_detection"
    prep_state = preprocessor.step(prep_state)
    if prep_state.get("error"):
        return {"error": f"Schema detection failed: {prep_state.get('error')}"}
    
    prep_state["current_substep"] = "clean_nulls"
    prep_state = preprocessor.step(prep_state)
    if prep_state.get("error"):
        return {"error": f"Clean nulls failed: {prep_state.get('error')}"}
    
    # Step 2: Create agent with config
    if config is None:
        cd_config = CONFIG.get("agents", {}).get("causal_discovery", {}).copy()
        cd_config["bootstrap_iterations"] = 2  # Reduced for faster testing
        cd_config["n_subsets"] = 2
    else:
        cd_config = config
    
    agent = CausalDiscoveryAgent(config=cd_config)
    
    # Step 3: Run agent pipeline
    state: Dict[str, Any] = {
        "df_preprocessed": preprocessor.df if preprocessor.df is not None else df,
        "variable_schema": prep_state.get("variable_schema", {}),
        "db_id": "test"
    }
    
    substeps = [
        "data_profiling",
        "algorithm_configuration",
        "run_algorithms_portfolio",
        "graph_scoring",
        "graph_evaluation",
        "ensemble_synthesis",
    ]
    
    start_time = time.time()
    result = {
        "error": None,
        "graph": None,
        "execution_time": 0.0,
        "agent_scores": None,
        "algorithm_found": False
    }
    
    try:
        for sub in substeps:
            state["current_substep"] = sub
            state = agent.step(state)
            
            if state.get("error"):
                result["error"] = state["error"]
                break
        
        execution_time = time.time() - start_time
        result["execution_time"] = execution_time
        
        # Extract graph for the specific algorithm from scorecard
        # The agent runs all algorithms, so we need to find the one we're testing
        scorecard = state.get("scorecard", [])
        algorithm_results = state.get("algorithm_results", {})
        
        # First, try to find the algorithm in scorecard (scored and evaluated)
        algorithm_found = False
        for entry in scorecard:
            if entry.get("algorithm") == algorithm_name:
                result["graph"] = entry.get("graph")
                result["agent_scores"] = {
                    "global_consistency": entry.get("global_consistency"),
                    "sampling_stability": entry.get("sampling_stability"),
                    "structural_stability": entry.get("structural_stability")
                }
                algorithm_found = True
                break
        
        # If not in scorecard, try algorithm_results (raw results)
        if not algorithm_found and algorithm_name in algorithm_results:
            alg_result = algorithm_results[algorithm_name]
            if "error" not in alg_result:
                result["graph"] = alg_result
                algorithm_found = True
        
        result["algorithm_found"] = algorithm_found
        
        # Also store the selected graph from ensemble (for reference)
        selected_graph = state.get("selected_graph")
        if selected_graph:
            result["selected_graph"] = selected_graph
        
    except Exception as e:
        result["error"] = str(e)
        result["execution_time"] = time.time() - start_time
    
    return result


def compare_algorithm_vs_agent(df: pd.DataFrame, ground_truth: Dict[str, Any],
                               algorithm_name: str, variable_schema: Optional[Dict[str, Any]] = None,
                               agent_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Compare individual algorithm vs agent pipeline for a single algorithm
    
    Args:
        df: DataFrame to analyze
        ground_truth: Ground truth graph
        algorithm_name: Algorithm to compare
        variable_schema: Optional variable schema
        agent_config: Optional agent configuration
        
    Returns:
        Comparison results dictionary
    """
    variables = list(df.columns)
    
    alg_params = {}
    if algorithm_name == "PC":
        alg_params = {"alpha": 0.05, "indep_test": "fisherz"}
    elif algorithm_name == "FCI":
        alg_params = {"alpha": 0.05, "indep_test": "fisherz"}
    elif algorithm_name == "GES":
        alg_params = {"score": "bic-g"}
    
    individual_result = run_individual_algorithm(
        algorithm_name, df, variable_schema=variable_schema, **alg_params
    )
    
    agent_result = run_agent_pipeline(df, ground_truth, algorithm_name, config=agent_config)
    
    individual_metrics = None
    if individual_result.get("graph") and not individual_result.get("error"):
        individual_metrics = evaluate_graph_with_sid(
            individual_result["graph"], ground_truth, variables
        )
    
    agent_metrics = None
    if agent_result.get("graph") and not agent_result.get("error"):
        agent_metrics = evaluate_graph_with_sid(
            agent_result["graph"], ground_truth, variables
        )
    
    improvement = {}
    if individual_metrics and agent_metrics:
        improvement = {
            "shd_diff": agent_metrics["normalized_shd"] - individual_metrics["normalized_shd"],
            "sid_diff": agent_metrics["sid"] - individual_metrics["sid"],
            "time_overhead": agent_result["execution_time"] - individual_result["execution_time"],
            "shd_improvement": individual_metrics["normalized_shd"] - agent_metrics["normalized_shd"],  # positive = improvement
            "sid_improvement": individual_metrics["sid"] - agent_metrics["sid"]  # positive = improvement
        }
    
    return {
        "algorithm": algorithm_name,
        "individual": {
            "shd": individual_metrics["normalized_shd"] if individual_metrics else None,
            "sid": individual_metrics["sid"] if individual_metrics else None,
            "execution_time": individual_result["execution_time"],
            "error": individual_result.get("error"),
            "graph": individual_result.get("graph")
        },
        "agent": {
            "shd": agent_metrics["normalized_shd"] if agent_metrics else None,
            "sid": agent_metrics["sid"] if agent_metrics else None,
            "execution_time": agent_result["execution_time"],
            "error": agent_result.get("error"),
            "graph": agent_result.get("graph"),
            "agent_scores": agent_result.get("agent_scores"),
            "algorithm_found": agent_result.get("algorithm_found", False)
        },
        "improvement": improvement if improvement else None
    }


def run_comparison_experiment(n_nodes: int = 5, edge_prob: float = 0.3, 
                             n_samples: int = 200, seed: int = 42,
                             algorithms: Optional[List[str]] = None) -> Dict[str, Any]:
    """Run full comparison experiment
    
    Args:
        n_nodes: Number of nodes in synthetic graph
        edge_prob: Edge probability for ER graph
        n_samples: Number of samples
        seed: Random seed
        algorithms: List of algorithms to test (None = all)
        
    Returns:
        Complete comparison results
    """
    if algorithms is None:
        algorithms = ["PC", "GES", "LiNGAM", "ANM", "CAM", "FCI"]
    
    print(f"Generating synthetic data: {n_nodes} nodes, {n_samples} samples, seed={seed}")
    df, meta = generate_er_synthetic(n_nodes=n_nodes, edge_prob=edge_prob, 
                                     n_samples=n_samples, seed=seed)
    
    # Extract ground truth
    ground_truth = {
        "edges": meta.get("edges", []),
        "variables": list(df.columns)
    }
    
    # Convert ground truth edges to dict format if needed
    if ground_truth["edges"] and isinstance(ground_truth["edges"][0], tuple):
        ground_truth["edges"] = [{"from": e[0], "to": e[1]} for e in ground_truth["edges"]]
    
    print(f"Ground truth: {len(ground_truth['edges'])} edges")
    print(f"Testing algorithms: {', '.join(algorithms)}\n")
    
    # Run preprocessing once
    preprocessor = DataPreprocessorAgent()
    preprocessor.df = df.copy()
    preprocessor._data_fetched = True
    
    prep_state: Dict[str, Any] = {
        "df_preprocessed": df.copy(),
        "db_id": "test",
        "skip_one_hot_encoding": True
    }
    
    prep_state["current_substep"] = "schema_detection"
    prep_state = preprocessor.step(prep_state)
    prep_state["current_substep"] = "clean_nulls"
    prep_state = preprocessor.step(prep_state)
    
    variable_schema = prep_state.get("variable_schema", {})
    
    # Agent config
    agent_config = CONFIG.get("agents", {}).get("causal_discovery", {}).copy()
    agent_config["bootstrap_iterations"] = 2
    agent_config["n_subsets"] = 2
    
    # Compare each algorithm
    results = []
    for alg in algorithms:
        print(f"Comparing {alg}...")
        try:
            result = compare_algorithm_vs_agent(
                df, ground_truth, alg, 
                variable_schema=variable_schema,
                agent_config=agent_config
            )
            results.append(result)
            
            # Print summary
            if result["individual"].get("error"):
                print(f"  Individual: ERROR - {result['individual']['error']}")
            else:
                print(f"  Individual: SHD={result['individual']['shd']:.4f}, "
                      f"SID={result['individual']['sid']:.4f}, "
                      f"Time={result['individual']['execution_time']:.2f}s")
            
            if result["agent"].get("error"):
                print(f"  Agent: ERROR - {result['agent']['error']}")
            elif not result["agent"].get("algorithm_found"):
                print(f"  Agent: Algorithm not found in results")
            else:
                print(f"  Agent: SHD={result['agent']['shd']:.4f}, "
                      f"SID={result['agent']['sid']:.4f}, "
                      f"Time={result['agent']['execution_time']:.2f}s")
            
            if result.get("improvement"):
                imp = result["improvement"]
                print(f"  Improvement: SHD={imp['shd_improvement']:+.4f}, "
                      f"SID={imp['sid_improvement']:+.4f}, "
                      f"Time overhead={imp['time_overhead']:.2f}s")
            print()
            
        except Exception as e:
            print(f"  ERROR: {e}\n")
            results.append({
                "algorithm": alg,
                "error": str(e),
                "individual": None,
                "agent": None,
                "improvement": None
            })
    
    return {
        "experiment_config": {
            "n_nodes": n_nodes,
            "edge_prob": edge_prob,
            "n_samples": n_samples,
            "seed": seed,
            "algorithms": algorithms
        },
        "ground_truth": {
            "n_edges": len(ground_truth["edges"]),
            "n_variables": len(ground_truth["variables"])
        },
        "results": results,
        "timestamp": datetime.now().isoformat()
    }


def generate_summary_report(experiment_results: Dict[str, Any]) -> pd.DataFrame:
    """Generate summary DataFrame from experiment results"""
    rows = []
    
    for result in experiment_results["results"]:
        if result.get("error"):
            continue
        
        alg = result["algorithm"]
        ind = result.get("individual", {})
        agent = result.get("agent", {})
        imp = result.get("improvement")
        
        # Handle case where improvement is None
        if imp is None:
            imp = {}
        
        rows.append({
            "algorithm": alg,
            "individual_shd": ind.get("shd") if ind else None,
            "individual_sid": ind.get("sid") if ind else None,
            "individual_time": ind.get("execution_time") if ind else None,
            "individual_error": ind.get("error") if ind else None,
            "agent_shd": agent.get("shd") if agent else None,
            "agent_sid": agent.get("sid") if agent else None,
            "agent_time": agent.get("execution_time") if agent else None,
            "agent_error": agent.get("error") if agent else None,
            "agent_found": agent.get("algorithm_found") if agent else False,
            "shd_improvement": imp.get("shd_improvement") if imp else None,
            "sid_improvement": imp.get("sid_improvement") if imp else None,
            "time_overhead": imp.get("time_overhead") if imp else None
        })
    
    return pd.DataFrame(rows)


def main():
    """Main function to run comparison experiment"""
    print("=" * 80)
    print("Algorithm vs Agent Performance Comparison")
    print("=" * 80)
    print()
    
    # Run experiment
    results = run_comparison_experiment(
        n_nodes=5,
        edge_prob=0.3,
        n_samples=200,
        seed=42
    )
    
    # Generate summary
    summary_df = generate_summary_report(results)
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_file = f"tests/comparison_results_{timestamp}.json"
    csv_file = f"tests/comparison_summary_{timestamp}.csv"
    
    # Save detailed JSON
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results saved to: {json_file}")
    
    # Save summary CSV
    summary_df.to_csv(csv_file, index=False)
    print(f"Summary saved to: {csv_file}")
    
    return results


if __name__ == "__main__":
    main()


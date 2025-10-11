from __future__ import annotations

import pprint
from typing import Dict, Any, List, Tuple
import numpy as np

from utils.settings import CONFIG
from agents.causal_discovery.agent import CausalDiscoveryAgent
from tests.synthetic_data import generate_er_synthetic

"""
This script demonstrates:
- Profile classification (quantile vs fixed thresholds)
- Algorithm tiering
- Detailed diagnostics for each pipeline stage
- Rejection reasons for candidate pruning
- Ground truth comparison and evaluation metrics
"""


def print_step(header: str, payload: Dict[str, Any]):
    print("\n===" , header, "===")
    pp = pprint.PrettyPrinter(indent=2, width=120, compact=True)
    pp.pprint(payload)


def extract_edges_from_graph(graph: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Extract edge list from graph structure"""
    if not graph or "graph" not in graph or "edges" not in graph["graph"]:
        return []
    
    edges = []
    for edge in graph["graph"]["edges"]:
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


def evaluate_against_ground_truth(discovered_graph: Dict[str, Any], 
                                ground_truth: Dict[str, Any], 
                                variables: List[str]) -> Dict[str, Any]:
    """Evaluate discovered graph against ground truth"""
    
    # Extract edges
    pred_edges = extract_edges_from_graph(discovered_graph)
    true_edges = ground_truth.get("edges", [])
    
    # Convert true edges to tuples if needed
    if true_edges and isinstance(true_edges[0], dict):
        true_edges = [(e["from"], e["to"]) for e in true_edges if "from" in e and "to" in e]
    
    # Calculate metrics
    metrics = calculate_graph_metrics(pred_edges, true_edges, variables)
    
    return {
        "metrics": metrics,
        "predicted_edges": pred_edges,
        "true_edges": true_edges,
        "edge_comparison": {
            "correct_edges": list(set(pred_edges) & set(true_edges)),
            "missing_edges": list(set(true_edges) - set(pred_edges)),
            "extra_edges": list(set(pred_edges) - set(true_edges))
        }
    }


def main():
    # Load config
    cd_config = (CONFIG.get("agents", {}) or {}).get("causal_discovery", {})

    # Generate synthetic ER dataset
    df, meta = generate_er_synthetic(n_nodes=8, edge_prob=0.25, n_samples=1500, seed=7)
    print_step("Synthetic meta (first 5 edges)", {"edges": meta.get("edges", [])[:5], "order": meta.get("order")})

    # Build agent
    agent = CausalDiscoveryAgent(config=cd_config)
    state: Dict[str, Any] = {"df_preprocessed": df}
    
    # Store ground truth for later evaluation
    ground_truth = {
        "edges": meta.get("edges", []),
        "variables": list(df.columns)
    }

    substeps = [
        "data_profiling",
        "algorithm_tiering",
        "run_algorithms_portfolio",
        "candidate_pruning",
        "scorecard_evaluation",
        "ensemble_synthesis",
    ]

    for sub in substeps:
        state["current_substep"] = sub
        state = agent.step(state)
        if state.get("error"):
            print_step(f"{sub} ERROR", {"error": state["error"]})
            break
        # Print concise diagnostics per substep
        if sub == "data_profiling":
            dp = state.get("data_profile", {})
            print_step("data_profiling", {
                "summary": dp.get("summary"),
                "n_pairs": dp.get("n_pairs"),
                "classifications": {
                    "linearity": dp.get("linearity"),
                    "non_gaussian": dp.get("non_gaussian"),
                    "anm_compatible": dp.get("anm_compatible"),
                    "gaussian": dp.get("gaussian"),
                    "equal_variance": dp.get("equal_variance")
                },
                "aggregated_scores": {
                    k: {"mean": round(v["mean"], 3), "std": round(v["std"], 3)} 
                    for k, v in dp.get("aggregated_scores", {}).items()
                }
            })
        elif sub == "algorithm_tiering":
            tiers = state.get("algorithm_tiers", {})
            reasoning = state.get("tiering_reasoning", "")
            print_step("algorithm_tiering", {
                "tier1": tiers.get("tier1", []), 
                "tier2": tiers.get("tier2", []), 
                "tier3": tiers.get("tier3", []),
                "reasoning": reasoning[:200] + "..." if len(reasoning) > 200 else reasoning
            })
        elif sub == "run_algorithms_portfolio":
            algos = state.get("algorithm_results", {})
            statuses = {k: ("ok" if "error" not in v else f"error: {v['error']}") for k, v in algos.items()}
            
            # Evaluate each algorithm against ground truth
            algorithm_evaluations = {}
            for algo_name, result in algos.items():
                if "error" not in result:
                    evaluation = evaluate_against_ground_truth(result, ground_truth, list(df.columns))
                    metrics = evaluation["metrics"]
                    algorithm_evaluations[algo_name] = {
                        "f1_score": round(metrics["f1_score"], 3),
                        "precision": round(metrics["precision"], 3),
                        "recall": round(metrics["recall"], 3),
                        "shd": metrics["shd"],
                        "edges_found": metrics["total_predicted"]
                    }
                else:
                    algorithm_evaluations[algo_name] = {"error": result["error"]}
            
            print_step("run_algorithms_portfolio", {
                "statuses": statuses,
                "ground_truth_evaluations": algorithm_evaluations
            })
        elif sub == "candidate_pruning":
            pruned = state.get("pruned_candidates", [])
            rejected = state.get("pruning_log", [])
            print_step("candidate_pruning", {
                "kept": len(pruned),
                "rejected": len(rejected),
                "rejection_reasons": {
                    reason: len([r for r in rejected if r.get("reason") == reason])
                    for reason in set(r.get("reason", "unknown") for r in rejected)
                } if rejected else {}
            })
        elif sub == "scorecard_evaluation":
            sc = state.get("scorecard", [])
            tops = [{"algorithm": x.get("algorithm"), "score": round(x.get("composite_score", 0.0), 3)} for x in sc[:3]]
            print_step("scorecard_evaluation", {"top": tops})
        elif sub == "ensemble_synthesis":
            pag = state.get("consensus_pag", {})
            dag = state.get("selected_graph", {})
            
            # Evaluate against ground truth
            if dag and isinstance(dag, dict):
                evaluation = evaluate_against_ground_truth(dag, ground_truth, list(df.columns))
                metrics = evaluation["metrics"]
                
                print_step("ensemble_synthesis", {
                    "pag_edges": len(pag.get("edges", [])) if isinstance(pag, dict) else None,
                    "dag_edges": len(dag.get("edges", [])) if isinstance(dag, dict) else None,
                    "ground_truth_evaluation": {
                        "precision": round(metrics["precision"], 3),
                        "recall": round(metrics["recall"], 3),
                        "f1_score": round(metrics["f1_score"], 3),
                        "shd": metrics["shd"],
                        "normalized_shd": round(metrics["normalized_shd"], 3),
                        "true_positives": metrics["true_positives"],
                        "false_positives": metrics["false_positives"],
                        "false_negatives": metrics["false_negatives"]
                    },
                    "edge_analysis": {
                        "correct_edges": evaluation["edge_comparison"]["correct_edges"],
                        "missing_edges": evaluation["edge_comparison"]["missing_edges"],
                        "extra_edges": evaluation["edge_comparison"]["extra_edges"]
                    }
                })
            else:
                print_step("ensemble_synthesis", {
                    "pag_edges": len(pag.get("edges", [])) if isinstance(pag, dict) else None,
                    "dag_edges": len(dag.get("edges", [])) if isinstance(dag, dict) else None,
                    "error": "No valid DAG found for evaluation"
                })


if __name__ == "__main__":
    main()



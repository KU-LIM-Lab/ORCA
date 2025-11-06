import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit  # Logistic/Sigmoid function
import json  # config 및 stats 저장을 위해 추가

# --- 1. Causal Discovery (CD) 벤치마크 생성기 ---

def _topological_sort(adj: np.ndarray) -> list[int]:
    """Return a topological order for a DAG adjacency matrix using networkx."""
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    
    rows, cols = np.nonzero(adj)
    G.add_edges_from(zip(rows, cols))
    
    return list(nx.topological_sort(G))

def _generate_graph_adj(
    n_nodes: int,
    rng: np.random.Generator,
    graph_type: str = "er",
    p: float | None = None, 
    avg_degree: float = 2.0, # 평균 차수 기본값
    m: int = 2,
) -> np.ndarray:
    """
    Generate a DAG adjacency matrix.
    """
    if graph_type == "er":
        if p is None:
            p = avg_degree / (n_nodes - 1) if n_nodes > 1 else 0.0
        Gu = nx.gnp_random_graph(n_nodes, p, seed=rng, directed=False)
    elif graph_type == "sf":
        m = min(m, max(1, n_nodes - 1))
        Gu = nx.barabasi_albert_graph(n_nodes, m, seed=rng)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    perm = rng.permutation(n_nodes)
    inv = np.empty(n_nodes, dtype=int)
    inv[perm] = np.arange(n_nodes) # inv[i] = 노드 i의 순열상 위치

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for u, v in Gu.edges():
        if inv[u] < inv[v]: # 위치를 O(1)로 조회
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)
    
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, j in G.edges():
        adj[i, j] = 1.0
    return adj

def _simulate_sem_cd(
    adj: np.ndarray,
    n_samples: int,
    rng: np.random.Generator,
    scm_type: str = "linear",
    noise_type: str = "gaussian",
) -> tuple[pd.DataFrame, nx.DiGraph]:
    """
    Simulate data from a Structural Causal Model (SCM).
    """
    n = adj.shape[0]
    order = _topological_sort(adj)
    weights = rng.uniform(0.5, 1.5, size=adj.shape) * (rng.choice([-1.0, 1.0], size=adj.shape))
    weights = weights * (adj > 0)

    X = np.zeros((n_samples, n), dtype=float)

    if noise_type == "non-gaussian":
        noise_data = rng.laplace(0.0, 1.0, size=(n_samples, n))
    else:
        noise_data = rng.normal(0.0, 1.0, size=(n_samples, n))

    for j in order:
        parents = np.where(adj[:, j] > 0)[0]
        
        if len(parents) == 0:
            X[:, j] = noise_data[:, j]
        else:
            parent_values = X[:, parents] @ weights[parents, j]
            
            if len(parents) > 0:
                parent_values /= np.sqrt(len(parents))
                
            if scm_type == "non-linear":
                X[:, j] = np.tanh(parent_values) + noise_data[:, j]
            else:
                X[:, j] = parent_values + noise_data[:, j]

    columns = [f"V{i}" for i in range(n)]
    df = pd.DataFrame(X, columns=columns)
    
    G_truth = nx.DiGraph()
    G_truth.add_nodes_from(columns) 
    
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                G_truth.add_edge(columns[i], columns[j], weight=weights[i, j])

    return df, G_truth

def generate_cd_benchmarks(base_dir: str = "data/synthetic_cd", main_seed: int = 42, n_repeats: int = 5):
    """
    Generates and saves all Causal Discovery (CD) benchmark datasets.
    """
    print(f"--- 1. Generating Causal Discovery (CD) Benchmarks ({n_repeats} repeats) ---")
    
    scenarios = {
        "CD-1_Baseline":   {"graph": "er", "scm": "linear", "noise": "gaussian", "n": 2000, "d": 20, "p": 0.2, "m": 2},
        "CD-2_Graph":      {"graph": "sf", "scm": "linear", "noise": "gaussian", "n": 2000, "d": 20, "p": 0.2, "m": 2},
        "CD-3_Noise":      {"graph": "er", "scm": "linear", "noise": "non-gaussian", "n": 2000, "d": 20, "p": 0.2, "m": 2},
        "CD-4_SCM":        {"graph": "er", "scm": "non-linear", "noise": "gaussian", "n": 2000, "d": 20, "p": 0.2, "m": 2},
        "CD-5_Complex":    {"graph": "sf", "scm": "non-linear", "noise": "non-gaussian", "n": 2000, "d": 20, "p": 0.2, "m": 2},
        "CD-6_Scaling_ER": {"graph": "er", "scm": "linear", "noise": "gaussian", "n": 100, "d": 500, "avg_degree": 4, "m": 2},
        "CD-7_Scaling_SF": {"graph": "sf", "scm": "linear", "noise": "gaussian", "n": 100, "d": 500, "p": 0.2, "m": 4},
    }

    master_rng = np.random.default_rng(main_seed)

    for i, (name, config) in enumerate(scenarios.items()):
        print(f"\nGenerating Scenario: {name} ({n_repeats} repeats)...")
        scenario_base_dir = os.path.join(base_dir, name)

        for r in range(n_repeats):
            run_seed = master_rng.integers(1e9)
            rng = np.random.default_rng(run_seed)

            run_dir = os.path.join(scenario_base_dir, f"run_{r:03d}")
            os.makedirs(run_dir, exist_ok=True)

            run_config = config.copy()
            run_config["seed"] = int(run_seed)

            adj = _generate_graph_adj(
                n_nodes=run_config["d"],
                rng=rng,
                graph_type=run_config["graph"],
                p=run_config.get("p"),
                avg_degree=run_config.get("avg_degree", 2.0),
                m=run_config.get("m", 2)
            )

            df, G_truth = _simulate_sem_cd(
                adj,
                n_samples=run_config["n"],
                rng=rng,
                scm_type=run_config["scm"],
                noise_type=run_config["noise"]
            )

            # --- 저장 ---
            df.to_csv(os.path.join(run_dir, "data.csv"), index=False)
            nx.write_gml(G_truth, os.path.join(run_dir, "graph_truth.gml"))

            adj_int = np.zeros((run_config["d"], run_config["d"]), dtype=int)
            for u_name, v_name in G_truth.edges():
                u_idx = int(u_name[1:])  # "V12" -> 12
                v_idx = int(v_name[1:])
                adj_int[u_idx, v_idx] = 1

            np.save(os.path.join(run_dir, "adj.npy"), adj_int)

            # adj_int에서 edges.csv 생성
            u_nodes, v_nodes = np.where(adj_int > 0)
            pd.DataFrame({"u": u_nodes, "v": v_nodes}).to_csv(
                os.path.join(run_dir, "edges.csv"), index=False
            )

            clean_config = {k: v for k, v in run_config.items() if isinstance(v, (int, float, str, bool, list, dict))}
            
            num_edges = int(adj_int.sum())
            clean_config["num_edges"] = num_edges
            clean_config["avg_degree_realized"] = float(num_edges / run_config["d"]) if run_config["d"] > 0 else 0.0

            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(clean_config, f, indent=2)

    print("Causal Discovery benchmarks generated.")


# --- 2. Causal Inference (CI) 벤치마크 생성기 ---

def generate_ci_synthetic(
    n_samples: int,
    n_confounders: int,
    scenario_id: str,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Generates a single Causal Inference benchmark dataset (W, T, Y).
    """
    
    W = rng.normal(0, 1, size=(n_samples, n_confounders))
    
    if scenario_id in ["CI-1", "CI-3", "CI-4"]:
        propensity_logit = W[:, 0] + 0.5 * W[:, 1]
    elif scenario_id in ["CI-2", "CI-5"]:
        propensity_logit = W[:, 0] + 0.5 * (W[:, 1]**2)
    
    propensity_logit -= propensity_logit.mean()
    propensity = np.clip(expit(propensity_logit), 0.05, 0.95)
    T = rng.binomial(1, propensity)
    
    Y_confounding = 2 * W[:, 0] + W[:, 1]
    
    if scenario_id in ["CI-3", "CI-5"]:
        Y_confounding += 0.5 * (W[:, 0]**2)
        
    if scenario_id in ["CI-4", "CI-5"]:
        ground_truth_cate = 1 + 0.5 * W[:, 2] # CATE
    else:
        ground_truth_cate = np.full(n_samples, 2.0) # ATE
        
    eps0 = rng.normal(0, 0.1, size=n_samples)
    eps1 = rng.normal(0, 0.1, size=n_samples)
    
    Y0 = Y_confounding + eps0
    Y1 = Y_confounding + ground_truth_cate + eps1
    
    Y = (T == 0) * Y0 + (T == 1) * Y1
    
    df_data = pd.DataFrame(W, columns=[f"W{i}" for i in range(n_confounders)])
    df_data["T"] = T
    df_data["Y"] = Y
    df_data["propensity"] = propensity
    df_data["tau_true"] = ground_truth_cate
    df_data["mu0"] = Y0
    df_data["mu1"] = Y1
    
    return df_data


def generate_ci_benchmarks(base_dir: str = "data/synthetic_ci", main_seed: int = 42, n_repeats: int = 5):
    """
    Generates and saves all Causal Inference (CI) benchmark datasets.
    """
    print(f"--- 2. Generating Causal Inference (CI) Benchmarks ({n_repeats} repeats) ---")
    
    scenarios = {
        "CI-1_Baseline":    {"id": "CI-1", "n": 2000, "d": 5},
        "CI-2_Nonlinear_T": {"id": "CI-2", "n": 2000, "d": 5},
        "CI-3_Nonlinear_Y": {"id": "CI-3", "n": 2000, "d": 5},
        "CI-4_CATE":        {"id": "CI-4", "n": 2000, "d": 5},
        "CI-5_Complex":     {"id": "CI-5", "n": 2000, "d": 5},
    }
    
    master_rng = np.random.default_rng(main_seed)

    for i, (name, config) in enumerate(scenarios.items()):
        print(f"\nGenerating Scenario: {name} ({n_repeats} repeats)...")
        scenario_base_dir = os.path.join(base_dir, name)

        for r in range(n_repeats):
            run_seed = master_rng.integers(1e9)
            rng = np.random.default_rng(run_seed)

            run_dir = os.path.join(scenario_base_dir, f"run_{r:03d}")
            os.makedirs(run_dir, exist_ok=True)

            run_config = config.copy()
            run_config["seed"] = int(run_seed)

            df_data = generate_ci_synthetic(
                n_samples=run_config["n"],
                n_confounders=run_config["d"],
                scenario_id=run_config["id"],
                rng=rng
            )

            # --- 저장 ---
            df_data.to_csv(os.path.join(run_dir, "data.csv"), index=False)

            clean_config = {k: v for k, v in run_config.items() if isinstance(v, (int, float, str, bool, list, dict))}
            with open(os.path.join(run_dir, "config.json"), "w") as f:
                json.dump(clean_config, f, indent=2)

            stats = {
                "ATE_true": float(df_data["tau_true"].mean()),
                "treat_rate_actual": float(df_data["T"].mean())
            }
            with open(os.path.join(run_dir, "stats.json"), "w") as f:
                json.dump(stats, f, indent=2)
            
    print("Causal Inference benchmarks generated.")


# --- 3. 메인 실행 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Causal Discovery (CD) and Causal Inference (CI) benchmark datasets."
    )
    parser.add_argument(
        "--suite",
        type=str,
        choices=["cd", "ci", "all"],
        default="all",
        help="Which benchmark suite to generate ('cd', 'ci', or 'all')."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Main seed for the random number generator."
    )
    parser.add_argument(
        '--n_repeats',
        type=int,
        default=5,
        help="Number of datasets (repeats/runs) to generate per scenario."
    )
    
    args = parser.parse_args()
    
    if args.suite == "cd" or args.suite == "all":
        generate_cd_benchmarks(base_dir="data/synthetic_cd", main_seed=args.seed, n_repeats=args.n_repeats)
        
    if args.suite == "ci" or args.suite == "all":
        generate_ci_benchmarks(base_dir="data/synthetic_ci", main_seed=args.seed, n_repeats=args.n_repeats)
        
    print("\nBenchmark generation complete.")
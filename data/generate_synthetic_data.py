import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
from scipy.special import expit  # Logistic/Sigmoid function
import json 

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
    
    adj = np.zeros((n_nodes, n_nodes), dtype=int)
    for i, j in G.edges():
        adj[i, j] = 1
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

def _parse_int_list_csv(s: str) -> list[int]:
    """Parse comma-separated integers like '3,5,10' into a list."""
    items = [x.strip() for x in s.split(",") if x.strip()]
    out: list[int] = []
    for x in items:
        try:
            out.append(int(x))
        except ValueError as e:
            raise ValueError(f"Invalid integer in list: '{x}' from '{s}'") from e
    if not out:
        raise ValueError("d_list cannot be empty")
    return out


def _determine_n_samples(d: int, n_policy: str, n_fixed: int, n_coef: float) -> int:
    """Return n as a function of d according to a policy."""
    if n_policy == "fixed":
        n = int(n_fixed)
    elif n_policy == "linear":
        n = int(round(n_coef * d))
    elif n_policy == "dlogd":
        n = int(round(n_coef * d * float(np.log(max(d, 2)))))
    else:
        raise ValueError(f"Unknown n_policy: {n_policy}. Choose from fixed|linear|dlogd")

    return max(10, n)

def generate_cd_benchmarks(
    base_dir: str = "data/synthetic_cd",
    main_seed: int = 42,
    n_repeats: int = 5,
    d_list: list[int] | None = None,
    n_policy: str = "linear",
    n_fixed: int = 2000,
    n_coef: float = 100.0,
    avg_degree: float = 8.0,
    m_override: int | None = None,
):
    """Generates and saves Causal Discovery (CD) benchmark datasets.

    This generator supports a grid over:
      - scenario templates (graph/scm/noise)
      - dimensions d in d_list
      - repeats (run_000..)

    n can be scaled with d via n_policy:
      - fixed:  n = n_fixed
      - linear: n = n_coef * d
      - dlogd:  n = n_coef * d * log(d)

    Graph sparsity is controlled via avg_degree (undirected average degree before orientation).
    For ER graphs, p is derived as avg_degree/(d-1). For SF graphs, m is set to round(avg_degree/2)
    unless m_override is provided.
    """
    print(f"--- 1. Generating Causal Discovery (CD) Benchmarks ({n_repeats} repeats) ---")

    if d_list is None:
        d_list = [2, 3, 5, 10, 30, 50, 100]

    # Scenario templates
    scenario_templates = {
        "CD-1_Baseline": {"graph": "er", "scm": "linear", "noise": "gaussian"},
        "CD-2_Graph": {"graph": "sf", "scm": "linear", "noise": "gaussian"},
        "CD-3_Noise": {"graph": "er", "scm": "linear", "noise": "non-gaussian"},
        "CD-4_SCM": {"graph": "er", "scm": "non-linear", "noise": "gaussian"},
        "CD-5_Complex": {"graph": "sf", "scm": "non-linear", "noise": "non-gaussian"},
    }

    master_rng = np.random.default_rng(main_seed)

    for name, tmpl in scenario_templates.items():
        print(f"\nGenerating Scenario: {name} over d_list={d_list} ({n_repeats} repeats each)...")

        for d in d_list:
            scenario_d_base_dir = os.path.join(base_dir, name, f"d_{d}")
            os.makedirs(scenario_d_base_dir, exist_ok=True)

            n_samples = _determine_n_samples(d=d, n_policy=n_policy, n_fixed=n_fixed, n_coef=n_coef)

            for r in range(n_repeats):
                run_seed = int(master_rng.integers(1_000_000_000))
                rng = np.random.default_rng(run_seed)

                run_dir = os.path.join(scenario_d_base_dir, f"run_{r:03d}")
                os.makedirs(run_dir, exist_ok=True)

                # Determine graph parameters
                graph_type = tmpl["graph"]
                p_used: float | None = None
                m_used: int | None = None

                if graph_type == "er":
                    p_used = float(avg_degree / (d - 1)) if d > 1 else 0.0
                    adj = _generate_graph_adj(
                        n_nodes=d,
                        rng=rng,
                        graph_type="er",
                        p=None,
                        avg_degree=avg_degree,
                        m=2,
                    )
                elif graph_type == "sf":
                    m_auto = max(1, int(round(avg_degree / 2.0)))
                    m_used = int(m_override) if m_override is not None else int(m_auto)
                    adj = _generate_graph_adj(
                        n_nodes=d,
                        rng=rng,
                        graph_type="sf",
                        p=None,
                        avg_degree=avg_degree,
                        m=m_used,
                    )
                else:
                    raise ValueError(f"Unknown graph_type: {graph_type}")

                df, G_truth = _simulate_sem_cd(
                    adj,
                    n_samples=n_samples,
                    rng=rng,
                    scm_type=tmpl["scm"],
                    noise_type=tmpl["noise"],
                )

                # Save data
                df.to_csv(os.path.join(run_dir, "data.csv"), index=False)
                nx.write_gml(G_truth, os.path.join(run_dir, "graph_truth.gml"))

                # Build integer adjacency from G_truth
                adj_int = np.zeros((d, d), dtype=int)
                for u_name, v_name in G_truth.edges():
                    u_idx = int(u_name[1:])  # "V12" -> 12
                    v_idx = int(v_name[1:])
                    adj_int[u_idx, v_idx] = 1

                np.save(os.path.join(run_dir, "adj.npy"), adj_int)

                u_nodes, v_nodes = np.where(adj_int > 0)
                pd.DataFrame({"u": u_nodes, "v": v_nodes}).to_csv(
                    os.path.join(run_dir, "edges.csv"), index=False
                )

                # Save config
                num_edges = int(adj_int.sum())
                # Realized average out-degree (directed) = num_edges/d
                avg_degree_realized = float(num_edges / d) if d > 0 else 0.0

                run_config = {
                    "scenario": name,
                    "graph": graph_type,
                    "scm": tmpl["scm"],
                    "noise": tmpl["noise"],
                    "n": int(n_samples),
                    "d": int(d),
                    "seed": int(run_seed),
                    "n_policy": n_policy,
                    "n_fixed": int(n_fixed),
                    "n_coef": float(n_coef),
                    "avg_degree_target": float(avg_degree),
                    "p_used": float(p_used) if p_used is not None else None,
                    "m_used": int(m_used) if m_used is not None else None,
                    "num_edges": num_edges,
                    "avg_degree_realized": avg_degree_realized,
                }

                # Remove Nones for cleaner json
                clean_config = {k: v for k, v in run_config.items() if v is not None}

                with open(os.path.join(run_dir, "config.json"), "w") as f:
                    json.dump(clean_config, f, indent=2)

    print("Causal Discovery benchmarks generated.")


# --- 2. Causal Inference (CI) Benchmark ---

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

            # Save data
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
    parser.add_argument(
        "--d_list",
        type=str,
        default="3,5,10,30,50,100",
        help="Comma-separated list of dimensions for CD (e.g., '3,5,10,30,50,100')."
    )
    parser.add_argument(
        "--n_policy",
        type=str,
        choices=["fixed", "linear", "dlogd"],
        default="linear",
        help="CD sample-size scaling policy as a function of d."
    )
    parser.add_argument(
        "--n_fixed",
        type=int,
        default=2000,
        help="CD fixed n when n_policy='fixed'."
    )
    parser.add_argument(
        "--n_coef",
        type=float,
        default=100.0,
        help="CD coefficient for n_policy (linear: n=n_coef*d, dlogd: n=n_coef*d*log(d))."
    )
    parser.add_argument(
        "--avg_degree",
        type=float,
        default=8.0,
        help="Target average degree (undirected before orientation). For ER this implies expected edges ≈ d*avg_degree/2."
    )
    parser.add_argument(
        "--m_override",
        type=int,
        default=None,
        help="Optional override for SF m (default is round(avg_degree/2))."
    )
    
    args = parser.parse_args()
    
    if args.suite == "cd" or args.suite == "all":
        d_list = _parse_int_list_csv(args.d_list)
        generate_cd_benchmarks(
            base_dir="data/synthetic_cd",
            main_seed=args.seed,
            n_repeats=args.n_repeats,
            d_list=d_list,
            n_policy=args.n_policy,
            n_fixed=args.n_fixed,
            n_coef=args.n_coef,
            avg_degree=args.avg_degree,
            m_override=args.m_override,
        )
        
    if args.suite == "ci" or args.suite == "all":
        generate_ci_benchmarks(base_dir="data/synthetic_ci", main_seed=args.seed, n_repeats=args.n_repeats)
        
    print("\nBenchmark generation complete.")
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import networkx as nx


def _topological_sort(adj: np.ndarray) -> List[int]:
    """Return a topological order for a DAG adjacency matrix using networkx."""
    n = adj.shape[0]
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    # add directed edges where adj[i, j] > 0
    edges = [(i, j) for i in range(n) for j in range(n) if adj[i, j] > 0]
    G.add_edges_from(edges)
    return list(nx.topological_sort(G))


def _simulate_sem(adj: np.ndarray, n_samples: int, rng: np.random.Generator) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n = adj.shape[0]
    order = _topological_sort(adj)
    # Edge weights in [0.5, 1.5] with random signs
    weights = rng.uniform(0.5, 1.5, size=adj.shape) * (rng.choice([-1.0, 1.0], size=adj.shape))
    weights = weights * (adj > 0)

    X = np.zeros((n_samples, n), dtype=float)
    noise = rng.normal(0.0, 1.0, size=(n_samples, n))
    for j in order:
        parents = np.where(adj[:, j] > 0)[0]
        if len(parents) == 0:
            X[:, j] = noise[:, j]
        else:
            X[:, j] = X[:, parents] @ weights[parents, j] + noise[:, j]

    columns = [f"V{i}" for i in range(n)]
    df = pd.DataFrame(X, columns=columns)
    edges = []
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0:
                edges.append({"from": columns[i], "to": columns[j], "weight": float(weights[i, j])})

    meta = {
        "variables": columns,
        "edges": edges,
        "order": order,
    }
    return df, meta



def generate_er_synthetic(
    n_nodes: int = 8,
    edge_prob: float = 0.25,
    n_samples: int = 1500,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Generate a DAG by sampling an undirected Erdos–Renyi graph with networkx,
    then orienting edges according to a random node permutation to ensure acyclicity.

    Returns (df, meta) where meta contains variables, edges, and topological order.
    """
    rng = np.random.default_rng(seed)

    # 1) Undirected ER graph
    Gu = nx.gnp_random_graph(n_nodes, edge_prob, seed=seed, directed=False)

    # 2) Random permutation -> induces a total order; direct edges from earlier to later in this order
    perm = rng.permutation(n_nodes)
    pos = {node: int(np.where(perm == node)[0][0]) for node in range(n_nodes)}

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))
    for u, v in Gu.edges():
        if pos[u] < pos[v]:
            G.add_edge(u, v)
        else:
            G.add_edge(v, u)

    # 3) Build adjacency matrix consistent with G
    adj = np.zeros((n_nodes, n_nodes), dtype=float)
    for i, j in G.edges():
        adj[i, j] = 1.0

    # 4) Simulate linear SEM with proper topological order
    df, meta = _simulate_sem(adj, n_samples, rng)

    columns = meta["variables"]
    # build edge metadata with weights already filled in by _simulate_sem
    # (meta already contains edges & order; add permutation and ensure order is true topo order)
    meta["perm"] = perm.tolist()

    return df, meta

# Convenience batch generator for multiple ER datasets
def generate_er_synthetic_batch(configs: List[Dict[str, Any]]) -> List[Tuple[pd.DataFrame, Dict[str, Any]]]:
    """Generate multiple ER synthetic datasets. Each config can override n_nodes, edge_prob, n_samples, seed."""
    out = []
    for cfg in configs:
        out.append(
            generate_er_synthetic(
                n_nodes=cfg.get("n_nodes", 8),
                edge_prob=cfg.get("edge_prob", 0.25),
                n_samples=cfg.get("n_samples", 1500),
                seed=cfg.get("seed", 42),
            )
        )
    return out
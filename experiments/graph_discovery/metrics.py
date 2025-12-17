from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Any, List

import networkx as nx
import numpy as np


def edges_to_adj(nodes: List[str], edges: List[Dict[str, Any]]) -> np.ndarray:
    """Convert edge list to adjacency matrix (d x d) in node order."""
    idx = {n: i for i, n in enumerate(nodes)}
    d = len(nodes)
    A = np.zeros((d, d), dtype=int)
    for e in edges:
        u = e.get("from")
        v = e.get("to")
        if u in idx and v in idx:
            A[idx[u], idx[v]] = 1
    return A


def compute_shd(
    nodes: List[str],
    edges_hat: List[Dict[str, Any]],
    edges_true: List[Dict[str, Any]],
    *,
    double_for_anticausal: bool = True,
) -> int:
    """Structural Hamming Distance (SHD) between two directed graphs.

    Parameters
    ----------
    double_for_anticausal:
        - True (default): a reversed edge counts as 2 mistakes (one missing + one extra).
        - False: a reversed edge counts as 1 mistake (R SID's allMistakesOne=TRUE).
    """
    A_hat = edges_to_adj(nodes, edges_hat).astype(int)
    A_true = edges_to_adj(nodes, edges_true).astype(int)

    # No self loops
    np.fill_diagonal(A_hat, 0)
    np.fill_diagonal(A_true, 0)

    if double_for_anticausal:
        # L1 distance on adjacency matrices (reversal -> 2)
        return int(np.abs(A_true - A_hat).sum())

    # allMistakesOne-style: reversal counts as 1.
    Gtmp = (A_true + A_hat) % 2
    Gtmp = Gtmp + Gtmp.T
    nr_reversals = int((Gtmp == 2).sum() // 2)
    nr_incl_del = int((Gtmp == 1).sum() // 2)
    return int(nr_reversals + nr_incl_del)


def compute_sid(
    nodes: List[str],
    edges_hat: List[Dict[str, Any]],
    edges_true: List[Dict[str, Any]],
) -> int | None:
    """Structural Intervention Distance (SID) between two DAGs using CDT.

    Notes
    -----
    - This uses `cdt.metrics.SID`, which is an R-backed wrapper.
    - SID is defined only for DAGs. If either graph is cyclic, returns None.

    Returns
    -------
    sid : int | None
        - int: SID value.
        - None: if CDT or required R dependencies are not available, graphs are not DAGs,
          or any error occurs.
    """
    # CDT relies on an R backend (Rscript + SID R package). If Rscript is not available, we cannot compute SID.
    if shutil.which("Rscript") is None:
        return None

    A_hat = edges_to_adj(nodes, edges_hat).astype(int)
    A_true = edges_to_adj(nodes, edges_true).astype(int)

    # No self loops
    np.fill_diagonal(A_hat, 0)
    np.fill_diagonal(A_true, 0)

    # SID is only defined on DAGs.
    try:
        G_hat = nx.DiGraph(A_hat)
        G_true = nx.DiGraph(A_true)
        if not nx.is_directed_acyclic_graph(G_hat) or not nx.is_directed_acyclic_graph(G_true):
            return None
    except Exception:
        return None

    try:
        from cdt.metrics import SID   # type: ignore

        sid_val = SID(A_true, A_hat)
        if sid_val is None:
            return None
        # CDT may return a numpy scalar / float; normalize to int.
        return int(float(sid_val))
    except Exception:
        return None


def _edge_set_from_dict_list(edges: List[Dict[str, Any]]) -> set[tuple[str, str]]:
    s: set[tuple[str, str]] = set()
    for e in edges:
        u = str(e.get("from"))
        v = str(e.get("to"))
        if u is not None and v is not None:
            s.add((u, v))
    return s


def compute_edge_metrics(
    edges_hat: List[Dict[str, Any]],
    edges_true: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Precision / recall / F1 over directed edges."""
    pred_set = _edge_set_from_dict_list(edges_hat)
    true_set = _edge_set_from_dict_list(edges_true)

    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_set) if true_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "n_pred": int(len(pred_set)),
        "n_true": int(len(true_set)),
    }


def compute_graph_metrics(
    nodes: List[str],
    edges_hat: List[Dict[str, Any]],
    edges_true: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute SHD, SID (if available), and basic edge-level metrics."""
    metrics: Dict[str, Any] = {}

    shd = compute_shd(nodes, edges_hat, edges_true, double_for_anticausal=True)
    metrics["shd"] = shd

    sid_val = compute_sid(nodes, edges_hat, edges_true)
    metrics["sid"] = sid_val

    edge_stats = compute_edge_metrics(edges_hat, edges_true)
    metrics.update(edge_stats)

    return metrics


def compute_node_level_records(
    nodes: List[str],
    edges_hat: List[Dict[str, Any]],
    edges_true: List[Dict[str, Any]],
    base_record: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Compute per-node precision/recall/F1 based on incident edges."""
    pred_set = _edge_set_from_dict_list(edges_hat)
    true_set = _edge_set_from_dict_list(edges_true)

    node_records: List[Dict[str, Any]] = []
    for node in nodes:
        pred_node = {e for e in pred_set if node in e}
        true_node = {e for e in true_set if node in e}

        tp = len(pred_node & true_node)
        fp = len(pred_node - true_node)
        fn = len(true_node - pred_node)

        precision = tp / len(pred_node) if pred_node else 0.0
        recall = tp / len(true_node) if true_node else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rec = {
            **base_record,
            "node": node,
            "precision_node": float(precision),
            "recall_node": float(recall),
            "f1_node": float(f1),
            "tp_node": int(tp),
            "fp_node": int(fp),
            "fn_node": int(fn),
        }
        node_records.append(rec)

    return node_records

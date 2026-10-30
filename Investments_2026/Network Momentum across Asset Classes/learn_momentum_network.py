#!/usr/bin/env python3
"""Fit the paper's graph objective and first-anchor five-network pilot.

Uses an equivalent N-variable convex dual with damped Newton steps. The
unmodified objective, all candidate edges, and a primal-dual certificate are
retained. Alpha=beta=1 is an explicitly untuned engineering pilot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import solve
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist, squareform

from build_momentum_features import FEATURES, LOOKBACKS, write_json
from prepare_sp500_daily import file_hash, require, write_parquet


ROOT = Path(__file__).resolve().parent


def graph_objective(adjacency, distances, alpha, beta):
    degree = adjacency.sum(axis=1)
    if (degree <= 0).any():
        return np.inf
    return float(.5 * np.sum(adjacency * distances) - alpha * np.log(degree).sum()
                 + beta * np.square(adjacency).sum())


def fit_graph(distances, alpha=1., beta=1., tolerance=1e-6, max_iterations=1000, initial_dual=None):
    """Solve z'w + 2*beta*w'w - alpha*sum(log(Sw)), w>=0.

    With y>0 the convex dual minimization is
      sum((y_i+y_j-z_ij)_+**2)/(8*beta) - alpha*sum(log(y)).
    Its gradient is S*w-alpha/y, with w=(y_i+y_j-z)_+/(4*beta).
    A degree/y complementarity certificate checks the complete edge problem.
    No distance rescaling, edge screening, or numerical degree floor is used.
    """
    zmat = np.asarray(distances, dtype=float)
    require(zmat.ndim == 2 and zmat.shape[0] == zmat.shape[1] and len(zmat) >= 2,
            "Distances must be a square matrix with at least two nodes")
    require(np.isfinite(zmat).all() and (zmat >= 0).all() and np.allclose(zmat, zmat.T)
            and np.all(np.diag(zmat) == 0), "Invalid squared distances")
    require(alpha > 0 and beta > 0 and tolerance > 0, "Positive graph parameters required")
    n = len(zmat)
    left, right = np.triu_indices(n, 1)
    z = zmat[left, right]
    y = np.full(n, max(float(z.max()) / 2, np.sqrt(alpha * beta / (n - 1))))
    if initial_dual is not None:
        initial_dual = np.asarray(initial_dual, dtype=float)
        require(initial_dual.shape == (n,) and np.isfinite(initial_dual).all() and (initial_dual > 0).all(),
                "Initial dual must be finite, positive and aligned to all current nodes")
        y = initial_dual.copy()
    trace = []

    def evaluate(v):
        excess = np.maximum(v[left] + v[right] - z, 0.)
        weight = excess / (4 * beta)
        degree = np.bincount(left, weights=weight, minlength=n) + np.bincount(right, weights=weight, minlength=n)
        objective = float(excess @ excess / (8 * beta) - alpha * np.log(v).sum())
        return objective, degree - alpha / v, weight, degree, excess

    started = time.monotonic()
    for iteration in range(max_iterations):
        objective, gradient, weights, degree, excess = evaluate(y)
        complement = degree * y / alpha
        residual = float(np.max(np.abs(complement - 1)))
        trace.append({"iteration": iteration, "dual_min_objective": objective,
                      "relative_degree_residual": residual, "active_edges": int((excess > 0).sum())})
        if residual <= tolerance:
            break
        active = excess > 0
        il, ir = left[active], right[active]
        hessian = np.zeros((n, n))
        hessian[il, ir] = 1 / (4 * beta)
        hessian[ir, il] = 1 / (4 * beta)
        counts = np.bincount(il, minlength=n) + np.bincount(ir, minlength=n)
        hessian.flat[::n + 1] = counts / (4 * beta) + alpha / y ** 2
        direction = solve(hessian, -gradient, assume_a="pos", check_finite=False)
        slope = float(gradient @ direction)
        require(np.isfinite(direction).all() and slope < 0, "Dual Newton direction failed")
        negative = direction < 0
        step = min(1., .99 * float(np.min(-y[negative] / direction[negative]))) if negative.any() else 1.
        for backtrack in range(80):
            candidate = y + step * direction
            new_objective, _, _, new_degree, _ = evaluate(candidate)
            # Near convergence the dual value can lose digits by cancellation.
            # Accept an Armijo step or a strict complementarity improvement when
            # the value difference is within roundoff of its numerical scale.
            new_residual = float(np.max(np.abs(new_degree * candidate / alpha - 1)))
            roundoff = 32 * np.finfo(float).eps * max(1., abs(objective))
            if (new_objective <= objective + 1e-4 * step * slope
                    or (abs(new_objective - objective) <= roundoff and new_residual < residual)):
                y = candidate
                break
            step *= .5
        else:
            raise RuntimeError(f"Dual line search failed at residual {residual}")
    else:
        raise RuntimeError(f"Graph did not converge after {max_iterations} iterations; residual={residual}")
    adjacency = np.zeros((n, n))
    adjacency[left, right] = weights
    adjacency[right, left] = weights
    require((degree > 0).all(), "Graph contains an isolated node")
    # At each dual point w minimizes the nonnegative-edge Lagrangian exactly.
    # The gap is alpha*sum(x-1-log(x)), x=d*y/alpha. Stable near x=1.
    log_complement = np.log(complement)
    gap = float(alpha * np.sum(np.expm1(log_complement) - log_complement))
    primal = graph_objective(adjacency, zmat, alpha, beta)
    dual_objective = float(-excess @ excess / (8 * beta) + alpha * np.log(y).sum()
                           + alpha * n * (1 - np.log(alpha)))
    stationarity = z + 4 * beta * weights - alpha * (1 / degree[left] + 1 / degree[right])
    violations = np.where(weights > 0, np.abs(stationarity), np.maximum(-stationarity, 0))
    relative_kkt = float(np.max(violations / np.maximum(1., z + alpha * (1 / degree[left] + 1 / degree[right]))))
    certificate = {"status": "converged", "solver": "exact convex dual / damped Newton",
        "iterations": iteration, "seconds": time.monotonic() - started,
        "tolerance": tolerance, "relative_degree_residual": residual, "primal_objective": primal,
        "dual_objective": dual_objective, "primal_dual_gap": gap,
        "direct_objective_gap": primal - dual_objective, "relative_kkt_residual": relative_kkt,
        "positive_edges": int((weights > 0).sum()), "candidate_edges": len(weights),
        "minimum_degree": float(degree.min()), "maximum_degree": float(degree.max()),
        "connected_components": int(connected_components(adjacency > 0, directed=False)[0]),
        "edge_threshold": 0., "degrees_floored": False, "warm_started": initial_dual is not None}
    return adjacency, y, certificate, trace


def ensemble_graph(adjacencies):
    require(len(adjacencies) > 0, "No graphs supplied")
    average = np.mean(np.stack(adjacencies), axis=0)
    degree = average.sum(axis=1)
    require((degree > 0).all(), "Ensemble has isolated nodes")
    normalized = average / np.sqrt(degree[:, None] * degree[None, :])
    return average, normalized


def build(feature_root, output, alpha=1., beta=1., tolerance=1e-6):
    require(not output.exists(), "Output exists; choose a new pilot directory")
    source = json.loads((feature_root / "manifest.json").read_text())
    require(source["status"] == "complete", "Incomplete feature build")
    required = {"anchor_feature_history.npz", "anchor_eligibility.csv", "policy.json", "validation.json"}
    for item in source["outputs"]:
        if item["path"] in required:
            p = feature_root / item["path"]
            require(p.stat().st_size == item["size_bytes"] and file_hash(p) == item["sha256"], "Feature fingerprint changed")
    with np.load(feature_root / "anchor_feature_history.npz", allow_pickle=False) as arrays:
        cube = arrays["features"]
        source_ids = arrays["source_security_ids"]
        native_ids = arrays["intrinio_security_ids"]
        tickers = arrays["membership_tickers"]
        dates = arrays["dates"]
        require(arrays["feature_names"].tolist() == FEATURES, "Feature order mismatch")
    require(cube.shape == (1260, len(source_ids), 8) and np.isfinite(cube).all(), "Invalid feature cube")
    require(len(set(source_ids)) == len(source_ids) and len(set(native_ids)) == len(native_ids), "Duplicate graph identities")
    output.mkdir(parents=True)
    nodes = pd.DataFrame({"node": np.arange(len(source_ids)), "source_security_id": source_ids,
                          "intrinio_security_id": native_ids, "membership_ticker": tickers})
    nodes.to_csv(output / "nodes.csv", index=False)
    graphs, results = [], []
    for window in LOOKBACKS:
        vectors = cube[-window:].transpose(1, 0, 2).reshape(len(source_ids), -1)
        # Direct pairwise squared differences avoid Gram cancellation for very
        # similar rows. No division by window length or number of features.
        distances = squareform(pdist(vectors, metric="sqeuclidean"))
        adjacency, dual, certificate, trace = fit_graph(distances, alpha, beta, tolerance)
        graphs.append(adjacency)
        np.savez_compressed(output / f"graph_{window}.npz", adjacency=adjacency,
            squared_distances=distances, dual_variables=dual, source_security_ids=source_ids)
        pd.DataFrame(trace).to_csv(output / f"solver_trace_{window}.csv", index=False)
        certificate.update(lookback_sessions=window, feature_dimensions=8 * window,
                           first_history_date=str(pd.Timestamp(dates[-window]).date()),
                           last_history_date=str(pd.Timestamp(dates[-1]).date()))
        results.append(certificate)
        print(json.dumps(certificate), flush=True)
    average, normalized = ensemble_graph(graphs)
    own = cube[-1]
    propagated = normalized @ own
    require(np.isfinite(propagated).all(), "Nonfinite propagated features")
    np.savez_compressed(output / "ensemble.npz", adjacency_average=average, adjacency_normalized=normalized,
        own_features=own, network_features=propagated, source_security_ids=source_ids, feature_names=np.array(FEATURES))
    signal = nodes.copy()
    signal["date"] = pd.Timestamp(dates[-1])
    signal[["own_" + x for x in FEATURES]] = own
    signal[["network_" + x for x in FEATURES]] = propagated
    signal["ensemble_degree"] = average.sum(axis=1)
    signal["normalized_row_sum"] = normalized.sum(axis=1)
    write_parquet(signal, output / "anchor_network_features.parquet")
    left, right = np.triu_indices(len(source_ids), 1)
    connected = average[left, right] > 0
    edges = pd.DataFrame({"source_security_id_i": source_ids[left[connected]],
        "source_security_id_j": source_ids[right[connected]], "ticker_i": tickers[left[connected]],
        "ticker_j": tickers[right[connected]], "average_weight": average[left[connected], right[connected]],
        "normalized_weight": normalized[left[connected], right[connected]]})
    edges.sort_values("normalized_weight", ascending=False).to_csv(output / "ensemble_edges.csv", index=False)
    validation = {"status": "passed", "all_graphs_converged": True,
        "symmetric_nonnegative_zero_diagonal": bool(all(np.array_equal(a, a.T) and np.all(a >= 0)
            and np.all(np.diag(a) == 0) for a in graphs)),
        "all_degrees_positive": bool(all(np.all(a.sum(axis=1) > 0) for a in graphs)),
        "normalized_degree_eigenvector_residual": float(np.max(np.abs(
            normalized @ np.sqrt(average.sum(axis=1)) - np.sqrt(average.sum(axis=1))))),
        "labels_or_post_anchor_data_used": False, "regression_or_backtest_run": False}
    write_json(output / "validation.json", validation)
    summary = {"status": "complete", "purpose": "untuned first-anchor numerical pilot; no performance claims",
        "anchor_date": source["anchor_date"], "members": source["anchor_members"], "nodes": len(source_ids),
        "lookbacks_sessions": list(LOOKBACKS), "alpha": alpha, "beta": beta, "hyperparameters_selected": False,
        "node_alignment": "common longest-window eligible current PIT members",
        "graph_settings": {"zero_diagonal": True, "distance_normalization": "none",
                           "ensemble_order": "average raw adjacency then symmetric degree normalization"},
        "graphs": results, "ensemble_positive_edges": len(edges),
        "ensemble_connected_components": int(connected_components(average > 0, directed=False)[0]),
        "input_feature_manifest_sha256": file_hash(feature_root / "manifest.json"),
        "script_sha256": file_hash(__file__), "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "regressions_fitted": False, "backtest_run": False,
        "limits": ["Alpha and beta are engineering defaults from the paper grid, without label-based selection.",
                   "Only the first anchor is fitted. No historical daily-graph series, OLS coefficients, positions or returns.",
                   "The source's provisional close and exploratory identity limitations remain."]}
    summary["outputs"] = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
                          for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"outputs", "graphs", "limits"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/momentum_features/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/network_pilot/2016-01-04_a1_b1_v1")
    parser.add_argument("--alpha", type=float, default=1.)
    parser.add_argument("--beta", type=float, default=1.)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    build(args.features.resolve(), args.output.resolve(), args.alpha, args.beta, args.tolerance)

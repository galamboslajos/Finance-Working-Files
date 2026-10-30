"""Numerically stable graph-grid solves, including very small alpha*beta.

For tiny quadratic penalties, recovering primal edges by subtracting dual
variables and distances loses digits. A primal active-set solve avoids that
subtraction and certifies the full edge problem with KKT and dual bounds.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
from scipy.linalg import solve, lstsq, LinAlgWarning
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components

from learn_momentum_network import fit_graph, graph_objective
from prepare_sp500_daily import require


def polish_primal(x, left, right, cost, penalty, n):
    """Newton-polish the positive support after bound-constrained discovery."""
    x = x.copy()
    for _ in range(40):
        degree = np.bincount(left, weights=x, minlength=n) + np.bincount(right, weights=x, minlength=n)
        gradient = cost + 4 * penalty * x - 1 / degree[left] - 1 / degree[right]
        free = np.flatnonzero(x > 0)
        residual = float(np.max(np.abs(gradient[free]) / np.maximum(1., cost[free] + 1/degree[left[free]] + 1/degree[right[free]])))
        if residual < 1e-10:
            break
        incidence = np.zeros((n, len(free)))
        incidence[left[free], np.arange(len(free))] = 1.
        incidence[right[free], np.arange(len(free))] = 1.
        hessian = incidence.T @ (incidence / degree[:, None] ** 2)
        hessian.flat[::len(free)+1] += 4 * penalty
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                direction = solve(hessian, -gradient[free], assume_a="pos", check_finite=False)
        except (np.linalg.LinAlgError, LinAlgWarning):
            direction = lstsq(hessian, -gradient[free], cond=1e-13, check_finite=False)[0]
        slope = float(gradient[free] @ direction)
        if not np.isfinite(direction).all() or slope >= 0:
            break
        negative = direction < 0
        step = min(1., float(np.min(-x[free[negative]] / direction[negative]))) if negative.any() else 1.
        value = cost @ x + 2 * penalty * (x @ x) - np.log(degree).sum()
        for _ in range(60):
            candidate = x.copy(); candidate[free] = np.maximum(x[free] + step * direction, 0.)
            d = np.bincount(left, weights=candidate, minlength=n) + np.bincount(right, weights=candidate, minlength=n)
            if (d > 0).all():
                new_value = cost @ candidate + 2 * penalty * (candidate @ candidate) - np.log(d).sum()
                if new_value <= value + 1e-4 * step * slope + 16*np.finfo(float).eps*max(1.,abs(value)):
                    x = candidate
                    break
            step *= .5
        else:
            break
    return x


def fit_primal(distances, beta, initial=None, tolerance=1e-6, max_rounds=30):
    zmat = np.asarray(distances, float)
    n = len(zmat)
    il, ir = np.triu_indices(n, 1)
    z = zmat[il, ir]
    positive = z[z > 0]
    require(len(positive) > 0 and beta > 0, "Primal solver requires positive distance scale and beta")
    nearest = zmat.copy(); np.fill_diagonal(nearest, np.inf)
    scale = float(np.median(nearest.min(axis=1)))
    require(scale > 0, "Zero nearest-neighbor distances require the dual solver")
    cost, penalty = z / scale, beta / scale ** 2
    weights = np.zeros(len(z))
    if initial is not None:
        require(np.asarray(initial).shape == zmat.shape, "Primal seed not aligned")
        weights = np.asarray(initial)[il, ir] * scale
    active = weights > 0
    # Every node needs a feasible positive degree before optimization starts.
    edge_lookup = np.full((n, n), -1, dtype=np.int32)
    edge_lookup[il, ir] = np.arange(len(z)); edge_lookup[ir, il] = np.arange(len(z))
    for i in range(n):
        if not np.any(active & ((il == i) | (ir == i))):
            edge = edge_lookup[i, np.argmin(nearest[i])]
            active[edge] = True; weights[edge] = 1.
    started, total_iterations = time.monotonic(), 0
    for round_number in range(max_rounds):
        edges = np.flatnonzero(active); left, right = il[edges], ir[edges]
        def objective_gradient(x):
            degree = np.bincount(left, weights=x, minlength=n) + np.bincount(right, weights=x, minlength=n)
            if (degree <= 0).any():
                return np.inf, np.zeros_like(x)
            value = cost[edges] @ x + 2 * penalty * (x @ x) - np.log(degree).sum()
            gradient = cost[edges] + 4 * penalty * x - 1 / degree[left] - 1 / degree[right]
            return value, gradient
        fit = minimize(objective_gradient, weights[edges], jac=True, method="L-BFGS-B",
            bounds=[(0., None)] * len(edges), options={"gtol": 1e-10, "ftol": 1e-15, "maxiter": 30000, "maxls": 60})
        total_iterations += int(fit.nit)
        fit.x = polish_primal(fit.x, left, right, cost[edges], penalty, n)
        weights[:] = 0; weights[edges] = fit.x
        degree = np.bincount(il, weights=weights, minlength=n) + np.bincount(ir, weights=weights, minlength=n)
        require((degree > 0).all(), "Primal solve lost a degree")
        gradient = cost + 4 * penalty * weights - 1 / degree[il] - 1 / degree[ir]
        denominator = np.maximum(1., cost + 1 / degree[il] + 1 / degree[ir])
        violation = np.where(weights > 0, np.abs(gradient), np.maximum(-gradient, 0))
        residual = float(np.max(violation / denominator))
        add = (~active) & (gradient / denominator < -tolerance / 10)
        if add.any():
            active |= add
            continue
        adjacency = np.zeros_like(zmat)
        adjacency[il, ir] = weights / scale; adjacency[ir, il] = weights / scale
        real_degree = adjacency.sum(axis=1)
        dual = 1 / real_degree
        excess = np.maximum(dual[il] + dual[ir] - z, 0)
        lower = float(-excess @ excess / (8 * beta) + np.log(dual).sum() + n)
        # A feasible beta=0 dual is also a lower bound for positive beta. This
        # bound stays stable when beta is too small for dual-edge subtraction.
        shrink = min(1., float(np.min(z / (dual[il] + dual[ir]))))
        if shrink > 0:
            feasible_dual = dual * np.nextafter(shrink, 0.)
            lower = max(lower, float(np.log(feasible_dual).sum() + n))
        primal = graph_objective(adjacency, zmat, 1., beta)
        gap = max(0., primal - lower)
        if residual <= tolerance and gap / n <= 2 * tolerance:
            certificate = {"status": "converged", "solver": "scaled primal active-set L-BFGS-B",
                "iterations": total_iterations, "active_set_rounds": round_number + 1,
                "seconds": time.monotonic() - started, "relative_kkt_residual": residual,
                "primal_objective": primal, "dual_objective": lower, "primal_dual_gap": gap,
                "gap_per_node": gap / n, "positive_edges": int((weights > 0).sum()),
                "candidate_edges": len(z), "minimum_degree": float(real_degree.min()),
                "maximum_degree": float(real_degree.max()), "connected_components": int(connected_components(adjacency > 0, directed=False)[0]),
                "edge_threshold": 0., "degrees_floored": False, "warm_started": initial is not None}
            return adjacency, dual, certificate
        # Tighten a nearly solved active problem before admitting a result.
        if residual > tolerance and not fit.success:
            raise RuntimeError(f"Primal graph failed: {fit.message}; residual={residual}, gap={gap}")
    raise RuntimeError(f"Primal graph did not converge: residual={residual}, gap={gap}")


def fit_grid_graph(distances, product, initial_adjacency=None, initial_dual=None, tolerance=1e-6):
    nearest = np.asarray(distances).copy(); np.fill_diagonal(nearest, np.inf)
    scale = float(np.median(nearest.min(axis=1)))
    if scale > 0 and product / scale ** 2 < 1e-8:
        return fit_primal(distances, product, initial_adjacency, tolerance)
    adjacency, dual, certificate, _ = fit_graph(distances, alpha=1., beta=product, tolerance=tolerance,
                                                initial_dual=initial_dual, max_iterations=2000)
    certificate["gap_per_node"] = certificate["primal_dual_gap"] / len(distances)
    return adjacency, dual, certificate

#!/usr/bin/env python3
"""Build causal daily five-network features on each date's PIT stock universe.

Daily checkpoints support restart. Pair distances roll exactly across dates,
with a direct numerical rebase every 21 sessions and on discontinuities. The
121 original parameter pairs are represented by 30 equivalent products.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist, squareform

from audit_first_window import read_slice
from build_momentum_features import FEATURES, LOOKBACKS, write_json
from extend_momentum_history import checked_object, load_native_history
from graph_grid import fit_grid_graph
from prepare_sp500_daily import canonical_hash, file_hash, require, write_parquet
from walk_forward import grid_groups


ROOT = Path(__file__).resolve().parent


class RollingDistances:
    def __init__(self, cube, lookbacks=LOOKBACKS, rebase_sessions=21):
        self.cube, self.lookbacks, self.rebase = cube, lookbacks, rebase_sessions
        self.previous_t, self.previous_ids, self.previous = None, None, {}

    def at(self, t, ids):
        ids = np.asarray(ids, dtype=int)
        require(t + 1 >= max(self.lookbacks), "Insufficient graph feature dates")
        require(np.isfinite(self.cube[t-max(self.lookbacks)+1:t+1, ids]).all(), "Graph history unavailable")
        reset = self.previous_t is None or t != self.previous_t + 1 or t % self.rebase == 0
        old_map = {} if reset else {int(native): i for i, native in enumerate(self.previous_ids)}
        overlap = np.array([i for i, native in enumerate(ids) if int(native) in old_map], dtype=int)
        old_positions = np.array([old_map[int(ids[i])] for i in overlap], dtype=int)
        new_positions = np.array([i for i, native in enumerate(ids) if int(native) not in old_map], dtype=int)
        distances = {}
        for window in self.lookbacks:
            if reset:
                vectors = self.cube[t-window+1:t+1, ids].transpose(1, 0, 2).reshape(len(ids), -1)
                matrix = squareform(pdist(vectors, "sqeuclidean"))
            else:
                matrix = np.zeros((len(ids), len(ids)))
                if len(overlap):
                    native = ids[overlap]
                    addition = squareform(pdist(self.cube[t, native], "sqeuclidean"))
                    removal = squareform(pdist(self.cube[t-window, native], "sqeuclidean"))
                    matrix[np.ix_(overlap, overlap)] = self.previous[window][np.ix_(old_positions, old_positions)] + addition - removal
                if len(new_positions):
                    vectors = self.cube[t-window+1:t+1, ids].transpose(1, 0, 2).reshape(len(ids), -1)
                    cross = cdist(vectors[new_positions], vectors, "sqeuclidean")
                    matrix[new_positions, :] = cross
                    matrix[:, new_positions] = cross.T
                require(float(matrix.min()) >= -1e-7, "Rolling distance drift exceeds tolerance")
                matrix = np.maximum(matrix, 0.)
                np.fill_diagonal(matrix, 0.)
            # Direct checks are independent of the rolling update and do not
            # inspect any feature after t.
            for i, j in ((0, len(ids)-1), (len(ids)//3, len(ids)//2)):
                expected = np.square(self.cube[t-window+1:t+1, ids[i]] - self.cube[t-window+1:t+1, ids[j]]).sum()
                np.testing.assert_allclose(matrix[i, j], expected, rtol=1e-10, atol=1e-7)
            distances[window] = matrix
        self.previous_t, self.previous_ids, self.previous = t, ids.copy(), distances
        return distances


def consecutive_ready(cube):
    good = np.isfinite(cube).all(axis=2)
    runs = np.zeros(good.shape, dtype=np.int32)
    for t in range(len(good)):
        previous = runs[t-1] if t else 0
        runs[t] = np.where(good[t], previous + 1, 0)
    return runs


def select_nodes(cohort, native_ids, runs, t):
    ledger = cohort.sort_values("source_security_id").reset_index(drop=True).copy()
    native_positions = native_ids.get_indexer(ledger.intrinio_security_id)
    admitted = ledger.identity_eligible.to_numpy(bool) & (native_positions >= 0)
    ready = np.zeros(len(ledger), dtype=np.int32)
    ready[admitted] = runs[t, native_positions[admitted]]
    for window in LOOKBACKS:
        ledger[f"eligible_{window}"] = admitted & (ready >= window)
    selected = ledger.eligible_1260.to_numpy()
    ledger["graph_eligibility_reason"] = np.select([~ledger.identity_eligible, ~admitted, selected, ready == 0],
        ["identity_withheld", "no_native_history", "eligible", "current_features_unavailable"], default="incomplete_feature_history")
    ledger["contiguous_feature_sessions"] = ready
    require(not ledger.loc[selected, "intrinio_security_id"].duplicated().any(), "Dated admitted-native collision")
    return ledger, native_positions[selected]


def align_seed(previous_ids, previous, ids):
    if previous is None:
        return None
    lookup = {int(native): i for i, native in enumerate(previous_ids)}
    new = np.array([i for i, native in enumerate(ids) if int(native) in lookup], dtype=int)
    old = np.array([lookup[int(ids[i])] for i in new], dtype=int)
    seed = np.zeros((len(ids), len(ids)))
    seed[np.ix_(new, new)] = previous[np.ix_(old, old)]
    return seed


def consolidate(output, days, products):
    output = Path(output)
    base_paths, network_paths = [], []
    for year in sorted({pd.Timestamp(d).year for d in days}):
        bases, networks = [], []
        for day in [d for d in days if pd.Timestamp(d).year == year]:
            with np.load(output / "daily" / f"{day}.npz", allow_pickle=False) as saved:
                count = len(saved["source_security_ids"])
                base = pd.DataFrame({"date": np.repeat(pd.Timestamp(day), count),
                    "source_security_id": saved["source_security_ids"], "intrinio_security_id": saved["intrinio_security_ids"],
                    "membership_ticker": saved["membership_tickers"], "daily_volatility": saved["daily_volatility"],
                    "forward_return": saved["forward_return"], "target": saved["target"],
                    "label_date": np.repeat(pd.Timestamp(saved["label_date"].item()), count)})
                base[["own_" + x for x in FEATURES]] = saved["own_features"]
                bases.append(base); networks.append(saved["network_features"])
        base = pd.concat(bases, ignore_index=True)
        require(not base.duplicated(["date", "source_security_id"]).any(), "Duplicate annual predictor keys")
        path = output / "base" / f"year={year}" / "part-00000.parquet"
        write_parquet(base, path); base_paths.append(path.relative_to(output).as_posix())
        values = np.concatenate(networks, axis=1)
        for j, product in enumerate(products):
            frame = base[["date", "source_security_id"]].copy()
            frame[["network_" + x for x in FEATURES]] = values[j]
            path = output / "network" / f"product={product:.12g}" / f"year={year}" / "part-00000.parquet"
            write_parquet(frame, path); network_paths.append(path.relative_to(output).as_posix())
    return base_paths, network_paths


def build(history, dataset, output, start, end, products=None, sample_dates=None):
    hm = json.loads((history / "manifest.json").read_text())
    dm = json.loads((dataset / "manifest.json").read_text())
    require(hm["status"] == dm["status"] == "complete", "Incomplete source")
    require(hm["input_dataset_manifest_sha256"] == file_hash(dataset / "manifest.json"), "PIT source changed")
    require(hm["start"] <= start <= end < hm["end"], "A future label session beyond signal end is required")
    groups = grid_groups()
    products = sorted(set(products if products is not None else [g["product"] for g in groups]))
    require(set(products).issubset({g["product"] for g in groups}), "Unknown grid products")
    config = {"history_manifest_sha256": file_hash(history / "manifest.json"), "dataset_manifest_sha256": file_hash(dataset / "manifest.json"),
        "start": start, "end": end, "products": products, "sample_dates": sample_dates,
        "graph_lookbacks": list(LOOKBACKS), "relative_kkt_tolerance": 1e-6, "dual_gap_per_node_tolerance": 2e-6,
        "direct_distance_rebase_sessions": 21, "missing_labels_change_nodes": False}
    if output.exists():
        require(json.loads((output / "run_config.json").read_text()) == config, "Existing run has a different configuration")
        require(not (output / "manifest.json").exists(), "Completed series is immutable")
    else:
        output.mkdir(parents=True); write_json(output / "run_config.json", config)
    # Read only finite-history inputs. The label table is deliberately separate
    # from node selection and graph estimation.
    sessions = xcals.get_calendar("XNYS", start=hm["start"], end=hm["end"]).sessions.tz_localize(None)
    label_end = sessions[sessions > pd.Timestamp(end)][0]
    sessions = sessions[sessions <= label_end]
    full = load_native_history(history, hm["start"], str(label_end.date()), columns=["date", "intrinio_security_id"] + FEATURES
                               + ["daily_volatility", "feature_return_1d"])
    native_ids = pd.Index(sorted(full.intrinio_security_id.unique()))
    times = sessions.get_indexer(full.date); natives = native_ids.get_indexer(full.intrinio_security_id)
    require((times >= 0).all() and not full.duplicated(["date", "intrinio_security_id"]).any(), "Invalid native grid")
    cube = np.full((len(sessions), len(native_ids), 8), np.nan)
    volatility = np.full(cube.shape[:2], np.nan); returns = np.full(cube.shape[:2], np.nan)
    cube[times, natives] = full[FEATURES].to_numpy()
    volatility[times, natives] = full.daily_volatility; returns[times, natives] = full.feature_return_1d
    del full, times, natives
    runs = consecutive_ready(cube)
    for entry in dm["output_files"]:
        if entry["path"].startswith("panel/") and int(start[:4]) <= int(entry["path"].split("/")[1][5:]) <= int(end[:4]):
            checked_object(dataset, entry)
    panel = read_slice(dataset, dm, "panel", ["date", "source_security_id", "intrinio_security_id", "membership_ticker", "identity_eligible", "identity_status"], start, end)
    if sample_dates is not None:
        panel = panel.loc[panel.date.isin(pd.to_datetime(sample_dates))]
    expected = sessions[(sessions >= start) & (sessions <= end)]
    if sample_dates is not None:
        expected = expected[expected.isin(pd.to_datetime(sample_dates))]
    require(set(panel.date.unique()) == set(expected), "Missing PIT dates in requested series")
    distances = RollingDistances(cube)
    seed_ids, seeds = None, {}
    summaries, daily_certificates, completed = [], [], []
    started = time.monotonic()
    for date, cohort in panel.groupby("date", sort=True):
        day = str(date.date()); t = sessions.get_loc(date)
        marker_path = output / "daily" / f"{day}.json"
        if marker_path.exists():
            marker = json.loads(marker_path.read_text())
            for entry in marker["objects"]:
                checked_object(output, entry)
            summaries.append(marker["coverage"]); daily_certificates.extend(marker["certificates"]); completed.append(day)
            continue
        ledger, ids = select_nodes(cohort, native_ids, runs, t)
        require(len(ids) >= 2, f"Too few graph nodes on {day}")
        z_by_window = distances.at(t, ids)
        averages = np.zeros((len(products), len(ids), len(ids)))
        certificates = []
        next_seeds = {}
        for window in LOOKBACKS:
            seed = align_seed(seed_ids, seeds.get(window), ids) if seed_ids is not None else None
            dual = None
            for j, product in enumerate(products):
                adjacency, dual, certificate = fit_grid_graph(z_by_window[window], product, initial_adjacency=seed, initial_dual=dual)
                require(certificate["relative_kkt_residual"] <= 1e-6 and certificate["gap_per_node"] <= 2e-6,
                        f"Uncertified graph on {day}/{window}/{product}")
                certificate.update(date=day, window=window, product=product)
                certificates.append(certificate)
                averages[j] += adjacency / len(LOOKBACKS)
                if j == 0:
                    next_seeds[window] = adjacency.copy()
                seed = adjacency
        own = cube[t, ids]
        propagated = np.empty((len(products), len(ids), 8))
        for j, average in enumerate(averages):
            degree = average.sum(axis=1)
            require((degree > 0).all() and np.array_equal(average, average.T) and np.all(np.diag(average) == 0), "Invalid ensemble")
            normalized = average / np.sqrt(np.outer(degree, degree))
            propagated[j] = normalized @ own
        require(np.isfinite(propagated).all(), "Unavailable propagated features")
        # Only now read future returns. These cannot alter the ordered IDs.
        chosen = ledger.loc[ledger.eligible_1260]
        label_date = sessions[t+1]
        forward = returns[t+1, ids]
        vol = volatility[t, ids]
        require(np.isfinite(vol).all() and (vol > 0).all(), "Invalid signal-time volatility")
        target = forward / vol
        daily_dir = output / "daily"; daily_dir.mkdir(exist_ok=True)
        array_path = daily_dir / f"{day}.npz"
        np.savez_compressed(array_path, source_security_ids=chosen.source_security_id.to_numpy(dtype=str),
            intrinio_security_ids=chosen.intrinio_security_id.to_numpy(dtype=str), membership_tickers=chosen.membership_ticker.to_numpy(dtype=str),
            own_features=own, network_features=propagated, daily_volatility=vol, forward_return=forward, target=target,
            label_date=np.array(label_date.to_datetime64()), products=np.array(products))
        ledger_path = daily_dir / f"{day}_eligibility.parquet"
        write_parquet(ledger, ledger_path)
        coverage = {"date": day, "members": len(cohort), "nodes": len(ids), "available_labels": int(np.isfinite(target).sum()),
                    **{f"eligible_{w}": int(ledger[f"eligible_{w}"].sum()) for w in LOOKBACKS},
                    "label_date": str(label_date.date()), "graph_seconds": float(sum(c["seconds"] for c in certificates))}
        marker = {"coverage": coverage, "certificates": certificates, "config_sha256": canonical_hash(config),
            "series_script_sha256": file_hash(__file__), "solver_script_sha256": file_hash(ROOT / "graph_grid.py"),
            "objects": [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)} for p in [array_path, ledger_path]]}
        write_json(marker_path, marker)
        summaries.append(coverage); daily_certificates.extend(certificates); completed.append(day)
        seed_ids, seeds = ids.copy(), next_seeds
        if len(completed) % 10 == 0 or len(completed) == 1:
            print(json.dumps({"completed_dates": len(completed), "total_dates": len(expected), "latest": day,
                              "nodes": len(ids), "elapsed_seconds": time.monotonic()-started}), flush=True)
    require(len(completed) == len(expected), "Incomplete daily series")
    bases, networks = consolidate(output, completed, products)
    coverage = pd.DataFrame(summaries)
    coverage.to_csv(output / "coverage_daily.csv", index=False)
    certificates = pd.DataFrame(daily_certificates)
    certificates.to_csv(output / "solver_certificates.csv", index=False)
    manifest = {"status": "complete", "purpose": "engineering sample" if sample_dates is not None else "causal daily graph/label series",
        **config, "dates": len(completed), "graph_fits": len(certificates), "predictor_rows": int(coverage.nodes.sum()),
        "member_rows": int(coverage.members.sum()), "label_rows": int(coverage.available_labels.sum()),
        "minimum_nodes": int(coverage.nodes.min()), "maximum_nodes": int(coverage.nodes.max()),
        "max_relative_kkt_residual": float(certificates.relative_kkt_residual.max()),
        "max_dual_gap_per_node": float(certificates.gap_per_node.max()),
        "elapsed_seconds": time.monotonic()-started, "base_files": bases, "network_files": networks,
        "history": str(history), "dataset": str(dataset), "ols_models_fitted": False, "out_of_sample_evaluated": False,
        "script_sha256": file_hash(__file__), "graph_solver_sha256": file_hash(ROOT / "graph_grid.py")}
    manifest["outputs"] = [{"path": p.relative_to(output).as_posix(), "size_bytes": p.stat().st_size, "sha256": file_hash(p)}
                           for p in sorted(output.rglob("*")) if p.is_file()]
    write_json(output / "manifest.json", manifest)
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"outputs", "network_files", "base_files"}}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=ROOT / "data/momentum_history/wfo_2016_2025_v1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--output", type=Path, default=ROOT / "data/daily_networks/wfo_grid_v1")
    parser.add_argument("--start", default="2016-01-04")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--products", type=float, nargs="+")
    parser.add_argument("--sample-dates", nargs="+")
    args = parser.parse_args()
    build(args.history.resolve(), args.dataset.resolve(), args.output.resolve(), args.start, args.end, args.products, args.sample_dates)

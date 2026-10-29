#!/usr/bin/env python3
"""Independently verify saved first-window features, PIT nodes and graph outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.sparse.csgraph import connected_components

from audit_first_window import read_slice
from build_momentum_features import FEATURES, LOOKBACKS, write_json
from prepare_sp500_daily import file_hash, require
from review_minor_closes import load_feature_prices


ROOT = Path(__file__).resolve().parent


def independent_features(close):
    """NumPy formula reference for the current uninterrupted run, without pandas EWM."""
    values = np.asarray(close, dtype=float)
    valid = np.isfinite(values) & (values > 0)
    start = np.flatnonzero(~valid)[-1] + 1 if (~valid).any() else 0
    p = values[start:]
    require(len(p) >= 314, "Sample lacks a full current feature window")
    r = p[1:] / p[:-1] - 1
    vol = np.full(len(p), np.nan)
    for t in range(60, len(p)):
        samples = r[:t]
        weights = (1 - 2 / 61) ** np.arange(t - 1, -1, -1)
        mean = weights @ samples / weights.sum()
        vol[t] = np.sqrt(weights @ ((samples - mean) ** 2) / weights.sum())
    raw = np.full((len(p), 8), np.nan)
    for j, horizon in enumerate((1, 21, 63, 126, 252)):
        raw[horizon:, j] = (p[horizon:] / p[:-horizon] - 1) / (vol[horizon:] * np.sqrt(horizon))
    price_std = np.full(len(p), np.nan)
    for t in range(62, len(p)):
        price_std[t] = np.std(p[t-62:t+1], ddof=0)
    for j, (short, long) in enumerate(((8, 24), (16, 48), (32, 96)), 5):
        ema_short, ema_long = np.empty(len(p)), np.empty(len(p))
        ema_short[0] = ema_long[0] = p[0]
        for t in range(1, len(p)):
            ema_short[t] = p[t] / short + (1 - 1 / short) * ema_short[t-1]
            ema_long[t] = p[t] / long + (1 - 1 / long) * ema_long[t-1]
        norm = (ema_short - ema_long) / price_std
        for t in range(313, len(p)):
            raw[t, j] = norm[t] / np.std(norm[t-251:t+1], ddof=0)
    clipped = np.full(8, np.nan)
    for j in range(8):
        available = np.flatnonzero(np.isfinite(raw[:, j]))
        samples = raw[available, j]
        weights = 2. ** (-(len(p) - 1 - available) / 252)
        mean = weights @ samples / weights.sum()
        std = np.sqrt(weights @ ((samples - mean) ** 2) / weights.sum())
        clipped[j] = np.clip(raw[-1, j], mean - 5 * std, mean + 5 * std)
    return raw[-1], clipped, vol[-1]


def verify(features, pilot, dataset, layer, output):
    require(not output.exists(), "Verification output exists")
    manifests = {}
    files_checked = 0
    for root in (features, pilot):
        manifest = json.loads((root / "manifest.json").read_text())
        require(manifest["status"] == "complete", "Incomplete build")
        for entry in manifest["outputs"]:
            p = root / entry["path"]
            require(p.stat().st_size == entry["size_bytes"] and file_hash(p) == entry["sha256"],
                    f"Output fingerprint mismatch: {p}")
            files_checked += 1
        manifests[str(root)] = manifest
    fm, gm = manifests[str(features)], manifests[str(pilot)]
    require(gm["input_feature_manifest_sha256"] == file_hash(features / "manifest.json"), "Graph lineage mismatch")
    require(fm["input_review_manifest_sha256"] == file_hash(layer / "manifest.json"), "Close layer lineage mismatch")
    require(fm["input_dataset_manifest_sha256"] == file_hash(dataset / "manifest.json"), "PIT lineage mismatch")
    source = json.loads((dataset / "manifest.json").read_text())
    anchor = read_slice(dataset, source, "panel", ["source_security_id", "intrinio_security_id", "identity_eligible"],
                        fm["anchor_date"], fm["anchor_date"]).set_index("source_security_id")
    nodes = pd.read_csv(pilot / "nodes.csv")
    require(len(anchor) == 505 and len(nodes) == fm["common_ensemble_nodes"], "Cohort count mismatch")
    require(anchor.loc[nodes.source_security_id].identity_eligible.all(), "Graph contains a withheld identity")
    require(anchor.loc[nodes.source_security_id, "intrinio_security_id"].tolist() == nodes.intrinio_security_id.tolist(),
            "Graph native identities differ from dated PIT panel")
    # Independently evaluate actual source prices for a fixed spread of stocks.
    selected = nodes.iloc[np.linspace(0, len(nodes)-1, 16, dtype=int)].intrinio_security_id.tolist()
    price = load_feature_prices(fm["start"], fm["end"], columns=["date", "intrinio_security_id", "adj_close",
                                "feature_close_eligible"], layer=layer)
    price = price.loc[price.intrinio_security_id.isin(selected)]
    paths = [str(features / e["path"]) for e in fm["outputs"] if e["path"].startswith("native_features/")]
    saved = ds.dataset(paths, format="parquet").to_table(filter=ds.field("intrinio_security_id").isin(selected)).to_pandas()
    max_raw_error, max_clipped_error, max_vol_error = 0., 0., 0.
    for native in selected:
        frame = saved.loc[saved.intrinio_security_id.eq(native)].sort_values("date").set_index("date")
        raw_prices = price.loc[price.intrinio_security_id.eq(native)].set_index("date").reindex(frame.index)
        masked = raw_prices.adj_close.where(raw_prices.feature_close_eligible.eq(True))
        np.testing.assert_allclose(masked, frame.feature_adj_close, equal_nan=True, rtol=0, atol=0)
        raw, clipped, vol = independent_features(masked)
        actual_raw = frame[["raw_" + x for x in FEATURES]].iloc[-1].to_numpy(float)
        actual_clipped = frame[FEATURES].iloc[-1].to_numpy(float)
        np.testing.assert_allclose(raw, actual_raw, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(clipped, actual_clipped, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(vol, frame.daily_volatility.iloc[-1], rtol=1e-12, atol=1e-14)
        max_raw_error = max(max_raw_error, float(np.max(abs(raw - actual_raw))))
        max_clipped_error = max(max_clipped_error, float(np.max(abs(clipped - actual_clipped))))
        max_vol_error = max(max_vol_error, float(abs(vol - frame.daily_volatility.iloc[-1])))
    graphs, relative_kkt = [], []
    with np.load(features / "anchor_feature_history.npz", allow_pickle=False) as archive:
        cube, source_ids = archive["features"], archive["source_security_ids"]
    for window in LOOKBACKS:
        with np.load(pilot / f"graph_{window}.npz", allow_pickle=False) as archive:
            adjacency, distances, dual = archive["adjacency"], archive["squared_distances"], archive["dual_variables"]
            np.testing.assert_array_equal(archive["source_security_ids"], source_ids)
        require(np.array_equal(adjacency, adjacency.T) and np.all(np.diag(adjacency) == 0)
                and (adjacency >= 0).all() and (adjacency.sum(axis=1) > 0).all(), "Invalid adjacency")
        # Direct edge stationarity uses d from stored A, independently of dual
        # solver stopping conditions. Check every edge, including zeros.
        degree = adjacency.sum(axis=1)
        il, ir = np.triu_indices(len(nodes), 1)
        grad = distances[il, ir] + 4 * gm["beta"] * adjacency[il, ir] - gm["alpha"] * (1/degree[il] + 1/degree[ir])
        violation = np.where(adjacency[il, ir] > 0, np.abs(grad), np.maximum(-grad, 0))
        scale = np.maximum(1., distances[il, ir] + gm["alpha"] * (1/degree[il] + 1/degree[ir]))
        error = float(np.max(violation / scale))
        require(error < 1e-6, "Graph KKT check failed")
        relative_kkt.append(error)
        # Compare selected direct distances to the original feature cube.
        for i, j in ((0, 1), (7, 50), (100, len(nodes)-1)):
            expected = np.square(cube[-window:, i] - cube[-window:, j]).sum()
            np.testing.assert_allclose(distances[i, j], expected, rtol=1e-12)
        np.testing.assert_allclose(degree * dual / gm["alpha"], 1., rtol=1e-6)
        graphs.append(adjacency)
    average = sum(graphs) / len(graphs)
    normalized = average / np.sqrt(np.outer(average.sum(axis=1), average.sum(axis=1)))
    with np.load(pilot / "ensemble.npz", allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["adjacency_average"], average, rtol=0, atol=0)
        np.testing.assert_allclose(archive["adjacency_normalized"], normalized, rtol=1e-14)
        np.testing.assert_allclose(archive["network_features"], normalized @ cube[-1], rtol=1e-13, atol=1e-13)
    _, component = connected_components(average > 0, directed=False)
    result = {"status": "passed", "sha256_output_objects_verified": files_checked,
        "dated_pit_members": len(anchor), "aligned_graph_nodes": len(nodes),
        "independent_numpy_feature_checks_securities": len(selected), "independent_feature_values_checked": len(selected) * 8,
        "maximum_raw_feature_absolute_error": max_raw_error, "maximum_clipped_feature_absolute_error": max_clipped_error,
        "maximum_daily_volatility_absolute_error": max_vol_error, "maximum_relative_graph_kkt_residual": max(relative_kkt),
        "ensemble_component_sizes": pd.Series(component).value_counts().tolist(),
        "feature_manifest_sha256": file_hash(features / "manifest.json"), "pilot_manifest_sha256": file_hash(pilot / "manifest.json"),
        "verification_script_sha256": file_hash(__file__),
        "limits": "Numerical/source integrity checks do not authenticate prices, identities, knowledge time or investment returns."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=ROOT / "data/momentum_features/v1")
    parser.add_argument("--pilot", type=Path, default=ROOT / "data/network_pilot/2016-01-04_a1_b1_v1")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/sp500_daily")
    parser.add_argument("--layer", type=Path, default=ROOT / "data/feature_close_review/v1")
    parser.add_argument("--output", type=Path, default=ROOT / "data/momentum_pilot_verification/v1/report.json")
    args = parser.parse_args()
    verify(args.features.resolve(), args.pilot.resolve(), args.dataset.resolve(), args.layer.resolve(), args.output.resolve())

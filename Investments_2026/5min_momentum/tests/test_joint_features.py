"""Exact duplicate policy and hard earliest-start checks, without fitting models."""
from pathlib import Path
import sys
import tempfile
import json
from datetime import datetime,timezone,timedelta
from unittest import mock
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from joint_features import MODEL_FEATURES,exact_duplicate_columns
import train_xgb_ret_ensemble as engine


class JointTests(unittest.TestCase):
    def test_union_has_unique_names_and_all_authorized_blocks(self):
        self.assertEqual(len(MODEL_FEATURES),30)
        self.assertEqual(len(set(MODEL_FEATURES)),30)
        self.assertEqual(sum(x.startswith("nm_") for x in MODEL_FEATURES),8)
        self.assertEqual(sum(x.startswith("vol_") for x in MODEL_FEATURES),6)
        self.assertEqual(MODEL_FEATURES[-1],"overnight_gap")

    def test_remove_identical_columns_but_keep_distinct_normalizations(self):
        frame=pd.DataFrame(dict(a=[1.,2.,np.nan],duplicate=[1.,2.,np.nan],scaled=[2.,4.,np.nan]))
        kept,removed=exact_duplicate_columns(frame,list(frame))
        self.assertEqual(kept,["a","scaled"])
        self.assertEqual(removed[0]["removed"],"duplicate")

    def test_deduplication_uses_actual_float32_model_precision(self):
        frame=pd.DataFrame(dict(a=[1.,2.],b=[1.+1e-10,2.+1e-10]))
        kept,_=exact_duplicate_columns(frame,["a","b"])
        self.assertEqual(kept,["a"])
        frame=pd.DataFrame(dict(a=[0.,np.nan],b=[-0.,np.nan]))
        self.assertEqual(exact_duplicate_columns(frame,["a","b"])[0],["a"])

    def test_start_gate_rejects_even_one_microsecond_early(self):
        gate=datetime(2026,11,6,14,40,tzinfo=timezone.utc)
        cfg={"not_before_utc":gate.isoformat()}
        with self.assertRaises(RuntimeError):engine.require_start_time(cfg,gate-timedelta(microseconds=1))
        engine.require_start_time(cfg,gate)
        with self.assertRaises(ValueError):engine.require_start_time({"not_before_utc":"2026-11-06T15:40:00"},gate)

    def test_batch_cannot_bypass_gate_before_loading_or_fitting(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/"CONFIG.json").write_text(json.dumps({"not_before_utc":"2099-01-01T00:00:00+00:00"}))
            with mock.patch.object(engine.base,"read_fold") as reader:
                with self.assertRaises(RuntimeError):engine.batch_worker(run,"2017Q1",[0])
                reader.assert_not_called()

    def test_exclusive_controller_prevents_a_duplicate_run(self):
        with tempfile.TemporaryDirectory() as folder:
            with engine.controller_lock(Path(folder)):
                with self.assertRaises(RuntimeError):
                    with engine.controller_lock(Path(folder)):pass

    def test_workers_launch_from_the_immutable_run_snapshot(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);(run/"logs").mkdir()
            frozen=run/"source_snapshot/scripts/train_xgb_ret_ensemble.py"
            frozen.parent.mkdir(parents=True);frozen.write_text("# pinned worker")
            with mock.patch.object(engine.subprocess,"Popen") as spawn:
                task=engine.start_task(run,"2017Q1",[0,1],{})
                command=spawn.call_args.args[0]
                self.assertEqual(Path(command[1]),frozen)
                task["log"].close()

    def test_joint_config_has_no_resource_or_round_budgets(self):
        with tempfile.TemporaryDirectory() as folder:
            project=Path(folder);run=project/"data/models/joint"
            f=project/"data/features/09_joint_momentum_volume_gap_v1";f.mkdir(parents=True)
            (f/"MANIFEST.json").write_text("{}")
            with mock.patch.object(engine,"prepare_joint_panel"),mock.patch.object(engine,"wait_for_start",side_effect=InterruptedError("pre-fit check")):
                with self.assertRaises(InterruptedError):engine.full_run(project,run,experiment="joint30",not_before="2026-11-06T14:40:00+00:00")
            cfg=json.loads((run/"CONFIG.json").read_text())
            for key in ["max_rounds","memory_limit_gib","hard_stop_utc"]:self.assertIsNone(cfg[key])
            self.assertEqual(cfg["minimum_available_memory_gib"],0.)
            self.assertEqual(cfg["members_per_quarter"],100)
            self.assertEqual(cfg["early_stopping_patience"],30)
            self.assertFalse(list(run.rglob("*.ubj")))


if __name__=="__main__":unittest.main()

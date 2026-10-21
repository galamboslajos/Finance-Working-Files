import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import control
import supervise


class Child:
    pid = 987654
    returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class RunningChild(Child):
    def poll(self):
        return None


class TerminatedChild(Child):
    def poll(self):
        supervise.signal.getsignal(supervise.signal.SIGTERM)(supervise.signal.SIGTERM, None)


class ControlTests(unittest.TestCase):
    def test_reservation_and_duplicate_are_not_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            p = root / "plan.json"
            control.save_new(p, {"scope": "synthetic"})
            with patch.object(control, "ROOT", root):
                control.reserve_read(root, "one", 12, {"plan_pin": control.pin(p)})
                control.require_reservation(root, "one", 12, control.pin(p))
                with self.assertRaises(ValueError):
                    control.reserve_read(root, "one", 12, {"plan_pin": control.pin(p)})
                self.assertEqual(control.read(root / "run/read_usage.json")["totals"]["compressed_source"], 12)

    def test_category_caps_and_plan_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            p = root / "plan.json"
            control.save_new(p, {})
            with patch.object(control, "ROOT", root):
                with self.assertRaises(ValueError):
                    control.reserve_read(root, "big", 12 * 1024**3 + 1, {})
                old = control.pin(p)
                p.write_text("changed")
                with self.assertRaises(ValueError):
                    control.reserve_read(root, "changed", 1, {"plan_pin": old})


class SupervisorTests(unittest.TestCase):
    def fixture(self, root):
        worker = root / "synthetic_worker.py"
        worker.write_text("# fabricated test worker\n")
        protocol = root / "protocol.json"
        control.save_new(protocol, {"status": "SEALED_BEFORE_FINANCIAL_READ",
            "python_path": sys.executable, "limits": {
                "active_seconds_by_category": {"synthetic": 60},
                "available_proxy_minimum_bytes": 7 * 1024**3,
                "free_disk_minimum_bytes": 0, "rss_warning_bytes": 16 * 1024**3,
                "rss_stop_bytes": 20 * 1024**3, "output_maximum_bytes": 1024**3}})
        launch = root / "launch.json"
        control.save_new(launch, {"schema": "dmv2-corrected-refit-phase-launch/v1",
            "protocol_pin": control.pin(protocol), "code_pins": [control.pin(worker)],
            "phase": "synthetic", "time_category": "synthetic", "maximum_seconds": 30,
            "command": [sys.executable, str(worker)]})
        return launch

    def fake_resources(self, available=8 * 1024**3):
        return {"macos_memory_pressure": 1, "available_proxy_bytes": available,
            "free_disk_bytes": 100 * 1024**3}

    def test_waiting_for_memory_does_not_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            with patch.object(supervise, "ROOT", root), patch.object(supervise, "resources", return_value=self.fake_resources(1)), patch.object(supervise.subprocess, "Popen") as popen:
                supervise.run(launch, control.pin(launch)["sha256"])
                popen.assert_not_called()
            self.assertEqual(control.read(root / "run/control/synthetic.json")["status"], "WAITING_FOR_MEMORY")

    def test_pause_does_not_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            control.save_new(root / "PAUSE_REQUESTED.json", {"synthetic": True})
            with patch.object(supervise, "ROOT", root), patch.object(supervise, "resources", return_value=self.fake_resources()), patch.object(supervise.subprocess, "Popen") as popen:
                supervise.run(launch, control.pin(launch)["sha256"])
                popen.assert_not_called()
            self.assertEqual(control.read(root / "run/control/synthetic.json")["status"], "PAUSED_BEFORE_DISPATCH")

    def test_success_cleans_owned_group_and_charges_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            with patch.object(supervise, "ROOT", root), patch.object(supervise, "resources", return_value=self.fake_resources()), patch.object(supervise.subprocess, "Popen", return_value=Child()), patch.object(supervise.os, "killpg") as kill:
                supervise.run(launch, control.pin(launch)["sha256"])
                self.assertEqual(kill.call_count, 2)
            self.assertEqual(control.read(root / "run/control/synthetic.json")["status"], "COMPLETE")
            self.assertGreater(control.read(root / "run/control/active_time.json")["consumed_seconds"]["synthetic"], 0)

    def test_rss_stop_preserves_failure_and_kills_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            with patch.object(supervise, "ROOT", root), patch.object(supervise, "resources", return_value=self.fake_resources()), patch.object(supervise.subprocess, "Popen", return_value=RunningChild()), patch.object(supervise, "tree_rss", return_value=21 * 1024**3), patch.object(supervise.subprocess, "check_output", return_value="1"), patch.object(supervise.os, "killpg") as kill:
                with self.assertRaises(ValueError):
                    supervise.run(launch, control.pin(launch)["sha256"])
                self.assertEqual(kill.call_count, 2)
            self.assertEqual(control.read(root / "run/control/synthetic.json")["status"], "FAILED_OR_STOPPED")

    def test_external_termination_cleans_owned_group_and_restores_handler(self):
        previous = supervise.signal.getsignal(supervise.signal.SIGTERM)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            with patch.object(supervise, "ROOT", root), patch.object(supervise, "resources", return_value=self.fake_resources()), patch.object(supervise.subprocess, "Popen", return_value=TerminatedChild()), patch.object(supervise.os, "killpg") as kill:
                with self.assertRaises(SystemExit):
                    supervise.run(launch, control.pin(launch)["sha256"])
                self.assertEqual(kill.call_count, 2)
            self.assertEqual(control.read(root / "run/control/synthetic.json")["status"], "FAILED_OR_STOPPED")
        self.assertEqual(supervise.signal.getsignal(supervise.signal.SIGTERM), previous)

    def test_existing_attempt_and_changed_worker_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            launch = self.fixture(root)
            (root / "synthetic_worker.py").write_text("changed")
            with patch.object(supervise, "ROOT", root), patch.object(supervise.subprocess, "Popen") as popen:
                with self.assertRaises(ValueError):
                    supervise.run(launch, control.pin(launch)["sha256"])
                popen.assert_not_called()


if __name__ == "__main__":
    unittest.main()

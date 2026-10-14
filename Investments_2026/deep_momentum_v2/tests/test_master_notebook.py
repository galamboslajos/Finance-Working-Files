"""Interactive rerun state checks; notebook cells execute with mocked workers."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock


def cell_containing(needle):
    path = Path(__file__).resolve().parents[1] / "notebooks/00_research_master.ipynb"
    notebook = json.loads(path.read_text())
    matches = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code" and needle in "".join(cell["source"])]
    if len(matches) != 1:
        raise AssertionError(f"Expected one notebook cell containing {needle!r}")
    return matches[0]


class InteractiveRerunTests(unittest.TestCase):
    def namespace(self):
        return {"catalog": Mock(), "settings": {}, "display": Mock(), "Markdown": str}

    def test_failed_monthly_rerun_clears_previous_results_before_worker_call(self):
        state = self.namespace()
        names = ("monthly_run", "daily_df", "monthly_df", "numerical_preview", "monthly_artifact_path")
        state.update({name: object() for name in names})
        state["run_monthly_sample"] = Mock(side_effect=ValueError("bounded read failed"))
        with self.assertRaisesRegex(ValueError, "bounded read failed"):
            exec(cell_containing("run_monthly_sample(catalog, settings)"), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_disabled_monthly_rerun_clears_previous_results(self):
        state = self.namespace()
        names = ("monthly_run", "daily_df", "monthly_df", "numerical_preview", "monthly_artifact_path")
        state.update({name: object() for name in names})
        state["run_monthly_sample"] = Mock(return_value=None)
        exec(cell_containing("run_monthly_sample(catalog, settings)"), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_failed_inspection_rerun_clears_previous_snapshot(self):
        state = self.namespace()
        state["inspection"] = object()
        state["run_inspection"] = Mock(side_effect=ValueError("membership read failed"))
        with self.assertRaisesRegex(ValueError, "membership read failed"):
            exec(cell_containing("run_inspection(catalog, settings)"), state)
        self.assertIsNone(state["inspection"])

    def test_preview_cell_clears_old_preview_when_monthly_frame_is_unavailable(self):
        state = {"monthly_df": None, "numerical_preview": object()}
        exec(cell_containing("numerical_preview = monthly_df.loc"), state)
        self.assertIsNone(state["numerical_preview"])


if __name__ == "__main__":
    unittest.main()

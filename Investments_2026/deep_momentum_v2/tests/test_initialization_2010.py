"""Selected chronology contracts; no financial observations or source reads."""
import json
from pathlib import Path
import unittest

from dmv2.chronology import build_annual_folds
from dmv2.workflow import load_workflow, workflow_identity_policy
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY


ROOT = Path(__file__).resolve().parents[1]


class Initialization2010Tests(unittest.TestCase):
    def test_current_workflow_keeps_ten_years_and_has_four_later_tests(self):
        settings = load_workflow(ROOT)
        train = settings["training"]
        folds = build_annual_folds(settings["research_start"], train["last_test_year"],
                                   train["train_years"], mode=train["mode"])
        self.assertEqual(settings["research_start"], "2010-01-01")
        self.assertEqual((train["mode"], train["train_years"], train["test_years"]),
                         ("rolling", 10, 1))
        self.assertEqual(len(folds), 4)
        self.assertEqual([f.test_start for f in folds],
                         [f"{year}-01-01" for year in range(2020, 2024)])
        self.assertEqual((folds[0].train_start, folds[0].train_end,
                          folds[0].label_cutoff),
                         ("2010-01-01", "2019-12-31", "2019-12-31"))
        self.assertEqual((folds[-1].train_start, folds[-1].train_end, folds[-1].test_end),
                         ("2013-01-01", "2022-12-31", "2023-12-31"))
        self.assertEqual(workflow_identity_policy(settings), PROVISIONAL_IDENTITY_POLICY)

    def test_start_change_does_not_enable_models_or_remove_warmup(self):
        settings = load_workflow(ROOT)
        self.assertEqual(settings["inspection"]["as_of"], "2010-01-29")
        self.assertEqual((settings["monthly_returns"]["start_month"],
                          settings["monthly_returns"]["end_month"],
                          settings["monthly_returns"]["sample_size"]),
                         ("2010-01", "2010-02", 2))
        self.assertEqual(settings["features"]["history_sessions"], 503)
        self.assertEqual(settings["warmup"]["policy"], "per_signal_exact_feature_windows")
        self.assertFalse(settings["execution"]["full_panel_build_enabled"])
        self.assertFalse(settings["execution"]["model_execution_enabled"])
        notebook = json.loads((ROOT / "notebooks/00_research_master.ipynb").read_text())
        self.assertTrue(all(cell.get("execution_count") is None and not cell.get("outputs")
                            for cell in notebook["cells"] if cell["cell_type"] == "code"))


if __name__ == "__main__":
    unittest.main()

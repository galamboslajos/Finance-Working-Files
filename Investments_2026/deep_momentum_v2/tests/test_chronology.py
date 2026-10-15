"""Calendar-only chronology contracts; no fabricated market data or model fits."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, timedelta
import unittest

from dmv2.chronology import build_annual_folds


class AnnualFoldTests(unittest.TestCase):
    def test_selected_rolling_10_1_has_four_folds_from_2020_through_2023(self):
        folds = build_annual_folds()
        self.assertEqual(len(folds), 4)
        self.assertEqual(folds[0].train_start, "2010-01-01")
        self.assertEqual(folds[0].train_end, "2019-12-31")
        self.assertEqual(folds[0].test_start, "2020-01-01")
        self.assertEqual(folds[0].test_end, "2020-12-31")
        self.assertEqual(folds[-1].train_start, "2013-01-01")
        self.assertEqual(folds[-1].train_end, "2022-12-31")
        self.assertEqual(folds[-1].test_start, "2023-01-01")
        self.assertEqual(folds[-1].test_end, "2023-12-31")

    def test_training_cutoff_is_last_completed_outcome_before_test_year(self):
        for fold in build_annual_folds():
            with self.subTest(fold=fold.fold_id):
                end = date.fromisoformat(fold.train_end)
                start = date.fromisoformat(fold.test_start)
                self.assertEqual(fold.label_cutoff, fold.train_end)
                self.assertEqual(end + timedelta(days=1), start)
                self.assertLess(fold.label_cutoff, fold.test_start)
                # A December signal's January label is a test OUTCOME.
                self.assertGreaterEqual(start.replace(day=31).isoformat(), fold.test_start)
                self.assertGreater(start.replace(day=31).isoformat(), fold.label_cutoff)

    def test_rolling_keeps_exactly_ten_calendar_years_including_leap_years(self):
        for fold in build_annual_folds(mode="rolling"):
            train_start = date.fromisoformat(fold.train_start)
            train_end = date.fromisoformat(fold.train_end)
            self.assertEqual(train_end.year - train_start.year + 1, 10)
            self.assertEqual((train_start.month, train_start.day), (1, 1))
            self.assertEqual((train_end.month, train_end.day), (12, 31))
        leap_window = build_annual_folds("2000-01-01", 2001, 1)[0]
        self.assertEqual((date.fromisoformat(leap_window.train_end)
                          - date.fromisoformat(leap_window.train_start)).days + 1, 366)

    def test_expanding_is_explicit_and_keeps_original_training_start(self):
        folds = build_annual_folds('2000-01-01',mode="expanding")
        self.assertEqual(len(folds), 14)
        self.assertTrue(all(fold.train_start == "2000-01-01" for fold in folds))
        self.assertEqual(folds[-1].train_end, "2022-12-31")
        self.assertEqual(folds[0].test_start, "2010-01-01")
        self.assertEqual(folds[-1].mode, "expanding")

    def test_test_years_are_adjacent_disjoint_full_years(self):
        folds = build_annual_folds()
        for previous, current in zip(folds, folds[1:]):
            self.assertEqual(date.fromisoformat(previous.test_end) + timedelta(days=1),
                             date.fromisoformat(current.test_start))
        self.assertTrue(all(fold.test_start.endswith("-01-01")
                            and fold.test_end.endswith("-12-31") for fold in folds))

    def test_one_fold_and_alternate_positive_training_length(self):
        folds = build_annual_folds("2000-01-01", 2003, 3, mode="rolling")
        self.assertEqual(len(folds), 1)
        self.assertEqual(folds[0].train_end, "2002-12-31")
        self.assertEqual(folds[0].test_start, "2003-01-01")

    def test_result_and_records_are_immutable(self):
        folds = build_annual_folds()
        self.assertIsInstance(folds, tuple)
        with self.assertRaises(FrozenInstanceError):
            folds[0].train_start = "2010-01-01"
        with self.assertRaises(TypeError):
            folds[0] = folds[1]

    def test_same_explicit_protocol_has_stable_ordered_labels(self):
        arguments = dict(history_start="2000-01-01", last_test_year=2023,
                         train_years=10, mode="rolling")
        first, second = build_annual_folds(**arguments), build_annual_folds(**arguments)
        self.assertEqual(first, second)
        self.assertEqual(first[0].fold_id, "rolling-10y-test-2010")
        self.assertEqual(first[-1].fold_id, "rolling-10y-test-2023")
        self.assertEqual(len({fold.fold_id for fold in first}), len(first))

    def test_mode_label_distinguishes_rolling_from_expanding(self):
        self.assertNotEqual(build_annual_folds(mode="rolling")[0].fold_id,
                            build_annual_folds(mode="expanding")[0].fold_id)

    def test_protected_2024_and_later_test_years_fail(self):
        for year in (2024, 2026, 9999):
            with self.subTest(year=year):
                with self.assertRaisesRegex(ValueError, "protected"):
                    build_annual_folds(last_test_year=year)

    def test_partial_or_noncanonical_history_start_fails(self):
        for value in ("2000-02-01", "2000-01-02", "2000-1-1", "20000101",
                      "2000-01-01T00:00:00", " 2000-01-01", "2000-01-01 ",
                      "2000-02-30", "not-a-date", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_annual_folds(history_start=value)

    def test_nonstring_history_start_fails(self):
        for value in (None, date(2000, 1, 1), 2000):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_annual_folds(history_start=value)

    def test_invalid_training_length_fails_instead_of_coercing(self):
        for value in (0, -1, 10.0, "10", True, False, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_annual_folds(train_years=value)

    def test_invalid_last_test_year_fails_instead_of_coercing(self):
        for value in (0, -1, 2023.0, "2023", True, False, None):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_annual_folds(last_test_year=value)

    def test_invalid_training_mode_fails_instead_of_falling_back(self):
        for value in ("Rolling", "expansion", "", None, 1, []):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    build_annual_folds(mode=value)

    def test_insufficient_nominal_history_fails_instead_of_returning_empty(self):
        for arguments in (dict(last_test_year=2009),
                          dict(history_start="2020-01-01"),
                          dict(history_start="9999-01-01"),
                          dict(train_years=10_000)):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "nominal training"):
                    build_annual_folds(**arguments)


if __name__ == "__main__":
    unittest.main()

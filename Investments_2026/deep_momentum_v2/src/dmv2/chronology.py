"""Nominal annual outcome folds; no data reads, feature builds or model fits.

V2 user direction is a rolling ten-year training / one-year test schedule,
initialized in 2000. ``expanding`` is available only as an explicit comparison
to the paper's classifier schedule. The paper's trailing ten-year RET payoff
window is a separate estimator; this module does not construct it.

All bounds refer to COMPLETED OUTCOME dates, not signal dates. For example,
a December 2009 signal predicting January 2010 belongs to the 2010 test;
its label cannot enter the training period ending December 2009.

These are calendar specifications, not coverage certification. Required
feature warmup, missing labels and delayed source availability can make the
first usable fit later than the nominal first test. Actual row selection must
also verify when each label became observable. Date-only cutoffs do not define
intraday execution or prove knowledge-time PIT correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal


TrainingMode = Literal["rolling", "expanding"]
LAST_DEVELOPMENT_YEAR = 2023


@dataclass(frozen=True, slots=True)
class AnnualFold:
    """Inclusive calendar bounds for one nominal annual outcome test.

    ``label_cutoff`` is the latest admissible completed outcome date. It is
    not a promise that every label was published by the fitting instant.
    The proposed V2 model uses outcomes ending before January 1 of its test
    year; this differs from the source paper's January-31 retraining date.
    """

    fold_id: str
    mode: TrainingMode
    train_start: str
    train_end: str
    label_cutoff: str
    test_start: str
    test_end: str


def build_annual_folds(
    history_start: str = "2000-01-01",
    last_test_year: int = LAST_DEVELOPMENT_YEAR,
    train_years: int = 10,
    *,
    mode: TrainingMode = "rolling",
) -> tuple[AnnualFold, ...]:
    """Return immutable, deterministic folds without inspecting any data.

    ``history_start`` must be an exact ISO January 1 and denotes the first
    nominal training OUTCOME year. Rolling advances the training start each
    year; expanding retains that start. Both end training on December 31 of
    the preceding year and test the full next calendar year.

    Defaults implement the selected V2 rolling 10/1 schedule: fourteen folds,
    training 2000-2009 / testing 2010 through training 2013-2022 / testing 2023.
    These bounds do not authorize market-data reads or guarantee a complete
    ten-year usable training panel. Protected 2024+ test years fail closed.
    """
    if not isinstance(history_start, str):
        raise ValueError("history_start must be an ISO YYYY-01-01 string")
    try:
        first = date.fromisoformat(history_start)
    except ValueError as exc:
        raise ValueError("history_start must be an ISO YYYY-01-01 string") from exc
    if first.isoformat() != history_start or (first.month, first.day) != (1, 1):
        raise ValueError("history_start must be an exact ISO January 1")
    if type(train_years) is not int or train_years < 1:
        raise ValueError("train_years must be a positive integer")
    if type(last_test_year) is not int or not 1 <= last_test_year <= LAST_DEVELOPMENT_YEAR:
        raise ValueError("last_test_year must be an integer in 1..2023; 2024+ is protected")
    if mode not in ("rolling", "expanding"):
        raise ValueError("mode must be 'rolling' or 'expanding'")
    first_test_year = first.year + train_years
    if first_test_year > last_test_year:
        raise ValueError("requested history does not provide a nominal training window before the last test year")

    folds = []
    for test_year in range(first_test_year, last_test_year + 1):
        train_start_year = test_year - train_years if mode == "rolling" else first.year
        train_end = date(test_year - 1, 12, 31).isoformat()
        folds.append(AnnualFold(
            fold_id=f"{mode}-{train_years}y-test-{test_year}",
            mode=mode,
            train_start=date(train_start_year, 1, 1).isoformat(),
            train_end=train_end,
            label_cutoff=train_end,
            test_start=date(test_year, 1, 1).isoformat(),
            test_end=date(test_year, 12, 31).isoformat(),
        ))
    return tuple(folds)

# Research maintenance rules

Read `docs/RESEARCH_CONTRACT.md` and `docs/AUDIT_FINDINGS.md` before changing
features, eligibility, labels, splits, execution or performance reporting.

- This repository is the canonical source. `data/` and `.venv/` are private
  storage links, not a second independently edited source tree.
- Source notebooks stay output-free. `scripts/run_notebook.py` writes private
  inspection outputs and must not start fits or rebuild absent releases.
- Never overwrite completed feature/model releases. Record new versions and
  source/input fingerprints; distinguish new code from historical results.
- Use security IDs and dense exchange calendars. No price filling, compressed
  dates, future-label eligibility or replacement of missing selected exits.
- Test future perturbations, feature units, provenance, early-close anchors and
  timestamps when changing construction. Keep exact row-level exclusions.
- Separate training gain from held-out return EV; label conditional metrics and
  unknown outcomes. Numerical checks do not certify PIT or tradability.
- Opening-price knowledge and same-price fills are optimistic research. Never
  certify execution while input delivery follows the assumed decision/fill.
- Respect 100 models, 12/3-month WFO, 30-round validation stopping and no arbitrary
  round/RAM/time ceilings unless the user changes them. Do not silently change
  the authorized random inner splits.
- Run relevant tests and `scripts/audit_research.py` after shared accounting,
  feature or timing changes.
- Cloud use is read/download only. Keep licensed outputs inside Galambos Capital;
  notify before downloads over 50 GB. Cleanup alone does not authorize one-minute work.

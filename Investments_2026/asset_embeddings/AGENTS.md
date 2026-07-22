# Repository Instructions

Read README.md, docs/PROJECT.md, and docs/DATA.md before changing research logic.

- Preserve point-in-time semantics; economic dates and availability dates are different.
- Never introduce silent imputations, filters, joins, or forward fills.
- Keep representation tests separate from strategy backtests.
- Start with simple baselines and require evidence before adding model complexity.
- Test look-ahead boundaries, identifiers, amendments, and portfolio accounting.
- Do not commit source data, licensed material, credentials, local tools, or generated artifacts.
- Keep the working tree understandable and avoid speculative scaffolding.
- Commit and push coherent milestones so collaborators can audit the full history.

# Repository Instructions

Read README.md, docs/PROJECT.md, and docs/DATA.md before changing research logic.

## Repository boundary

- This directory belongs to the existing Finance-Working-Files monorepo.
- The Git top-level directory must be named Finance-Working-Files.
- The required origin is https://github.com/galamboslajos/Finance-Working-Files.git.
- Never run git init inside Investments_2026/asset_embeddings.
- Never create or connect a standalone asset_embeddings repository.
- Do not stage or modify files outside Investments_2026/asset_embeddings unless the user explicitly
  expands the task.

Before work, verify the boundary with:

~~~bash
git rev-parse --show-toplevel
git remote get-url origin
git status --short --branch
~~~

## Commit and review policy

- Start substantive work from an updated main branch and use an agent/ feature branch.
- Keep commits task-sized and use messages that state the research or implementation outcome.
- Stage explicit asset_embeddings paths; never use a repository-wide git add -A.
- Review the staged file list, diff summary, and git diff --cached --check before committing.
- Run relevant tests or validation before pushing.
- Push each coherent milestone in the same working session.
- Use a draft pull request against Finance-Working-Files/main for substantive changes.
- At handoff, report the branch, commit hash, validation, and pull-request URL.
- If authentication, push, or PR creation fails, state the blocker clearly; never imply that local
  work is already visible on GitHub.
- Do not leave substantive completed work uncommitted or unpushed without an explicit reason.

## Public repository safety

- Treat every tracked file, commit, branch, pull request, issue, and log excerpt as public.
- Never commit credentials, tokens, cookies, service-account files, private keys, account emails,
  cloud project IDs, bucket names or URIs, internal object paths, non-public inventory statistics,
  or licensed data.
- Use placeholders and environment-variable names in documentation. Exchange real access details
  only through an approved private channel.
- Keep local values in ignored `.env`, `.gcloud`, or `.tools` paths.
- Review the complete staged diff before every commit. If a secret is exposed, stop, revoke or
  rotate it, and assess Git history; deleting it only from the latest version is insufficient.

## Research integrity

- Preserve point-in-time semantics; economic dates and availability dates are different.
- Never introduce silent imputations, filters, joins, or forward fills.
- Keep representation tests separate from strategy backtests.
- Start with simple baselines and require evidence before adding model complexity.
- Test look-ahead boundaries, identifiers, amendments, and portfolio accounting.
- Do not commit source data, licensed material, credentials, local tools, or generated artifacts.
- Keep the working tree understandable and avoid speculative scaffolding.

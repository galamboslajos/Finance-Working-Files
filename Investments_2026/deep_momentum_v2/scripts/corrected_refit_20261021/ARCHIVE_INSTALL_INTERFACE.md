The installer is staged only. No real gate plan, source inventory, destination copy or repository write has been performed by the implementation agent.

The concrete source is `/private/tmp/dmv2-corrected-refit-20261021`. The new destination is `/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/outputs/corrected_refit/2020-2023-20261021-v1`. All installer control records belong outside that source, under `/private/tmp/dmv2-corrected-refit-install-20261021`, so plan/receipt creation cannot alter an already sealed source inventory.

Root supplies one closed metadata scope JSON after every numerical stage is terminal. Its schema is `dmv2-corrected-refit-archive-scope/v1`, status `SEALED_AFTER_NUMERICAL_COMPLETION_BEFORE_ARCHIVE`, and fields are:

- `source_root`, `destination`, `all_numerical_stages_terminal: true`.
- `gates`: raw pins named `progress`, `models`, `backtest`, `model_report`, at the exact canonical paths in `validate_scope`.
- `model_protocol_pin`: the frozen model protocol raw pin.
- `backtest_expected_pins`: exact raw pins for `source_plan_pin`, `backtest_protocol_pin`, `selection_audit_pin`, `preflight_audit_pin`, `replay_audit_pin` in the independent backtest proof.
- `artifacts`: explicit locator labels to raw pins inside the new source attempt. All other source files are still preserved and indexed, including every model, failed attempt, checkpoint, accounting table, and withheld-path diagnostic.
- `original_input_pins`: explicit repository input pins to recheck before and after installation; an empty list makes no claim about unlisted external files.

Every raw pin has absolute `path`, `size_bytes` and `sha256`. The independent backtest contract is `dmv2-independent-corrected-backtest-verification/v1`, `PASS`,120 retained scenarios,1,006 sessions,48 months and15 selections. Any split between complete and withheld scenarios is accepted when the total is120 and all preservation checks pass. No minimum completed-path count is imposed.

Use four explicit steps:

1. `plan-gates --scope EXTERNAL_SCOPE --output EXTERNAL_GATE_PLAN`: path/size/hash checks only for metric-bearing proof JSON; no proof scores are parsed.
2. Root reserves `projected_compressed_source_bytes` in the existing shared `compressed_source` ledger, binding the exact gate-plan raw pin. `prepare --plan EXTERNAL_GATE_PLAN --plan-sha256 SHA --reservation LABEL --output EXTERNAL_INSTALL_PLAN` then parses the precharged proof JSON once, uses only terminal/provenance fields, checks terminal guard status, and freezes every regular source file. The reservation has an immutable consumed marker outside the source. Gate JSON can contain numerical summaries, so its complete file bytes are charged; the installer does not use those summaries.
3. Root reviews the frozen inventory. `execute --plan EXTERNAL_INSTALL_PLAN --plan-sha256 SHA` requires filesystem escalation through the calling tool to write the repository destination. It performs only raw hashing and opaque byte copies. The source is checked before and after, each destination file is verified, and the final archive including generated metadata must fit within20GiB. Existing or failed partial destinations are preserved and never overwritten or removed.
4. Optional, separate publication: `publish-doc --plan EXTERNAL_INSTALL_PLAN --plan-sha256 SHA --source-relative README.md --destination ABSOLUTE_REPOSITORY_DOCS_MARKDOWN_PATH`. This requires a successful matching archive, an explicit fresh docs path, and byte-identical Markdown from the sealed archive. It performs no Git operation.

The archive preserves embedded legacy provenance unchanged. Separate `relocation_index.json` and `installed_catalog.json` supply new absolute locations. The catalog lists every archived source file plus named artifact pins; external original inputs retain their original paths. Copies of the exact external installation plan, gate plan and scope are archived too. No active pointer is promoted.

After the inventory is sealed, no task, monitor or controller may write into the source. Any changed, added or missing source file fails installation. Raw opaque hashing/copy I/O is measured by the installer counter and is separate from numerical source decoding; no dataset table or model object is deserialized.

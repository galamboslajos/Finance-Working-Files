"""Explicit fresh corrected-source evidence rebind, preflight or 64-path replay."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('rebind', 'preflight', 'backtest'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--identity-sha256', required=True)
    p.add_argument('--reference-directory', type=Path, required=True)
    p.add_argument('--reference-audit-sha256', required=True)
    p.add_argument('--reference-phase-sha256', required=True)
    p.add_argument('--source-correction-manifest', type=Path)
    p.add_argument('--registry-output', type=Path)
    p.add_argument('--backtest-output', type=Path)
    p.add_argument('--event-registry', type=Path)
    p.add_argument('--identity-handover-registry', type=Path)
    p.add_argument('--identity-handover-decision', type=Path)
    p.add_argument('--allow-withheld-paths', action='store_true')
    args = p.parse_args(argv)
    root = args.root.resolve(); run = (root / args.run).resolve()
    if not run.is_relative_to(root / 'outputs/walkforward_source_corrected'):
        p.error('New run must be inside the separate source-corrected walk-forward outputs')
    sys.path.insert(0, str(root / 'src'))
    from dmv2.catalog import canonical_sha256
    from dmv2.backtest_loading import _pin
    from dmv2.source_corrected_backtest import build_rebound_execution_evidence, run_source_corrected_backtest
    identity = json.loads((run / 'build_identity.json').read_text())
    if canonical_sha256(identity) != args.identity_sha256:
        p.error('Explicit new root identity does not match')
    reference = dict(reference_directory=(root / args.reference_directory).resolve(),
        reference_audit_sha256=args.reference_audit_sha256, reference_phase_sha256=args.reference_phase_sha256)
    if args.phase == 'rebind':
        if args.registry_output is None: p.error('Rebind requires an explicit new private registry directory')
        result = build_rebound_execution_evidence(root, run, args.identity_sha256, **reference,
            output_directory=(root / args.registry_output).resolve())
        print(json.dumps({key: str(value) for key, value in result.items()}, sort_keys=True))
        return
    if args.backtest_output is None or args.source_correction_manifest is None:
        p.error('Preflight/backtest requires explicit new output and corrected-source manifest')
    registries = {}
    for key in ('event_registry', 'identity_handover_registry', 'identity_handover_decision'):
        value = getattr(args, key)
        if value is None: p.error('Explicit rebound registries and decision are required')
        path = (root / value).resolve()
        if not path.is_relative_to(root / 'data/derived'):
            p.error('Rebound execution registries must remain in private derived-data evidence')
        registries[key] = path
    output = (root / args.backtest_output).resolve()
    if output.parent != run or output.exists():
        p.error('Backtest output must be a new direct child directory of the new run')
    result = run_source_corrected_backtest(root, run, identity['configuration'], args.identity_sha256,
        **reference, **registries, output_directory=output,
        source_correction_manifest_pin=_pin(root, (root / args.source_correction_manifest).resolve()),
        preflight_only=args.phase == 'preflight', allow_withheld_paths=args.allow_withheld_paths)
    print(json.dumps({'phase': args.phase, 'directory': str(result), 'new_model_fits': 0,
                      'source_identity_promoted': False}, sort_keys=True))


if __name__ == '__main__': main()

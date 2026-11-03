#!/usr/bin/env python3
"""Export the verified held-valuation correction without the large raw dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from prepare_sp500_daily import file_hash, require

ROOT = Path(__file__).resolve().parent


def export(output, performance):
    require(not output.exists(), 'Export exists; choose a fresh directory')
    results = ROOT / 'data/walk_forward/valuation_results_v1'
    verification = ROOT / 'data/walk_forward/valuation_verification_v1/report.json'
    result = json.loads((results / 'manifest.json').read_text())
    checked = json.loads(verification.read_text())
    chart = json.loads((performance / 'manifest.json').read_text())
    require(result['status'] == chart['status'] == 'complete' and checked['status'] == 'passed', 'Completed verified outputs required')
    require(file_hash(results / 'manifest.json') == checked['result_manifest_sha256'] == chart['results_manifest_sha256'], 'Mixed output versions')
    copied = []

    def copy(source, relative):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        require(file_hash(source) == file_hash(destination), 'Export copy differs')
        copied.append({'path':relative, 'source_relative_path':source.relative_to(ROOT).as_posix(),
                       'sha256':file_hash(destination), 'size_bytes':destination.stat().st_size})

    for directory, manifest, prefix in [(results, result, 'results'), (performance, chart, 'performance')]:
        for item in manifest['outputs']:
            source = directory / item['path']
            require(file_hash(source) == item['sha256'], 'Verified output changed')
            if source.suffix != '.parquet':
                copy(source, prefix + '/' + item['path'])
        copy(directory / 'manifest.json', prefix + '/manifest.json')
    copy(verification, 'verification/valuation_report.json')
    copy(ROOT / 'data/performance_gap_audit/v1/reconstruction_verification.json', 'verification/evidence_reconstruction.json')
    for name in ('valuation_events_v1.json', 'valuation_manual_quotes_v1.json'):
        copy(ROOT / 'configs' / name, 'evidence/' + name)
    readme = '''# Repaired held valuations on the original WFO

The original full eligible PIT S&P 500 strategy and every forecast remain
unchanged. This bundle corrects realized valuation gaps only; it has no top-N
selector or ranker.

**60 of 62 missing stock/date outcomes recovered; unavailable portfolio days
fall from 38 to 2; complete months increase from 53/84 to 82/84.**

## Start here

- [Repair explanation, evidence and limits](../../docs/WFO_VALUATION_REPAIRS.md)
- [Four-page chart PDF](performance/performance_charts.pdf)
- [Self-contained HTML report](performance/performance_report.html): download and open locally; GitHub shows HTML source.
- [Detailed performance definitions](../../docs/WFO_VALUATION_REPAIR_PERFORMANCE.md)
- [Unchanged original output bundle](../original_wfo/README.md)

![Revised monthly returns and annual performance](performance/performance_overview.png)

## Remaining gaps and equity interpretation

December 2022 and August 2025 remain incomplete because the ABMD and WBA
mergers included non-tradeable contingent rights. Cash-only or maximum-right
marks are not substituted into primary returns. Their GMOM one-day portfolio
mark ranges are 0.380 and 5.893 basis points, with sign accounted for. These
are hypothetical one-day marks, not a liquidation or carried-rights ledger.

![Three complete equity and drawdown segments](performance/equity_drawdown_segments.png)

The three complete runs reset independently to 100. The longest spans 1,002
sessions, 2019-01-02 through 2022-12-21. Continuous full-period CAGR and maximum
drawdown are withheld across the remaining two gaps.

## Tables and verification

| Output | File |
| --- | --- |
| Exact corrections and two unresolved keys | [valuation_ledger.csv](results/valuation_ledger.csv) |
| Daily returns and unchanged intended exposures | [daily_paper_proxy_returns.csv](results/daily_paper_proxy_returns.csv) |
| Annual and aggregate moments | [paper_proxy_metrics.csv](results/paper_proxy_metrics.csv) |
| Extra risk and tail metrics | [extended_metrics.csv](performance/extended_metrics.csv) |
| Monthly compounded returns | [monthly_returns.csv](performance/monthly_returns.csv) |
| Six-strategy contingent-right sensitivity | [contingent_claim_sensitivity.csv](results/contingent_claim_sensitivity.csv) |
| Repair policy and limitations | [policy.json](results/policy.json) |
| Independently replayed portfolio sums | [valuation_report.json](verification/valuation_report.json) |
| Source-evidence reconstruction check | [evidence_reconstruction.json](verification/evidence_reconstruction.json) |
| Primary merger references and final terms | [valuation_events_v1.json](evidence/valuation_events_v1.json) |
| Three reviewed predecessor quote transcriptions | [valuation_manual_quotes_v1.json](evidence/valuation_manual_quotes_v1.json) |

GMOM complete-date Sharpe is 0.736, LinReg 0.753 and matched LongOnly 0.776,
before costs. They now use 1,758 dates, versus 1,722 originally; all previously
observed returns remain unchanged. The verifier replayed every strategy/date
sum and checked original-artifact hashes. All 69 structural tests pass.

This retrospective outcome correction does not rebuild feature histories,
refit networks or OLS, or change validation selection or the forecast cohort.
It does not certify executable trading, original historical identities,
complete training data, settlement cash timing, or subsequent rights payouts.
Large raw/source objects, feature matrices and stock/date forecast Parquets
are omitted. The export manifest lists every copied file's checksum.
'''
    (output / 'README.md').write_text(readme)
    manifest = {'status':'complete', 'purpose':'Compact review of retrospective held-valuation corrections',
        'source_result_manifest_sha256':file_hash(results / 'manifest.json'),
        'source_performance_manifest_sha256':file_hash(performance / 'manifest.json'),
        'export_script_sha256':file_hash(__file__), 'forecasts_changed':False, 'raw_data_included':False,
        'large_prediction_tables_included':False, 'copied_outputs':copied, 'readme_sha256':file_hash(output / 'README.md')}
    (output / 'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':'complete','copied_files':len(copied),'bytes':sum(x['size_bytes'] for x in copied),'output':str(output)},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT / 'outputs/valuation_repaired_wfo')
    p.add_argument('--performance',type=Path,default=ROOT / 'data/walk_forward/valuation_performance_v2')
    a=p.parse_args(); export(a.output.resolve(),a.performance.resolve())

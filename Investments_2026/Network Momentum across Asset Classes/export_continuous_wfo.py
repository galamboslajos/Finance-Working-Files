#!/usr/bin/env python3
"""Export the verified continuous accounting scenario for GitHub review."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from prepare_sp500_daily import file_hash, require

ROOT=Path(__file__).resolve().parent


def export(results,performance,verification,output):
    require(not output.exists(),'Continuous review export exists; use a fresh directory')
    result=json.loads((results/'manifest.json').read_text())
    chart=json.loads((performance/'manifest.json').read_text())
    proof=json.loads(verification.read_text())
    require(result['status']==chart['status']=='complete' and proof['status']=='passed','Verified complete scenario required')
    require(file_hash(results/'manifest.json')==chart['results_manifest_sha256']==proof['result_manifest_sha256'],'Mixed scenario versions')
    copied=[]

    def copy(source,relative):
        target=output/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        require(file_hash(source)==file_hash(target),'Export copy differs')
        copied.append({'path':relative,'source_relative_path':source.relative_to(ROOT).as_posix(),
            'sha256':file_hash(target),'size_bytes':target.stat().st_size})

    for directory,manifest,prefix in [(results,result,'results'),(performance,chart,'performance')]:
        for item in manifest['outputs']:
            source=directory/item['path'];require(file_hash(source)==item['sha256'],'Verified artifact changed')
            copy(source,prefix+'/'+item['path'])
        copy(directory/'manifest.json',prefix+'/manifest.json')
    copy(verification,'verification/report.json')
    copy(ROOT/'configs/contingent_rights_policy_v1.json','evidence/contingent_rights_policy_v1.json')
    readme='''# Continuous WFO equity and performance

**1,760 / 1,760 sessions, 84 / 84 months, one continuous equity path.**
The original forecasts and positions remain frozen. This is a **conservative
contingent-right accounting scenario**: two unavailable market valuations use
explicit assumptions. Cash is recognized; long rights are marked zero and
short rights reserved at their contractual payout cap. Subsequent rights
cash flows, financing, revaluations and reserve releases are excluded.

## Current charts

- [Two-page PDF](performance/performance_charts.pdf)
- [Self-contained HTML report](performance/performance_report.html): download and open locally; GitHub shows HTML source.
- [Definitions, results and limitations](../../docs/CONTINUOUS_WFO_PERFORMANCE.md)

![Continuous equity and drawdown](performance/equity_drawdown.png)

Equity starts at 100 once and compounds every session. Drawdown uses the
full running high-water mark. Dashed lines mark the two assumption dates;
they do not break the curves. Narrow shaded envelopes compare four possible
zero/face event marks, all sharing the exclusion of subsequent claim P&L.
They are not bounds on actual wealth with later rights cash flows.

![All 84 months, annual performance and rolling diagnostics](performance/performance_overview.png)

## Conditional full-period results

| Strategy | CAGR | Maximum drawdown | Sharpe |
| --- | ---: | ---: | ---: |
| GMOM | 5.65% | -16.05% | 0.733 |
| LinReg | 5.87% | -16.18% | 0.746 |
| LongOnly | 6.43% | -16.40% | 0.766 |

All figures are before trading costs and conditional on the rights policy.
LongOnly is the inverse-volatility-sized matched cohort, not an S&P 500 index.
GMOM CAGR across all four event-mark assumptions is 5.6523%–5.6618%.

The two assumptions affect ABMD on 2022-12-22 and WBA on 2025-08-28. GMOM's
portfolio-day returns are -0.03224% and -0.12456%, respectively. Neither
whole portfolio day is zero-filled, dropped or reweighted. All other 1,758
portfolio days and all original intended exposures stay unchanged.

## Tables and evidence

| Output | File |
| --- | --- |
| Complete daily portfolio returns with both assumption dates flagged | [daily_paper_proxy_returns.csv](results/daily_paper_proxy_returns.csv) |
| Continuous equity and full-history drawdown | [equity_drawdown.csv](results/equity_drawdown.csv) |
| All 84 monthly returns | [monthly_returns.csv](results/monthly_returns.csv) |
| Annual/aggregate return, CAGR, volatility, Sharpe, Sortino and Calmar | [paper_proxy_metrics.csv](results/paper_proxy_metrics.csv) |
| Rights and short-reserve register for all six strategies | [claim_register.csv](results/claim_register.csv) |
| Four alternative initial-mark combinations | [claim_mark_scenario_metrics.csv](results/claim_mark_scenario_metrics.csv) |
| Equity sensitivity envelope | [claim_mark_sensitivity_envelope.csv](results/claim_mark_sensitivity_envelope.csv) |
| Exact accounting policy and primary event sources | [policy.json](results/policy.json) |
| Independent holdings/curve/metrics verification | [report.json](verification/report.json) |

The verifier independently derives the 12 assumed strategy/event returns
from original holdings and cash/right terms, preserves every other observed
or recovered day, checks all 504 strategy/month returns and verifies one
uninterrupted path for each strategy. All 75 structural tests pass.

The [earlier valuation audit](../valuation_repaired_wfo/README.md) retains the
two unknown source marks. No raw stock label, feature, model or tradeable
price is fabricated here. No non-transferable right is assumed sold.
This scenario does not supply executable fills, complete later rights cash
flows, trading costs, borrow/financing or a rebuilt clean training dataset.
The large raw data and prediction tables are excluded from this review
export. Viewing the report requires no local raw data; reconstruction does.
'''
    (output/'README.md').write_text(readme)
    record={'status':'complete','purpose':'Continuous WFO accounting-scenario review, with two explicit rights assumptions',
        'accounting_scenario':True,'source_stock_labels_still_unobserved':2,
        'source_result_manifest_sha256':file_hash(results/'manifest.json'),
        'source_performance_manifest_sha256':file_hash(performance/'manifest.json'),
        'script_sha256':file_hash(__file__),'raw_data_included':False,'stock_date_predictions_included':False,
        'copied_outputs':copied,'readme_sha256':file_hash(output/'README.md')}
    write=output/'export_manifest.json';write.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'status':'complete','copied_files':len(copied),'bytes':sum(x['size_bytes'] for x in copied),'output':str(output)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'data/walk_forward/continuous_results_v1')
    p.add_argument('--performance',type=Path,default=ROOT/'data/walk_forward/continuous_performance_v2')
    p.add_argument('--verification',type=Path,default=ROOT/'data/walk_forward/continuous_verification_v1/report.json')
    p.add_argument('--output',type=Path,default=ROOT/'outputs/continuous_wfo')
    a=p.parse_args();export(a.results.resolve(),a.performance.resolve(),a.verification.resolve(),a.output.resolve())

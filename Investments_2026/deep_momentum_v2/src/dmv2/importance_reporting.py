"""Small saved-model importance aggregates and standalone scientific plots.

Consumes only derived feature-importance rows and metadata, never X, y or models.
"""
from collections import defaultdict
import math
from pathlib import Path


def need(ok, message):
    if not ok:
        raise ValueError(message)


def summarize_importance(rows, mapping):
    """Conserve each forest's total while separating values/missingness indicators."""
    names = [r['feature_name'] for r in mapping]
    need(names and len(names) == len(set(names)), 'Unique mapping names required')
    by_name = {r['feature_name']: r for r in mapping}
    by_year = defaultdict(dict)
    for row in rows:
        name, year = row['feature_name'], row['year']
        need(type(year) is int and name in by_name and name not in by_year[year], 'Unknown or duplicate importance key')
        need(type(row['retained']) is bool, 'Explicit retained flag required')
        for field in ('value_importance', 'missing_importance', 'total_importance'):
            value = row[field]
            if row['retained']:
                need(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid importance scalar')
            else:
                need(value is None, 'Unestimated feature cannot be represented as zero evidence')
        if row['retained']:
            need(abs(row['total_importance'] - math.fsum((row['value_importance'], row['missing_importance']))) <= 1e-14,
                 'Value/missingness importance does not conserve total')
        by_year[year][name] = row
    need(by_year, 'No importance rows')
    for year, part in by_year.items():
        need(set(part) == set(names), 'Annual original feature population changed')
        need(abs(math.fsum(r['total_importance'] for r in part.values() if r['retained']) - 1.) <= 1e-10,
             'Annual feature importance does not sum to one')
    groups = {}
    for entry in mapping:
        key = entry['group_id']
        metadata = {k: entry[k] for k in ('group_id', 'group_kind', 'group_label', 'sort_key')}
        if key in groups:
            need(groups[key]['metadata'] == metadata, 'Group metadata conflict')
        else:
            groups[key] = dict(metadata=metadata, names=[])
        groups[key]['names'].append(entry['feature_name'])
    summaries, annuals = [], []
    years = sorted(by_year)
    for group in sorted(groups.values(), key=lambda r: r['metadata']['sort_key']):
        saved = []
        for year in years:
            part = [by_year[year][name] for name in group['names']]
            observed = [r for r in part if r['retained']]
            item = {**group['metadata'], 'year': year, 'candidate_feature_count': len(part), 'retained_feature_count': len(observed)}
            for field in ('value_importance', 'missing_importance', 'total_importance'):
                item[field] = math.fsum(r[field] for r in observed) if observed else None
            item['importance_per_retained_feature'] = item['total_importance'] / len(observed) if observed else None
            annuals.append(item);saved.append(item)
        item = {**group['metadata'], 'candidate_feature_count': len(group['names']),
                'years': years, 'estimated_year_count': sum(r['retained_feature_count'] > 0 for r in saved),
                'retained_count_by_year': {str(r['year']): r['retained_feature_count'] for r in saved}}
        for field in ('value_importance', 'missing_importance', 'total_importance', 'importance_per_retained_feature'):
            values = [r[field] for r in saved if r[field] is not None]
            item['mean_' + field] = math.fsum(values) / len(values) if values else None
        totals = [r['total_importance'] for r in saved if r['total_importance'] is not None]
        item['minimum_annual_total'] = min(totals) if totals else None
        item['maximum_annual_total'] = max(totals) if totals else None
        summaries.append(item)
    for year in years:
        need(abs(math.fsum(r['total_importance'] for r in annuals if r['year'] == year and r['total_importance'] is not None) - 1) <= 1e-10,
             'Horizon grouping loses or duplicates importance')
    return dict(years=years, feature_count=len(names), group_count=len(groups), grouped=summaries, annual=annuals,
                year_missingness_share={str(year): math.fsum(r['missing_importance'] for r in part.values() if r['retained'])
                                       for year, part in by_year.items()},
                definition='Mean decrease in impurity; global forest normalization; equal annual weights; no predictions or returns attributed')


def _short_label(row):
    label = row['group_label'].replace(' sessions', 'd').replace(' session', 'd')
    label = label.replace('Abnormal volume ', 'Volume ratio: ').replace('Paper nominal ', 'Paper: ').replace(' months', 'm').replace(' month', 'm')
    return f"{label}   (n={row['candidate_feature_count']})"


def plot_importance(summary, output, *, title='Expanded RF: importance by feature lookback'):
    """Three aligned descriptive panels; dates are refit years, not future forecasts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, MaxNLocator
    import numpy as np
    groups = summary['grouped'];years = summary['years'];n = len(groups)
    need(years == [2020, 2021, 2022, 2023] and summary['feature_count'] == 715,
         'This saved endpoint figure requires the declared four-year 715-input scope')
    need(all(g['estimated_year_count'] == 4 for g in groups), 'Unestimated horizon needs a separately labelled view')
    y = np.arange(n)
    value = np.asarray([g['mean_value_importance'] for g in groups]) * 100
    missing = np.asarray([g['mean_missing_importance'] for g in groups]) * 100
    total = value + missing
    per_feature = np.asarray([g['mean_importance_per_retained_feature'] for g in groups]) * 100
    matrix = np.array([[next(a['total_importance'] for a in summary['annual'] if a['group_id'] == g['group_id'] and a['year'] == year)
                         for year in years] for g in groups]) * 100
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                         'axes.labelsize': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(17.6, max(10.8, n * .29 + 2)), sharey=True,
                             gridspec_kw={'width_ratios': [1.4, 1.03, 1.03], 'wspace': .20})
    fig.subplots_adjust(left=.275, right=.965, top=.835, bottom=.115)
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white');ax.set_ylim(n-.5, -.5)
        for name in ('top', 'right', 'left'): ax.spines[name].set_visible(False)
        ax.spines['bottom'].set_color('#c9ced6');ax.tick_params(axis='y', length=0)
    for i, g in enumerate(groups):
        if g['group_kind'] == 'paper_months':
            for ax in axes: ax.axhspan(i-.5, i+.5, color='#eaf3f6', zorder=-1)
        if i and g['sort_key'][0] != groups[i-1]['sort_key'][0]:
            for ax in axes: ax.axhline(i-.5, color='#c9ced6', lw=.8, zorder=5)
    left, heat, right = axes
    left.barh(y, value, height=.66, color='#23628a', label='Feature values', zorder=3)
    left.barh(y, missing, left=value, height=.66, color='#de9b3f', label='Missingness indicators', zorder=3)
    left.set_yticks(y, [_short_label(g) for g in groups], fontsize=9)
    maximum=max(total);left.set_xlim(0, maximum*1.27)
    for i, v in enumerate(total):left.text(v+maximum*.025, i, ('<0.01%' if 0 < v < .01 else f'{v:.2f}%'), va='center', fontsize=8.7, color='#1f2937')
    left.set_title('Total importance per lookback\nMean across four annual forests', loc='left', pad=12)
    left.set_xlabel('Share of total forest importance')
    left.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0));left.xaxis.set_major_locator(MaxNLocator(4))
    left.grid(axis='x', color='#e7e9ed', zorder=0)
    image=heat.imshow(matrix, aspect='auto', cmap='Blues', vmin=0, vmax=float(matrix.max()), origin='upper')
    heat.set_xticks(range(4), [str(v) for v in years]);heat.set_xlabel('Annual refit / forecast year')
    heat.set_title('Annual importance\nCell values = share of total (%)', loc='left', pad=12)
    for i in range(n):
        for j in range(4):heat.text(j, i, ('<.01' if 0 < matrix[i,j] < .005 else f'{matrix[i,j]:.2f}'), ha='center', va='center', fontsize=8,
                                    color='white' if matrix[i,j] > matrix.max()*.56 else '#17324d')
    right.scatter(per_feature, y, color='#23628a', s=27, zorder=4)
    right.hlines(y, 0, per_feature, color='#adc4d4', lw=1.2)
    right.set_xlim(0, max(per_feature)*1.22)
    right.set_title('Average per base feature\nTotal divided by feature count', loc='left', pad=12)
    right.set_xlabel('Mean share per feature')
    right.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=2));right.xaxis.set_major_locator(MaxNLocator(3))
    right.grid(axis='x', color='#e7e9ed', zorder=0)
    for i, v in enumerate(per_feature):right.text(v+max(per_feature)*.035, i, f'{v:.3f}%', va='center', fontsize=8.5, color='#1f2937')
    fig.suptitle(title, x=.035, y=.975, ha='left', fontsize=21, weight='bold', color='#142b40')
    fig.text(.035, .939, 'Endpoint-return classifier • 2020–2023 annual refits • 715 base features + 715 missingness indicators',
             fontsize=11.5, color='#46586b')
    handles,labels=left.get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.272,.918),frameon=False,ncol=2,fontsize=10)
    fig.text(.035, .064, 'Training split importance (MDI), not out-of-sample return contribution. All groups together sum to 100%; n = base feature count.',
             fontsize=10, color='#46586b')
    fig.text(.035, .039, 'Daily windows, skip windows, 252-session volume baselines and the paper’s nominal monthly horizons are kept distinct. Paper rows are shaded.',
             fontsize=10, color='#46586b')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ('.png','.pdf'):
        target=output.with_suffix(suffix)
        need(not target.exists(), 'Preserve existing figure')
        fig.savefig(target, dpi=180, facecolor='white')
    plt.close(fig)
    return [str(output.with_suffix(s)) for s in ('.png','.pdf')]


def _report_artifact(root, name, item):
    import hashlib
    root = Path(root).resolve()
    need(type(name) is str and Path(name).name == name and not name.startswith('.'), 'Closed report filename required')
    need(set(item) == {'size_bytes', 'sha256'}, 'Closed relative artifact pin required')
    path = (root / name).resolve()
    need(path.parent == root and path.is_file() and 0 <= item['size_bytes'] <= 8*1024**2
         and path.stat().st_size == item['size_bytes'], 'Bounded report artifact changed')
    need(hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Report artifact digest changed')
    return path


def read_importance_plot(directory):
    """Authenticate small saved artifacts; no model/panel deserialization or rendering."""
    import json
    import hashlib
    root = Path(directory).resolve()
    def small_json(path):
        need(path.stat().st_size <= 8*1024**2, 'Report JSON exceeds cap')
        def invalid(value):raise ValueError('Nonfinite report metadata: '+value)
        return json.loads(path.read_text(),parse_constant=invalid)
    receipt = small_json(root/'plot_receipt.json')
    need(receipt.get('schema') == 'dmv2-rf-importance-report/v1'
         and receipt.get('status') == 'COMPLETE_SAVED_MODEL_IMPORTANCE'
         and receipt.get('head') == 'endpoint' and receipt.get('years') == [2020,2021,2022,2023]
         and receipt.get('base_features') == 715 and receipt.get('model_loads') == 4
         and all(receipt.get(k) == 0 for k in ('new_estimator_fits','prediction_calls','source_panel_reads','backtest_replays')),
         'Exact saved-model importance scope required')
    for item in (receipt['source_model_plan_pin'],receipt['source_models_proof_pin']):
        path=Path(item['path']);need(path.stat().st_size <= 8*1024**2 and path.stat().st_size == item['size_bytes']
             and hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256'], 'Source model metadata pin changed')
    proof=small_json(Path(receipt['source_models_proof_pin']['path']))
    need(proof['status']=='PASS' and proof['phase']=='models' and proof['verified_estimator_fits']==20
         and proof['verified_shared_preprocessors']==4 and proof['model_plan_pin']==receipt['source_model_plan_pin']
         and proof['financial_2024_plus_values_read'] is False,
         'Original model verification proof changed')
    paths={name:_report_artifact(root,name,item) for name,item in receipt['artifacts'].items()}
    required={'analysis_read_plan.json','extraction_receipt.json','feature_importances.json','horizon_mapping.json',
              'horizon_summary.json','render_runtime.json','rf_importance_by_lookback.png','rf_importance_by_lookback.pdf'}
    need(set(paths)==required,'Exact derived report artifact set required')
    extracted=small_json(paths['extraction_receipt.json']);plan=small_json(paths['analysis_read_plan.json'])
    need(extracted['status']=='COMPLETE_MODEL_ONLY_IMPORTANCE' and extracted['model_loads']==4
         and extracted['feature_rows']==2860 and extracted['years']==receipt['years']
         and extracted['head']=='endpoint' and extracted['model_read_bytes']==receipt['model_read_bytes']
         and all(extracted.get(k)==0 for k in ('new_estimator_fits','prediction_calls','source_panel_reads','backtest_replays'))
         and plan['model_plan_pin']==receipt['source_model_plan_pin']
         and plan['independent_models_proof_pin']==receipt['source_models_proof_pin'], 'Copied extraction evidence changed')
    summary=small_json(paths['horizon_summary.json'])
    need(summary['years']==receipt['years'] and summary['feature_count']==715 and summary['group_count']==33,
         'Saved group scope changed')
    return dict(receipt=receipt,summary=summary,png=str(paths['rf_importance_by_lookback.png']),
                pdf=str(paths['rf_importance_by_lookback.pdf']),automatic_model_loading=False)

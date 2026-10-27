"""Passive composite reports; original withheld controls remain visible."""
from pathlib import Path
from .fixed_forecast_reporting import checked,metadata,need,read_fixed_forecast_reconciliation


def read_fixed_forecast_completed(directory,completion_directory):
    root=Path(completion_directory).resolve();ledger=metadata(root/'ledger.json')
    result={k:ledger.get(k) for k in ('status','phase','active_seconds','peak_worker_rss_bytes','error')}
    result.update(new_fits=0,maximum_fits=0,maximum_scenarios=28)
    if ledger['status']!='COMPLETE_VERIFIED':return result
    need(ledger['phase'] is None and ledger['fits_started']==ledger['fits_completed']==0
        and len(ledger['completed'])==2,'Zero-fit completion phase schedule')
    phases={}
    for row in ledger['completed']:
        path=checked(row['receipt']);need(path.parent==root/'phases','Unsafe completion phase path')
        phases[row['key']]=metadata(path)
    need(set(phases)=={'complete','verify_completion'} and all(x['status']=='COMPLETE_PHASE' for x in phases.values()),'Independent completion schedule')
    path=checked(phases['verify_completion']['result']);need(path==root/'verification/completion.json','Wrong completion proof')
    proof=metadata(path)
    need(proof['status']=='PASS' and proof['plan_pin']==ledger['plan_pin'] and proof['verified_replays']==6
         and proof['original_rows_byte_equivalent']==22 and proof['combined_scenarios']==28
         and proof['observed_points_added']==1 and proof['high_low_values_added'] is False
         and proof['new_fits']==proof['new_predictions']==0 and proof['learning_inputs_changed'] is False
         and proof['selected_baskets_changed'] is False and proof['financial_2024_plus_read'] is False,
         'Independent completion scope changed')
    plan=metadata(checked(ledger['plan_pin']));audit=metadata(checked(proof['audit_pin']))
    vp=metadata(checked(proof['verification_read_plan_pin']))
    parent=Path(directory).resolve()
    need(Path(plan['parent_directory']).resolve()==parent and checked(plan['parent_ledger_pin'])==parent/'ledger.json'
         and checked(proof['parent_proof_pin'])==parent/'verification/replay.json','Wrong parent reconciliation')
    need(audit['plan_pin']==ledger['plan_pin'] and audit['parent_proof_pin']==proof['parent_proof_pin']
         and audit['replayed_paths']==6 and audit['original_rows_reused']==22
         and audit['new_fits']==0 and audit['forecasts_changed'] is False and audit['selection_changed'] is False,
         'Completion audit scope changed')
    base=read_fixed_forecast_reconciliation(parent)
    need(base['status']=='COMPLETE_VERIFIED' and base['complete_paths']==22 and base['withheld_paths']==6,'Original coverage reference changed')
    import pandas as pd
    import pyarrow.parquet as pq
    def report(key,rows):
        item=audit['artifacts'][key];need(item==vp['artifacts'][key],'Report not independently bound')
        path=checked(item,maximum=2*1024**2);need(path.is_relative_to(root/'completion'),'Wrong composite report path')
        pf=pq.ParquetFile(path);need(pf.metadata.num_rows==rows,'Incomplete composite grid')
        return pf.read().to_pandas()
    comp=report('combined_comparison',28);replacement=report('replacement_comparison',6)
    cases={(a,s) for a in ('xgb_paper15_5y','rf_paper15_10y','xgb_paper15_10y')
                 for s in ('execution_close_gross','execution_close_open_units')}
    need(not comp.duplicated(['arm','stage']).any() and set(zip(replacement.arm,replacement.stage))==cases,'Wrong replacement grid')
    mask=[(r.arm,r.stage) in cases for r in comp.itertuples()]
    pd.testing.assert_frame_equal(comp.loc[mask].sort_values(['arm','stage']).reset_index(drop=True),
        replacement.sort_values(['arm','stage']).reset_index(drop=True),check_dtype=False,check_exact=True)
    original=base['comparison'];oldmask=[(r.arm,r.stage) not in cases for r in original.itertuples()]
    pd.testing.assert_frame_equal(comp.loc[[not x for x in mask]].sort_values(['arm','stage']).reset_index(drop=True),
        original.loc[oldmask].sort_values(['arm','stage']).reset_index(drop=True),check_dtype=False,check_exact=True)
    need(int(comp.complete_path.sum())==proof['complete_paths'] and 28-int(comp.complete_path.sum())==proof['withheld_paths'],'Composite coverage mismatch')
    result.update(base)
    result.update(status='COMPLETE_VERIFIED',comparison=comp,original_comparison=original,
        replacement_comparison=replacement,original_complete_paths=22,original_withheld_paths=6,
        complete_paths=proof['complete_paths'],withheld_paths=proof['withheld_paths'],
        completion_active_seconds=ledger['active_seconds'],completion_peak_rss_bytes=ledger['peak_worker_rss_bytes'],
        all_completed_controls_reconcile=proof['all_completed_controls_reconcile'],
        completion_reference_differences=pd.DataFrame(proof['reference_differences']),
        observed_points_added=1,source_dates=proof['source_dates'],source_natives=proof['source_natives'])
    # Curves remain the original independently verified 22 complete paths.
    # The six added diagnostic controls are reported through verified aggregate metrics.
    return result

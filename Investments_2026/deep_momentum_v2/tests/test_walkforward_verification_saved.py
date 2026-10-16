"""End-to-end saved 1006-session/64-scenario fixture, using synthetic quotes."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.backtest_loading import ARMS,_pin
from dmv2.catalog import canonical_sha256
from dmv2.daily_backtest import summarize_daily
from dmv2.walkforward_accounting import run_walkforward_accounting
from dmv2.walkforward_backtest import yearly_breakdown
from dmv2.walkforward_clock import walkforward_clock
from dmv2.walkforward_selection import select_walkforward_basket
from test_walkforward_preflight_runner import fixture,save
from test_walkforward_verification import verify


def saved_fixture(root):
    run,cfg,_,_=fixture(root)
    cfg['execution'].update(verification_maximum_peak_rss_bytes=8*1024**3,verification_maximum_wall_seconds=1800,verification_model_fits=0)
    save(root/'configs/walkforward_2020_2023.json',cfg)
    parent={'configuration':cfg}; identity=canonical_sha256(parent); save(run/'build_identity.json',parent)
    save(run/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity})
    sessions,windows,clock=walkforward_clock(); prices=[]; source_pins=[]
    for year in range(2020,2024):
        directory=run/'years'/str(year); path=directory/'daily_prices_raw.parquet'
        frame=pd.read_parquet(path)
        if year==2023:
            frame=frame.loc[~(frame.trading_date.eq('2023-12-29')&frame.intrinio_security_id.eq('A'))]
            frame.to_parquet(path,index=False)
        prices.append(frame)
        audit={'status':'COMPLETE','year':year,'build_identity_sha256':identity,
            'feature_provenance':{'requested_ids':['A','B','C','D']},'artifacts':{'daily_prices_raw':_pin(directory,path)}}
        save(directory/'audit.json',audit); source_pins.extend([_pin(root,path),_pin(root,directory/'audit.json')])
    prices=pd.concat(prices,ignore_index=True); backtest=run/'backtest'; artifacts={}; comparisons=[]; annuals=[]; monthlys=[]; result_cache={}
    def table(path,frame):
        path=backtest/path; path.parent.mkdir(parents=True,exist_ok=True); frame.to_parquet(path,index=False)
        artifacts[str(path.relative_to(backtest))]=_pin(backtest,path)
    for arm in ARMS:
        forecast_path=run/'models/arms'/arm/'predictions.parquet'; forecast=pd.read_parquet(forecast_path); source_pins.append(_pin(root,forecast_path))
        for basket in ['class_1_10','class_2_9']:
            selected=select_walkforward_basket(forecast,basket=basket,sessions=clock['selection_sessions'],configuration=cfg)
            for name,frame in [('original_ledger',selected.ledger),('selected',selected.selections),('rebalances',selected.rebalances),('monthly',selected.monthly)]:table(Path('selections')/arm/basket/(name+'.parquet'),frame)
            monthlys.append(selected.monthly.assign(arm=arm,basket=basket))
            for variant,stop in [('unstopped',None),('asset_stop_10pct',.1)]:
                for fee in [0,5,10,25]:
                    # All four estimator forecasts in this fixture coincide;
                    # save an independent complete path in every required slot.
                    if (basket,variant,fee) not in result_cache:
                        result_cache[(basket,variant,fee)]=run_walkforward_accounting(selected.selections,prices,sessions,rebalances=selected.rebalances,terminal_date='2023-12-29',
                            policy=DailyAccountingPolicy(cost_bps=fee,required_names_per_side=None,asset_stop_fraction=stop))
                    result=result_cache[(basket,variant,fee)]
                    summary,monthly=summarize_daily(result,arm=arm,basket=basket,variant=variant,cost_bps=fee,expected_sessions=1006)
                    comparisons.append(summary); annuals.append(yearly_breakdown(result,arm=arm,basket=basket,variant=variant,cost_bps=fee))
                    relative=Path('arms')/arm/basket/variant/f'cost_{fee:02d}bps'
                    for name,frame in result.tables.items():table(relative/(name+'.parquet'),frame)
                    table(relative/'monthly.parquet',monthly)
    old=root/cfg['historical_retirement_registry']['directory']; historical=verify.json_file(old/'build_identity.json')['events']
    source_pins.extend([_pin(root,old/'audit.json'),_pin(root,old/'build_identity.json')])
    phase={'parent_identity_sha256':identity,'configuration_sha256':canonical_sha256(cfg),'source_pins':source_pins,
        'historical_retirements':historical,'event_registry':{'schema':'dmv2-walkforward-actions/v1','events':[]}}
    save(backtest/'build_identity.json',phase); artifacts['build_identity.json']=_pin(backtest,backtest/'build_identity.json')
    table(Path('comparison.parquet'),pd.DataFrame(comparisons)); table(Path('yearly.parquet'),pd.concat(annuals,ignore_index=True));table(Path('selection_coverage.parquet'),pd.concat(monthlys,ignore_index=True))
    save(backtest/'audit.json',{'status':'COMPLETE','phase':'walkforward_continuous_backtest_v1','build_identity_sha256':identity,
        'phase_identity_sha256':canonical_sha256(phase),'artifacts':artifacts,'models_fitted':False,'preprocessing_fitted':False,
        'annual_liquidation':False,'financial_2024_plus_read':False,'sessions':1006,'outcome_months':48,'scenarios':64,
        'complete_proxy_paths':32,'failed_closed_paths':32})
    return run,backtest


@unittest.skipUnless(os.environ.get('DMV2_FULL_VERIFICATION_TEST')=='1','Explicit 64-path synthetic integration test')
class SavedWalkforwardVerificationTests(unittest.TestCase):
    def test_complete_and_withheld_grid_is_independently_audited_without_model_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); run,backtest=saved_fixture(root); output=run/'independent_audit.json'
            with redirect_stdout(StringIO()):result=verify.verify_saved_run(root,run,backtest,output=output)
            self.assertEqual(result['status'],'PASS');self.assertEqual(len(result['scenario_checks']),64)
            self.assertEqual(result['complete_proxy_paths'],32);self.assertEqual(result['withheld_paths'],32)
            self.assertFalse(result['withheld_economic_performance_certified'])
            self.assertEqual(result['root_audit_sha256'],verify.digest(run/'audit.json'))
            self.assertEqual(result['backtest_audit_sha256'],verify.digest(backtest/'audit.json'))
            self.assertTrue(output.exists());self.assertEqual(result['models_fitted'],0)
            with self.assertRaisesRegex(verify.VerificationError,'Preserve existing'):
                # Existing output refusal is verified without rerunning 64 paths.
                # The public entry point should check this before financial I/O.
                verify.verify_saved_run(root,run,backtest,output=output)

            original=(root/'configs/walkforward_2020_2023.json').read_bytes()
            cfg=verify.json_file(root/'configs/walkforward_2020_2023.json'); cfg['execution']['verification_model_fits']=1
            save(root/'configs/walkforward_2020_2023.json',cfg)
            with patch.object(verify,'_quote_inputs',side_effect=AssertionError('Financial input decoded after protocol drift')):
                with self.assertRaisesRegex(verify.VerificationError,'configuration changed'):
                    verify.verify_saved_run(root,run,backtest)
            (root/'configs/walkforward_2020_2023.json').write_bytes(original)
            historical=verify.json_file(run/'build_identity.json')['configuration']['historical_retirement_registry']
            path=root/historical['directory']/'build_identity.json'; path.write_bytes(path.read_bytes()+b'\n')
            with patch.object(verify,'_quote_inputs',side_effect=AssertionError('Financial input decoded after original drift')):
                with self.assertRaisesRegex(verify.VerificationError,'hash/size'):
                    verify.verify_saved_run(root,run,backtest)


if __name__=='__main__':unittest.main()

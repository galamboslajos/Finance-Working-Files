"""Freeze only the previously selected statistical settings for twenty refits."""
from pathlib import Path
import importlib.metadata
import json
import sys
from control import ROOT, check, pin, read, save_new, need
import model_worker


def main():
    r = Path('/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2')
    catalog_path = ROOT / 'input_catalog.json'
    catalog = read(catalog_path)
    need(catalog['status'] == 'VERIFIED_PROVISIONAL_FROZEN', 'Input freeze missing')
    need(read(check(catalog['independent_verification']))['status'] == 'PASS', 'Input verification missing')
    need(pin(ROOT / 'model_worker.py')['sha256'] == '1006a1ea658027f258bbc061bad0ad77ec443acfb93ff3c04c617704b05e0bfe',
         'Reviewed model worker changed')
    ref = r / 'outputs/rf_expanded_tuning/20261020T092321879709Z-792c5bb9b38e/artifacts'
    tuned = {str(year): pin(ref / f'{year}_outer_train__fit/fit_manifest.json') for year in (2020, 2021, 2022, 2023)}
    for year, item in tuned.items():
        model_worker.tuned_parameters(int(year), read(check(item)))
    runtime = {name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'pyarrow',
        'exchange-calendars', 'scikit-learn', 'xgboost', 'joblib', 'threadpoolctl')}
    source_protocol = read(ROOT / 'source_protocol.json')
    policy = {'schema': 'dmv2-corrected-refit-protocol/v1', 'status': 'SEALED_BEFORE_FINANCIAL_READ',
        'user_authority': 'ok, ready: refit existing annual models and rerun probability/RET/daily comparison with unchanged rules',
        'worker_pin': pin(ROOT / 'model_worker.py'),
        'helper_pins': {name: pin(r / 'src/dmv2' / name) for name in model_worker.HELPERS},
        'catalog_pin': pin(catalog_path), 'catalog_schema': catalog['schema'], 'catalog_status': catalog['status'],
        'input_verification_pin': catalog['independent_verification'],
        'terminal_guard_pin': pin(ROOT / 'run/terminal_guard_v2.json'),
        'baseline_recipe_pin': pin(r / 'configs/classifier_pilot.json'), 'tuned_fit_manifest_pins': tuned,
        'runtime': runtime, 'run_directory': str(ROOT / 'run/models'),
        'python_path': source_protocol['python_path'], 'outcome_years': [2020, 2021, 2022, 2023],
        'parameter_search': False, 'protected_2024_plus': True, 'limits': source_protocol['limits'],
        'compressed_source_maximum_bytes': 12 * 1024**3, 'derived_matrix_dispatch_maximum_bytes': 32 * 1024**3,
        'resource_guard_pin': pin(ROOT / 'supervise.py'), 'read_control_pin': pin(ROOT / 'control.py'),
        'pipeline_pin': pin(ROOT / 'run_model_pipeline.py'),
        'verification_worker_pin': pin(ROOT / 'verify_models.py'),
        'control_pin': pin(ROOT / 'control.py'), 'supervisor_pin': pin(ROOT / 'supervise.py'),
        'label_policy_pin': pin(ROOT / 'MONTHLY_ENDPOINT_LABEL_POLICY.md'),
        'old_runs_and_checkpoints_preserved': True, 'active_pointer_promoted': False,
        'backtest_policy': 'Existing RET and separate class1/10,2/9;up-to10balancednames,50/50,next-open,10%assetstop,cost0/5/10/25bp,noannualreset',
        'interpretation': 'Fixed old annual tuning winners, newly fitted corrected-input states; not a new hyperparameter search or fresh untouched test'}
    print(json.dumps(save_new(ROOT / 'model_protocol.json', policy)))


if __name__ == '__main__':
    main()

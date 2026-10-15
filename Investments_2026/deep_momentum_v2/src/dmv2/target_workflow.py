"""Bounded shared target/payoff checks on already loaded notebook samples."""
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

from .catalog import canonical_sha256
from .features import FeaturePanel
from .monthly import MonthlyReturnPanel
from .ret import RET_POLICY_DEFINITION, build_ret_payoffs
from .targets import TARGET_POLICY_DEFINITION, build_monthly_targets
from .universe import DataContractError


RET_DEFINITION = RET_POLICY_DEFINITION


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise DataContractError('Duplicate shared target protocol field: ' + key)
        value[key] = item
    return value


def validate_target_ret_protocol(value):
    """Keep the notebook contract equal to the implemented worker definitions."""
    if (not isinstance(value, dict) or type(value.get('schema_version')) is not int
            or value['schema_version'] != 1
            or value.get('protocol_id') != 'dmv2-shared-monthly-target-ret-v1'
            or canonical_sha256(value.get('targets')) != canonical_sha256(TARGET_POLICY_DEFINITION)
            or canonical_sha256(value.get('ret')) != canonical_sha256(RET_DEFINITION)):
        raise DataContractError('Shared target/RET protocol differs from its worker contracts')
    execution = value.get('execution', {})
    expected = {
        'bounded_existing_sample_enabled': True, 'maximum_sample_outcome_months': 3,
        'maximum_sample_original_rows': 3000, 'full_history_build_enabled': False,
        'model_fitting_enabled': False, 'preprocessing_fitting_enabled': False,
        'backtest_enabled': False, 'raw_downloads_enabled': False,
        'cloud_mutations_enabled': False,
    }
    if (execution != expected or any(type(execution[k]) is not type(v)
                                      for k, v in expected.items())):
        raise DataContractError('Notebook target checks must stay bounded and cannot fit or launch work')
    coverage = value.get('coverage', {})
    if ('target_coverage_fraction_gate' not in coverage
            or coverage.get('target_coverage_fraction_gate') is not None
            or coverage.get('context_gate_applies_to_targets') is not False):
        raise DataContractError('Context coverage cannot silently become a target completeness gate')
    chronology = value.get('model_chronology', {})
    if chronology != {
        'training': 'rolling_preceding_10_completed_outcome_years', 'refit': 'annual',
        'prediction': 'monthly', 'nominal_outer_test_years': [2020, 2021, 2022, 2023],
        'primary_universe': 'historical_sp500_sp400_union',
        'primary_context': 'available_cohort_80pct',
        'sensitivity_context': 'available_cohort_90pct', 'paper15_changed': False,
    }:
        raise DataContractError('Shared target workflow changed the selected cohort or chronology')
    if value.get('certification') != {
        'knowledge_time_verified': False, 'accounting_certified': False,
        'model_ready': False, 'survivorship_bias_free_certified': False,
    }:
        raise DataContractError('Engineering targets cannot certify source identity or economics')
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise DataContractError('Target protocol must be finite JSON') from error


def load_target_ret_protocol(root):
    path = Path(root) / 'configs/target_ret_protocol.json'
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_unique)
    except (OSError, json.JSONDecodeError) as error:
        raise DataContractError('Cannot read the shared target/RET protocol') from error
    return validate_target_ret_protocol(value)


@dataclass(frozen=True)
class SharedTargetRun:
    targets: object
    payoffs: object
    summary: dict


def run_shared_target_sample(feature_panel, monthly_returns, protocol):
    """Reuse only existing <=3-month samples; never load prices or full panels.

    The bounded two-ID engineering sample cannot supply representative ranks
    or a ten-year payoff history. Correctly withheld outputs are evidence that
    the guards worked, not successful model coverage.
    """
    protocol = validate_target_ret_protocol(protocol)
    if not isinstance(feature_panel, FeaturePanel) or not isinstance(monthly_returns, MonthlyReturnPanel):
        raise DataContractError('Shared sample requires existing feature and monthly-return panels')
    if monthly_returns.provenance.get('universe') != 'combined':
        raise DataContractError('Shared modelling targets require the selected historical SP500/SP400 union')
    limits = protocol['execution']
    if (not monthly_returns.windows
            or len(monthly_returns.windows) > limits['maximum_sample_outcome_months']
            or feature_panel.coverage.num_rows > limits['maximum_sample_original_rows']
            or monthly_returns.table.num_rows > limits['maximum_sample_original_rows']):
        raise DataContractError('Shared target sample exceeds the bounded existing-sample plan')
    keys = ['source_security_id', 'signal_date']
    feature_keys = feature_panel.table.select(keys).to_pylist()
    coverage_keys = feature_panel.coverage.select(keys).to_pylist()
    if (len(feature_keys) != len(coverage_keys)
            or {tuple(r[k] for k in keys) for r in feature_keys}
            != {tuple(r[k] for k in keys) for r in coverage_keys}):
        raise DataContractError('Feature coverage changed the original historical keys')
    cutoff = max(window.outcome_end for window in monthly_returns.windows)
    targets = build_monthly_targets(feature_panel.coverage, monthly_returns,
        label_observed_through=cutoff, policy=protocol['targets']['policy_id'])
    payoffs = build_ret_payoffs(targets, signal_date=cutoff, label_observed_through=cutoff)
    rows = targets.table.select(['prediction_eligible', 'target_available',
                                'observed_return_available']).to_pylist()
    rank_status = Counter(targets.diagnostics['rank_status'].to_pylist())
    return SharedTargetRun(targets, payoffs, {
        'target_ret_protocol_sha256': canonical_sha256(protocol),
        'original_rows_retained': len(rows),
        'prediction_eligible_rows': sum(r['prediction_eligible'] for r in rows),
        'observed_return_rows': sum(r['observed_return_available'] for r in rows),
        'class_available_rows': sum(r['target_available'] for r in rows),
        'rank_status_counts': dict(rank_status),
        'ret_vector_available': payoffs.provenance['vector_available'],
        'ret_available_month_class_cells': payoffs.provenance['available_month_class_cells'],
        'ret_expected_month_class_cells': 1200,
        'raw_source_reads_performed': False, 'full_history_build_performed': False,
        'preprocessing_fitted': False, 'model_fitted': False,
        'daily_data_modified': False, 'model_ready': False,
    })

"""Explicit provisional admission, separate from original identity evidence.

The user accepts identity uncertainty for exploration. This changes research
admission only; it never clears source flags, certifies historical knowledge,
selects survivors or resolves conflicting/missing routes by price continuity.
"""
from collections import Counter
import math

import pandas as pd

from .universe import DataContractError, iso_date


STRICT_IDENTITY_POLICY = 'strict_unique_unflagged'
PROVISIONAL_IDENTITY_POLICY = 'provisional_unique_resolved_link'
IDENTITY_WARNING = ('PROVISIONAL RESEARCH: historical security identities and '
    'survivorship are unverified; selection and look-ahead bias may affect '
    'results. Results require rerunning after identity repair.')


def validate_identity_policy(policy):
    if policy not in (STRICT_IDENTITY_POLICY, PROVISIONAL_IDENTITY_POLICY):
        raise DataContractError('Unknown identity eligibility policy')
    return policy


def identity_policy_provenance(policy):
    validate_identity_policy(policy)
    return {'identity_policy': policy, 'identity_policy_version': 1,
            'identity_status': 'UNVERIFIED', 'knowledge_time_verified': False,
            'temporal_identity_verified': False, 'survivorship_verified': False,
            'identity_uncertainty_accepted': policy == PROVISIONAL_IDENTITY_POLICY,
            'evidence_scope': 'PROVISIONAL_EXPLORATION', 'rerun_after_identity_repair': True,
            'warning': IDENTITY_WARNING}


def _identifier(value):
    return isinstance(value, str) and bool(value.strip())


def _missing(value):
    return value is None or value is pd.NA or (isinstance(value,float) and math.isnan(value))


def qualify_identity_records(records, *, identity_policy=STRICT_IDENTITY_POLICY):
    """Recompute admission from original metadata; claimed eligibility is ignored.

    Input is one row per source/date. All native routes, including blocked ones,
    count toward same-date collisions. Global/later aliases and prices never
    enter this decision. Strict reduced legacy fixtures remain supported when
    optional provider fields are absent; explicit provisional admission requires
    both native/unified IDs and affirmative published resolution metadata.
    """
    validate_identity_policy(identity_policy)
    if isinstance(records,(str,bytes)):
        raise DataContractError('Identity qualification requires explicit metadata records')
    output=[];seen=set();targets=Counter()
    for original in records:
        if not isinstance(original,dict):
            raise DataContractError('Identity records must be metadata dictionaries')
        row=dict(original)
        for field in ('intrinio_security_id','unified_security_id','identity_review_required','resolution_status'):
            if field in row and _missing(row[field]):
                row[field]=None
        sid=row.get('source_security_id')
        day=row.get('signal_date',row.get('as_of'))
        if not _identifier(sid):
            raise DataContractError('Identity eligibility requires a source security ID')
        if day is not None:
            iso_date(day,'identity cohort date')
            if day>='2024-01-01':
                raise DataContractError('Protected 2024+ identity cohorts are outside research admission')
        if (day,sid) in seen:
            raise DataContractError('Duplicate source/date identity record')
        seen.add((day,sid))
        status=row.get('identity_status')
        if not _identifier(status):
            raise DataContractError('Missing original identity status')
        if not _identifier(row.get('source_identity_status')):
            row['source_identity_status']=status
        pid=row.get('intrinio_security_id')
        if pid not in (None,'') and not _identifier(pid):
            raise DataContractError('Invalid native identity route')
        if status=='resolved' and not _identifier(pid):
            raise DataContractError('Resolved identity has no provider ID')
        if _identifier(pid):
            targets[(day,pid)]+=1
        if status=='resolved' and (
            ('identity_review_required' in row and row['identity_review_required'] is not False)
            or ('resolution_status' in row and row['resolution_status']!='resolved')):
            row['identity_status']='needs_review'
        output.append(row)
    for row in output:
        pid=row.get('intrinio_security_id');day=row.get('signal_date',row.get('as_of'))
        if _identifier(pid) and targets[(day,pid)]>1:
            row['identity_status']='target_collision'
        status=row['identity_status']
        admitted=False;reason='blocked_'+status
        if status=='resolved':
            admitted=True;reason='strict_unique_unflagged'
        if identity_policy==PROVISIONAL_IDENTITY_POLICY and status in ('resolved','needs_review'):
            strong_structure=(_identifier(pid) and _identifier(row.get('unified_security_id'))
                and (row.get('identity_review_required') is False or row.get('identity_review_required') is True)
                and row.get('resolution_status')=='resolved'
                and (status=='resolved' or row.get('identity_review_required') is True))
            admitted=bool(strong_structure)
            reason=('provisional_review_flag_accepted' if admitted and row['identity_review_required'] is True
                    else 'unique_unflagged_link' if admitted else 'blocked_missing_or_unresolved_link_metadata')
        row['identity_eligible']=admitted
        row['identity_eligibility_policy']=identity_policy
        row['identity_eligibility_reason']=reason
    return output


def select_engineering_ids(snapshot, sample_size, *, identity_policy=STRICT_IDENTITY_POLICY):
    """Select before prices; explicit provisional smoke includes a flagged route.

    This deterministic balanced smoke is not a model/universe selection rule.
    It exercises at most the declared sample size and never replaces a failed
    price history with a different ID after inspecting availability.
    """
    if type(sample_size) is not int or not 1<=sample_size<=10:
        raise DataContractError('Engineering identity sample size must be 1..10')
    rows=qualify_identity_records(snapshot.table.to_pylist(),identity_policy=identity_policy)
    eligible=[r for r in rows if r['identity_eligible']]
    if identity_policy==STRICT_IDENTITY_POLICY:
        return tuple(sorted({r['intrinio_security_id'] for r in eligible})[:sample_size])
    clean=sorted(r['intrinio_security_id'] for r in eligible if r['identity_review_required'] is False)
    flagged=sorted(r['intrinio_security_id'] for r in eligible if r['identity_review_required'] is True)
    selected=[]
    if clean:
        selected.append(clean[0])
    if flagged and len(selected)<sample_size:
        selected.append(flagged[0])
    selected.extend(v for v in sorted(set(clean+flagged)) if v not in selected)
    return tuple(selected[:sample_size])


def qualify_audit_ledger(table, *, identity_policy=STRICT_IDENTITY_POLICY):
    """Separate research-admission view of the non-promoting evidence audit."""
    columns=('source_security_id','signal_date','unified_security_id','intrinio_security_id',
             'original_identity_status','provider_review_required','link_resolution_status',
             'route_agrees_with_snapshot')
    if not hasattr(table,'column_names') or set(columns)-set(table.column_names):
        raise DataContractError('Research admission requires the original identity audit ledger')
    records=[]
    for r in table.select(columns).to_pylist():
        records.append({'source_security_id':r['source_security_id'],'signal_date':r['signal_date'],
            'unified_security_id':r['unified_security_id'],'intrinio_security_id':r['intrinio_security_id'],
            'identity_status':r['original_identity_status'] if r['route_agrees_with_snapshot'] is not False else 'audit_route_conflict',
            'source_identity_status':r['original_identity_status'],
            'identity_review_required':r['provider_review_required'],'resolution_status':r['link_resolution_status']})
    rows=qualify_identity_records(records,identity_policy=identity_policy)
    return rows,[{'signal_date':day,'original_members':len(group),
        'identity_eligible_members':sum(r['identity_eligible'] for r in group),
        'review_flag_accepted_members':sum(r['identity_eligibility_reason']=='provisional_review_flag_accepted' for r in group),
        'blocked_members':sum(not r['identity_eligible'] for r in group)}
        for day in sorted({r['signal_date'] for r in rows})
        for group in ([r for r in rows if r['signal_date']==day],)]

"""Pure metadata-only triage. No result authorizes a route change or price fill."""


def triage(*, requested, source_dates, cache_dates, same_pinned_source, complete_metadata_scope):
    if type(requested) is not bool or type(same_pinned_source) is not bool or type(complete_metadata_scope) is not bool:
        raise ValueError('Explicit Boolean evidence flags required')
    if source_dates is None or cache_dates is None:
        return {'state':'insufficient_metadata','mapping_change_authorized':False}
    source,cache=set(source_dates),set(cache_dates)
    if any(not isinstance(x,str) or len(x)!=10 for x in source|cache): raise ValueError('Canonical date-key metadata required')
    if not same_pinned_source:
        state='release_or_lineage_comparison_unresolved'
    elif not requested:
        state='request_or_admission_omission_candidate'
    elif cache-source:
        state='cache_contains_keys_not_in_compared_source_requires_lineage_review'
    elif source-cache:
        state='cache_or_loader_loss_candidate'
    elif source:
        state='matching_source_cache_key_inventory_check_endpoint_quality_and_feature_window_next'
    elif complete_metadata_scope:
        state='no_exact_native_keys_in_this_pinned_source_window'
    else:
        state='partial_metadata_empty_result_not_source_absence'
    return {'state':state,'source_dates':len(source),'cache_dates':len(cache),
        'source_keys_missing_from_cache':sorted(source-cache),'cache_keys_missing_from_source':sorted(cache-source),
        'historical_identity_verified':False,'financial_value_validity_verified':False,
        'dataset_absent_claimed':False,'mapping_change_authorized':False}

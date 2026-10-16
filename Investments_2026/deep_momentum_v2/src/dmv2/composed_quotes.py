"""Compose immutable prior and independent quote overlays in a fixed order."""
from dataclasses import dataclass

from .catalog import canonical_sha256
from .universe import DataContractError


@dataclass(frozen=True)
class ComposedQuoteAmendments:
    parent: object
    independent: object
    fingerprint: str

    @property
    def provenance(self):
        return {'policy_id': 'prior12_then_global_independent_review_v1',
            'fingerprint': self.fingerprint, 'parent': self.parent.provenance,
            'independent': self.independent.provenance,
            'parent_first': True, 'no_overlapping_target_keys': True,
            'original_source_files_changed': False, 'financial_2020_plus_read': False}

    def verify(self, root):
        self.parent.verify(root)
        self.independent.verify(root)
        expected = _fingerprint(self.parent, self.independent)
        if expected != self.fingerprint:
            raise DataContractError('Composed quote bundle identity changed')

    def apply(self, table, *, start=None, end=None):
        if self.fingerprint != _fingerprint(self.parent, self.independent):
            raise DataContractError('Composed quote bundle changed before application')
        original = self.parent.apply(table, start=start, end=end)
        return self.independent.apply(original, start=start, end=end)

    def apply_table(self, table, *, start=None, end=None):
        return self.apply(table, start=start, end=end)


def _fingerprint(parent, independent):
    def keys(bundle):
        return {(row['intrinio_security_id'], row['trading_date'])
            for row in bundle.amendments.select(['intrinio_security_id', 'trading_date']).to_pylist()}
    if keys(parent) & keys(independent):
        raise DataContractError('Independent amendments overlap a frozen parent target')
    return canonical_sha256({'policy_id': 'prior12_then_global_independent_review_v1',
        'parent': parent.fingerprint, 'independent': independent.fingerprint,
        'application_order': ['parent', 'independent']})


def compose_quote_amendments(parent, independent):
    return ComposedQuoteAmendments(parent, independent, _fingerprint(parent, independent))

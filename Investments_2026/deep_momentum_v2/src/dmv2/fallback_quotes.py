"""Compile reviewed actual-price amendments once for the new restricted run.

Frozen source files and earlier recipes stay unchanged. The same finite keyed
bundle applies to feature history, outcomes and execution, retaining all source
rows and unchanged volume columns. Only separately reviewed missing bars append.
No cloud requests, synthetic OHLC or post-2019 financial values are used.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .catalog import canonical_sha256
from .ohlc_repairs import ORIGINAL_FIELDS, original_bar_sha256
from .universe import DataContractError

POLICY_ID = "dmv2-fallback-all-reviewed-actual-price-amendments-20261016-v1"
REPLACEMENT_FIELDS = ("open", "high", "low", "close", "adj_open", "adj_high", "adj_low", "adj_close",
                      "price_quality_status", "raw_ohlc_bounds_valid", "adjusted_ohlc_bounds_valid")
KEYS = ("intrinio_security_id", "trading_date")
AMENDMENT_SCHEMA = pa.schema([
    ("intrinio_security_id", pa.string()), ("trading_date", pa.string()),
    ("kind", pa.string()), ("original_bar_sha256", pa.string()),
    ("replacement_bar_sha256", pa.string()), ("fields_json", pa.string()),
    ("policy_id", pa.string()),
])


def _day(value):
    if isinstance(value, date):
        value = value.isoformat()
    if not isinstance(value, str):
        raise DataContractError("Explicit ISO date required")
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise DataContractError("Invalid amendment date") from exc
    return value


def _safe(value):
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _snapshot(amendments, source_pins, sessions):
    return {"policy_id": POLICY_ID, "amendments": amendments.to_pylist(),
            "source_pins": list(source_pins), "sessions": list(sessions)}


@dataclass(frozen=True)
class FallbackQuoteAmendments:
    amendments: pa.Table
    ledger: pd.DataFrame
    source_pins: tuple
    sessions: tuple
    fingerprint: str

    @property
    def provenance(self):
        kinds = self.amendments["kind"].to_pylist()
        return {"schema": "dmv2-compiled-reviewed-actual-quote-amendments/v1",
                "policy_id": POLICY_ID, "fingerprint": self.fingerprint,
                "amendment_count": len(kinds), "replacement_count": kinds.count("replace"),
                "append_count": kinds.count("append"), "source_pins": list(self.source_pins),
                "original_source_files_changed": False, "replacement_volume_changed": False,
                "source_rows_dropped": False, "fabricated_raw_prices": False,
                "financial_2020_plus_read": False, "knowledge_time_certified": False,
                "adjustment_unit_transport_provisional": True}

    def verify(self, root):
        from .backtest_loading import _verify_pin
        if canonical_sha256(_snapshot(self.amendments, self.source_pins, self.sessions)) != self.fingerprint:
            raise DataContractError("Compiled quote amendments changed")
        for pin in self.source_pins:
            _verify_pin(Path(root).resolve(), pin)

    def apply(self, raw_table: pa.Table, *, start=None, end=None) -> pa.Table:
        """Apply within explicit requested dates and IDs already in the request.

        Metadata scope checks precede every financial conversion. Original row
        order/schema survive. Replacement volume is untouched; exact reviewed
        append rows retain their previously declared missing volume/action data.
        """
        if not isinstance(raw_table, pa.Table) or not set(KEYS) <= set(raw_table.column_names):
            raise DataContractError("Raw Arrow table requires native/date keys")
        if canonical_sha256(_snapshot(self.amendments, self.source_pins, self.sessions)) != self.fingerprint:
            raise DataContractError("Compiled quote amendment identity changed")
        days = [_day(day) for day in raw_table["trading_date"].to_pylist()]
        if any(day > "2019-12-31" for day in days):
            raise DataContractError("Fallback amendments cannot decode 2020+ financial rows")
        lower = _day(start) if start is not None else min(days, default="2018-01-01")
        upper = _day(end) if end is not None else max(days, default="2019-12-31")
        if lower > upper or upper > "2019-12-31":
            raise DataContractError("Amendment request leaves authorized dates")
        if any(not lower <= day <= upper for day in days):
            raise DataContractError("Input rows exceed declared amendment bounds")
        ids = raw_table["intrinio_security_id"].to_pylist()
        if any(not isinstance(native, str) or not native for native in ids):
            raise DataContractError("Invalid native identity in amendment input")
        native_ids = set(ids)
        keys = list(zip(ids, days, strict=True))
        if len(set(keys)) != len(keys):
            raise DataContractError("Duplicate source keys in amendment input")
        index = {key: position for position, key in enumerate(keys)}
        replacements, appends = {}, []
        for item in self.amendments.to_pylist():
            day, native = item["trading_date"], item["intrinio_security_id"]
            if not lower <= day <= upper or native not in native_ids:
                continue
            if day not in self.sessions:
                raise DataContractError("Amendment is not an independent exchange session")
            values = json.loads(item["fields_json"])
            key = native, day
            if item["kind"] == "replace":
                if key not in index:
                    raise DataContractError("Pinned original replacement key is absent")
                if not set(ORIGINAL_FIELDS) <= set(raw_table.column_names):
                    raise DataContractError("Original full OHLC/quality fields required for amendment verification")
                original = raw_table.slice(index[key], 1).select(list(ORIGINAL_FIELDS)).to_pylist()[0]
                actual = original_bar_sha256(original)
                if actual == item["replacement_bar_sha256"]:
                    continue
                if actual != item["original_bar_sha256"]:
                    raise DataContractError("Original source bar differs from compiled amendment pin")
                if set(values) - set(REPLACEMENT_FIELDS):
                    raise DataContractError("Replacement attempts to alter unapproved fields")
                for field, value in values.items():
                    if field in raw_table.column_names:
                        replacements.setdefault(field, {})[index[key]] = value
            elif item["kind"] == "append":
                if key in index:
                    raise DataContractError("Append-only amendment collides with an existing source key")
                row = {field.name: values.get(field.name) for field in raw_table.schema}
                for field in raw_table.schema:
                    if field.name == "trading_date" and pa.types.is_date(field.type):
                        row[field.name] = date.fromisoformat(day)
                appends.append(row)
            else:
                raise DataContractError("Unknown compiled amendment operation")
        output = raw_table
        for field, changed in replacements.items():
            mask = pa.array([position in changed for position in range(len(output))], type=pa.bool_())
            values = pa.array([changed[position] for position in sorted(changed)], type=output.schema.field(field).type)
            updated = pc.replace_with_mask(output[field].combine_chunks(), mask, values)
            output = output.set_column(output.schema.get_field_index(field), output.schema.field(field), updated)
        if appends:
            output = pa.concat_tables([output, pa.Table.from_pylist(appends, schema=raw_table.schema)])
        if len(output) != len(raw_table) + len(appends) or output.schema != raw_table.schema:
            raise DataContractError("Quote amendments changed original row count or schema")
        return output

    def apply_table(self, raw_table, *, start=None, end=None):
        return self.apply(raw_table, start=start, end=end)


def build_quote_amendments(original, corrected, *, ledger, source_pins=(), sessions=()):
    """Pure compiler seam; source admission happens in the reviewed adapters."""
    if (not isinstance(original, pd.DataFrame) or not isinstance(corrected, pd.DataFrame)
            or not set((*KEYS, *ORIGINAL_FIELDS)) <= set(original.columns)
            or not set(original.columns) <= set(corrected.columns)):
        raise DataContractError("Complete original/corrected quote frames required")
    for frame in (original, corrected):
        if frame.duplicated(list(KEYS)).any():
            raise DataContractError("Duplicate amendment source keys")
        if not frame.trading_date.map(lambda day: "2018-01-01" <= _day(day) <= "2019-12-31").all():
            raise DataContractError("Compiled amendments restricted to 2018–2019")
    if len(corrected) < len(original):
        raise DataContractError("Quote amendment cannot drop original rows")
    before = original.set_index(list(KEYS), drop=False)
    after = corrected.set_index(list(KEYS), drop=False)
    if not set(before.index) <= set(after.index):
        raise DataContractError("Quote amendment dropped an original source key")
    records = []
    for key in sorted(after.index):
        row = after.loc[key]
        if key in before.index:
            old = before.loc[key]
            for field in ("volume", "adj_volume", "factor", "split_ratio", "dividend"):
                if field in original and _safe(old[field]) != _safe(row[field]):
                    raise DataContractError("Replacement changed original volume/action fields")
            old_hash, new_hash = original_bar_sha256(old), original_bar_sha256(row)
            if old_hash == new_hash:
                continue
            kind = "replace"
            fields = {name: _safe(row[name]) for name in REPLACEMENT_FIELDS}
        else:
            old_hash, new_hash, kind = None, original_bar_sha256(row), "append"
            fields = {name: _safe(row[name]) for name in original.columns}
        records.append(dict(zip(KEYS, key), kind=kind, original_bar_sha256=old_hash,
                            replacement_bar_sha256=new_hash,
                            fields_json=json.dumps(fields, sort_keys=True, allow_nan=False), policy_id=POLICY_ID))
    if len(records) > 128:
        raise DataContractError("Reviewed quote amendment count exceeds finite bound")
    amendments = pa.Table.from_pylist(records, schema=AMENDMENT_SCHEMA)
    sessions = tuple(sorted(set(_day(day) for day in sessions)))
    pins = tuple(source_pins)
    fingerprint = canonical_sha256(_snapshot(amendments, pins, sessions))
    return FallbackQuoteAmendments(amendments, ledger.copy(deep=True), pins, sessions, fingerprint)


def compile_fallback_quote_amendments(root, *, evidence_dir=None):
    """Reuse all reviewed admission workers, then compile their accepted deltas."""
    from .action_backtest import plan_action_backtest, _prepare_prices
    from .action_loading import load_action_inputs
    from .backtest_loading import _json, _pin, _verify_pin
    from .bracket_ohlc_repairs import apply_bracketed_sip_ohlc_repairs
    from .supplemental_bars import admit_reviewed_missing_bars
    root = Path(root).resolve()
    plan = plan_action_backtest(root, evidence_dir=evidence_dir)
    bundle = load_action_inputs(root, plan=plan["input_plan"], unquoted_successor_policy="withhold_affected_paths")
    original = bundle.daily_prices
    prices, old_ledger, review = _prepare_prices(bundle, plan["reviewed_manifest"], plan["configuration"],
        plan["evidence_dir"], additional_manifests=plan.get("additional_reviewed_manifests", ()))
    source = plan["reviewed_manifest"]
    supplemental = admit_reviewed_missing_bars(prices, evidence_root=plan["evidence_dir"],
        projection_evidence=source["projection_evidence"], source_provenance=source["source_provenance"],
        native_mapping=source["native_mapping"], review_notes=source["review_notes"])
    bracket_root = root / "data/derived/data_repairs/2018-2019-resolution-20261016-v1/ohlc"
    bracket_path = bracket_root / "proposed_bracketed_ohlc_inputs.json"
    bracket = apply_bracketed_sip_ohlc_repairs(supplemental.prices, evidence_root=bracket_root,
                                             manifest=_json(bracket_path))
    pins = {pin["path"]: pin for pin in plan["identity"]["source_pins"]}
    for path in [bracket_path, *[bracket_root / pin["path"] for pin in bracket.evidence_pins]]:
        pin = _pin(root, path)
        pins[pin["path"]] = pin
    for pin in pins.values():
        _verify_pin(root, pin)
    ledgers = [old_ledger.assign(amendment_stage="old_reviewed_overlay"),
               supplemental.ledger.assign(amendment_stage="reviewed_append"),
               bracket.ledger.assign(amendment_stage="new_three_full_bar_overlay")]
    return build_quote_amendments(original, bracket.prices,
        ledger=pd.concat(ledgers, ignore_index=True),
        source_pins=tuple(pins[key] for key in sorted(pins)), sessions=bundle.sessions)

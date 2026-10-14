"""Reviewed source adapters; the source-ID cohort precedes all price selection."""
from dataclasses import dataclass, replace

from .loading import BoundedTableReader, ReadBudget
from .universe import DataContractError, build_snapshot, iso_date


MEMBERSHIP_COLUMNS = (
    "index_id", "trading_date", "security_id", "membership_ticker", "is_member",
    "membership_effective_date", "membership_end_date",
    "membership_availability_date", "identity_confidence", "pit_policy",
)
BRIDGE_COLUMNS = (
    "source_product_id", "source_security_id", "unified_security_id",
    "bridge_method", "bridge_confidence",
)
LINK_COLUMNS = (
    "unified_security_id", "intrinio_security_id", "identity_review_required",
    "resolution_status",
)
CALENDAR_COLUMNS = ("trading_date", "calendar_name", "is_trading_session")


@dataclass(frozen=True)
class LoadedUniverse:
    snapshot: object
    loading_provenance: dict
    loading_stats: dict


def session_dates(calendar):
    """Return sorted, unique source sessions without filling missing dates."""
    for name in CALENDAR_COLUMNS:
        if name not in calendar.column_names:
            raise DataContractError(f"Calendar missing column: {name}")
    names, dates = set(), []
    for row in calendar.select(CALENDAR_COLUMNS).to_pylist():
        day = iso_date(row["trading_date"], "calendar trading_date")
        names.add(row["calendar_name"])
        if row["is_trading_session"] is True:
            dates.append(day)
        elif row["is_trading_session"] is not False:
            raise DataContractError("Calendar contains unknown session state")
    if len(names) != 1 or any(not isinstance(name, str) or not name.strip() for name in names):
        raise DataContractError("Calendar must have one explicit calendar_name")
    if len(set(dates)) != len(dates):
        raise DataContractError("Duplicate source trading sessions")
    return tuple(sorted(dates))


def month_end_sessions(calendar, *, start, end):
    """Choose last listed sessions, then filter bounds; never invent calendar days.

    Obvious missing months and partial final source months fail. Source-calendar
    completeness within a month is not independently established by this helper.
    """
    iso_date(start, "start")
    iso_date(end, "end")
    if start > end:
        raise DataContractError("start is later than end")
    sessions = session_dates(calendar)
    last = {}
    for day in sessions:
        last[day[:7]] = day
    month = start[:7]
    while month <= end[:7]:
        if month not in last:
            raise DataContractError(f"Calendar missing requested month: {month}")
        year, number = map(int, month.split("-"))
        month = f"{year + (number == 12):04d}-{1 if number == 12 else number + 1:02d}"
    # Partial boundary months cannot be certified from a finite calendar alone.
    # Require the supplied calendar to extend into the next month for each result.
    result = []
    for month, day in sorted(last.items()):
        if start <= day <= end:
            if not any(other[:7] > month for other in sessions):
                raise DataContractError("Calendar ends within the requested signal month")
            result.append(day)
    return tuple(result)


def load_universe(catalog, *, as_of, universe, mode, budget=None):
    """Load one declared historical snapshot; never use quotes to select members."""
    iso_date(as_of, "as_of")
    if mode == "knowledge_time":
        # Fail before touching sources. Native contracts lack the required evidence.
        from .universe import KnowledgeTimeUnavailable
        raise KnowledgeTimeUnavailable(
            "The pinned sources cannot certify historical knowledge-time PIT. "
            "Choose historical_reconstruction explicitly for a disclosed reconstruction.")
    if universe not in ("sp500", "sp400", "combined"):
        raise DataContractError("universe must be sp500, sp400 or combined")
    if mode != "historical_reconstruction":
        raise DataContractError("Unknown PIT mode")
    reader = BoundedTableReader(catalog, budget or ReadBudget())
    calendar = reader.read_metadata(
        "siblis", "trading_calendar_sessions", columns=CALENDAR_COLUMNS,
        allowed_columns=CALENDAR_COLUMNS, max_rows=20_000,
        key_columns=("trading_date",))
    if as_of not in session_dates(calendar.table):
        raise DataContractError("as_of is not a session in the pinned Siblis calendar")
    indices = ("sp500", "sp400") if universe == "combined" else (universe,)
    members = reader.read(
        "siblis", "membership_by_index_by_date", start=as_of, end=as_of,
        columns=MEMBERSHIP_COLUMNS, allowed_columns=MEMBERSHIP_COLUMNS,
        key_columns=("index_id", "security_id", "trading_date"),
        filters={"index_id": indices})
    if members.table.num_rows == 0:
        raise DataContractError("No membership observations for the requested session/cohort")
    bridge = reader.read_metadata(
        "identity", "security_identity_bridge", columns=BRIDGE_COLUMNS,
        allowed_columns=BRIDGE_COLUMNS, max_rows=30_000,
        key_columns=("source_product_id", "source_security_id"))
    links = reader.read_metadata(
        "intrinio", "intrinio_identity_links", columns=LINK_COLUMNS,
        allowed_columns=LINK_COLUMNS, max_rows=20_000,
        key_columns=("unified_security_id",))
    snapshot = build_snapshot(members.table, bridge.table, links.table,
                              as_of=as_of, universe=universe, mode=mode)
    if not snapshot.table.num_rows:
        raise DataContractError("All observed membership rows failed date/availability gates")
    provenance = {"calendar": calendar.provenance, "membership": members.provenance,
                  "bridge": bridge.provenance, "links": links.provenance}
    snapshot = replace(snapshot, provenance={**snapshot.provenance,
                                            "source_pins": catalog.provenance()})
    return LoadedUniverse(snapshot, provenance,
                          {"calendar": calendar.stats, "membership": members.stats,
                           "bridge": bridge.stats, "links": links.stats})

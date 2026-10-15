# Historical identity evidence audit

VERIFIED LOCAL, 2026-10-14. Implemented in
[`identity_audit.py`](../src/dmv2/identity_audit.py), called by the
[`master notebook`](../notebooks/00_research_master.ipynb). This is an original
V2 metadata diagnostic. It does not approve mappings or change the existing
universe, feature, normalization or model execution rules.

The later user-selected [provisional admission policy](PROVISIONAL_IDENTITY_POLICY.md)
permits uniquely resolved flagged routes in research without changing this
original evidence ledger or certifying identities. Its separate admission
ledger distinguishes the accepted uncertainty from a completed mapping repair.

## Scope and retained populations

Three independent XNYS month-end anchors within the first 2000–2009 training
window are checked. Each cohort is formed from Siblis historical SP500/SP400
membership **before** inspecting identity or price-inventory evidence. The
union collapses only identical Siblis source IDs across sleeves. Original
source IDs, membership denominators and existing identity states are preserved.

| Signal | Original union | Existing resolved routes | Review flagged | Same-date collisions | Missing links |
|---|---:|---:|---:|---:|---:|
| 2000-01-31 | 900 | 302 | 529 | 32 | 37 |
| 2005-01-31 | 899 | 192 | 668 | 28 | 11 |
| 2009-01-30 | 902 | 142 | 736 | 16 | 8 |

These mutually exclusive existing states sum to each original population.
`resolved` means a unique declared route without the producer review flag; it
does not mean historically certified identity. A flagged link is a review task,
not proof of a wrong security or absent prices. Collision rows may themselves
carry a producer review flag; do not add flag counts to the exclusive states.
All **2,701 constituent/date rows** remain in both the evidence ledger and the
review queue, including currently resolved routes that still lack dated proof.

## Evidence classes and exact interpretation

1. `source_isin_status` validates the literal ISIN attached to the dated source
   membership: uppercase two-letter prefix, nine alphanumeric characters, one
   numeric check digit, then expanded Luhn. Missing, conflicting and invalid
   source values remain explicit. This does not independently verify an issuer,
   share class or identifier assignment interval.
2. `source_alias_isin_agreement` compares that source literal with the published
   link alias list. It is **platform/source-alias agreement**, not independent
   provider-returned ISIN confirmation. In the pinned product, all 14,297 link
   ISIN sets exactly equal the upstream bridge ISIN set for their unified ID,
   including empty sets. No additional ISIN values were observed. The generic
   variable dictionary supplies no alias derivation.
3. `exact_provider_isin_query` requires a checksum-valid source ISIN, the same
   literal in `query_identifier`, declared query type `isin`, a nonblank native
   target, `resolution_status=resolved` and HTTP 200. The name denotes a
   **declared successful literal query**; the original HTTP request/response
   bytes were not verified. A type label alone is insufficient. This can guide
   review of retrieval-time resolution, not certify historical validity.
4. `source_ticker_alias_agreement` is a separate weaker source-alias diagnostic.
   Ticker matching does not prove historical ticker ownership or share class.
5. `dated_identifier_support` requires a matching valid source ISIN and both
   strictly ISO date boundaries covering the signal under conservative
   half-open `[valid_from, valid_to)` arithmetic. Blank ends are not infinite
   ends; malformed/reversed intervals are counted. This arithmetic is a V2
   diagnostic convention, **not a demonstrated source interval contract**.
   It differs from the separately audited inclusive membership end contract.
   Even a complete source interval would not date the provider link or certify
   when that link became knowable. Every one of the 24,616 current history rows
   has missing validity boundaries, so support is zero at all three anchors.
6. `same_date_provider_collision` is recomputed across **all original members
   at that signal**, including flagged links. `global_provider_alias_count`
   reports release-wide unified aliases separately. Lifetime/future aliases
   never become historical eligibility criteria.
7. `acquisition_reported_positive_rows` means the linked native ID has at least
   one complete HTTP-200 acquisition record reporting positive whole-history
   row counts. It does not verify exact-date quotes, pre-signal warmup, survival,
   identity or terminal economics. All 9,266 acquisition requested identifiers
   are native Intrinio IDs; none matches an original link ticker/ISIN query.
   `exact_query_acquisition_record_matched` preserves that separate distinction.
   Successful zero-row acquisition records remain empty, not recorded failures.

The current [Intrinio Lookup Security documentation](https://docs.intrinio.com/documentation/web_api/get_security_by_id_v2)
accepts ISIN as an **input** but lists native IDs, ticker and FIGI fields in
its output, without an ISIN field. The local security master likewise has no
ISIN column. Current documentation does not reconstruct a July/August 2026
response. Provider FIGIs exist locally, but Siblis has no independently dated
FIGI identifiers for a source-side comparison. Matching fields from the same
capture or upstream aliases must not be counted as independent observations.

## Actual diagnostic counts

| Signal | Valid dated-source ISINs | Matching declared ISIN queries | Source-alias ISIN agreements | Positive native-ID acquisition inventory | Dated identity support |
|---|---:|---:|---:|---:|---:|
| 2000-01-31 | 315 | 268 | 304 | 769 | 0 |
| 2005-01-31 | 194 | 181 | 194 | 850 | 0 |
| 2009-01-30 | 147 | 142 | 147 | 890 | 0 |

Query totals include collided rows: respectively 2, 4 and 2. The priority queue
places those conflicts first, so the non-conflicting matching-query review
groups are **266, 177 and 140**. These are evidence groups, not approved repairs.
The groups can include existing resolved routes. No identity was promoted.

For January 2000 specifically, native acquisition inventory reports positive
rows for 483/529 review-flagged members, 256/302 existing resolved members and
30/32 collision members. The remaining 46, 46 and 2 have complete HTTP-200
records reporting zero rows, one page and no explanatory message. This retained
table cannot explain why the response was empty. The 37 missing links have no
native route with which to join an acquisition record. None of these counts
demonstrates prices at the historical signal or across a required feature window.

## Review queue and safety contract

Priorities are structural conflicts (same-date collisions, ambiguous routes,
conflicting source ISINs or original-versus-recomputed route differences),
missing routes, matching declared ISIN queries awaiting provenance, then weaker
or mismatched identifiers. Original and recomputed unified/native IDs remain
separate. Equally missing routes are unknown agreement, not positive evidence.

All rows have `qualification=REVIEW_REQUIRED_NOT_PROMOTED` and false
`independent_provider_isin_verified`, `raw_response_content_verified`,
`temporal_identity_verified`, `knowledge_time_verified` and
`price_history_coverage_verified`. This is an evidence-review status; it does
not rewrite the original `resolved` or quarantine states. The audit flags,
queue priority, response hashes and acquisition counts are never predictors.
`features_df`, `monthly_df` and their population rules remain separate.

Source membership effective/availability dates must be present and no later
than the signal; inclusive membership ends cannot precede it. The standalone
audit validates these clocks as well as the source loader. It retains malformed
identifier intervals as diagnostics but rejects missing/duplicate original
member evidence, malformed alias arrays, financial/target inputs, protected
signal dates and exceeded budgets rather than quietly dropping constituents.

## Reproduce and inspect

From the configured V2 folder, start Python with `PYTHONPATH=src .venv/bin/python`
or use the master's historical identity evidence cell:

```python
from pathlib import Path
from dmv2.catalog import SourceCatalog
from dmv2.identity_audit import build_identity_audit, save_identity_audit

audit = build_identity_audit(SourceCatalog(Path.cwd()))
audit.summary                    # aggregate date populations, no financial values
audit.review_queue              # full private constituent/date evidence
path = save_identity_audit(Path.cwd(), audit)
```

The metadata loader projects six local pinned tables: source calendar,
historical membership, unified bridge, native links, Siblis identifier history
and acquisition evidence. Membership filtering returns **only the three exact
dates**, although the security-sorted Parquet layout requires larger metadata
column scans. Measured: six footers, six data files, 9,520,091 candidate rows,
85,608 returned metadata rows and 34,875,548 projected compressed bytes.
Cumulative ceilings: 128 footers, 12 data files, 20 million candidate rows,
100,000 returned rows and 128 MiB. These are metadata-only bounds, not approval
for a full financial panel, all-month audit or backtest.

`summary.json`, `constituent_ledger.parquet` and `review_queue.parquet` are saved
under ignored `outputs/identity_audits/<evidence-hash>/`. Evidence includes exact
release/manifest/object pins, reviewed projections, requests, per-stage measured
budgets, independent calendar provenance, original cohort/table fingerprints,
worker hashes and claim limits. Identical evidence reuses the same artifact;
changed evidence under identical pins fails. No raw response or data row is
published in Git. The Intrinio release is
`professional-reference-20260807-r1-pdrs-v2-r3`; the membership and identity
releases are `professional-reference-20260807-r1`.

## Remaining evidence request

The current local sources cannot finish historical identity qualification. A
forwardable request is in
[`IDENTITY_EVIDENCE_REQUEST.md`](IDENTITY_EVIDENCE_REQUEST.md). It asks for the
producer flag rule/reasons, original identity lookup lineage and a dated source
to native-ID crosswalk. No message has been sent and no IAM change is proposed.
Any subsequent repair requires explicit evidence and a versioned reviewed
mapping policy; this diagnostic is not a waiver. Knowledge-time certification
also needs publication/revision evidence beyond an effective-dated crosswalk.

The [latest organizational GitHub review](SADIG_IDENTITY_REVIEW_20261014.md)
checks Sadig's October 14 research head and other branch differences. Its
completed diagnostics add zero dated strong identities; current source delivery
still retains IDENTITY/PIT UNVERIFIED. The three core identity workers are
unchanged from our reference snapshot. The draft request is narrowed to newer
external artifacts and original producer/capture evidence; their provisional
research exception has not been adopted by our V2.

The local retained-lineage check found no raw identity response payloads or
producer resolver code. The pinned release inventories published tables;
`raw_page_inventory` inventories stock-price pages, not identity responses.
The exact prior-release manifest was followed as a small remote locator:
`gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/intrinio_market_daily/v1/_releases/professional-reference-20260807-r1/MANIFEST.json`,
generation `1786102843388189`, declared size 74,655 bytes. Its attempted read
using configuration `deep-momentum-research`, project
`trusty-agility-492109-v0` and account `info@galamboscapital.com` failed with
`Reauthentication failed. cannot prompt during non-interactive execution.`
No object was read in that failed attempt. This was a CLI authentication-refresh
issue; no missing storage permission, absent dataset or owner IAM action was
established. A subsequent forced interactive browser login completed as the
intended account, and the **same exact-generation read succeeded**. Its 74,655
bytes match the recorded raw SHA-256
`sha256:1f0e4ca5b6e17381ad0f58bf6ed2bfda30a1bb1588d9b9b36fe7169d8de56a7e`.

That manifest references an older retained metadata object:
`gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/intrinio_market_daily/v1/stable/MANIFEST.json`,
generation `1785655514257064`. Its size was checked before reading; the read
succeeded and its 46,576 bytes match raw SHA-256
`sha256:e07789d4dcd03963ac8bfb6bbb3ad541c324158538cc5b7d6a16798430289810`.
Neither JSON manifest supplies a raw identity-capture URI or dated native-ID
crosswalk. The older manifest inventories published table paths and describes
provider-adjusted history as retrieval-time data. No referenced table or price
payload was downloaded in this follow-up, and no broad bucket listing was run.
The two metadata reads total 121,231 bytes. Successful CLI reads verify these
objects, not Python ADC access, broader permissions or historical identity.
Original failed-read evidence is preserved separately from the success record.

The installed 15-code-cell master completed, all 298 structural tests passed,
and the identity artifact matched its pinned local audit. These checks verify
the diagnostic workflow, not historical identity or economic certification.

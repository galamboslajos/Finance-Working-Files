# Identity evidence request — narrowed after latest GitHub review

Draft for the user to forward; no message has been sent.

Sadig, we reviewed GalambosC/deep-momentum on codex/deep-momentum-v2-phase1 at
f8f705a08ef06443c0aed3ff68216f713a35eef7 (October 14, 2026).
We saw the identity repair/local construction diagnostics, R14 inspection,
G1 assessment and DEC-0143 provisional identity exception. We understand that
the current source build still carries IDENTITY/PIT UNVERIFIED; we are not
treating new catalogue acceptance as repaired historical stock mappings.

Our separate V2 uses the same August releases, starts research in 2000 and
currently audits SP500/SP400 historical membership. January 2000 retains
900 original members: 529 review-flagged routes, 32 shared-native-ID conflicts,
37 missing native links and 302 unique unflagged routes. Those flags do not
by themselves prove wrong mappings. Most flagged native IDs have reported
price history, which does not resolve historical identity.

Before repeating work, could you point us to existing evidence outside the
checked branch for:

1. A newer completed historical source-ID/share-class → Intrinio-native-ID
   crosswalk, with identifier validity boundary semantics, availability/revision
   timestamps or immutable historical vintages and adjudication evidence. If
   one exists, please share its exact read-accessible URI, version, manifest
   and hashes. If not, confirmation that the documented gap remains is enough;
   we are not requesting another same-input reconstruction.
2. Original July identity capture manifests/raw lookup receipts and the
   resolver/normalizer rule or per-record reasons behind
   intrinio_identity_links.identity_review_required. Our links name run
   intrinio-identity-full-20260725-r4-provider-native-v1 and manifest hash
   sha256:fc5932ba14fe5fe405aa45ee8047e79e74d45906e177a965aea489bc8e809327.
   Per-row source_relative_path/source_response_hash references exist, but the
   exact capture location and hash serialization contract are missing from
   the published manifests.
3. The producer derivation of observed_isins_json and observed_tickers_json.
   Every current ISIN alias set matches its upstream unified bridge set, so
   we cannot count that agreement as independent provider-returned evidence.
   Price acquisition receipts use native IDs; they do not authenticate the
   original ticker/ISIN lookup literal.

We successfully read the pinned August release manifest and linked July
retained manifest in GCS; their sizes and raw hashes matched. Neither JSON
manifest supplies the raw capture or dated crosswalk. This is an evidence
location/semantics request, not a broader permissions or price-download
request. No provider contact, purchase, dataset rebuild or cloud run is being
requested. Our private January 2000 conflict/missing-link queue can narrow any
existing case-level evidence.

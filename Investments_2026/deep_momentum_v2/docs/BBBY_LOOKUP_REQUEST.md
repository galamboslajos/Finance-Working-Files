# Historical BBBY provider lookup request

Draft for the owner; **not sent**. This asks for a specific missing identity
lookup, without requesting credentials or cloud mutations.

> Could you supply the correct Intrinio native security ID and original lookup
> evidence for **historical Bed Bath & Beyond Inc., ISIN US0758961009** (old BBBY,
> later BBBYQ), separately from Overstock/current BBBY, ISIN US6903701018?
>
> Our pinned Siblis metadata identifies the old issuer as US0758961009. However,
> its published Intrinio link is labelled `query_identifier_type=isin` while the
> actual saved query literal is **BBBY**, with retained path
> `raw_pages/query_identifier=BBBY-4ce678ab13/security.json.gz`. It returns
> `sec_ydbQnj`, also linked to the US6903701018 lookup. Both references record
> response SHA-256
> `78fde89da078e7297d325c3be59800f2bb6443d204e9f4d668e3778367510cd0`.
> The original response bodies are unavailable locally, so the old ISIN alias
> does not establish a successful exact-old-ISIN provider lookup.
>
> Please provide an exact lookup of **US0758961009 / CUSIP 075896100** (or its
> confirmed historical BBBYQ identifier), including the lookup literal, native ID, retained
> response path and content hash, issuer/share-class/FIGI evidence, and dated
> ticker/listing history. Please also identify the associated pre-2020 price
> source and stated coverage so we can rebuild affected inputs. If this lookup
> is unresolved or unsupported, please state that explicitly. We will preserve
> the original runs and keep the replacement route blocked until reviewed.

## Local verification record

- Reviewed old source IDs:
  `siblis_isin_c297a5f81a23c9f90503` and
  `siblis_ticker_company_00326c7d9ad13903e5e3`.
- Old ISIN unified ID: `equity_us_aacab20bc16f70ab265fd1e1`.
- The source identifier-history validity boundaries are blank; membership
  intervals do not certify dated identifier ownership.
- Independent contemporary issuer/class evidence: a
  [2019 SEC-hosted Form 13F information table](https://www.sec.gov/Archives/edgar/data/101199/000010119919000056/xslForm13F_X01/ufg13f03-31x19informationt.xml)
  reports **BED BATH & BEYOND INC / Common / CUSIP 075896100**. This supports the
  historical issuer/CUSIP association; it does not establish an Intrinio native
  ID, complete identifier-validity interval or original provider response.
- Published Intrinio release:
  `professional-reference-20260807-r1-pdrs-v2-r3`; verified canonical manifest
  SHA-256 `467c97c83ee6b326d87bcb9ade3bc8e49f44efdbf6ed0afc872c748005d309db`.
- Published identity-links table SHA-256:
  `ba51dd1259ffdb3097b2c55c747ffeab889af41c0bf4f6f0fcea8c82dea7d758`.
- Producer identity run:
  `intrinio-identity-full-20260725-r4-provider-native-v1`; declared producer
  manifest SHA-256
  `fc5932ba14fe5fe405aa45ee8047e79e74d45906e177a965aea489bc8e809327`.
- Bounded metadata-only audit:
  `data/derived/identity_amendments/old-bbby-native-conflict-20261016-v1/local_identity_lookup.json`, SHA-256
  `893d0a497bb830bce412a8501912439be757c6b32f99b542f11fd4483466713c`.

This request does not claim that the vendor lacks the old instrument. The
current code amendment disables a known wrong route and retains all source
rows; the correct replacement native ID and price route remain unresolved.

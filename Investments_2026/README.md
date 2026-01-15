# About this directory

* **Strategic layer of the portfolio:** The objective was to have a layer of the portfolio that has robust out of sample features. Stacked/cross validated portfolio weights over different splits and simulatied paths. The strategic layer therefore serves as the spine of the asset allocation (read more on the asset preselection logic bellow).

Preselection: Investments_2026/Preselection_accross_splits.ipynb

Robust Weights for Strategic Layer of the Portfolio:Investments_2026/10_PortOpt_ParamUncertainty.ipynb














## State-Aware Strategic Asset Allocation Model

This project implements a quantitative portfolio construction framework from Kristensen and Vorobets. Unlike traditional static optimizers, this model uses a dynamic, regime-aware approach to build tactical portfolios.

The core objective is to generate an optimal portfolio that is **conditioned on the current market state** (defined by the VIX) and **robust to parameter uncertainty** (managed via a regime-based bootstrap). The model uses modern techniques, including Entropy Pooling and Conditional Value-at-Risk (CVaR), to build high-conviction, risk-managed portfolios.

## Theoretical Framework & Key Assumptions

This model is built on several key hypotheses about the nature of financial markets (based on Vorobets, 2025: https://github.com/fortitudo-tech/pcrm-book)

### 1. The Future is Uncertain (Parameter Uncertainty)
* **Hypothesis:** Even if we correctly identify the regime, the *true* expected returns for the next period are unknowable. Using a single historical average ("point estimate") is brittle and leads to "noise-mining."
* **Implication:** We must model our uncertainty. The model uses a Monte Carlo bootstrap to generate a *distribution* of plausible future return scenarios, building a portfolio that is robust across many of those futures, not just one.

### 2. Time and State Conditional Framework 
* **Hypothesis:** The recent past is more relevant for forecasting the near-term future than the distant past/ and different market regimes suggest different PnL outcomes
* **Implication:** We use an exponentially-weighted time-decay prior (`p_exp`) as our baseline, making the model more responsive to recent market dynamics.

### 3. Risk is in the Tail (CVaR, not Variance)
* **Hypothesis:** Volatility (variance) is an incomplete measure of risk. It treats upside and downside deviation equally and, by assuming a normal distribution, fails to capture the true risk of extreme, non-normal losses.
* **Implication:** We use **CVaR (Expected Shortfall)** as the core risk measure. CVaR focuses on the *average magnitude of losses in the tail*, providing a more coherent and realistic measure of true risk.

## Practical Roadmap to Narrow Down Portfolio Size

We want to achive proper diversification in the portfolio, but also, we do not want over diversification. To tackle this we would select single best (Mean-CVaR optimum on resmapled data) allocations sector-by-sector. That leaves us with a feasible amount of participants for the final portfolio optimisation, stress testings.

# Sectors

| Sector                     | Count  |
| -------------------------- | ------ |
| **Technology**             | **66** |
| **Health Care**            | **69** |
| **Financials**             | **79** |
| **Consumer Discretionary** | **92** |
| **Consumer Staples**       | **34** |
| **Industrials**            | **95** |
| **Energy**                 | **38** |
| **Materials**              | **33** |
| **Utilities**              | **30** |
| **Communication Services** | **27** |
| **Real Estate**            | **33** |

All together, there 596 tickers alicve in the specified sets.

    'Technology': # Active tech stocks
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'INTC', 'CSCO', 'ORCL', 'IBM', 'ADBE', 'CRM', 
    'NOW', 'INTU', 'AMAT', 'LRCX', 'KLAC', 'MCHP', 'TXN', 'ADI', 'QCOM', 'AVGO',
    'NXPI', 'MU', 'STX', 'WDC', 'HPQ', 'HPE', 'DELL', 'ANET', 'FFIV', 'JNPR',
    'CTSH', 'IT', 'ACN', 'EPAM', 'FTNT', 'PANW', 'CRWD', 'SNPS', 'CDNS', 'ANSS',
    'KEYS', 'TER', 'IPGP', 'GLW', 'APH', 'TEL', 'AKAM', 'ADSK', 'CIEN', 'SMCI',
    'APP', 'DDOG', 'PLTR', 'FICO', 'GEN', 'BR', 'JKHY', 'FIS', 'FI', 'CSGP',
    'TYL', 'MPWR', 'ON', 'SWKS'],

Removed (delisted/acquired):
MXIM - acquired by ADI (2021)
XLNX - acquired by AMD (2022)
ALTR - acquired by Intel (2015)
LLTC - acquired by ADI (2017)
BRCM - acquired by Avago/Broadcom (2016)
CA - acquired by Broadcom (2018)
EMC - acquired by Dell (2016)
NVLS - acquired by Lam Research (2012)
LSI - acquired by Avago (2014)  
    
    # Health Care
    'Health Care': ['JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'LLY', 'BMY', 'AMGN', 'GILD', 'BIIB',
    'REGN', 'VRTX', 'MRNA', 'ABT', 'MDT', 'SYK', 'BSX', 'EW', 'ISRG', 'DHR',
    'TMO', 'A', 'IDXX', 'IQV', 'CRL', 'BIO', 'HOLX', 'DXCM', 'PODD', 'ALGN',
    'ZBH', 'BAX', 'BDX', 'COO', 'XRAY', 'HSIC', 'CAH', 'MCK', 'CVS', 'CI',
    'HUM', 'CNC', 'MOH', 'DVA', 'HCA', 'UHS', 'THC', 'DGX', 'LH', 'TECH',
    'RVTY', 'MTD', 'WAT', 'ILMN', 'INCY', 'PRGO', 'NKTR', 'ELV', 'COR', 
    'VTRS', 'CTLT', 'OGN', 'GEHC', 'KVUE', 'SOLV'],

Removed (delisted/acquired):
ALXN - acquired by AstraZeneca (2021)
CELG - acquired by Bristol-Myers Squibb (2019)
CERN - acquired by Oracle (2022)
AET - acquired by CVS Health (2018)
ESRX - acquired by Cigna (2018)
BCR - acquired by Becton Dickinson (2017)
STJ - acquired by Abbott (2017)
VAR - acquired by Siemens Healthineers (2021)
LIFE - acquired by Thermo Fisher (2014)
FRX - acquired by Actavis (2014)
BEAM - acquired by Daiichi Sankyo (2019)
ABMD - acquired by Johnson & Johnson (2023)
ENDP - bankruptcy (2022)
    
    # Financials
    'Financials': [ 'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'COF', 'DFS',
    'V', 'MA', 'PYPL', 'ICE', 'CME', 'CBOE', 'NDAQ', 'SPGI', 'MCO', 'MSCI',
    'BK', 'STT', 'NTRS', 'BEN', 'IVZ', 'TROW', 'AMP', 'RJF', 'LNC', 'PRU',
    'MET', 'AFL', 'AIG', 'ALL', 'TRV', 'CB', 'PGR', 'CINF', 'L', 'AJG',
    'MMC', 'AON', 'BRO', 'WTW', 'BRK.B', 'USB', 'PNC', 'TFC', 'MTB', 'FITB',
    'KEY', 'RF', 'CFG', 'HBAN', 'CMA', 'ZION', 'FHN', 'ACGL', 'GL',
    'AIZ', 'UNM', 'BHF', 'GNW', 'AMG', 'JEF', 'SYF', 'ALLY', 'COIN',
    'HOOD', 'IBKR', 'MKTX', 'FDS', 'WRB', 'ERIE', 'HIG', 'BX', 'KKR', 'APO'],

Removed (delisted/acquired):
PBCT – acquired by M&T Bank (2022)
ETFC – acquired by Morgan Stanley (2020)
INFO – merged into S&P Global (2022)
FRC – failed; assets assumed by JPMorgan (2023)
SIVB – failed; Silicon Valley Bank collapse (2023)
SBNY – failed; Signature Bank collapse (2023)
ACAS – acquired by Ares Capital (2017)
    
    # Consumer Discretionary
    'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'LOW', 'TJX', 'ROST', 'NKE', 'SBUX', 'MCD', 'YUM',
    'DRI', 'CMG', 'DPZ', 'BKNG', 'MAR', 'HLT', 'LVS', 'WYNN', 'MGM', 'CCL',
    'RCL', 'NCLH', 'F', 'GM', 'APTV', 'BWA', 'LKQ', 'AN', 'KMX', 'AZO',
    'ORLY', 'BBY', 'ULTA', 'LULU', 'NVR', 'LEN', 'DHI', 'PHM', 'WHR', 'POOL',
    'TSCO', 'DG', 'DLTR', 'TGT', 'EBAY', 'ETSY', 'EXPE', 'TRIP', 'CZR',
    'PENN', 'GRMN', 'MHK', 'RL', 'PVH', 'TPR', 'HAS', 'MAT', 'DECK',
    'BBWI', 'GPS', 'ANF', 'URBN', 'FL', 'JWN', 'M', 'KSS', 'BIG', 'GPC',
    'AAP', 'FBHS', 'WSM', 'RH', 'W', 'GME', 'ADT', 'COTY',
    'ABNB', 'DASH', 'UBER', 'CPRT'],

Removed (delisted/acquired):
JCP – J.C. Penney, bankrupt & delisted (2020)
BBBY – Bed Bath & Beyond, bankrupt & delisted (2023)
SHLD – Sears Holdings, bankrupt & delisted (2018)
ADS – Alliance Data Systems; renamed to BFH (Bread Financial) (2022)
DTV – DirecTV, no longer publicly traded after AT&T acquisition (2015)
DISH – Dish Network still exists but now trading OTC after bankruptcy filing (2024) — effectively delisted
TWC – Time Warner Cable, acquired by Charter (2016)
HOT – Starwood Hotels, acquired by Marriott (2016)
WYN – Wyndham Worldwide split; legacy ticker no longer active (2018)
TIF – Tiffany & Co., acquired by LVMH (2021)
KSU – Kansas City Southern, acquired by Canadian Pacific (2023)
CPRI – Capri Holdings, acquired by Tapestry (2024, closing)
MAT – Mattel is active (correcting confusion — KEEP)
LEG – Leggett & Platt suspended dividend & facing distress, still listed (KEEP)
BIG – Big Lots is still listed but near-distress; active (KEEP)

    # Consumer Staples
    'Consumer Staples': ['PG', 'KO', 'PEP', 'COST', 'WMT', 'PM', 'MO', 'KMB', 'CL', 'CLX',
    'CHD', 'EL', 'K', 'GIS', 'CAG', 'SJM', 'MKC', 'HSY', 'HRL', 'CPB',
    'KHC', 'MDLZ', 'MNST', 'KR', 'SYY', 'ADM', 'BG', 'TSN', 'TAP', 'STZ',
    'BF.B', 'KDP', 'LW', 'WBA'],

Removed (delisted/acquired):
RAI – Reynolds American, acquired by British American Tobacco (2017)
WFM – Whole Foods Market, acquired by Amazon (2017)
SLE – Sara Lee, acquired/split; no longer publicly traded (2012)
DF – Dean Foods, bankrupt & delisted (2019)
AVP – Avon Products, acquired by Natura (2020); ADR delisted
SVU – SuperValu, acquired by UNFI (2018)
MJN – Mead Johnson, acquired by Reckitt Benckiser (2017)
RAD – Rite Aid, bankrupt again, delisted (2023)   

    # Industrials
    'Industrials': ['CAT', 'DE', 'HON', 'GE', 'MMM', 'EMR', 'ROK', 'ETN', 'PH', 'ITW',
    'IR', 'DOV', 'AME', 'XYL', 'GNRC', 'PCAR', 'CTAS', 'FAST', 'GWW', 'NDSN',
    'SNA', 'SWK', 'LII', 'TT', 'CARR', 'OTIS', 'JCI', 'LHX', 'NOC', 'LMT',
    'RTX', 'GD', 'BA', 'TXT', 'HII', 'TDG', 'HWM', 'WAB', 'CSX', 'NSC',
    'UNP', 'UPS', 'FDX', 'EXPD', 'CHRW', 'JBHT', 'ODFL', 'UAL', 'DAL', 'LUV',
    'ALK', 'AAL', 'WM', 'RSG', 'ROL', 'VRSK', 'EFX', 'AOS', 'ALLE', 'CMI',
    'FTV', 'IEX', 'RHI', 'MAS', 'LDOS', 'J', 'BAH', 'AXON', 'PWR', 'EME',
    'FLR', 'JBL', 'MLM', 'VMC', 'URI', 'BLDR', 'HUBB', 'AYI', 'TRMB', 'TDY',
    'FLS', 'PNR', 'DNB', 'ITT'],

Removed (delisted/acquired):
COL – Rockwell Collins, acquired by UTC (2018)
RTN – Raytheon, merged with UTC into RTX (2020)
UTX – United Technologies, merged into RTX (2020)
LLL – L3 Technologies, merged with Harris to form LHX (2019)
NLSN – Nielsen Holdings, taken private; delisted (2022)
RRD – R.R. Donnelley, acquired by Chatham; delisted (2022)
JOY – Joy Global, acquired by Komatsu (2017)
CAM – Cameron International, acquired by Schlumberger (2016)
FDC – First Data, acquired by Fiserv (2019)
LM – Legg Mason, acquired by Franklin Templeton (2020)
MIL – MFC Industrial / “Millicom old ADR” (multiple restructurings); legacy ticker delisted
GEV – General Electric Ventures/GenVec (merged & delisted)
DAY – Legacy “Day International”; acquired by BAIN (2005), long-terminated
GRN – Old “Greenhill Renewable Energy”; delisted
CPAY – “Cardtronics / CompuPay” legacy ticker; acquired by NCR (2021); delisted  

    # Energy
    'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'PXD', 'DVN', 'MRO', 'APA', 'OXY', 'HES',
    'FANG', 'CTRA', 'MPC', 'VLO', 'PSX', 'HAL', 'SLB', 'BKR', 'NOV', 'FTI',
    'WMB', 'KMI', 'OKE', 'TRGP', 'EQT', 'RRC', 'SWN', 'CNX', 'BTU',
    'CLF', 'X', 'MUR', 'RIG', 'DO', 'HP', 'NBR', 'SUN'],

Removed (delisted/acquired):
APC – Anadarko Petroleum, acquired by Occidental (2019)
EP – El Paso Corp., acquired by Kinder Morgan (2012)
NBL – Noble Energy, acquired by Chevron (2020)
NFX – Newfield Exploration, acquired by Encana/Ovintiv (2019)
CXO – Concho Resources, acquired by ConocoPhillips (2021)
WPX – WPX Energy, merged with Devon Energy (2021)
QEP – QEP Resources, acquired by Diamondback Energy (2021)
DNR – Denbury Resources, bankrupt (2020), relisted then acquired by Exxon (2023), old ticker gone
ESV – Ensco, bankrupted & merged into Valaris; old ticker delisted
NE – Noble Corp. (old), bankrupt (2020); reorganized, new ticker is NOBL
RDC – Rowan Companies, merged with Ensco (2019); delisted
ANDV – Andeavor, acquired by Marathon Petroleum (2018)
TSO – Tesoro, rebranded into Andeavor (2017), then acquired by MPC
DYN – Dynegy, acquired by Vistra Energy (2018)
BHI – Baker Hughes Inc., merged with GE Oil & Gas into BHGE (2017), replaced by BKR
HFC – HollyFrontier, reorganized into HF Sinclair (DINO) (2022)
CEG – Legacy Constellation Energy (2000s), delisted; modern CEG is not this ticker (you should remove this one)    

    # Materials
    'Materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'DOW', 'PPG', 'NEM', 'FCX', 'NUE',
    'STLD', 'VMC', 'MLM', 'BALL', 'PKG', 'IP', 'AVY', 'SEE', 'CE', 'EMN',
    'ALB', 'FMC', 'MOS', 'CF', 'IFF', 'CTVA', 'CCK', 'WRK', 'ATI', 'AA',
    'OI', 'LYB'],

Removed (delisted/acquired):
AKS – AK Steel, acquired by Cleveland-Cliffs (2020)
ARG – Airgas, acquired by Air Liquide (2016)
ARNC – Arconic (old), split/spun and legacy ticker terminated (2020); new Arconic bought by Apollo (2023), delisted
GRA – W.R. Grace, acquired by Standard Industries (2021), delisted
MON – Monsanto, acquired by Bayer (2018)

    # Utilities
    'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'PEG', 'ED',
    'WEC', 'ES', 'AEE', 'CMS', 'CNP', 'NI', 'DTE', 'ETR', 'FE', 'PPL',
    'ATO', 'NRG', 'EVRG', 'AWK', 'LNT', 'PNW', 'PCG', 'EIX', 'VST', 'AES'],

Removed (delisted/acquired):
DRE – Duke Realty, acquired by Prologis (2022)
GAS – AGL Resources, acquired by Southern Company (2016)
POM – Pepco Holdings, acquired by Exelon (2016)
TE – TECO Energy, acquired by Emera (2016)
SCG – SCANA, acquired by Dominion Energy (2019)
FTR – Frontier Communications, bankruptcy (2020); old ticker delisted
WIN – Windstream, bankruptcy (2019); delisted
LUMN – Lumen Technologies, bankruptcy (2024), NASDAQ delisted (OTC now → remove)
LVLT – Level 3 Communications, acquired by CenturyLink (2017)
CTX – CTX = Old Centex, acquired by PulteGroup (2009)

    # Communication Services
    'Communication Services': ['META', 'GOOGL', 'GOOG', 'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR',
    'WBD', 'FOX', 'FOXA', 'OMC', 'IPG', 'EA', 'TTWO', 'ATVI', 'LYV', 'MTCH',
    'NWSA', 'NWS', 'NYT', 'PARA', 'TGNA', 'LSXMA', 'TKO'],

Removed (delisted/acquired):
VIAB – Viacom, merged with CBS to form ViacomCBS/Paramount (PARA) (2019)
DISCA – Discovery, merged with WarnerMedia into WBD (2022)
DISCK – Discovery Class C, also merged into WBD (2022)
SNI – Scripps Networks, acquired by Discovery (2018), then absorbed into WBD
TWX – Time Warner, acquired by AT&T (2018)
TWTR – Twitter, taken private by Elon Musk (2022), delisted
YHOO – Yahoo, sold to Verizon, later acquired by Apollo; old ticker long gone (2017)

    # Real Estate
    'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'DLR', 'PSA', 'SPG', 'O', 'WELL', 'AVB',
    'EQR', 'ESS', 'MAA', 'UDR', 'VTR', 'VNO', 'SLG', 'BXP', 'ARE', 'CBRE',
    'VICI', 'INVH', 'KIM', 'REG', 'FRT', 'HST', 'AIV', 'CPT',
    'IRM', 'EXR', 'SBAC', 'MAC', 'WY'],

Removed (delisted/acquired):                    
DOC – merged into PEAK in 2024, ticker eliminated
GGP – acquired by Brookfield (2018)
PCL – merged into WY (2016)

    # Commodities (your non-stock assets)
    'Commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', '^GSPC', '^NDX', '^GDAXI'],
    
    # Currencies
    'Currencies': ['EUR=X', 'JPY=X', 'GBP=X', 'AUD=X', 'CHF=X']

## Dependencies

 `fortitudo.tech` (Custom library for Entropy Pooling & Mean-CVaR) and Anton Vorobets' Portfolio Construction and Risk Management: https://github.com/fortitudo-tech/pcrm-book 

"""
Deep Momentum — Configuration
Replication of Han & Qin (2026), "Bimodality Everywhere: International Evidence of Deep Momentum"

All constants match the paper exactly. Do not modify without referencing the paper.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ═══ API ═══════════════════════════════════════════════
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
FMP_BASE = "https://financialmodelingprep.com/stable"
FMP_RATE_LIMIT = 50  # calls per second (3000/min)

# ═══ COUNTRIES — Table 1 of the paper ═════════════════
# suffix: (country_code, country_name, first_date_paper, avg_stocks_paper)
# first_date_paper is from Table 1 — our FMP data may start later
COUNTRIES = {
    "US":  ("US", "United States",   "1965-01-31", 4985),
    "L":   ("GB", "United Kingdom",  "1965-01-31", 1573),
    "AX":  ("AU", "Australia",       "1973-02-28",  925),
    "AT":  ("AT", "Austria",         "1973-02-28",   74),
    "BR":  ("BE", "Belgium",         "1973-02-28",  177),
    "SA":  ("BR", "Brazil",          "1990-02-28",  309),
    "TO":  ("CA", "Canada",          "1973-02-28", 1994),
    "SS":  ("CN", "China (Shanghai)","1992-06-30", 1911),
    "SZ":  ("CN", "China (Shenzhen)","1992-06-30",    0),  # combined with SS in paper
    "CO":  ("DK", "Denmark",         "1973-02-28",  140),
    "HE":  ("FI", "Finland",         "1988-04-30",  131),
    "PA":  ("FR", "France",          "1973-02-28",  594),
    "DE":  ("DE", "Germany",         "1973-02-28",  526),
    "HK":  ("HK", "Hong Kong",       "1973-02-28",  808),
    "BO":  ("IN", "India (BSE)",     "1990-02-28", 2502),
    "NS":  ("IN", "India (NSE)",     "1990-02-28",    0),  # combined with BO in paper
    "JK":  ("ID", "Indonesia",       "1990-05-31",  333),
    "TA":  ("IL", "Israel",          "1986-02-28",  407),
    "MI":  ("IT", "Italy",           "1973-02-28",  250),
    "T":   ("JP", "Japan",           "1973-02-28", 2474),
    "KS":  ("KR", "South Korea",     "1988-02-29",  967),
    "KL":  ("MY", "Malaysia",        "1986-02-28",  682),
    "MX":  ("MX", "Mexico",          "1988-02-29",  115),
    "AS":  ("NL", "Netherlands",     "1973-02-28",  170),
    "NZ":  ("NZ", "New Zealand",     "1988-02-29",  109),
    "OL":  ("NO", "Norway",          "1980-02-29",  171),
    "WA":  ("PL", "Poland",          "1995-01-31",  436),
    "LS":  ("PT", "Portugal",        "1988-02-29",   75),
    "SI":  ("SG", "Singapore",       "1983-02-28",  404),
    "JO":  ("ZA", "South Africa",    "1973-02-28",  266),
    "MC":  ("ES", "Spain",           "1988-02-29",  140),
    "ST":  ("SE", "Sweden",          "1982-01-31",  371),
    "SW":  ("CH", "Switzerland",     "1973-02-28",  229),
    "TW":  ("TW", "Taiwan",          "1988-02-29",  788),
    "BK":  ("TH", "Thailand",        "1988-02-29",  397),
    "IS":  ("TR", "Turkey",          "1988-02-29",  217),
}

# ═══ DATA FILTERS — Section 3.1 of the paper ══════════
# "Observations with monthly returns greater than 300% or lower than -95% are removed"
MAX_MONTHLY_RETURN = 3.00   # 300%
MIN_MONTHLY_RETURN = -0.95  # -95%

# "the remaining returns are winsorized at 1% and 99% within each country"
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# "the market capitalization is below the bottom 5% within a country in any month"
# Paper uses 5% on clean exchange-listed data (Datastream/CRSP).
# FMP includes OTC/pink sheets for US, so we use 25% for US to approximate
# the same universe. Other countries use 5% (their symbols are exchange-specific).
MCAP_BOTTOM_PCT = 0.05
MCAP_BOTTOM_PCT_US = 0.25

# ═══ MODEL — Section 3.3 of the paper ═════════════════
N_CLASSES = 10              # "ten return classes" (deciles)
MIN_TRAIN_YEARS = 10        # "We require at least ten years of data to train XGBoost"
N_ENSEMBLE = 100            # "we train the algorithm 100 times in each training month"
TRAIN_VAL_RATIO = 0.8       # "randomly split into a training set and a validation set in a ratio of 8:2"
RETRAIN_FREQUENCY = 12      # "retrain the algorithm every year" (12 months)

# XGBoost defaults (paper uses defaults except early stopping)
XGB_PARAMS = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "verbosity": 0,
    "n_estimators": 10000,       # large number, early stopping will cut
    "early_stopping_rounds": 50,
    "random_state": None,        # will be set per ensemble member
}

# ═══ FEATURES — Section 3.3.1 ═════════════════════════
MOMENTUM_HORIZONS = [1, 3, 6, 9, 12]  # months

# ═══ PORTFOLIO — Section 3.2 & 4.2.2 ══════════════════
PORTFOLIO_DECILE_LONG = 10   # buy top decile
PORTFOLIO_DECILE_SHORT = 1   # sell bottom decile
OOS_START = "2010-01-31"     # "out-of-sample period of January 2010 to December 2023"

# Transaction costs (not in paper — paper reports break-even costs instead)
# 15 bps one-way is conservative for liquid stocks post-2010
TC_BPS = 15

# ═══ RECLASSIFICATION — Section 3.3.2 ═════════════════
CLASS_RETURN_LOOKBACK_YEARS = 10  # "sample analogue over the past ten years"

# ═══ PATHS ═════════════════════════════════════════════
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CACHE_DIR = os.path.join(PROJECT_DIR, "cache")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

for d in [DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

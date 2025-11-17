# State-Aware Strategic Asset Allocation Model

This project implements a quantitative portfolio construction framework from Kristensen and Vorobets. Unlike traditional static optimizers, this model uses a dynamic, regime-aware approach to build tactical portfolios.

The core objective is to generate an optimal portfolio that is **conditioned on the current market state** (defined by the VIX) and **robust to parameter uncertainty** (managed via a regime-based bootstrap). The model uses modern techniques, including Entropy Pooling and Conditional Value-at-Risk (CVaR), to build high-conviction, risk-managed portfolios.

## Theoretical Framework & Key Assumptions

This model is built on several key hypotheses about the nature of financial markets.

### 1. Markets are Not Stationary (Regime-Switching)
* **Hypothesis:** The statistical properties of asset returns (mean, volatility, correlation) are not constant over time. They are functions of the underlying market regime.
* **Implication:** A static "long-term optimal" portfolio is suboptimal. The "best" portfolio is tactical and must adapt to the current regime. This model explicitly uses the VIX to identify the regime and condition the portfolio accordingly.

### 2. The Future is Uncertain (Parameter Uncertainty)
* **Hypothesis:** Even if we correctly identify the regime, the *true* expected returns for the next period are unknowable. Using a single historical average ("point estimate") is brittle and leads to "noise-mining."
* **Implication:** We must model our uncertainty. The model uses a Monte Carlo bootstrap to generate a *distribution* of plausible future return scenarios, building a portfolio that is robust across many of those futures, not just one.

### 3. History is Biased (Time-Weighting)
* **Hypothesis:** The recent past is more relevant for forecasting the near-term future than the distant past.
* **Implication:** We use an exponentially-weighted time-decay prior (`p_exp`) as our baseline, making the model more responsive to recent market dynamics.

### 4. Risk is in the Tail (CVaR, not Variance)
* **Hypothesis:** Volatility (variance) is an incomplete measure of risk. It treats upside and downside deviation equally and, by assuming a normal distribution, fails to capture the true risk of extreme, non-normal losses.
* **Implication:** We use **CVaR (Expected Shortfall)** as the core risk measure. CVaR focuses on the *average magnitude of losses in the tail*, providing a more coherent and realistic measure of true risk.

---

## Investment Mandate: The Risk-Based Leverage Cap

The primary mandate of this strategy is not to just maximize returns, but to do so within a **strictly defined tail-risk budget.**

The portfolio's **maximum acceptable loss** is defined as a **-50% loss to equity** in a "worst 5%" (CVaR 95%) annual scenario. This non-negotiable risk budget is the ultimate constraint that determines the strategy's leverage cap.

The chart below shows the trade-off between the strategy's Net Return and its CVaR 95% tail risk as leverage increases.

![Net Return vs. CVaR by Leverage Cap](Leverage.png)

As shown, while returns (blue line) increase with leverage, the tail risk (orange line) accelerates. The mandate's **-50% budget (green line)** is breached at approximately **2.2x leverage**.

Therefore, any leveraged implementation of this strategy must operate at or below this 2.2x cap to remain compliant with the core risk mandate.

## Dependencies
* `numpy`
* `pandas`
* `yfinance`
* `cvxopt`
* `matplotlib`
* `fortitudo.tech` (Custom library for Entropy Pooling & Mean-CVaR)

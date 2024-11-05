# Portfolio Optimization 

In this project, I take a general overview of global stock market indices, and I attempt to optimize a US stock portfolio among the theoretical framework of Modern Portfolio Theory. First, I begin by taking a sample of the more well-known stock market indicies of different countries'. In a longer historical snapshot, I observe their overall risk and return profiles. Second, I select the US for my broader portfolio allocation. I fetch data from https://finance.yahoo.com, I take a sample of 40 stocks. I take returns, volatilty and covariance of these equities, and I create optimised portfolios to get: the Minimum Variance Portfolio; the Max Return Portfolio; and the Tangetial Portfolio (the one that maximizes the Sharpe Ratio). Third, I add an additional contstraint: I want to create optimised portfolios that have maximum 15 assets (thy shrink down the selected assets). 



```python
# !pip install yfinance
```


```python
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
```

## Overview of Global Stock Market Indicies 

I take an overview of well-known stockmarket indicies from different countries. I want to see the overall risk-return pattern of major stockmarkets. I take data from the year 2000 up until 2024. I plot average annual returns against avarage annual volatilities. 

### Downloading the indicies with the help of y.finance (yf.download)
```python
indices = {
    'S&P 500': '^GSPC',
    'Dow Jones': '^DJI',
    'NASDAQ': '^IXIC',
    'FTSE 100': '^FTSE',
    'DAX': '^GDAXI',
    'CAC 40': '^FCHI',
    'Nikkei 225': '^N225',
    'Hang Seng': '^HSI',
    'Shanghai Composite': '000001.SS',
    'BSE Sensex': '^BSESN',
    'ASX 200': '^AXJO',
    'KOSPI': '^KS11',
    'Taiwan Weighted': '^TWII',
    'Bovespa': '^BVSP',
    'TSX Composite': '^GSPTSE',
    'Mexican IPC': '^MXX',
    'Argentine Merval': '^MERV',
    'South Africa 40': 'JSE.JO',
    'Swiss Market Index': '^SSMI',
    'AEX': '^AEX',
    'IBEX 35': '^IBEX',
    'OMX Stockholm 30': '^OMX',
    'OMX Helsinki 25': '^OMXH25',
    'BUX': '^BUX',
    'PSI 20': '^PSI20',
    'ATX': '^ATX',
    'BEL 20': '^BFX',
    'FTSE MIB': 'FTSEMIB.MI'
}
data = {}
for name, ticker in indices.items():
    data[name] = yf.download(ticker, start='2000-01-01', end='2024-01-01')['Adj Close']
```

### Computing Returns and Volatility 

First, there are two empty dictionaries created for Returns and Std_dev, which than filled with the help of a for loop (takes the computation of returns and stdev through the enterie series) and stores in new columns dedicated for Returns and Std_dev.

```python
returns = {}
std_devs = {}

for name, prices in data.items():
    annual_returns = prices.resample('Y').ffill().pct_change().dropna()
    returns[name] = annual_returns.mean()
    std_devs[name] = annual_returns.std()

returns_df = pd.DataFrame.from_dict(returns, orient='index', columns=['Return'])
std_devs_df = pd.DataFrame.from_dict(std_devs, orient='index', columns=['StdDev'])
```

### Plotting the y~x, the average annualized returns of stockmarket (y) against their avarage annualized volatility (x)

```python
plt.figure(figsize=(14, 8))
plt.scatter(std_devs_df['StdDev'], returns_df['Return'])

for i in returns_df.index:
    plt.annotate(i, (std_devs_df['StdDev'][i], returns_df['Return'][i]))

plt.title('Stock Market Indices: Return vs Standard Deviation (2000-2024)')
plt.xlabel('Standard Deviation')
plt.ylabel('Annual Return')
plt.grid(True)
plt.show()
```


    
![png](output_6_0.png)
    

### Creating a tabe that containst the indicies, average annualized returns, and average annualized volatility

```python
# Combine returns and standard deviations into a single DataFrame
summary_df = pd.concat([returns_df, std_devs_df], axis=1)

summary_df.reset_index(inplace=True)
summary_df.rename(columns={'index': 'Index'}, inplace=True)
summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Return</th>
      <th>StdDev</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>S&amp;P 500</td>
      <td>0.073943</td>
      <td>0.179858</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dow Jones</td>
      <td>0.066691</td>
      <td>0.146816</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NASDAQ</td>
      <td>0.113466</td>
      <td>0.256594</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FTSE 100</td>
      <td>0.019516</td>
      <td>0.139340</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DAX</td>
      <td>0.068484</td>
      <td>0.220158</td>
    </tr>
    <tr>
      <th>5</th>
      <td>CAC 40</td>
      <td>0.030042</td>
      <td>0.191230</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Nikkei 225</td>
      <td>0.062511</td>
      <td>0.221349</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Hang Seng</td>
      <td>0.033689</td>
      <td>0.243565</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Shanghai Composite</td>
      <td>0.088960</td>
      <td>0.435482</td>
    </tr>
    <tr>
      <th>9</th>
      <td>BSE Sensex</td>
      <td>0.173820</td>
      <td>0.294475</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ASX 200</td>
      <td>0.050513</td>
      <td>0.152220</td>
    </tr>
    <tr>
      <th>11</th>
      <td>KOSPI</td>
      <td>0.099598</td>
      <td>0.231159</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Taiwan Weighted</td>
      <td>0.086664</td>
      <td>0.242805</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Bovespa</td>
      <td>0.139952</td>
      <td>0.322931</td>
    </tr>
    <tr>
      <th>14</th>
      <td>TSX Composite</td>
      <td>0.050333</td>
      <td>0.158716</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Mexican IPC</td>
      <td>0.123320</td>
      <td>0.203404</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Argentine Merval</td>
      <td>0.551016</td>
      <td>0.814909</td>
    </tr>
    <tr>
      <th>17</th>
      <td>South Africa 40</td>
      <td>0.111171</td>
      <td>0.313720</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Swiss Market Index</td>
      <td>0.029225</td>
      <td>0.175893</td>
    </tr>
    <tr>
      <th>19</th>
      <td>AEX</td>
      <td>0.032899</td>
      <td>0.203889</td>
    </tr>
    <tr>
      <th>20</th>
      <td>IBEX 35</td>
      <td>0.022592</td>
      <td>0.190260</td>
    </tr>
    <tr>
      <th>21</th>
      <td>OMX Stockholm 30</td>
      <td>0.101565</td>
      <td>0.167679</td>
    </tr>
    <tr>
      <th>22</th>
      <td>OMX Helsinki 25</td>
      <td>0.052765</td>
      <td>0.107631</td>
    </tr>
    <tr>
      <th>23</th>
      <td>BUX</td>
      <td>0.132085</td>
      <td>0.280397</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PSI 20</td>
      <td>-0.011407</td>
      <td>0.152377</td>
    </tr>
    <tr>
      <th>25</th>
      <td>ATX</td>
      <td>0.094349</td>
      <td>0.281281</td>
    </tr>
    <tr>
      <th>26</th>
      <td>BEL 20</td>
      <td>0.033193</td>
      <td>0.208045</td>
    </tr>
    <tr>
      <th>27</th>
      <td>FTSE MIB</td>
      <td>0.007548</td>
      <td>0.206600</td>
    </tr>
  </tbody>
</table>
</div>


## What is the relationship of returns and risks for the sample of stockmarkets worldwide?

Fitting a linear line to the scatter might inidcate the direction of the relationship (and the strength of it). Althought the sample size is very small, still, it can help to imagine the nature of the relationship. 


```python
from sklearn.linear_model import LinearRegression

X = std_devs_df['StdDev'].values.reshape(-1, 1)
y = returns_df['Return'].values

reg = LinearRegression().fit(X, y)

slope = reg.coef_[0]

print(f"Slope of the fitted line: {slope:.4f}")
```
The fitted slope only serves demonstrational purposes to show the average positive relationship between risks and returns. We see that in the last 24 year, US stockmarkets experienced average annual returns of 6.6% - 7.3 % - 11.3% (depending on which market index are we talking about) with average annual volatilites of 14.6% - 17.9% - 25.6% (respectively). 

    Slope of the fitted line: 0.6894

A simplistic understanding of the relationship of the annualized average risks and returns for the sample of global stockmarket indicies is that 1 % point increase of risk is associated with 0.69 % point increase in returns (on average).

## Portfolio Optimization of US Stock Portfolio 🇺🇸📈$

In this section, I take a sample of 40 stocks from the largest US public firms, I take a timeframe of the last 10 years, I take returns, volatility, which I annualize. I perform a portfolio optimisiation in the mean-variance dimension: I create Minimum Variance Portfolio, Max Return Portfolio, and Tangency Portfolio (that maximizes the Sharpe Ratio). I plot the Efficiency Frontier, the Capital Market Line and extract the weights and key statistics of each optimised portfolio.


### Downloading the sample of US Stocks using yf.download

```python
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'V', 'WMT', 'JPM',
    'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'MA', 'PYPL', 'VZ', 'ADBE', 'NFLX',
    'INTC', 'CMCSA', 'KO', 'PEP', 'PFE', 'MRK', 'T', 'CSCO', 'XOM', 'ABT',
    'CVX', 'NKE', 'LLY', 'MCD', 'DHR', 'MDT', 'WFC', 'BMY', 'COST', 'TMO'
]

data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close']
```

### Computing key variables: Returns, Volatities, and annualizing them

The prices time series is resampled to a yearly frequency using 'Y'. This takes the last available price within each year, and ffill() ensures any missing data is filled by carrying the last observed value forward.

```python
returns_std_list = []

for ticker, prices in data.items():
    annual_returns = prices.resample('Y').ffill().pct_change().dropna()
    mean_annual_return = annual_returns.mean()
    std_dev = annual_returns.std()
    
    returns_std_list.append({
        'Ticker': ticker,
        'Annual Return': mean_annual_return,
        'Standard Deviation': std_dev
    })

returns_std_df = pd.DataFrame(returns_std_list)
returns_std_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Ticker</th>
      <th>Annual Return</th>
      <th>Standard Deviation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAPL</td>
      <td>0.312286</td>
      <td>0.399265</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSFT</td>
      <td>0.313368</td>
      <td>0.274847</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GOOGL</td>
      <td>0.249063</td>
      <td>0.328202</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AMZN</td>
      <td>0.384492</td>
      <td>0.499767</td>
    </tr>
    <tr>
      <th>4</th>
      <td>META</td>
      <td>0.349379</td>
      <td>0.710907</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BRK-B</td>
      <td>0.107807</td>
      <td>0.129346</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JNJ</td>
      <td>0.079604</td>
      <td>0.106610</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V</td>
      <td>0.185791</td>
      <td>0.181425</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WMT</td>
      <td>0.111552</td>
      <td>0.213600</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JPM</td>
      <td>0.167219</td>
      <td>0.213238</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PG</td>
      <td>0.093472</td>
      <td>0.149932</td>
    </tr>
    <tr>
      <th>11</th>
      <td>UNH</td>
      <td>0.228030</td>
      <td>0.152923</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NVDA</td>
      <td>0.954135</td>
      <td>0.987619</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HD</td>
      <td>0.194159</td>
      <td>0.254904</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DIS</td>
      <td>0.029555</td>
      <td>0.224811</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MA</td>
      <td>0.216630</td>
      <td>0.205938</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PYPL</td>
      <td>0.199296</td>
      <td>0.578537</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VZ</td>
      <td>0.031556</td>
      <td>0.119957</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ADBE</td>
      <td>0.317286</td>
      <td>0.357582</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NFLX</td>
      <td>0.389533</td>
      <td>0.512008</td>
    </tr>
    <tr>
      <th>20</th>
      <td>INTC</td>
      <td>0.124038</td>
      <td>0.386990</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CMCSA</td>
      <td>0.088897</td>
      <td>0.211495</td>
    </tr>
    <tr>
      <th>22</th>
      <td>KO</td>
      <td>0.073949</td>
      <td>0.077387</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PEP</td>
      <td>0.102941</td>
      <td>0.105435</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PFE</td>
      <td>0.070465</td>
      <td>0.291724</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MRK</td>
      <td>0.129874</td>
      <td>0.203615</td>
    </tr>
    <tr>
      <th>26</th>
      <td>T</td>
      <td>0.050819</td>
      <td>0.227922</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CSCO</td>
      <td>0.118349</td>
      <td>0.197142</td>
    </tr>
    <tr>
      <th>28</th>
      <td>XOM</td>
      <td>0.108817</td>
      <td>0.388806</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ABT</td>
      <td>0.147632</td>
      <td>0.234091</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CVX</td>
      <td>0.113028</td>
      <td>0.302318</td>
    </tr>
    <tr>
      <th>31</th>
      <td>NKE</td>
      <td>0.134405</td>
      <td>0.250911</td>
    </tr>
    <tr>
      <th>32</th>
      <td>LLY</td>
      <td>0.313039</td>
      <td>0.233484</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MCD</td>
      <td>0.173435</td>
      <td>0.143280</td>
    </tr>
    <tr>
      <th>34</th>
      <td>DHR</td>
      <td>0.263474</td>
      <td>0.294672</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MDT</td>
      <td>0.048849</td>
      <td>0.151288</td>
    </tr>
    <tr>
      <th>36</th>
      <td>WFC</td>
      <td>0.055349</td>
      <td>0.295837</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BMY</td>
      <td>0.027157</td>
      <td>0.177969</td>
    </tr>
    <tr>
      <th>38</th>
      <td>COST</td>
      <td>0.235929</td>
      <td>0.238945</td>
    </tr>
    <tr>
      <th>39</th>
      <td>TMO</td>
      <td>0.198928</td>
      <td>0.233930</td>
    </tr>
  </tbody>
</table>
</div>


### Scatter Plot for Annualized Average Returns and Volatilities for the sample of 40 US Stocks

```python
plt.figure(figsize=(14, 8))
plt.scatter(returns_std_df['Standard Deviation'], returns_std_df['Annual Return'])

for i in range(len(returns_std_df)):
    plt.annotate(returns_std_df['Ticker'][i], 
                 (returns_std_df['Standard Deviation'][i], returns_std_df['Annual Return'][i]))

plt.title('Annual Return vs Standard Deviation (2014-2024) for 40 Largest US Firms')
plt.xlabel('Standard Deviation')
plt.ylabel('Annual Return')
plt.grid(True)
plt.show()
```


    
![png](output_14_0.png)
    

### Relationship Analysis of Annualized Average Risks and Retunrs for the sample of 40 US Stocks

```python
from sklearn.linear_model import LinearRegression

X = returns_std_df['Standard Deviation'].values.reshape(-1, 1)
y = returns_std_df['Annual Return'].values

reg = LinearRegression().fit(X, y)
slope = reg.coef_[0]

print(f"Slope of the fitted line: {slope:.4f}")
```
A simplistic understanding of the relationship (slope result) of the annualized average risks and returns for the sample of 40 US Stocks is that 1 % point increase of risk is associated with 0.73 % point increase in returns (on average). 

    Slope of the fitted line: 0.7293

### I want to take a look at the covariance matrix.

In Markowitz’s Mean-Variance Optimization (MVO), the goal is to find portfolios that offer the highest return for a given level of risk or the lowest risk for a given level of return. The covariance matrix is essential in constructing the efficient frontier of portfolios because it helps us identify combinations of assets that best balance risk and return

```python
annual_returns_df = pd.DataFrame()

for ticker, prices in data.items():
    annual_returns = prices.resample('Y').ffill().pct_change().dropna()
    annual_returns_df[ticker] = annual_returns

# annual_returns_df
# by "uncommenting" the # annual_returns_df one can take first a look into annualized return series
```


```python
# The covariance matrix
cov_matrix = annual_returns_df.cov()
# print(cov_matrix)
# by "uncommenting" the # print(cov_matrix) one can take first a look into covariance matrix (detailed version)
```

### Creating an interactive plot for the covariance matrix with the use of color coded values (heatmap)

```python
mask = np.triu(np.ones_like(cov_matrix, dtype=bool))

plt.figure(figsize=(16, 14))  
sns.heatmap(cov_matrix, mask=mask, annot=False, cmap='coolwarm', cbar=True, square=True, linewidths=.5)


plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)

plt.title('Covariance Matrix of Annual Returns (2014-2024) for 40 Largest US Firms')
plt.tight_layout()  
plt.show()
```


    
![png](output_18_0.png)
    
The diagonal values represent the covariance of each asset with itself, which is just its variance. These values tend to be higher because they measure the volatility of each stock individually.
In the heatmap, these elements may appear in a slightly warmer color compared to others, indicating higher values (variances).

Positive and Negative Covariances:
Blue shades represent negative covariances, indicating that some stocks tend to move in opposite directions. Negative covariance can be beneficial in a portfolio, as it can reduce overall portfolio volatility when assets don’t move together.
Red shades represent positive covariances, indicating that the stocks tend to move in the same direction. These are more common among stocks in similar sectors, as they may react similarly to market conditions.

High Positive Covariance (Notable Pairs):
For instance, NVDA and UNH appear to have a high positive covariance (seen as a darker red cell). This could suggest that these two stocks had similar performance trends or responses to market conditions during this period.
Similarly, pairs like XOM and CVX (both energy sector companies) have a stronger positive covariance, indicating they often move in tandem.

Implications for Portfolio Optimization:
In portfolio optimization, you'd ideally combine assets with low or negative covariance to reduce risk through diversification. Assets with high positive covariance contribute more to the overall portfolio risk because their returns are more likely to fluctuate together.
From this heatmap, you could identify pairs of assets that might add more diversification benefit (i.e., those with lighter or blue-colored covariances).

Sectoral Patterns:
There may be subtle patterns related to sectors. For example, companies in the technology sector (AAPL, MSFT, GOOG, NVDA) could have higher covariances among themselves, as they may be more affected by similar market and industry conditions. Similarly, consumer goods or energy sectors might show their own clustering.

Low Covariance Values (Neutral Relationships):
Most pairs are close to zero covariance, indicating no strong relationship between their returns. These are often indicated by the lighter blue or neutral tones, suggesting that these assets could have a smaller effect on each other’s risk when combined in a portfolio.
This visualization provides a useful summary for identifying potential diversification benefits among assets, with red areas indicating where risk could be more concentrated if these stocks are paired in a portfolio.



## Portfolio Optimization for 40 US Stocks (no constraints on the number of assets in the portfolio BUT no short selling condition applied)


```python
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'V', 'WMT', 'JPM',
    'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'MA', 'PYPL', 'VZ', 'ADBE', 'NFLX',
    'INTC', 'CMCSA', 'KO', 'PEP', 'PFE', 'MRK', 'T', 'CSCO', 'XOM', 'ABT',
    'CVX', 'NKE', 'LLY', 'MCD', 'DHR', 'MDT', 'WFC', 'BMY', 'COST', 'TMO'
]

data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close']

prices_df = pd.DataFrame(data)
```

### Important assumption about the risk free! Change it if it has changed!

It might be a bit repetitive, but I want to make sure that the steps are staightforward and transparent. The return series can be checked here too, to avoid errors.

The risk free is an important input for computing the Sharpe ratio: the excess returns are defined as returns above the the risk-free returns.

```python
# Set the risk-free rate
risk_free_rate = 0.03

annual_returns = prices_df.resample('Y').last().pct_change().dropna()
mean_annual_returns = annual_returns.mean()
annual_cov_matrix = annual_returns.cov()

# Display the results 
# mean_annual_returns, annual_cov_matrix
# "Uncomment" if want to check!
```

### The main variables of portfolio optimization is defined

Returns, Risk (Covariance matrix), and the Sharpe ratio are the key inputs for portfolio analysis.

```python
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    returns = np.sum(weights * mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

```

### A simlation of 10.000 portfolios (with randomly assugned weigts) begin

The simulation is also recored, so we can call the optimized specific portfolios (based on return, risk, or Sharpe-ratio) later. It is crucial to record them so we can also extract the weights of optimized portfolios. When locating the Min.Risk, Max.Return and Tangency Portfolios one can also see a preliminary display of key portfolio metrics. It is important to note that at this part of the optimization there are no constraints on the number of assets included in each portfolio. Each portfolios included some positive fraction of EACH asset!

```python
# Simulation with randomly assigned weights
num_portfolios = 10000

results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(len(mean_annual_returns))
    weights /= np.sum(weights)
    weights_record.append(weights)
    
    portfolio_return, portfolio_std_dev, portfolio_sharpe = portfolio_performance(weights, mean_annual_returns, annual_cov_matrix, risk_free_rate)
    
    results[0, i] = portfolio_return
    results[1, i] = portfolio_std_dev
    results[2, i] = portfolio_sharpe

results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])
```


```python
# Locate the optimal portfolios
max_sharpe_idx = results_df['Sharpe Ratio'].idxmax()
max_return_idx = results_df['Return'].idxmax()
min_vol_idx = results_df['Volatility'].idxmin()

max_sharpe_portfolio = results_df.loc[max_sharpe_idx]
min_vol_portfolio = results_df.loc[min_vol_idx]
max_return_portfolio = results_df.loc[max_return_idx]

# max_sharpe_portfolio, min_vol_portfolio, max_return_portfolio
# "Uncomment" these lines to see the preliminary metrics of portfolios! They will be displayed later also.
```

```python
# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier with Optimal Portfolios')

# Highlight optimal portfolios
plt.scatter(min_vol_portfolio[1], min_vol_portfolio[0], color='blue', marker='*', s=200, label='Minimum Variance')
plt.scatter(max_sharpe_portfolio[1], max_sharpe_portfolio[0], color='red', marker='*', s=200, label='Tangency (Max Sharpe)')
plt.scatter(max_return_portfolio[1], max_return_portfolio[0], color='green', marker='*', s=200, label='Max Return')

# Capital Market Line
cml_x = np.linspace(0, max(results_df['Volatility']), 100)
cml_y = risk_free_rate + max_sharpe_portfolio['Sharpe Ratio'] * cml_x
plt.plot(cml_x, cml_y, color='orange', linestyle='--', linewidth=2, label='Capital Market Line (CML)')

plt.legend()
plt.show()
```


    
![png](output_25_0.png)
    

### Extracting the weights and summarising the key metrics of each optimized portfolio

```python
min_vol_weights = weights_record[min_vol_idx]
max_sharpe_weights = weights_record[max_sharpe_idx]
max_return_weights = weights_record[max_return_idx]

min_vol_weights_pct = min_vol_weights * 100
max_sharpe_weights_pct = max_sharpe_weights * 100
max_return_weights_pct = max_return_weights * 100

weights_summary_df = pd.DataFrame({
    'Company': tickers,
    'Min Variance Weight (%)': min_vol_weights_pct,
    'Tangency Weight (%)': max_sharpe_weights_pct,
    'Max Return Weight (%)': max_return_weights_pct
})

weights_summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Min Variance Weight (%)</th>
      <th>Tangency Weight (%)</th>
      <th>Max Return Weight (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AAPL</td>
      <td>0.081583</td>
      <td>0.114214</td>
      <td>4.276596</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSFT</td>
      <td>3.519509</td>
      <td>0.108309</td>
      <td>4.587986</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GOOGL</td>
      <td>1.473942</td>
      <td>0.361988</td>
      <td>1.799842</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AMZN</td>
      <td>3.195749</td>
      <td>2.125384</td>
      <td>4.567230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>META</td>
      <td>1.162981</td>
      <td>0.138945</td>
      <td>3.952132</td>
    </tr>
    <tr>
      <th>5</th>
      <td>BRK-B</td>
      <td>3.812512</td>
      <td>5.133285</td>
      <td>2.922937</td>
    </tr>
    <tr>
      <th>6</th>
      <td>JNJ</td>
      <td>1.104742</td>
      <td>0.986295</td>
      <td>2.596409</td>
    </tr>
    <tr>
      <th>7</th>
      <td>V</td>
      <td>3.263741</td>
      <td>3.243781</td>
      <td>0.479209</td>
    </tr>
    <tr>
      <th>8</th>
      <td>WMT</td>
      <td>0.908422</td>
      <td>4.285134</td>
      <td>3.241203</td>
    </tr>
    <tr>
      <th>9</th>
      <td>JPM</td>
      <td>1.867314</td>
      <td>2.401480</td>
      <td>3.847273</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PG</td>
      <td>4.514170</td>
      <td>0.199187</td>
      <td>3.572482</td>
    </tr>
    <tr>
      <th>11</th>
      <td>UNH</td>
      <td>1.965109</td>
      <td>3.199415</td>
      <td>3.986338</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NVDA</td>
      <td>0.619349</td>
      <td>2.946620</td>
      <td>4.997038</td>
    </tr>
    <tr>
      <th>13</th>
      <td>HD</td>
      <td>0.223122</td>
      <td>1.641726</td>
      <td>4.029570</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DIS</td>
      <td>1.912170</td>
      <td>1.438102</td>
      <td>0.204488</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MA</td>
      <td>3.007580</td>
      <td>4.717937</td>
      <td>2.091831</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PYPL</td>
      <td>0.978925</td>
      <td>0.681884</td>
      <td>4.207684</td>
    </tr>
    <tr>
      <th>17</th>
      <td>VZ</td>
      <td>3.858607</td>
      <td>0.439491</td>
      <td>1.461131</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ADBE</td>
      <td>1.281528</td>
      <td>2.078691</td>
      <td>1.673553</td>
    </tr>
    <tr>
      <th>19</th>
      <td>NFLX</td>
      <td>0.820733</td>
      <td>4.292689</td>
      <td>4.701652</td>
    </tr>
    <tr>
      <th>20</th>
      <td>INTC</td>
      <td>0.102161</td>
      <td>3.059684</td>
      <td>0.438159</td>
    </tr>
    <tr>
      <th>21</th>
      <td>CMCSA</td>
      <td>1.723881</td>
      <td>0.697544</td>
      <td>0.188475</td>
    </tr>
    <tr>
      <th>22</th>
      <td>KO</td>
      <td>0.976331</td>
      <td>1.593449</td>
      <td>0.243436</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PEP</td>
      <td>3.959898</td>
      <td>5.512227</td>
      <td>2.136640</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PFE</td>
      <td>0.326155</td>
      <td>5.402865</td>
      <td>2.487195</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MRK</td>
      <td>4.359233</td>
      <td>5.797818</td>
      <td>3.289597</td>
    </tr>
    <tr>
      <th>26</th>
      <td>T</td>
      <td>4.097115</td>
      <td>0.844299</td>
      <td>0.539624</td>
    </tr>
    <tr>
      <th>27</th>
      <td>CSCO</td>
      <td>2.998862</td>
      <td>0.119579</td>
      <td>2.645292</td>
    </tr>
    <tr>
      <th>28</th>
      <td>XOM</td>
      <td>2.711125</td>
      <td>5.735188</td>
      <td>1.171336</td>
    </tr>
    <tr>
      <th>29</th>
      <td>ABT</td>
      <td>4.001475</td>
      <td>5.254832</td>
      <td>4.987620</td>
    </tr>
    <tr>
      <th>30</th>
      <td>CVX</td>
      <td>4.045198</td>
      <td>2.317689</td>
      <td>0.235607</td>
    </tr>
    <tr>
      <th>31</th>
      <td>NKE</td>
      <td>0.157765</td>
      <td>2.002458</td>
      <td>2.351389</td>
    </tr>
    <tr>
      <th>32</th>
      <td>LLY</td>
      <td>4.486440</td>
      <td>1.515372</td>
      <td>4.955350</td>
    </tr>
    <tr>
      <th>33</th>
      <td>MCD</td>
      <td>4.623078</td>
      <td>5.207922</td>
      <td>1.215102</td>
    </tr>
    <tr>
      <th>34</th>
      <td>DHR</td>
      <td>3.025570</td>
      <td>2.070525</td>
      <td>0.862781</td>
    </tr>
    <tr>
      <th>35</th>
      <td>MDT</td>
      <td>3.757697</td>
      <td>3.808731</td>
      <td>0.465208</td>
    </tr>
    <tr>
      <th>36</th>
      <td>WFC</td>
      <td>3.486465</td>
      <td>0.758551</td>
      <td>1.241402</td>
    </tr>
    <tr>
      <th>37</th>
      <td>BMY</td>
      <td>4.490658</td>
      <td>0.929061</td>
      <td>0.837196</td>
    </tr>
    <tr>
      <th>38</th>
      <td>COST</td>
      <td>3.881946</td>
      <td>4.802853</td>
      <td>3.590950</td>
    </tr>
    <tr>
      <th>39</th>
      <td>TMO</td>
      <td>3.217158</td>
      <td>2.034795</td>
      <td>2.921057</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Portfolio Metrics
min_vol_return = min_vol_portfolio['Return']
min_vol_std_dev = min_vol_portfolio['Volatility']
min_vol_sharpe = min_vol_portfolio['Sharpe Ratio']

max_sharpe_return = max_sharpe_portfolio['Return']
max_sharpe_std_dev = max_sharpe_portfolio['Volatility']
max_sharpe_sharpe = max_sharpe_portfolio['Sharpe Ratio']

max_return_return = max_return_portfolio['Return']
max_return_std_dev = max_return_portfolio['Volatility']
max_return_sharpe = max_return_portfolio['Sharpe Ratio']

portfolio_summary_df = pd.DataFrame({
    'Portfolio': ['Minimum Variance', 'Tangency (Max Sharpe)', 'Maximum Return'],
    'Return': [min_vol_return, max_sharpe_return, max_return_return],
    'Volatility': [min_vol_std_dev, max_sharpe_std_dev, max_return_std_dev],
    'Sharpe Ratio': [min_vol_sharpe, max_sharpe_sharpe, max_return_sharpe]
})

portfolio_summary_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Portfolio</th>
      <th>Return</th>
      <th>Volatility</th>
      <th>Sharpe Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Minimum Variance</td>
      <td>0.164351</td>
      <td>0.107138</td>
      <td>1.254000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tangency (Max Sharpe)</td>
      <td>0.187916</td>
      <td>0.111467</td>
      <td>1.416712</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Maximum Return</td>
      <td>0.244518</td>
      <td>0.196991</td>
      <td>1.088973</td>
    </tr>
  </tbody>
</table>
</div>




```python
indices = {
    'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'NASDAQ': '^IXIC',
    'FTSE 100': '^FTSE', 'DAX': '^GDAXI', 'CAC 40': '^FCHI',
    'Nikkei 225': '^N225', 'Hang Seng': '^HSI', 'Shanghai Composite': '000001.SS',
    'BSE Sensex': '^BSESN', 'ASX 200': '^AXJO', 'KOSPI': '^KS11', 'Taiwan Weighted': '^TWII',
    'Bovespa': '^BVSP', 'TSX Composite': '^GSPTSE', 'Mexican IPC': '^MXX', 'Argentine Merval': '^MERV',
    'South Africa 40': 'JSE.JO', 'Swiss Market Index': '^SSMI', 'AEX': '^AEX', 'IBEX 35': '^IBEX',
    'OMX Stockholm 30': '^OMX', 'OMX Helsinki 25': '^OMXH25', 'BUX': '^BUX', 'PSI 20': '^PSI20',
    'ATX': '^ATX', 'BEL 20': '^BFX', 'FTSE MIB': 'FTSEMIB.MI'
}

# Fetch data and calculate annualized returns and volatility for each index
index_data = {}
for name, ticker in indices.items():
    data = yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close']
    returns = data.pct_change().dropna()  # Calculate daily returns
    annualized_return = (1 + returns.mean())**252 - 1
    annualized_std_dev = returns.std() * np.sqrt(252)
    index_data[name] = {'Return': annualized_return, 'Volatility': annualized_std_dev}

# Convert to DataFrame for plotting
indices_df = pd.DataFrame(index_data).T

# Extract tangency portfolio's return and volatility from previous calculations
tangency_return = max_sharpe_return  # Extracted return from optimization result
tangency_volatility = max_sharpe_std_dev  # Extracted std dev from optimization result

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(indices_df['Volatility'], indices_df['Return'], color='blue', s=50, label="Global Indices")
plt.scatter(tangency_volatility, tangency_return, color='red', s=100, marker='*', label="Tangency Portfolio")

# Adding labels for each index
for name, row in indices_df.iterrows():
    plt.text(row['Volatility'], row['Return'], name, fontsize=8)

# Plot settings
plt.xlabel("Standard Deviation (Volatility)")
plt.ylabel("Annualized Return")
plt.title("Annualized Returns vs. Volatility of Global Indices with Tangency Portfolio")
plt.legend()
plt.grid(True)
plt.show()
```
    
![png](output_28_1.png)
    


## Portfolio Optimization US Stocks with constraint in the number of assets involved 


```python
# List of 40 largest public firms in the US (example tickers, updated for Meta)
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'BRK-B', 'JNJ', 'V', 'WMT', 'JPM',
    'PG', 'UNH', 'NVDA', 'HD', 'DIS', 'MA', 'PYPL', 'VZ', 'ADBE', 'NFLX',
    'INTC', 'CMCSA', 'KO', 'PEP', 'PFE', 'MRK', 'T', 'CSCO', 'XOM', 'ABT',
    'CVX', 'NKE', 'LLY', 'MCD', 'DHR', 'MDT', 'WFC', 'BMY', 'COST', 'TMO'
]

# Fetch data for each firm
data = {}
for ticker in tickers:
    data[ticker] = yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close']

# Combine into a single DataFrame
prices_df = pd.DataFrame(data)

```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed



```python
# Calculate daily returns
daily_returns = prices_df.pct_change().dropna()

# Calculate annualized mean returns and covariance matrix
mean_annual_returns = daily_returns.mean() * 252
annual_cov_matrix = daily_returns.cov() * 252
```


```python
# Portfolio performance calculation
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    returns = np.sum(weights * mean_returns)
    std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = (returns - risk_free_rate) / std_dev
    return returns, std_dev, sharpe_ratio

# Objective functions for each portfolio type
def min_variance(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def max_return(weights, mean_returns):
    return -np.sum(weights * mean_returns)

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.03):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)[2]

# Sparsity constraint (limit number of assets with significant weights)
def sparsity_constraint(weights, max_assets=15):
    return max_assets - np.sum(weights > 0.01)

```


```python
def optimize_portfolio(mean_returns, cov_matrix, objective, max_assets=15, risk_free_rate=0.03):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Set constraints and bounds
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'ineq', 'fun': sparsity_constraint, 'args': (max_assets,)}  # Limit number of assets
    ]
    bounds = tuple((0, 1) for asset in range(num_assets))  # Weights between 0 and 1
    initial_guess = num_assets * [1. / num_assets]

    # Run optimization
    result = minimize(objective, initial_guess, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

```


```python
# Minimum Variance Portfolio
min_var_result = optimize_portfolio(mean_annual_returns, annual_cov_matrix, lambda w, *args: min_variance(w, args[1]), max_assets=15)
min_var_weights = min_var_result.x
min_var_return, min_var_volatility, _ = portfolio_performance(min_var_weights, mean_annual_returns, annual_cov_matrix)

# Maximum Return Portfolio
max_return_result = optimize_portfolio(mean_annual_returns, annual_cov_matrix, lambda w, *args: max_return(w, args[0]), max_assets=15)
max_return_weights = max_return_result.x
max_return, max_volatility, _ = portfolio_performance(max_return_weights, mean_annual_returns, annual_cov_matrix)

# Tangency Portfolio (Max Sharpe Ratio)
tangency_result = optimize_portfolio(mean_annual_returns, annual_cov_matrix, lambda w, *args: negative_sharpe_ratio(w, *args), max_assets=15, risk_free_rate=0.03)
tangency_weights = tangency_result.x
tangency_return, tangency_volatility, tangency_sharpe = portfolio_performance(tangency_weights, mean_annual_returns, annual_cov_matrix)

```


```python
# Define initial guess and bounds for efficient frontier calculation
num_assets = len(mean_annual_returns)
initial_guess = num_assets * [1. / num_assets]
bounds = tuple((0, 1) for asset in range(num_assets))  # Ensure weights between 0 and 1

# Generate target returns for the efficient frontier
target_returns = np.linspace(min_var_return, max_return, 100)
frontier_volatilities = []

for target_return in target_returns:
    # Define constraints for the target return and sparsity constraint
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights must sum to 1
        {'type': 'eq', 'fun': lambda x: np.sum(x * mean_annual_returns) - target_return},  # Target return
        {'type': 'ineq', 'fun': sparsity_constraint, 'args': (15,)}  # Limit number of assets
    ]
    # Minimize variance for each target return
    result = minimize(min_variance, initial_guess, args=(annual_cov_matrix,), method='SLSQP', bounds=bounds, constraints=constraints)
    frontier_volatilities.append(result.fun)


```


```python
# Plot
plt.figure(figsize=(12, 8))
plt.plot(frontier_volatilities, target_returns, 'g--', label="Efficient Frontier")
plt.scatter(min_var_volatility, min_var_return, color='blue', s=100, marker='o', label="Minimum Variance")
plt.scatter(tangency_volatility, tangency_return, color='red', s=100, marker='*', label="Tangency Portfolio")
plt.scatter(max_volatility, max_return, color='green', s=100, marker='^', label="Maximum Return")

# Capital Market Line (CML)
cml_x = np.linspace(0, max(frontier_volatilities), 100)
cml_y = 0.03 + tangency_sharpe * cml_x
plt.plot(cml_x, cml_y, color='orange', linestyle='--', label="Capital Market Line")

plt.xlabel("Standard Deviation (Volatility)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier with Optimal Portfolios and Capital Market Line")
plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_36_0.png)
    



```python
# Generate random portfolios to plot the efficient frontier with color-coded Sharpe Ratios
num_portfolios = 5000
results = np.zeros((3, num_portfolios))  # To store returns, volatility, and Sharpe ratios
weights_record = []

# Generate random portfolios with Dirichlet distribution to encourage diversity
for i in range(int(num_portfolios * 0.7)):  # 70% of portfolios from a Dirichlet distribution
    weights = np.random.dirichlet(np.ones(num_assets), 1)[0]  # Dirichlet sampling
    weights_record.append(weights)
    
    portfolio_return, portfolio_volatility, portfolio_sharpe = portfolio_performance(weights, mean_annual_returns, annual_cov_matrix, risk_free_rate=0.03)
    
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = portfolio_sharpe

# Add portfolios with concentrated weights for extreme values
for i in range(int(num_portfolios * 0.3), num_portfolios):  # 30% of portfolios with concentrated weights
    weights = np.zeros(num_assets)
    selected_assets = np.random.choice(range(num_assets), size=np.random.randint(2, 6), replace=False)
    weights[selected_assets] = np.random.dirichlet(np.ones(len(selected_assets)), 1)[0]  # Allocate only to selected assets
    weights_record.append(weights)
    
    portfolio_return, portfolio_volatility, portfolio_sharpe = portfolio_performance(weights, mean_annual_returns, annual_cov_matrix, risk_free_rate=0.03)
    
    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = portfolio_sharpe

# Convert results to DataFrame
results_df = pd.DataFrame(results.T, columns=['Return', 'Volatility', 'Sharpe Ratio'])

# Create points along the CML (without adding to results_df)
cml_x = np.linspace(0, max(results_df['Volatility']) * 1.5, 100)  # Extend further for visual alignment
cml_y = risk_free_rate + tangency_sharpe * cml_x  # Use tangency Sharpe ratio

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(results_df['Volatility'], results_df['Return'], c=results_df['Sharpe Ratio'], cmap='viridis', marker='o')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.title('Efficient Frontier with Optimal Portfolios')

# Highlight optimal portfolios
plt.scatter(min_var_volatility, min_var_return, color='blue', marker='*', s=200, label='Minimum Variance')
plt.scatter(tangency_volatility, tangency_return, color='red', marker='*', s=200, label='Tangency (Max Sharpe)')
plt.scatter(max_volatility, max_return, color='green', marker='*', s=200, label='Max Return')

# Plot the CML as a separate line
plt.plot(cml_x, cml_y, color='orange', linestyle='--', linewidth=2, label='Capital Market Line (CML)')

plt.legend()
plt.grid(True)
plt.show()
```


    
![png](output_37_0.png)
    



```python
# Convert weights to percentages for readability and filter significant weights
min_var_weights_pct = min_var_weights * 100
tangency_weights_pct = tangency_weights * 100
max_return_weights_pct = max_return_weights * 100

# Create separate DataFrames for each portfolio's non-zero weights
min_var_df = pd.DataFrame({
    'Company': selected_tickers[min_var_weights > 0.01],
    'Min Variance Weight (%)': min_var_weights_pct[min_var_weights > 0.01]
})

tangency_df = pd.DataFrame({
    'Company': selected_tickers[tangency_weights > 0.01],
    'Tangency Weight (%)': tangency_weights_pct[tangency_weights > 0.01]
})

max_return_df = pd.DataFrame({
    'Company': selected_tickers[max_return_weights > 0.01],
    'Max Return Weight (%)': max_return_weights_pct[max_return_weights > 0.01]
})

```


```python
# Merge all portfolios into a single DataFrame, filling NaNs with 0%
weights_summary_df = (
    min_var_df
    .merge(tangency_df, on='Company', how='outer')
    .merge(max_return_df, on='Company', how='outer')
    .fillna(0)  # Replace NaN with 0%
)

# Display the updated DataFrame
weights_summary_df


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Company</th>
      <th>Min Variance Weight (%)</th>
      <th>Tangency Weight (%)</th>
      <th>Max Return Weight (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AMZN</td>
      <td>2.949132</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JNJ</td>
      <td>12.465438</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WMT</td>
      <td>12.898277</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PG</td>
      <td>5.931174</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VZ</td>
      <td>19.753034</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>KO</td>
      <td>11.775471</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PFE</td>
      <td>3.458369</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MRK</td>
      <td>4.626106</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>XOM</td>
      <td>1.663117</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>MCD</td>
      <td>11.188587</td>
      <td>10.488430</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>BMY</td>
      <td>8.488440</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>COST</td>
      <td>4.129106</td>
      <td>17.751814</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>NVDA</td>
      <td>0.000000</td>
      <td>33.088989</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>LLY</td>
      <td>0.000000</td>
      <td>32.051129</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>DHR</td>
      <td>0.000000</td>
      <td>6.619637</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Create summary for overall portfolio metrics
portfolio_summary_df = pd.DataFrame({
    'Portfolio': ['Minimum Variance', 'Tangency (Max Sharpe)', 'Maximum Return'],
    'Return': [min_var_return, tangency_return, max_return],
    'Volatility': [min_var_volatility, tangency_volatility, max_volatility],
    'Sharpe Ratio': [min_var_return / min_var_volatility, tangency_sharpe, max_return / max_volatility]
})

portfolio_summary_df

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Portfolio</th>
      <th>Return</th>
      <th>Volatility</th>
      <th>Sharpe Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Minimum Variance</td>
      <td>0.108223</td>
      <td>0.139680</td>
      <td>0.774793</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tangency (Max Sharpe)</td>
      <td>0.388196</td>
      <td>0.241951</td>
      <td>1.480451</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Maximum Return</td>
      <td>0.664079</td>
      <td>0.490258</td>
      <td>1.354551</td>
    </tr>
  </tbody>
</table>
</div>




```python
tangency_data = portfolio_summary_df[portfolio_summary_df['Portfolio'] == 'Tangency (Max Sharpe)']
tangency_return = tangency_data['Return'].values[0]
tangency_volatility = tangency_data['Volatility'].values[0]

# List of global stock indices with tickers
indices = {
    'S&P 500': '^GSPC', 'Dow Jones': '^DJI', 'NASDAQ': '^IXIC',
    'FTSE 100': '^FTSE', 'DAX': '^GDAXI', 'CAC 40': '^FCHI',
    'Nikkei 225': '^N225', 'Hang Seng': '^HSI', 'Shanghai Composite': '000001.SS',
    'BSE Sensex': '^BSESN', 'ASX 200': '^AXJO', 'KOSPI': '^KS11', 'Taiwan Weighted': '^TWII',
    'Bovespa': '^BVSP', 'TSX Composite': '^GSPTSE', 'Mexican IPC': '^MXX', 'Argentine Merval': '^MERV',
    'South Africa 40': 'JSE.JO', 'Swiss Market Index': '^SSMI', 'AEX': '^AEX', 'IBEX 35': '^IBEX',
    'OMX Stockholm 30': '^OMX', 'OMX Helsinki 25': '^OMXH25', 'BUX': '^BUX', 'PSI 20': '^PSI20',
    'ATX': '^ATX', 'BEL 20': '^BFX', 'FTSE MIB': 'FTSEMIB.MI'
}

# Fetching data and calculating annualized returns and volatility for each index
index_data = {}
for name, ticker in indices.items():
    data = yf.download(ticker, start='2014-01-01', end='2024-01-01')['Adj Close']
    returns = data.pct_change().dropna()  # Calculate daily returns
    annualized_return = (1 + returns.mean())**252 - 1
    annualized_std_dev = returns.std() * np.sqrt(252)
    index_data[name] = {'Return': annualized_return, 'Volatility': annualized_std_dev}

# Convert to DataFrame for plotting
indices_df = pd.DataFrame(index_data).T

# Plotting
plt.figure(figsize=(12, 8))
plt.scatter(indices_df['Volatility'], indices_df['Return'], color='blue', s=50, label="Global Indices")
plt.scatter(tangency_volatility, tangency_return, color='red', s=100, marker='*', label="Tangency Portfolio")

# Adding labels for each index
for name, row in indices_df.iterrows():
    plt.text(row['Volatility'], row['Return'], name, fontsize=8)

# Plot settings
plt.xlabel("Standard Deviation (Volatility)")
plt.ylabel("Annualized Return")
plt.title("Annualized Returns vs. Volatility of Global Indices with Tangency Portfolio")
plt.legend()
plt.grid(True)
plt.show()


```

    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed
    [*********************100%***********************]  1 of 1 completed



    
![png](output_41_1.png)
    



```python

```

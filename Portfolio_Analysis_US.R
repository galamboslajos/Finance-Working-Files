###### Efficient Portfolio Optimization in R ######

# Load required packages
library(quantmod)
library(tidyverse)
library(PerformanceAnalytics)
library(quadprog)
library(lubridate)
library(knitr)
library(kableExtra)

# Define stock tickers for the largest companies and the S&P 500 index
# AAPL  - Apple Inc.
# MSFT  - Microsoft Corporation
# GOOGL - Alphabet Inc. (Class A shares, Google’s parent company)
# AMZN  - Amazon.com, Inc.
# NVDA  - NVIDIA Corporation
# TSLA  - Tesla, Inc.
# BRK-B - Berkshire Hathaway Inc. (Class B shares)
# META  - Meta Platforms, Inc. (formerly Facebook, Inc.)
# JPM   - JPMorgan Chase & Co.
# V     - Visa Inc.
# ^GSPC - S&P 500 Index (SPX)

# Define stock tickers for the largest companies and the S&P 500 index
tickers <- c("AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", 
             "TSLA", "BRK-B", "META", "JPM", "V", "^GSPC")

# Set date range: from 10 years ago to yesterday
start_date <- Sys.Date() - years(10)
end_date <- Sys.Date() - 1

# Initialize an empty list to store stock data
stock_data <- list()

# Loop through each ticker to download price data
for (ticker in tickers) {
  # Get stock data
  stock_prices <- getSymbols(ticker, src = "yahoo", 
                             from = start_date, 
                             to = end_date, 
                             auto.assign = FALSE)
  
  # Store the adjusted closing prices in the list with the ticker as the name
  stock_data[[ticker]] <- Ad(stock_prices)
}

# Combine all data into a single data frame
combined_data <- do.call(merge, stock_data)

# Rename columns for easier reference
colnames(combined_data) <- tickers

# Convert to data frame and add a date column
combined_data_df <- data.frame(date = index(combined_data), coredata(combined_data))

# Calculate daily returns for each stock and the S&P 500 index
returns_df <- combined_data_df %>%
  mutate(across(-date, ~ (./lag(.) - 1), .names = "return_{col}")) %>%
  select(date, starts_with("return_"))

# Convert returns_df to an xts object for easier manipulation
returns_xts <- xts(returns_df[-1], order.by = as.Date(returns_df$date))

# Calculate annualized returns from daily returns
annual_returns <- returns_df %>%
  group_by(year = lubridate::year(date)) %>%
  summarize(across(starts_with("return_"), ~ prod(1 + ., na.rm = TRUE) - 1))

# Drop the 'year' column to work with just the returns data
mean_annual_returns <- colMeans(annual_returns[-1], na.rm = TRUE)

# Calculate the daily covariance matrix
daily_cov_matrix <- cov(returns_xts, use = "complete.obs")

# Annualize the covariance matrix by multiplying by 252 (trading days)
annual_cov_matrix <- daily_cov_matrix * 252

# Function to calculate portfolio returns and risk
portfolio_stats <- function(weights, mean_returns, cov_matrix) {
  port_return <- sum(weights * mean_returns)
  port_risk <- sqrt(t(weights) %*% cov_matrix %*% weights)
  c(port_return, port_risk)
}

# Function for minimum risk portfolio
min_risk_portfolio <- function(cov_matrix) {
  n <- ncol(cov_matrix)
  dvec <- rep(0, n)
  Amat <- cbind(1, diag(n))  # Constraint: sum of weights = 1 and no short selling
  bvec <- c(1, rep(0, n))
  solve.QP(Dmat = cov_matrix, dvec = dvec, Amat = Amat, bvec = bvec, meq = 1)$solution
}

# Define risk-free rate (assume a 4% risk-free rate for example)
risk_free_rate <- 0.04

# Find the minimum risk portfolio
w_min_risk <- min_risk_portfolio(annual_cov_matrix)
min_risk_stats <- portfolio_stats(w_min_risk, mean_annual_returns, annual_cov_matrix)

# Create a sequence of target returns for the efficient frontier
target_returns <- seq(min(mean_annual_returns), max(mean_annual_returns), length.out = 100)
efficient_frontier <- sapply(target_returns, function(tr) {
  Amat <- cbind(1, mean_annual_returns)
  bvec <- c(1, tr)
  result <- solve.QP(Dmat = annual_cov_matrix, dvec = rep(0, ncol(annual_cov_matrix)), Amat = Amat, bvec = bvec, meq = 2)
  portfolio_stats(result$solution, mean_annual_returns, annual_cov_matrix)
})

# Convert to data frame for plotting
frontier_df <- data.frame(Risk = efficient_frontier[2, ], Return = efficient_frontier[1, ])

# Compute the Sharpe ratio for each point on the efficient frontier relative to the risk-free rate
sharpe_ratios <- (frontier_df$Return - risk_free_rate) / frontier_df$Risk

# Find the index of the maximum Sharpe ratio, which gives the tangency point
max_sharpe_idx <- which.max(sharpe_ratios)
tangency_return <- frontier_df$Return[max_sharpe_idx]
tangency_risk <- frontier_df$Risk[max_sharpe_idx]

# Find the corresponding weights for the tangency portfolio using the tangency return
Amat <- cbind(1, mean_annual_returns)
bvec <- c(1, tangency_return)
tangency_solution <- solve.QP(Dmat = annual_cov_matrix, dvec = rep(0, ncol(annual_cov_matrix)), Amat = Amat, bvec = bvec, meq = 2)
w_tangency <- tangency_solution$solution
names(w_tangency) <- gsub("return_", "", names(mean_annual_returns))

# Find the maximum return portfolio
w_max_return <- max_return_portfolio(mean_annual_returns)
max_return_stats <- portfolio_stats(w_max_return, mean_annual_returns, annual_cov_matrix)

# S&P 500 is "^GSPC"
spx_return <- mean_annual_returns["return_X.GSPC"]
spx_risk <- sqrt(annual_cov_matrix["return_X.GSPC", "return_X.GSPC"])

# Calculate individual asset statistics (annualized)
individual_asset_stats <- data.frame(
  Stock = gsub("return_", "", names(mean_annual_returns)),
  Return = mean_annual_returns,
  Risk = sqrt(diag(annual_cov_matrix))
)

# Plot the efficient frontier and highlight key portfolios, SPX, and individual assets
ggplot(frontier_df, aes(x = Risk, y = Return)) +
  geom_line(color = "blue") +
  geom_point(aes(x = min_risk_stats[2], y = min_risk_stats[1]), color = "red", size = 3, label = "Min Risk") +
  geom_point(aes(x = tangency_risk, y = tangency_return), color = "green", size = 3, label = "Tangency") +
  geom_abline(intercept = risk_free_rate, slope = sharpe_ratios[max_sharpe_idx], color = "darkgreen", linetype = "dashed") +
  geom_point(aes(x = max_return_stats[2], y = max_return_stats[1]), color = "purple", size = 3, label = "Max Return") +
  geom_point(aes(x = spx_risk, y = spx_return), color = "orange", size = 3, label = "SPX") +
  geom_point(data = individual_asset_stats, aes(x = Risk, y = Return), color = "black", size = 2) +
  geom_text(data = individual_asset_stats, aes(x = Risk, y = Return, label = Stock), hjust = -0.1, vjust = -0.5, size = 3) +
  labs(title = "Efficient Frontier with Tangency Line, SPX, and Individual Stocks",
       x = "Portfolio Risk (Annualized Standard Deviation)",
       y = "Portfolio Return (Annualized)") +
  theme_minimal() +
  geom_text(aes(x = min_risk_stats[2], y = min_risk_stats[1], label = "Min Risk"), hjust = -0.1) +
  geom_text(aes(x = tangency_risk, y = tangency_return, label = "Tangency"), hjust = -0.1) +
  geom_text(aes(x = max_return_stats[2], y = max_return_stats[1], label = "Max Return"), hjust = -0.1) +
  geom_text(aes(x = spx_risk, y = spx_return, label = "SPX"), hjust = -0.1)


# Print the key portfolio statistics
print("Minimum Risk Portfolio:")
print(min_risk_stats)
print(w_min_risk)

print("Tangency Portfolio:")
print(tangency_stats)
print(w_tangency)

print("Maximum Return Portfolio:")
print(max_return_stats)
print(w_max_return)

# Ensure weights have correct names corresponding to assets
asset_names <- gsub("return_", "", names(mean_annual_returns))

# Assign the names to the weights if not already present
names(w_min_risk) <- asset_names
names(w_tangency) <- asset_names
names(w_max_return) <- asset_names

# Create a data frame for the weights of each key portfolio
weights_df <- data.frame(
  Asset = asset_names,
  `Min Risk Weights` = round(w_min_risk, 4),
  `Tangency Weights` = round(w_tangency, 4),
  `Max Return Weights` = round(w_max_return, 4)
)

# Display the table
print(weights_df)

# Print the individual asset statistics
print("Individual Asset Statistics:")
print(individual_asset_stats)


# Create a data frame for the key portfolio statistics
key_portfolio_stats <- data.frame(
  Portfolio = c("Minimum Risk", "Tangency", "Maximum Return"),
  Return = c(min_risk_stats[1], tangency_stats[1], max_return_stats[1]),
  Risk = c(min_risk_stats[2], tangency_stats[2], max_return_stats[2])
)

# Create a data frame for the weights of each key portfolio
weights_df <- data.frame(
  Asset = gsub("return_", "", names(w_min_risk)),
  `Min Risk Weights` = round(w_min_risk, 4),
  `Tangency Weights` = round(w_tangency, 4),
  `Max Return Weights` = round(w_max_return, 4)
)

# Create a table for individual asset statistics
individual_asset_stats_table <- individual_asset_stats %>%
  rename(
    `Asset` = Stock,
    `Annualized Return` = Return,
    `Annualized Risk` = Risk
  )

# Combine the tables into a single HTML output
key_portfolio_stats %>%
  kable("html", caption = "Key Portfolio Statistics") %>%
  kable_styling(full_width = FALSE) %>%
  save_kable("key_portfolio_statistics.html")

weights_df %>%
  kable("html", caption = "Weights of Each Portfolio") %>%
  kable_styling(full_width = FALSE) %>%
  save_kable("portfolio_weights.html")

individual_asset_stats_table %>%
  kable("html", caption = "Individual Asset Statistics") %>%
  kable_styling(full_width = FALSE) %>%
  save_kable("individual_asset_statistics.html")







# Prediction_Automated
Lajos Galambos

# Description

In a [previous
file](https://github.com/galamboslajos/Finance-Working-Files/blob/main/Investment_2025/Index_Modelling.md),
we tried out multiple methods to come up with precise prediction results
for Nasdaq 100 daily returns (direction: +/-). This time we want to
implement those results form the most accurate Logistic Regression model
to a more automated process. We will work with code that feeds the model
with new data and returns the predictions.

    Recent NASDAQ Closing Prices:

                     Date IXIC.Close
    2025-04-23 2025-04-23   16708.05
    2025-04-24 2025-04-24   17166.04
    2025-04-25 2025-04-25   17382.94
    2025-04-28 2025-04-28   17366.13
    2025-04-29 2025-04-29   17461.32


    === DAILY NASDAQ SIGNAL ===

    Prediction is for market day: 2025-04-30 

    Based on data available up to: 2025-04-29 

    Today's predicted probability of positive return: 92.86 %

    Today's trading signal: LONG (Buy)


    Features used for today's prediction:

            date       r_lag1     r_lag2     r_lag3  abs_r_lag1 roll_mean5
    1 2025-04-29 -0.000967427 0.01255628 0.02704227 0.000967427 0.01375946
        roll_sd5
    1 0.01207507

    [1] "2025-04-25" "2025-04-28" "2025-04-29"


    === SIGNAL PERFORMANCE TRACKER ===

    Total predictions: 1 

    Correct predictions: 1 

    Accuracy: 100 %

            date predicted actual    outcome
    1 2025-04-29         1      1 🟢 Correct

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


    ### Sample of Feature Set:



    |   |date       |     return|     r_lag1|     r_lag2|     r_lag3| abs_r_lag1| roll_mean5|  roll_sd5| target|
    |:--|:----------|----------:|----------:|----------:|----------:|----------:|----------:|---------:|------:|
    |5  |2015-05-08 |  0.0116615|  0.0052508| -0.0039943| -0.0155885|  0.0052508| -0.0005341| 0.0102602|      1|
    |6  |2015-05-11 | -0.0019966|  0.0116615|  0.0052508| -0.0039943|  0.0116615| -0.0009334| 0.0102731|      0|
    |7  |2015-05-12 | -0.0034865| -0.0019966|  0.0116615|  0.0052508|  0.0019966|  0.0014870| 0.0067934|      0|
    |8  |2015-05-13 |  0.0011047| -0.0034865| -0.0019966|  0.0116615|  0.0034865|  0.0025068| 0.0061136|      1|
    |9  |2015-05-14 |  0.0137774|  0.0011047| -0.0034865| -0.0019966|  0.0011047|  0.0042121| 0.0079759|      1|


    ### Logistic Regression Model Summary:

    ```

    Call:
    glm(formula = target ~ ., family = binomial(link = "logit"), 
        data = model_data)

    Coefficients:
                 Estimate Std. Error z value Pr(>|z|)    
    (Intercept)  -0.01002    0.09780  -0.102  0.91840    
    r_lag1      -93.93157    6.66082 -14.102  < 2e-16 ***
    r_lag2      -86.13001    6.55000 -13.150  < 2e-16 ***
    r_lag3      -88.10004    6.43672 -13.687  < 2e-16 ***
    abs_r_lag1  -15.84373    7.85910  -2.016  0.04380 *  
    roll_mean5  453.87832   24.84774  18.266  < 2e-16 ***
    roll_sd5     29.36321   10.56513   2.779  0.00545 ** 
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1

    (Dispersion parameter for binomial family taken to be 1)

        Null deviance: 2652.0  on 1926  degrees of freedom
    Residual deviance: 2045.6  on 1920  degrees of freedom
    AIC: 2059.6

    Number of Fisher Scoring iterations: 5

    ```

Coefficients:

r_lag1, r_lag2, and r_lag3 have highly significant negative coefficients
(p-values \< 2e-16), indicating that lagged returns are strong
predictors of the target variable. roll_mean5 has a highly significant
positive coefficient, suggesting that the 5-day rolling mean is also a
strong predictor. abs_r_lag1 and roll_sd5 are not statistically
significant (p-values \> 0.05), meaning they may not contribute much to
the model.

Variables with \*\*\* are highly significant, meaning they strongly
influence the target variable.


    ### Confusion Matrix (Training Data):



    |   |   0|   1|
    |:--|---:|---:|
    |0  | 582| 181|
    |1  | 285| 879|


    ### Accuracy (Training Data):

     Accuracy 
    0.7581733 


    ### Confusion Matrix (Testing Data):



    |   |   0|   1|
    |:--|---:|---:|
    |0  | 166|  58|
    |1  |  88| 270|


    ### Accuracy (Testing Data):

     Accuracy 
    0.7491409 


    ### AUC (Training Data):

    Area under the curve: 0.8151


    ### AUC (Testing Data):

    Area under the curve: 0.811

![](Prediction_Automated.markdown_strict_files/figure-markdown_strict/unnamed-chunk-2-1.png)


    ### Root Mean Squared Error (RMSE) - Training Data:

    [1] 0.4184326


    ### Root Mean Squared Error (RMSE) - Testing Data:

    [1] 0.4182009


    ### Akaike Information Criterion (AIC):

    [1] 2059.573


    ### Feature Importance:

![](Prediction_Automated.markdown_strict_files/figure-markdown_strict/unnamed-chunk-2-2.png)

    ### Trading Suggestion for Today:

    - Date: 2025-04-30

    - Probability market goes UP: 99.27%

    - Suggested action: **BUY NASDAQ**

    ### Features Used for Today’s Prediction:



    |    r_lag1|     r_lag2|    r_lag3| abs_r_lag1| roll_mean5|  roll_sd5|
    |---------:|----------:|---------:|----------:|----------:|---------:|
    | 0.0054664| -0.0009674| 0.0125563|  0.0054664|  0.0137595| 0.0120751|


    ### Last 5 NASDAQ Closing Prices:



    |Date       |    Close|
    |:----------|--------:|
    |2025-04-23 | 16708.05|
    |2025-04-24 | 17166.04|
    |2025-04-25 | 17382.94|
    |2025-04-28 | 17366.13|
    |2025-04-29 | 17461.32|


    ### Model Sanity Check (Yesterday):

    - Return on 2025-04-29 was: 0.0055 → Market actually went UP

    - Model suggests **BUY (1)** today based on that.

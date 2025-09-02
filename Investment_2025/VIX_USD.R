

########## PLOT 1: # Daily VIX & DXY, Jan 2025–Today ##########
# install.packages(c("quantmod","zoo"))  # if needed
library(quantmod)
library(zoo)

# 1. Date range
start_date <- "2025-01-01"
end_date   <- Sys.Date()

# 2. Fetch both series (no auto.assign)
vix <- getSymbols("^VIX",   src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "VIX.Close"]
dxy <- getSymbols("DX-Y.NYB", src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "DX-Y.NYB.Close"]

# 3. Merge into one xts; this will introduce NA where either series is missing
both <- merge(vix, dxy)
colnames(both) <- c("VIX", "DXY")

# 4. Fill or interpolate NAs. Two common options:
#    a) Last observation carried forward:
both_locf <- na.locf(both)

#    b) Linear interpolation:
both_interp <- na.approx(both)

# — pick one for plotting, e.g. linear interpolation:
data <- both_interp

# 5. Plot
par(mar = c(5, 4, 4, 4) + 0.3)

# 5. Plot VIX on the left
plot(index(data), coredata(data[, "VIX"]),
     type="l", col="red",
     xlab="Date", ylab="VIX",
     main="Daily VIX (left) & DXY (right), Jan 2025–Today")

# 6. Overlay DXY on the same panel
par(new = TRUE)
plot(index(data), coredata(data[, "DXY"]),
     type="l", col="blue",
     axes=FALSE, xlab="", ylab="")

# 7. Right‑hand axis for DXY
axis(side = 4)
mtext("DXY", side = 4, line = 3)

# 8. Add dashed vertical “liberation day” line
liberation_date <- as.Date("2025-04-02")  # ← put your actual date here
abline(v = liberation_date,
       lty = 2,        # dashed
       lwd = 1,        # line width
       col = "darkgray")
mtext("Liberation Day", 
      side = 3, 
      at   = liberation_date, 
      line = -1, 
      cex  = 0.8, 
      col  = "black")

# 9. Legend
legend("topright",
       legend = c("VIX", "DXY", "Liberation Day"),
       col    = c("red", "blue", "darkgray"),
       lty    = c(1,      1,       2),
       bty    = "n")
####################################################

########### PLOT 2: # DXY vs. VIX, Rolling correlation ###########
# install.packages(c("quantmod","zoo"))  # if needed
library(quantmod)
library(zoo)

# 1. Date range
start_date <- "2022-01-01"
end_date   <- Sys.Date()

# 2. Fetch both series (no auto.assign)
vix <- getSymbols("^VIX",     src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "VIX.Close"]
dxy <- getSymbols("DX-Y.NYB", src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "DX-Y.NYB.Close"]

# 3. Merge, interpolate NAs
both <- merge(vix, dxy)
colnames(both) <- c("VIX", "DXY")
both <- na.locf(both)       # carry‑forward
both <- na.approx(both)     # then linear interp

# 4. Compute 30‑day rolling correlation
#    align="right" so that corr[t] uses the last 30 obs up to t
roll_corr <- rollapply(
  data = both,
  width = 30,
  FUN   = function(x) cor(x[, "DXY"], x[, "VIX"]),
  by.column = FALSE,
  align = "right",
  fill  = NA
)

# 5. Liberation Day
liberation_date <- as.Date("2025-04-02")

# 6. Set up 2‑row layout and margins
par(mfrow = c(2,1),
    mar   = c(4,4,2,4) + 0.3)

# —— Top plot: VIX & DXY —— 
plot(index(both), both$VIX,
     type = "l", col = "red",
     xlab = "", ylab = "VIX",
     main = "VIX (left) & DXY (right), Jan 2022–Today")
par(new = TRUE)
plot(index(both), both$DXY,
     type = "l", col = "blue",
     axes = FALSE, xlab = "", ylab = "")
axis(side = 4)
mtext("DXY", side = 4, line = 3)
abline(v = liberation_date, lty = 2, col = "darkgray")
mtext("Liberation Day", side = 3,
      at = liberation_date, line = -1, cex = 0.8)
legend("topright",
       legend = c("VIX", "DXY", "Liberation Day"),
       col    = c("red", "blue", "darkgray"),
       lty    = c(1,      1,      2),
       bty    = "n")

# —— Bottom plot: 30‑day rolling correlation —— 
plot(index(roll_corr), roll_corr,
     type = "l", lwd = 1.2,
     xlab = "Date", ylab = "Corr(VIX, DXY, 30d)",
     main = "30‑Day Rolling Correlation")
abline(h = 0, col = "gray60", lty = 3)  # zero line for ref
abline(v = liberation_date, lty = 2, col = "darkgray")
mtext("Liberation Day", side = 3,
      at = liberation_date, line = -1, cex = 0.8)


######### Running the analysis #########

# install.packages(c("quantmod","zoo"))  # if needed
library(quantmod)
library(zoo)

# 1. Date range
start_date <- "2021-01-01"
end_date   <- Sys.Date()

# 2. Download DXY & VIX
dxy <- getSymbols("DX-Y.NYB", src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "DX-Y.NYB.Close"]
vix <- getSymbols("^VIX",    src="yahoo",
                  from=start_date, to=end_date,
                  auto.assign=FALSE)[, "VIX.Close"]

# 3. FRED tickers for 2‑yr & 10‑yr government yields
#    US: DGS2 (2y), DGS10 (10y)
#    Foreign: replace the placeholders with the correct FRED series
foreign_2y <- c(
  EUR = "IR2TIB01EZM156N",  # ← example: 2‑yr German gov’t yield
  JPY = "IR2TIB01JPM156N",  # ← example: 2‑yr Japan gov’t yield
  GBP = "IR2TIB01GBM156N",  # ← example: 2‑yr UK gov’t yield
  CAD = "IR2TIB01CAM156N",  # ← example: 2‑yr Canada gov’t yield
  CHF = "IR2TIB01CHM156N",  # ← example: 2‑yr Swiss gov’t yield
  SEK = "IR2TIB01SEM156N"   # ← example: 2‑yr Sweden gov’t yield
)
foreign_10y <- gsub("2", "10", foreign_2y)  # assume same naming

# 4. Pull yields from FRED
symbols <- c(
  US2  = "DGS2",
  US10 = "DGS10",
  foreign_2y,
  foreign_10y
)
getSymbols(symbols, src="FRED",
           from=start_date, to=end_date)

# 5. Extract and merge into one xts
us2  <- DGS2
us10 <- DGS10
f2   <- do.call( merge, lapply(foreign_2y,  function(x) get(x)) )
f10  <- do.call( merge, lapply(foreign_10y, function(x) get(x)) )
colnames(f2)  <- names(foreign_2y)
colnames(f10) <- names(foreign_10y)

all.yields <- merge(us2, us10, f2, f10)

# 6. DXY weights
w <- c(EUR=0.576, JPY=0.136, GBP=0.119,
       CAD=0.091, CHF=0.036, SEK=0.042)

# 7. Compute diffs on each date
# 7a. Weighted foreign 2y & 10–2 slope
f2_w    <- f2 %*% w
slope_us    <- us10 - us2
slope_foreign <- (f10 - f2) %*% w

# 7b. Diff1 & Diff2
diff1 <- us2 - f2_w
diff2 <- slope_us - slope_foreign

# 8. Merge everything and fill gaps
dat <- merge(dxy, vix, diff1, diff2)
colnames(dat) <- c("DXY", "VIX", "Diff1", "Diff2")

# fill forward then linear interpolate
dat <- na.locf(dat)
dat <- na.approx(dat)

# 9. Build the regression data.frame
df <- data.frame(
  date = index(dat),
  logDXY = log(dat$DXY * 100),
  Diff1  = coredata(dat$Diff1),
  Diff2  = coredata(dat$Diff2),
  VIX    = coredata(dat$VIX)
)

# 10. Run the OLS
fit <- lm(logDXY ~ Diff1 + Diff2 + VIX, data = df)
summary(fit)



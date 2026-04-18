"""
FIN F414 - Financial Risk Analytics and Management
Project: Volatility, Liquidity, and Derivatives Risk Analysis
=============================================================
Covers:
  Part A  - Market Data, Returns & Liquidity Analysis      (Step 5 & 6)
  Part B  - Option Pricing & Volatility Inputs             (Step 7 & 8)
  Part C  - Greeks, Portfolio Construction & Hedging       (Step 7)
  Part D  - Risk Measurement & Stress Analysis             (Step 9)

Run each section independently or top-to-bottom in a Jupyter notebook.
"""

# -----------------------------------------------------------------------------
# FIX 1: Force UTF-8 output on Windows terminals to avoid UnicodeEncodeError
# -----------------------------------------------------------------------------
import sys
import io

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# -----------------------------------------------------------------------------
# DEPENDENCIES
# pip install yfinance pandas numpy scipy matplotlib seaborn arch
# -----------------------------------------------------------------------------

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta

# Optional - GARCH (Step 8)
try:
    from arch import arch_model
    ARCH_AVAILABLE = True
except ImportError:
    ARCH_AVAILABLE = False
    print("WARNING: 'arch' library not found.  GARCH section will be skipped.")
    print("Install with:  pip install arch")

# --- Plotting defaults -------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
COLORS = ["#2563EB", "#DC2626", "#16A34A", "#D97706"]


# =============================================================================
# PART A  -  STEP 5 & 6: MARKET DATA, RETURNS & LIQUIDITY
# =============================================================================

# -----------------------------------------------------------------------------
# A-1  Download NIFTY 50 data and rank by average daily turnover
# -----------------------------------------------------------------------------

# Full NIFTY 50 tickers (NSE suffix for yfinance)
NIFTY50_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "POWERGRID.NS", "TECHM.NS", "NTPC.NS", "ONGC.NS",
    "M&M.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "BAJAJFINSV.NS",
    "GRASIM.NS", "BPCL.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "INDUSINDBK.NS", "SBILIFE.NS", "HDFCLIFE.NS", "BRITANNIA.NS",
    "BAJAJ-AUTO.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "LTIM.NS",
    "UPL.NS", "SHRIRAMFIN.NS",
]

END_DATE   = datetime.today()
START_DATE = END_DATE - timedelta(days=182)    # ~6 months

print(f"Downloading NIFTY 50 data from {START_DATE.date()} to {END_DATE.date()} ...")
raw = yf.download(
    NIFTY50_TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False,
)

close_all  = raw["Close"].dropna(axis=1, how="all")
volume_all = raw["Volume"].dropna(axis=1, how="all")
common     = close_all.columns.intersection(volume_all.columns)

# -----------------------------------------------------------------------------
# FIX 2: Drop any ticker with fewer than 60 valid rows (delisted / 404 error)
# -----------------------------------------------------------------------------
MIN_ROWS = 60
valid_tickers = [
    t for t in common
    if close_all[t].dropna().shape[0] >= MIN_ROWS
    and volume_all[t].dropna().shape[0] >= MIN_ROWS
]

close_all  = close_all[valid_tickers]
volume_all = volume_all[valid_tickers]

print(f"  Valid tickers after filtering: {len(valid_tickers)} / {len(NIFTY50_TICKERS)}")

# Turnover = Close x Volume  (proxy for INR daily traded value)
turnover_all = (close_all * volume_all).mean()    # avg daily turnover per stock

q75 = turnover_all.quantile(0.75)
q25 = turnover_all.quantile(0.25)

liquid_candidates   = turnover_all[turnover_all >= q75].sort_values(ascending=False)
illiquid_candidates = turnover_all[turnover_all <= q25].sort_values()


def pick_stock(candidates, close_df, min_rows=MIN_ROWS):
    """Return the first ticker in candidates that has enough valid rows."""
    for ticker in candidates.index:
        series = close_df[ticker].dropna()
        if len(series) >= min_rows:
            return ticker
    return candidates.index[0]


LIQUID_TICKER   = pick_stock(liquid_candidates,   close_all)
ILLIQUID_TICKER = pick_stock(illiquid_candidates, close_all)

# FIX 3: Use "Rs." instead of the rupee symbol to avoid cp1252 encoding errors
print(
    f"\n  Liquid   stock selected : {LIQUID_TICKER}"
    f"  (avg daily turnover ~ Rs. {liquid_candidates[LIQUID_TICKER] / 1e7:.1f} Cr)"
)
print(
    f"  Illiquid stock selected : {ILLIQUID_TICKER}"
    f"  (avg daily turnover ~ Rs. {illiquid_candidates[ILLIQUID_TICKER] / 1e7:.1f} Cr)"
)


# -----------------------------------------------------------------------------
# A-2  Build per-stock DataFrames
# -----------------------------------------------------------------------------

def build_stock_df(ticker, close_df, volume_df):
    """Return a DataFrame with returns, volatility, and liquidity columns."""
    df = pd.DataFrame({
        "Close" : close_df[ticker],
        "Volume": volume_df[ticker],
    }).dropna()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)

    # Step 5: Returns & Volatility
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))
    df["RolVol_20d"] = df["Log_Return"].rolling(20).std() * np.sqrt(252)

    # Step 6: Liquidity
    df["Turnover_INR"]   = df["Close"] * df["Volume"]
    df["Turnover_Ratio"] = (
        df["Turnover_INR"] / df["Turnover_INR"].rolling(60).mean()
    )

    # Amihud (2002) illiquidity = |R| / Volume_INR
    df["Amihud"] = (
        df["Log_Return"].abs() / df["Turnover_INR"]
    ).replace([np.inf, -np.inf], np.nan)
    df["Amihud_MA"] = df["Amihud"].rolling(20).mean()

    return df.dropna()


df_liquid   = build_stock_df(LIQUID_TICKER,   close_all, volume_all)
df_illiquid = build_stock_df(ILLIQUID_TICKER, close_all, volume_all)


# -----------------------------------------------------------------------------
# A-3  Step 5: Summary statistics
# -----------------------------------------------------------------------------

def summary_stats(df, name):
    r = df["Log_Return"].dropna()
    return pd.Series({
        "Stock"             : name,
        "N (trading days)"  : len(r),
        "Mean Return"       : r.mean(),
        "Std Dev"           : r.std(),
        "Skewness"          : stats.skew(r),
        "Kurtosis (excess)" : stats.kurtosis(r),
        "Min"               : r.min(),
        "Max"               : r.max(),
        "Ann. Vol (hist)"   : r.std() * np.sqrt(252),
        "Avg Turnover (Cr)" : df["Turnover_INR"].mean() / 1e7,
        "Avg Amihud"        : df["Amihud"].mean(),
    })


summary = pd.DataFrame([
    summary_stats(df_liquid,   LIQUID_TICKER),
    summary_stats(df_illiquid, ILLIQUID_TICKER),
]).set_index("Stock").T

print("\n" + "=" * 60)
print("PART A  -  SUMMARY STATISTICS")
print("=" * 60)
print(summary.to_string())


# -----------------------------------------------------------------------------
# A-4  Plots: Closing Price, Returns, Volatility, Amihud (Steps 5 & 6)
# -----------------------------------------------------------------------------

fig = plt.figure(figsize=(16, 14))
gs  = gridspec.GridSpec(4, 2, hspace=0.45, wspace=0.35)

# Row 0 - Closing prices
for col, (df_plot, ticker, color) in enumerate(zip(
        [df_liquid, df_illiquid],
        [LIQUID_TICKER, ILLIQUID_TICKER],
        [COLORS[0], COLORS[1]])):
    ax = fig.add_subplot(gs[0, col])
    ax.plot(df_plot.index, df_plot["Close"], color=color, linewidth=1.2)
    ax.set_title(f"{ticker} - Closing Price", fontweight="bold")
    ax.set_ylabel("Price (Rs.)")
    ax.tick_params(axis="x", rotation=30)

# Row 1 - Log returns
for col, (df_plot, ticker, color) in enumerate(zip(
        [df_liquid, df_illiquid],
        [LIQUID_TICKER, ILLIQUID_TICKER],
        [COLORS[0], COLORS[1]])):
    ax = fig.add_subplot(gs[1, col])
    ax.plot(df_plot.index, df_plot["Log_Return"],
            color=color, alpha=0.7, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_title(f"{ticker} - Daily Log Returns", fontweight="bold")
    ax.set_ylabel("Log Return")
    ax.tick_params(axis="x", rotation=30)

# Row 2 - 20-day rolling realised volatility
ax_vol = fig.add_subplot(gs[2, :])
ax_vol.plot(
    df_liquid.index, df_liquid["RolVol_20d"],
    label=f"{LIQUID_TICKER} (Liquid)", color=COLORS[0], linewidth=1.4
)
ax_vol.plot(
    df_illiquid.index, df_illiquid["RolVol_20d"],
    label=f"{ILLIQUID_TICKER} (Illiquid)", color=COLORS[1], linewidth=1.4
)
ax_vol.set_title("20-Day Rolling Realised Volatility (annualised)", fontweight="bold")
ax_vol.set_ylabel("Volatility")
ax_vol.legend()
ax_vol.tick_params(axis="x", rotation=30)

# Row 3 - Amihud illiquidity
ax_ami = fig.add_subplot(gs[3, :])
ax_ami.plot(
    df_liquid.index, df_liquid["Amihud_MA"],
    label=f"{LIQUID_TICKER} (Liquid)", color=COLORS[0], linewidth=1.4
)
ax_ami.plot(
    df_illiquid.index, df_illiquid["Amihud_MA"],
    label=f"{ILLIQUID_TICKER} (Illiquid)", color=COLORS[1], linewidth=1.4
)
ax_ami.set_title("20-Day Rolling Amihud Illiquidity Ratio", fontweight="bold")
ax_ami.set_ylabel("Amihud Illiquidity")
ax_ami.legend()
ax_ami.tick_params(axis="x", rotation=30)

plt.suptitle("Part A - Volatility & Liquidity Analysis",
             fontsize=14, fontweight="bold", y=1.01)
plt.savefig("part_a_volatility_liquidity.png", bbox_inches="tight")
plt.show()
print("\nFigure saved: part_a_volatility_liquidity.png")

# Correlation: volatility vs liquidity
for df_corr, ticker in [(df_liquid, LIQUID_TICKER),
                         (df_illiquid, ILLIQUID_TICKER)]:
    merged = df_corr[["RolVol_20d", "Turnover_Ratio", "Amihud_MA"]].dropna()
    corr   = merged.corr()
    print(f"\nCorrelation matrix for {ticker}:")
    print(corr.round(3).to_string())


# -----------------------------------------------------------------------------
# A-5  Step 6: Classify stocks into top / bottom 25% by turnover
# -----------------------------------------------------------------------------

def classify_liquidity(df):
    q75_tr = df["Turnover_Ratio"].quantile(0.75)
    q25_tr = df["Turnover_Ratio"].quantile(0.25)
    df     = df.copy()
    df["Liq_Class"] = "Mid"
    df.loc[df["Turnover_Ratio"] >= q75_tr, "Liq_Class"] = "High (top 25%)"
    df.loc[df["Turnover_Ratio"] <= q25_tr, "Liq_Class"] = "Low (bottom 25%)"
    return df


df_liquid   = classify_liquidity(df_liquid)
df_illiquid = classify_liquidity(df_illiquid)

print(f"\nLiquidity classification counts - {LIQUID_TICKER}:")
print(df_liquid["Liq_Class"].value_counts().to_string())
print(f"\nLiquidity classification counts - {ILLIQUID_TICKER}:")
print(df_illiquid["Liq_Class"].value_counts().to_string())


# =============================================================================
# PART B  -  STEP 7 & 8: OPTION PRICING & VOLATILITY INPUTS
# =============================================================================

# -----------------------------------------------------------------------------
# B-1  BSM helper functions  (Step 7)
# -----------------------------------------------------------------------------

def bsm_d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 for the BSM model."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def bsm_price(S, K, T, r, sigma, option_type="call"):
    """Black-Scholes-Merton price for a European call or put."""
    d1, d2 = bsm_d1_d2(S, K, T, r, sigma)
    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price


def bsm_greeks(S, K, T, r, sigma, option_type="call"):
    """Compute Delta, Gamma, and Vega for a European option."""
    d1, d2  = bsm_d1_d2(S, K, T, r, sigma)
    delta_call = norm.cdf(d1)
    delta_put  = delta_call - 1
    gamma      = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega       = S * norm.pdf(d1) * np.sqrt(T) / 100    # per 1% change in sigma
    delta = delta_call if option_type.lower() == "call" else delta_put
    return {"delta": delta, "gamma": gamma, "vega": vega}


# -----------------------------------------------------------------------------
# B-2  Build option set for both stocks  (Step 7)
# -----------------------------------------------------------------------------

RISK_FREE_RATE  = 0.068           # RBI repo rate proxy - adjust as needed
MATURITIES_DAYS = [30, 60]


def build_option_set(df, ticker):
    """
    Build ATM / OTM-Call / OTM-Put options at 30d and 60d maturities.
    Uses last closing price as spot and historical vol as sigma.
    """
    S     = df["Close"].iloc[-1]
    sigma = df["Log_Return"].dropna().std() * np.sqrt(252)
    r     = RISK_FREE_RATE

    records    = []
    option_legs = [
        ("call", "ATM",      1.000),
        ("call", "OTM Call", 1.075),    # ~7.5% OTM call
        ("put",  "OTM Put",  0.925),    # ~7.5% OTM put
    ]

    for T_days in MATURITIES_DAYS:
        T = T_days / 252
        for opt_type, moneyness, K_factor in option_legs:
            K      = round(S * K_factor, 2)
            price  = bsm_price(S, K, T, r, sigma, opt_type)
            greeks = bsm_greeks(S, K, T, r, sigma, opt_type)
            records.append({
                "Ticker"    : ticker,
                "Maturity"  : f"{T_days}d",
                "Type"      : opt_type.upper(),
                "Moneyness" : moneyness,
                "Spot (S)"  : round(S,     2),
                "Strike (K)": K,
                "T (years)" : round(T,     4),
                "r"         : r,
                "sigma_hist": round(sigma, 4),
                "BSM Price" : round(price, 2),
                "Delta"     : round(greeks["delta"], 4),
                "Gamma"     : round(greeks["gamma"], 6),
                "Vega"      : round(greeks["vega"],  4),
            })

    return pd.DataFrame(records)


opts_liquid   = build_option_set(df_liquid,   LIQUID_TICKER)
opts_illiquid = build_option_set(df_illiquid, ILLIQUID_TICKER)
option_table  = pd.concat([opts_liquid, opts_illiquid], ignore_index=True)

print("\n" + "=" * 60)
print("PART B  -  OPTION PRICING TABLE (BSM, historical vol)")
print("=" * 60)
display_cols = [
    "Ticker", "Maturity", "Moneyness", "Type",
    "Spot (S)", "Strike (K)", "sigma_hist", "BSM Price",
    "Delta", "Gamma", "Vega",
]
print(option_table[display_cols].to_string(index=False))


# -----------------------------------------------------------------------------
# B-3  Step 8: GARCH(1,1) re-pricing - optional
# -----------------------------------------------------------------------------

def fit_garch_and_reprice(df, ticker, opt_df):
    """
    Fit GARCH(1,1), extract 1-step conditional vol, re-price all options,
    and return an augmented DataFrame.
    """
    if not ARCH_AVAILABLE:
        return opt_df

    returns_pct = df["Log_Return"].dropna() * 100    # arch expects % units

    model  = arch_model(returns_pct, vol="Garch", p=1, q=1,
                        dist="normal", rescale=False)
    result = model.fit(disp="off")

    # 1-step-ahead conditional volatility forecast (annualised, decimal)
    forecast  = result.forecast(horizon=1, reindex=False)
    garch_var = forecast.variance.iloc[-1, 0]
    garch_vol = np.sqrt(garch_var) * np.sqrt(252) / 100

    hist_vol = df["Log_Return"].dropna().std() * np.sqrt(252)
    print(
        f"\n  {ticker}  GARCH(1,1) params:"
        f"  omega={result.params['omega']:.4f}"
        f"  alpha={result.params['alpha[1]']:.4f}"
        f"  beta={result.params['beta[1]']:.4f}"
    )
    print(f"  1-step GARCH vol = {garch_vol:.4f}  |  Historical vol = {hist_vol:.4f}")

    subset = opt_df[opt_df["Ticker"] == ticker].copy()
    subset["sigma_GARCH"] = round(garch_vol, 4)
    subset["BSM_GARCH"]   = subset.apply(
        lambda row: round(
            bsm_price(
                row["Spot (S)"], row["Strike (K)"],
                row["T (years)"], row["r"],
                garch_vol, row["Type"].lower()
            ), 2
        ),
        axis=1,
    )
    subset["Price_Diff"] = (subset["BSM_GARCH"] - subset["BSM Price"]).round(2)
    return subset


if ARCH_AVAILABLE:
    print("\n" + "=" * 60)
    print("PART B (Optional)  -  GARCH(1,1) Re-pricing")
    print("=" * 60)
    garch_liquid   = fit_garch_and_reprice(df_liquid,   LIQUID_TICKER,   option_table)
    garch_illiquid = fit_garch_and_reprice(df_illiquid, ILLIQUID_TICKER, option_table)
    garch_table    = pd.concat([garch_liquid, garch_illiquid], ignore_index=True)
    garch_cols = [
        "Ticker", "Maturity", "Moneyness", "Type",
        "BSM Price", "sigma_GARCH", "BSM_GARCH", "Price_Diff",
    ]
    print(garch_table[garch_cols].to_string(index=False))


# =============================================================================
# PART C  -  GREEKS, PORTFOLIO CONSTRUCTION & DELTA HEDGING
# =============================================================================

# -----------------------------------------------------------------------------
# C-1  Build one portfolio per stock
#      Legs: Long 10 ATM Calls (30d) + Long 10 OTM Puts (30d)
#            + Short 5 OTM Calls (60d)
# -----------------------------------------------------------------------------

def build_portfolio(opt_df, ticker):
    """
    Select specific legs and compute aggregate Greeks for the portfolio.
    Edit the `legs` list below to change composition.
    """
    legs = [
        # (maturity, moneyness, type, quantity)
        ("30d", "ATM",      "CALL", +10),
        ("30d", "OTM Put",  "PUT",  +10),
        ("60d", "OTM Call", "CALL",  -5),
    ]

    portfolio_rows = []
    for mat, mon, opt_type, qty in legs:
        mask = (
            (opt_df["Ticker"]    == ticker)  &
            (opt_df["Maturity"]  == mat)     &
            (opt_df["Moneyness"] == mon)     &
            (opt_df["Type"]      == opt_type)
        )
        row = opt_df[mask].iloc[0].copy()
        row["Quantity"]      = qty
        row["Position Cost"] = round(row["BSM Price"] * qty, 2)
        row["Pos Delta"]     = round(row["Delta"] * qty, 4)
        row["Pos Gamma"]     = round(row["Gamma"] * qty, 6)
        row["Pos Vega"]      = round(row["Vega"]  * qty, 4)
        portfolio_rows.append(row)

    portfolio = pd.DataFrame(portfolio_rows)

    agg = {
        "Total Cost" : portfolio["Position Cost"].sum(),
        "Net Delta"  : portfolio["Pos Delta"].sum(),
        "Net Gamma"  : portfolio["Pos Gamma"].sum(),
        "Net Vega"   : portfolio["Pos Vega"].sum(),
    }
    return portfolio, agg


print("\n" + "=" * 60)
print("PART C  -  PORTFOLIO CONSTRUCTION & GREEKS")
print("=" * 60)

port_liquid,   agg_liquid   = build_portfolio(option_table, LIQUID_TICKER)
port_illiquid, agg_illiquid = build_portfolio(option_table, ILLIQUID_TICKER)

show_cols = [
    "Moneyness", "Maturity", "Type", "BSM Price",
    "Quantity", "Position Cost", "Pos Delta", "Pos Gamma", "Pos Vega",
]

for name, port, agg in [
    (LIQUID_TICKER,   port_liquid,   agg_liquid),
    (ILLIQUID_TICKER, port_illiquid, agg_illiquid),
]:
    print(f"\n  Portfolio - {name}")
    print(port[show_cols].to_string(index=False))
    print(
        f"  {'Aggregate':16s}  "
        f"Cost={agg['Total Cost']:9.2f}  "
        f"Net Delta={agg['Net Delta']:8.4f}  "
        f"Net Gamma={agg['Net Gamma']:.6f}  "
        f"Net Vega={agg['Net Vega']:8.4f}"
    )


# -----------------------------------------------------------------------------
# C-2  Delta hedging (with optional liquidity adjustment)
# -----------------------------------------------------------------------------

def delta_hedge(agg, df, ticker, liquidity_adjusted=True):
    """
    Determine shares of the underlying needed to delta-neutralise the portfolio.
    Liquidity adjustment scales the hedge down by the current turnover ratio.
    """
    net_delta     = agg["Net Delta"]
    S             = df["Close"].iloc[-1]
    shares_needed = -net_delta

    liq_ratio  = df["Turnover_Ratio"].iloc[-1]
    adj_factor = min(liq_ratio, 1.0) if liquidity_adjusted else 1.0
    shares_adj = shares_needed * adj_factor

    return {
        "Ticker"                : ticker,
        "Net Portfolio Delta"   : round(net_delta,     4),
        "Shares to Hedge (raw)" : round(shares_needed, 4),
        "Liquidity Ratio"       : round(liq_ratio,     4),
        "Shares to Hedge (adj)" : round(shares_adj,    4),
        "Hedge Cost Raw (Rs.)"  : round(abs(shares_needed) * S, 2),
        "Hedge Cost Adj (Rs.)"  : round(abs(shares_adj)    * S, 2),
    }


print("\n" + "=" * 60)
print("PART C  -  DELTA HEDGING (with liquidity adjustment)")
print("=" * 60)

hedge_liquid   = delta_hedge(agg_liquid,   df_liquid,   LIQUID_TICKER)
hedge_illiquid = delta_hedge(agg_illiquid, df_illiquid, ILLIQUID_TICKER)

for h in [hedge_liquid, hedge_illiquid]:
    print(f"\n  {h['Ticker']}")
    for k, v in h.items():
        if k != "Ticker":
            print(f"    {k:35s}: {v}")


# -----------------------------------------------------------------------------
# C-3  PnL simulation under price and volatility shocks
# -----------------------------------------------------------------------------

def simulate_pnl(port, df):
    """
    Approximate portfolio PnL using Delta-Gamma (price shocks)
    and Vega (vol shocks) Taylor expansion.
    """
    S         = df["Close"].iloc[-1]
    net_delta = port["Pos Delta"].sum()
    net_gamma = port["Pos Gamma"].sum()
    net_vega  = port["Pos Vega"].sum()

    price_shocks = [-0.02, -0.01, +0.01, +0.02]
    vol_shocks   = [-0.20, +0.20]

    rows     = []
    sigma_now = df["Log_Return"].dropna().std() * np.sqrt(252)

    for dS_pct in price_shocks:
        dS  = S * dS_pct
        pnl = net_delta * dS + 0.5 * net_gamma * dS ** 2
        rows.append({"Shock Type": f"Price {dS_pct:+.0%}", "PnL (Rs.)": round(pnl, 2)})

    for dVol_pct in vol_shocks:
        dVol = sigma_now * dVol_pct
        pnl  = net_vega * dVol * 100    # Vega expressed per 1% sigma change
        rows.append({"Shock Type": f"Vol {dVol_pct:+.0%}", "PnL (Rs.)": round(pnl, 2)})

    return pd.DataFrame(rows)


print("\n" + "=" * 60)
print("PART C  -  PnL SIMULATION")
print("=" * 60)

for name, port, df_s in [
    (LIQUID_TICKER,   port_liquid,   df_liquid),
    (ILLIQUID_TICKER, port_illiquid, df_illiquid),
]:
    pnl_df = simulate_pnl(port, df_s)
    print(f"\n  PnL scenarios - {name}")
    print(pnl_df.to_string(index=False))


# -----------------------------------------------------------------------------
# C-4  Plot: Position Deltas per portfolio leg
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (name, port) in zip(axes, [
    (LIQUID_TICKER,   port_liquid),
    (ILLIQUID_TICKER, port_illiquid),
]):
    labels  = port["Moneyness"] + "\n" + port["Maturity"] + "\n" + port["Type"]
    barvals = port["Pos Delta"].values
    colors  = [COLORS[0] if v >= 0 else COLORS[1] for v in barvals]
    ax.bar(labels, barvals, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title(f"{name} - Position Deltas", fontweight="bold")
    ax.set_ylabel("Position Delta")

plt.suptitle("Part C - Portfolio Greeks", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("part_c_portfolio_greeks.png", bbox_inches="tight")
plt.show()
print("\nFigure saved: part_c_portfolio_greeks.png")


# =============================================================================
# PART D  -  STEP 9: VALUE-AT-RISK & STRESS ANALYSIS
# =============================================================================

# -----------------------------------------------------------------------------
# D-1  Parametric (Model-Building) VaR at 95% and 99%
#      Regime split: normal vs high-volatility days (top 25% rolling vol)
# -----------------------------------------------------------------------------

CONFIDENCE_LEVELS = [0.95, 0.99]


def compute_parametric_var(returns, confidence_levels):
    """Standard parametric (normal) VaR for a return series."""
    mu    = returns.mean()
    sigma = returns.std()
    var   = {}
    for cl in confidence_levels:
        z = norm.ppf(1 - cl)
        var[f"VaR_{int(cl * 100)}%"] = -(mu + z * sigma)
    return var


def var_by_regime(df, ticker):
    """
    Split returns into normal and high-vol regimes (top 25% rolling vol days)
    and compute parametric VaR for each.
    """
    ret      = df["Log_Return"].dropna()
    q75_vol  = df["RolVol_20d"].quantile(0.75)
    hv_mask  = df["RolVol_20d"] >= q75_vol
    hv_dates = df.index[hv_mask]

    ret_normal = ret[~ret.index.isin(hv_dates)].dropna()
    ret_hv     = ret[ ret.index.isin(hv_dates)].dropna()

    rows = []
    for regime, series in [
        ("All data",        ret),
        ("Normal regime",   ret_normal),
        ("High-vol regime", ret_hv),
    ]:
        var = compute_parametric_var(series, CONFIDENCE_LEVELS)
        row = {
            "Ticker"        : ticker,
            "Regime"        : regime,
            "N days"        : len(series),
            "Mean (daily)"  : round(series.mean(), 6),
            "Sigma (daily)" : round(series.std(),  6),
        }
        row.update({k: round(v * 100, 4) for k, v in var.items()})
        rows.append(row)

    return pd.DataFrame(rows)


print("\n" + "=" * 60)
print("PART D  -  PARAMETRIC VaR TABLE (% of portfolio value)")
print("=" * 60)

var_liquid   = var_by_regime(df_liquid,   LIQUID_TICKER)
var_illiquid = var_by_regime(df_illiquid, ILLIQUID_TICKER)
var_table    = pd.concat([var_liquid, var_illiquid], ignore_index=True)
print(var_table.to_string(index=False))


# -----------------------------------------------------------------------------
# D-2  Monte Carlo VaR  (optional - targeted)
# -----------------------------------------------------------------------------

def monte_carlo_var(returns, n_simulations=50_000, confidence_levels=None):
    """
    Simulate 1-day portfolio returns under a normality assumption and
    extract empirical VaR at given confidence levels.
    """
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    mu    = returns.mean()
    sigma = returns.std()
    sim   = np.random.normal(mu, sigma, n_simulations)

    var = {}
    for cl in confidence_levels:
        pct = (1 - cl) * 100
        var[f"MC_VaR_{int(cl * 100)}%"] = round(
            -np.percentile(sim, pct) * 100, 4
        )
    return var, sim


print("\n" + "=" * 60)
print("PART D (Optional)  -  MONTE CARLO VaR")
print("=" * 60)

np.random.seed(42)
mc_rows = []
for df_s, ticker in [(df_liquid, LIQUID_TICKER), (df_illiquid, ILLIQUID_TICKER)]:
    ret            = df_s["Log_Return"].dropna()
    mc_var, mc_sim = monte_carlo_var(ret)
    row = {"Ticker": ticker}
    row.update(mc_var)
    mc_rows.append(row)
    print(f"  {ticker}:  {mc_var}")

mc_table = pd.DataFrame(mc_rows)


# -----------------------------------------------------------------------------
# D-3  VaR Plots: return distribution + Monte Carlo
# -----------------------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for row_i, (df_s, ticker) in enumerate([
    (df_liquid,   LIQUID_TICKER),
    (df_illiquid, ILLIQUID_TICKER),
]):
    ret = df_s["Log_Return"].dropna()

    # Left panel: empirical distribution with parametric VaR lines
    ax = axes[row_i][0]
    ax.hist(ret, bins=50, color=COLORS[row_i], alpha=0.65,
            edgecolor="white", density=True, label="Returns")
    x_range = np.linspace(ret.min(), ret.max(), 500)
    ax.plot(x_range, norm.pdf(x_range, ret.mean(), ret.std()),
            "k--", linewidth=1.3, label="Normal fit")

    var95 = compute_parametric_var(ret, [0.95])["VaR_95%"] / 100
    var99 = compute_parametric_var(ret, [0.99])["VaR_99%"] / 100
    ax.axvline(-var95, color=COLORS[2], linewidth=1.5, linestyle="-.",
               label=f"VaR 95% = {var95 * 100:.2f}%")
    ax.axvline(-var99, color=COLORS[3], linewidth=1.5, linestyle=":",
               label=f"VaR 99% = {var99 * 100:.2f}%")
    ax.set_title(f"{ticker} - Return Distribution & VaR", fontweight="bold")
    ax.set_xlabel("Daily Log Return")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

    # Right panel: Monte Carlo simulation histogram
    ax2 = axes[row_i][1]
    _, mc_sim_plot = monte_carlo_var(ret, n_simulations=50_000)
    ax2.hist(mc_sim_plot, bins=80, color=COLORS[row_i], alpha=0.55,
             edgecolor="white", density=True)
    mc95 = np.percentile(mc_sim_plot, 5)
    mc99 = np.percentile(mc_sim_plot, 1)
    ax2.axvline(mc95, color=COLORS[2], linewidth=1.5, linestyle="-.",
                label=f"MC VaR 95% = {-mc95 * 100:.2f}%")
    ax2.axvline(mc99, color=COLORS[3], linewidth=1.5, linestyle=":",
                label=f"MC VaR 99% = {-mc99 * 100:.2f}%")
    ax2.set_title(f"{ticker} - Monte Carlo VaR (50k sims)", fontweight="bold")
    ax2.set_xlabel("Simulated 1-Day Return")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)

plt.suptitle("Part D - VaR Analysis", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("part_d_var_analysis.png", bbox_inches="tight")
plt.show()
print("\nFigure saved: part_d_var_analysis.png")


# -----------------------------------------------------------------------------
# D-4  GARCH-based time-varying VaR  (optional)
# -----------------------------------------------------------------------------

def garch_var(df, ticker, confidence_levels=None):
    """
    Compute parametric VaR using GARCH(1,1)-estimated conditional volatility
    for the most recent trading day.
    """
    if not ARCH_AVAILABLE:
        return None
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    ret_pct = df["Log_Return"].dropna() * 100
    model   = arch_model(ret_pct, vol="Garch", p=1, q=1,
                         dist="normal", rescale=False)
    result  = model.fit(disp="off")

    # Conditional volatility series (daily, decimal)
    cond_vol = result.conditional_volatility / 100

    rows = []
    for cl in confidence_levels:
        z   = norm.ppf(1 - cl)
        gv  = cond_vol.iloc[-1]
        var = -(ret_pct.mean() / 100 + z * gv)
        rows.append({
            "Ticker"             : ticker,
            "Confidence"         : f"{int(cl * 100)}%",
            "GARCH sigma (last)" : round(gv,        6),
            "GARCH VaR (%)"      : round(var * 100,  4),
        })
    return pd.DataFrame(rows)


if ARCH_AVAILABLE:
    print("\n" + "=" * 60)
    print("PART D (Optional)  -  GARCH-based VaR")
    print("=" * 60)
    gv_liquid   = garch_var(df_liquid,   LIQUID_TICKER)
    gv_illiquid = garch_var(df_illiquid, ILLIQUID_TICKER)
    if gv_liquid is not None and gv_illiquid is not None:
        garch_var_table = pd.concat([gv_liquid, gv_illiquid], ignore_index=True)
        print(garch_var_table.to_string(index=False))


# -----------------------------------------------------------------------------
# D-5  Final consolidated VaR comparison
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("PART D  -  CONSOLIDATED VaR COMPARISON")
print("=" * 60)
print("\nParametric VaR:")
print(var_table.to_string(index=False))
print("\nMonte Carlo VaR:")
print(mc_table.to_string(index=False))

print("\n" + "=" * 60)
print("ALL SECTIONS COMPLETE")
print(
    "Figures saved:\n"
    "  part_a_volatility_liquidity.png\n"
    "  part_c_portfolio_greeks.png\n"
    "  part_d_var_analysis.png"
)
print("=" * 60)

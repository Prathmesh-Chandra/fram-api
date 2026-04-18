import yfinance as yf
from functools import lru_cache

REQUIRED_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def _is_missing(value):
    return value is None or value != value

@lru_cache(maxsize=50)
def get_history(ticker: str, period: str = "6mo"):
    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False,
            multi_level_index=False,
        )
    except Exception:
        return None

    if df.empty:
        return None

    # yfinance can still return MultiIndex columns depending on version/settings.
    if getattr(df.columns, "nlevels", 1) > 1:
        df.columns = df.columns.get_level_values(0)

    for column in REQUIRED_COLUMNS:
        if column not in df.columns:
            return None

    df = df.loc[:, list(REQUIRED_COLUMNS)]
    df = df.dropna(how="all", subset=["Open", "High", "Low", "Close", "Volume"])
    if df.empty:
        return None

    def _float_series(name: str):
        series = df[name]
        if getattr(series, "ndim", 1) > 1:
            series = series.iloc[:, 0]
        return [None if _is_missing(v) else round(float(v), 2) for v in series.tolist()]

    def _int_series(name: str):
        series = df[name]
        if getattr(series, "ndim", 1) > 1:
            series = series.iloc[:, 0]
        return [None if _is_missing(v) else int(float(v)) for v in series.tolist()]

    df.index = df.index.strftime("%Y-%m-%d")
    return {
        "ticker": ticker,
        "dates": df.index.tolist(),
        "open": _float_series("Open"),
        "high": _float_series("High"),
        "low": _float_series("Low"),
        "close": _float_series("Close"),
        "volume": _int_series("Volume"),
    }
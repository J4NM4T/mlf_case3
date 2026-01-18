import numpy as np
import pandas as pd
import yfinance as yf
import talib as ta

def _as_1d_series(x, index=None) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        x = x.squeeze("columns")
    arr = np.asarray(x).squeeze()
    return pd.Series(arr, index=index if index is not None else getattr(x, "index", None), dtype=float)

def load_data(ticker:str, start_dte:str, end_dte:str) -> pd.DataFrame:
    ticker = yf.Ticker(ticker)
    historical_data = ticker.history(interval="1d", start=start_dte, end=end_dte)

    df = historical_data.copy()
    df.index = df.index.tz_localize(None)

    close = _as_1d_series(df['Close'], df.index)
    high = _as_1d_series(df['High'], df.index)
    low  = _as_1d_series(df['Low'], df.index)

    df['feature_rsi_14'] = ta.RSI(close, timeperiod=14)
    macd_line, macd_signal, macd_hist = ta.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['feature_macd'] = macd_line
    df['feature_macd_signal'] = macd_signal
    df['feature_macd_hist'] = macd_hist

    bb_up, bb_mid, bb_low = ta.BBANDS(close, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, matype=0)
    df['feature_bb_low'] = _as_1d_series(bb_low, df.index)
    df['feature_bb_mid'] = _as_1d_series(bb_mid, df.index)
    df['feature_bb_up']  = _as_1d_series(bb_up,  df.index)
    df['feature_close_bb_pos'] = ((close - df['feature_bb_mid']) / (df['feature_bb_up'] - df['feature_bb_low'] + 1e-12)).astype(float)

    df['feature_ret_1d'] = close.pct_change()
    df['feature_ret_std_30'] = df['feature_ret_1d'].rolling(window=30, min_periods=30).std()

    df['feature_sma_5'] = close.rolling(window=5, min_periods=5).mean()
    df['feature_sma_20'] = close.rolling(window=20, min_periods=20).mean()
    df['feature_sma_5_over_20'] = (df['feature_sma_5'] / df['feature_sma_20']) - 1.0

    df['feature_roc_20'] = close.pct_change(20)

    df['feature_ema_20'] = close.ewm(span=20, adjust=False).mean()
    df['feature_ema_50'] = close.ewm(span=50, adjust=False).mean()
    df['feature_ema_20_over_50'] = (df['feature_ema_20'] / df['feature_ema_50']) - 1.0

    df['feature_adx_14'] = ta.ADX(high, low, close, timeperiod=14)

    df['feature_atr_14'] = ta.ATR(high, low, close, timeperiod=14)
    df['feature_atr_pct'] = df['feature_atr_14'] / close

    rolling_mean = df['feature_ret_1d'].rolling(20).mean() 
    rolling_std = df['feature_ret_1d'].rolling(20).std() 
    df['feature_sharpe_20'] = rolling_mean - 0.02 / (rolling_std + 1e-12)

    df = df.dropna().copy()

    df_scaled_only = df.drop(['Close', 'High', 'Open', 'Low'], axis=1)
    df_scaled_only = rolling_zscore(df=df_scaled_only, cols=df_scaled_only.columns, window=30)

    df_model = df.copy()
    df_model[df_scaled_only.columns] = df_scaled_only[df_scaled_only.columns]
    
    df_model.columns = [x.lower() for x in df_model.columns]
    df_model.index.name = df_model.index.name.lower()

    df_model.drop(['dividends', 'stock splits'], axis=1, inplace=True)

    return df_model.dropna()


def rolling_zscore(df: pd.DataFrame, cols: list[str], window: int):
    roll = df[cols].rolling(window=window, min_periods=window)
    mu = roll.mean().shift(1)
    sigma = roll.std(ddof=0).replace(0, 1.0).shift(1)
    z = (df[cols] - mu) / sigma
    out = df.copy()
    out[cols] = z
    return out.dropna()
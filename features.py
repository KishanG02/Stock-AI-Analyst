import pandas as pd
import ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators for stock analysis.
    """

    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    df = df.copy()

    try:
        # -------------------------
        # Trend Indicators
        # -------------------------
        df["SMA_20"] = ta.trend.sma_indicator(
            close=df["Close"],
            window=20
        )

        df["SMA_50"] = ta.trend.sma_indicator(
            close=df["Close"],
            window=50
        )

        df["EMA_20"] = ta.trend.ema_indicator(
            close=df["Close"],
            window=20
        )

        # MACD
        df["MACD"] = ta.trend.macd(
            close=df["Close"]
        )

        df["MACD_SIGNAL"] = ta.trend.macd_signal(
            close=df["Close"]
        )

        # -------------------------
        # Momentum Indicators
        # -------------------------
        df["RSI"] = ta.momentum.rsi(
            close=df["Close"],
            window=14
        )

        # Stochastic Oscillator
        df["STOCH"] = ta.momentum.stoch(
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )

        # -------------------------
        # Volatility Indicators
        # -------------------------
        bb = ta.volatility.BollingerBands(
            close=df["Close"],
            window=20,
            window_dev=2
        )

        df["BB_High"] = bb.bollinger_hband()
        df["BB_Low"] = bb.bollinger_lband()
        df["BB_Mid"] = bb.bollinger_mavg()

        # Average True Range
        df["ATR"] = ta.volatility.average_true_range(
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )

        # -------------------------
        # Volume Indicators
        # -------------------------
        df["OBV"] = ta.volume.on_balance_volume(
            close=df["Close"],
            volume=df["Volume"]
        )

        # Volume Moving Average
        df["VOL_MA20"] = (
            df["Volume"]
            .rolling(window=20)
            .mean()
        )

        # -------------------------
        # Cleaning
        # -------------------------

        # Forward fill first
        df = df.ffill()

        # Remove only rows where Close is missing
        df = df[df["Close"].notna()]

        # Remove remaining NaN rows caused by indicator warmup
        df = df.iloc[50:]

        if df.empty:
            raise ValueError(
                "All rows removed during indicator calculation."
            )

        return df

    except Exception as e:
        raise ValueError(
            f"Indicator calculation failed: {str(e)}"
        )
import yfinance as yf
import pandas as pd


def fetch_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical stock data safely.
    """

    try:
        df = yf.download(
            ticker,
            period=period,
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"No data found for {ticker}")
            return pd.DataFrame()

        # Handle MultiIndex columns from newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Open", "High", "Low", "Close", "Volume"]

        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return pd.DataFrame()

        df = df[required_cols]

        df.index = pd.to_datetime(df.index)

        # Remove rows where Close is missing
        df = df[df["Close"].notna()]

        print(f"{ticker}: {len(df)} rows downloaded")

        return df

    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()


def fetch_stock_info(ticker: str) -> dict:
    """
    Fetch company information safely.
    """

    try:
        stock = yf.Ticker(ticker)

        try:
            info = stock.info
        except Exception:
            info = {}

        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "52w_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52w_low": info.get("fiftyTwoWeekLow", "N/A"),
        }

    except Exception as e:
        print(f"Info fetch error: {e}")

        return {
            "name": ticker,
            "sector": "N/A",
            "pe_ratio": "N/A",
            "market_cap": "N/A",
            "52w_high": "N/A",
            "52w_low": "N/A",
        }
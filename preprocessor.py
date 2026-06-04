import numpy as np
from sklearn.preprocessing import MinMaxScaler


def preprocess(df, look_back=60):
    """
    Prepare stock data for LSTM.

    Returns:
        X_train
        X_test
        y_train
        y_test
        scaler
    """

    if df is None or df.empty:
        raise ValueError("Input dataframe is empty.")

    if "Close" not in df.columns:
        raise ValueError("Close column not found.")

    close_prices = df["Close"].copy()

    # Remove NaN values
    close_prices = close_prices.dropna()

    if len(close_prices) <= look_back:
        raise ValueError(
            f"Not enough rows. Need > {look_back}, got {len(close_prices)}."
        )

    close_prices = close_prices.values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)

    X = []
    y = []

    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # Ensure proper shape for LSTM
    X = X.reshape(
        X.shape[0],
        X.shape[1],
        1
    )

    split_idx = int(len(X) * 0.8)

    X_train = X[:split_idx]
    X_test = X[split_idx:]

    y_train = y[:split_idx]
    y_test = y[split_idx:]

    if len(X_train) == 0:
        raise ValueError("Training dataset is empty.")

    if len(X_test) == 0:
        raise ValueError("Testing dataset is empty.")

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler
    )
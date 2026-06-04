import math
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)


# --------------------------------------------------
# Evaluate Model
# --------------------------------------------------
def predict(
    model,
    X_test,
    y_test,
    scaler
):
    """
    Generate predictions and evaluation metrics.
    """

    if len(X_test) == 0:
        raise ValueError("X_test is empty.")

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    )

    predictions = scaler.inverse_transform(
        predictions_scaled
    ).flatten()

    actuals = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    rmse = math.sqrt(
        mean_squared_error(
            actuals,
            predictions
        )
    )

    mae = mean_absolute_error(
        actuals,
        predictions
    )

    # Avoid division by zero
    non_zero_mask = actuals != 0

    if np.sum(non_zero_mask) > 0:
        mape = np.mean(
            np.abs(
                (
                    actuals[non_zero_mask]
                    - predictions[non_zero_mask]
                )
                / actuals[non_zero_mask]
            )
        ) * 100
    else:
        mape = 0.0

    return {
        "predictions": predictions,
        "actuals": actuals,
        "rmse": rmse,
        "mae": mae,
        "mape": mape
    }


# --------------------------------------------------
# Next Day Prediction
# --------------------------------------------------
def predict_next_day(
    model,
    df,
    scaler,
    look_back=60
):
    """
    Predict next trading day's close price.
    """

    if len(df) < look_back:
        raise ValueError(
            f"Need at least {look_back} rows."
        )

    close_prices = df["Close"].values

    recent_prices = close_prices[-look_back:]

    scaled = scaler.transform(
        recent_prices.reshape(-1, 1)
    )

    X = scaled.reshape(
        1,
        look_back,
        1
    )

    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    prediction = scaler.inverse_transform(
        prediction_scaled
    )[0][0]

    return float(prediction)


# --------------------------------------------------
# Multi-Day Forecast
# --------------------------------------------------
def forecast_future(
    model,
    df,
    scaler,
    look_back=60,
    days=7
):
    """
    Recursive future forecasting.
    """

    if len(df) < look_back:
        raise ValueError(
            f"Need at least {look_back} rows."
        )

    sequence = (
        scaler.transform(
            df["Close"]
            .values[-look_back:]
            .reshape(-1, 1)
        )
        .flatten()
        .tolist()
    )

    future_predictions = []

    for _ in range(days):

        X = np.array(
            sequence[-look_back:]
        ).reshape(
            1,
            look_back,
            1
        )

        pred_scaled = model.predict(
            X,
            verbose=0
        )[0][0]

        sequence.append(pred_scaled)

        pred_price = scaler.inverse_transform(
            np.array([[pred_scaled]])
        )[0][0]

        future_predictions.append(
            float(pred_price)
        )

    return future_predictions


# --------------------------------------------------
# Forecast Summary
# --------------------------------------------------
def forecast_summary(
    model,
    df,
    scaler,
    look_back=60
):
    """
    Returns 1-day, 7-day, and 30-day forecasts.
    """

    next_day = predict_next_day(
        model,
        df,
        scaler,
        look_back
    )

    week_forecast = forecast_future(
        model,
        df,
        scaler,
        look_back,
        days=7
    )

    month_forecast = forecast_future(
        model,
        df,
        scaler,
        look_back,
        days=30
    )

    return {
        "next_day": next_day,
        "7_day": week_forecast,
        "30_day": month_forecast
    }
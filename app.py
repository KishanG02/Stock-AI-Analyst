import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv

from data_fetcher import (
    fetch_stock_data,
    fetch_stock_info
)

from features import (
    add_technical_indicators
)

from preprocessor import (
    preprocess
)

from model import (
    build_model,
    train_model
)

from predictor import (
    predict,
    predict_next_day,
    forecast_future
)

from analyst import (
    get_ai_analysis
)

load_dotenv()

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Stock AI Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock AI Analyst")
st.caption(
    "LSTM Stock Prediction + Technical Analysis + AI Financial Insights"
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    ticker = st.text_input(
        "Ticker Symbol",
        value="AAPL"
    ).upper()

    period = st.selectbox(
        "Historical Data",
        ["1y", "2y", "5y"],
        index=1
    )

    look_back = st.slider(
        "Look Back Window",
        min_value=30,
        max_value=120,
        value=60
    )

    run_analysis = st.button(
        "🚀 Run Analysis",
        use_container_width=True
    )

# --------------------------------------------------
# MAIN APP
# --------------------------------------------------

if run_analysis:

    try:

        # --------------------------------------------------
        # DOWNLOAD DATA
        # --------------------------------------------------

        with st.spinner("Downloading stock data..."):

            df = fetch_stock_data(
                ticker=ticker,
                period=period
            )

            if df.empty:
                st.error(
                    f"No stock data found for '{ticker}'."
                )
                st.stop()

            info = fetch_stock_info(ticker)

        # --------------------------------------------------
        # TECHNICAL INDICATORS
        # --------------------------------------------------

        with st.spinner("Calculating indicators..."):

            df = add_technical_indicators(df)

            if df.empty:
                st.error(
                    "Indicator calculation returned no data."
                )
                st.stop()

        # --------------------------------------------------
        # HEADER
        # --------------------------------------------------

        company_name = info.get(
            "name",
            ticker
        )

        st.subheader(
            f"{company_name} ({ticker})"
        )

        current_price = float(
            df["Close"].iloc[-1]
        )

        rsi = float(
            df["RSI"].iloc[-1]
        )

        macd = float(
            df["MACD"].iloc[-1]
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Current Price",
            f"${current_price:.2f}"
        )

        col2.metric(
            "RSI (14)",
            f"{rsi:.2f}"
        )

        col3.metric(
            "Sector",
            info.get(
                "sector",
                "N/A"
            )
        )

        col4.metric(
            "P/E Ratio",
            str(
                info.get(
                    "pe_ratio",
                    "N/A"
                )
            )
        )

        # --------------------------------------------------
        # PRICE CHART
        # --------------------------------------------------

        st.subheader(
            "📊 Price & Technical Indicators"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["Close"],
                name="Close"
            )
        )

        if "SMA_20" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_20"],
                    name="SMA 20"
                )
            )

        if "SMA_50" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["SMA_50"],
                    name="SMA 50"
                )
            )

        if "BB_High" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_High"],
                    name="BB High"
                )
            )

        if "BB_Low" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df["BB_Low"],
                    name="BB Low"
                )
            )

        fig.update_layout(
            height=550,
            xaxis_title="Date",
            yaxis_title="Price"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # PREPROCESS
        # --------------------------------------------------

        if len(df) < look_back + 50:

            st.error(
                f"Need at least {look_back + 50} rows."
            )

            st.stop()

        # --------------------------------------------------
        # TRAIN MODEL
        # --------------------------------------------------

        with st.spinner(
            "Training LSTM model..."
        ):

            (
                X_train,
                X_test,
                y_train,
                y_test,
                scaler
            ) = preprocess(
                df,
                look_back
            )

            model = build_model(
                look_back
            )

            model, history = train_model(
                model,
                X_train,
                y_train
            )

        # --------------------------------------------------
        # EVALUATION
        # --------------------------------------------------

        results = predict(
            model,
            X_test,
            y_test,
            scaler
        )

        preds = results["predictions"]
        actuals = results["actuals"]

        rmse = results["rmse"]
        mae = results["mae"]
        mape = results["mape"]

        st.subheader(
            "📈 Model Performance"
        )

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "RMSE",
            f"${rmse:.2f}"
        )

        m2.metric(
            "MAE",
            f"${mae:.2f}"
        )

        m3.metric(
            "MAPE",
            f"{mape:.2f}%"
        )

        # --------------------------------------------------
        # PREDICTION CHART
        # --------------------------------------------------

        st.subheader(
            "🔮 Actual vs Predicted"
        )

        pred_fig = go.Figure()

        pred_fig.add_trace(
            go.Scatter(
                y=actuals,
                name="Actual"
            )
        )

        pred_fig.add_trace(
            go.Scatter(
                y=preds,
                name="Predicted"
            )
        )

        pred_fig.update_layout(
            height=500
        )

        st.plotly_chart(
            pred_fig,
            use_container_width=True
        )

        # --------------------------------------------------
        # FORECASTS
        # --------------------------------------------------

        next_day_price = predict_next_day(
            model,
            df,
            scaler,
            look_back
        )

        forecast_7 = forecast_future(
            model,
            df,
            scaler,
            look_back,
            days=7
        )

        delta_pct = (
            (
                next_day_price
                - current_price
            )
            / current_price
        ) * 100

        st.subheader(
            "📅 Forecast"
        )

        st.metric(
            "Next Trading Day",
            f"${next_day_price:.2f}",
            delta=f"{delta_pct:+.2f}%"
        )

        forecast_df = {
            "Day": list(
                range(
                    1,
                    len(forecast_7) + 1
                )
            ),
            "Predicted Price": forecast_7
        }

        st.line_chart(
            forecast_df,
            x="Day",
            y="Predicted Price"
        )

        # --------------------------------------------------
        # AI ANALYSIS
        # --------------------------------------------------

        st.subheader(
            "🤖 AI Financial Analyst"
        )

        with st.spinner(
            "Generating AI report..."
        ):

            analysis = get_ai_analysis(
                ticker=ticker,
                current_price=current_price,
                predicted_price=next_day_price,
                rsi=rsi,
                macd=macd,
                stock_info=info,
                rmse=rmse,
                mae=mae,
                mape=mape
            )

        st.markdown(
            analysis
        )

        # --------------------------------------------------
        # RAW DATA
        # --------------------------------------------------

        with st.expander(
            "📄 View Processed Data"
        ):
            st.dataframe(
                df.tail(50),
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Application Error: {str(e)}"
        )

        st.exception(e)
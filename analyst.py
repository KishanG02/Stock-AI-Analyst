import os
from groq import Groq


def get_ai_analysis(
    ticker,
    current_price,
    predicted_price,
    rsi,
    macd,
    stock_info,
    rmse,
    mae=None,
    mape=None
):
    """
    Generate AI-powered stock analysis using Groq.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        return """
## ❌ Configuration Error

GROQ_API_KEY not found.

Create a `.env` file:

GROQ_API_KEY=your_api_key_here

Then restart the application.
"""

    try:
        client = Groq(api_key=api_key)

        change_pct = (
            (predicted_price - current_price)
            / current_price
        ) * 100

        company_name = stock_info.get(
            "longName",
            ticker
        )

        sector = stock_info.get(
            "sector",
            "Unknown"
        )

        market_cap = stock_info.get(
            "marketCap",
            "Unknown"
        )

        prompt = f"""
You are a professional financial analyst.

Analyze the following stock data and provide:

1. Executive Summary
2. Technical Analysis
3. Risk Assessment
4. Model Performance Review
5. Buy / Hold / Sell Recommendation
6. Short-Term Outlook

Stock Information:

Ticker: {ticker}
Company: {company_name}
Sector: {sector}

Current Price: ${current_price:.2f}
Predicted Price: ${predicted_price:.2f}
Expected Change: {change_pct:.2f}%

RSI: {rsi:.2f}
MACD: {macd:.4f}

RMSE: {rmse:.4f}
MAE: {mae if mae is not None else 'N/A'}
MAPE: {mape if mape is not None else 'N/A'}

Market Cap: {market_cap}

Respond in markdown format.
"""

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"""
## ❌ AI Analysis Error

Error:

{str(e)}

Possible causes:

- Invalid GROQ_API_KEY
- No internet connection
- Groq API service unavailable
- Rate limit exceeded
"""
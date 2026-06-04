# 📈 Stock AI Analyst

> AI-Powered Stock Market Analysis, Forecasting & Financial Insights

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-LSTM-orange.svg)]()
[![Groq](https://img.shields.io/badge/Groq-LLM-green.svg)]()

## 🚀 Live Demo

🔗 **Try the App Here**

**https://stock-ai-analyst-lhqgrsswysheqtuwa8gxyx.streamlit.app/**

---

## 📌 Overview

Stock AI Analyst is an intelligent stock market analysis platform that combines:

* 📊 Technical Analysis
* 🤖 Generative AI Financial Insights
* 📈 LSTM Stock Price Forecasting
* 📰 Real-Time Market Data
* 📉 Performance Evaluation Metrics
* 🎯 Interactive Visualizations

The application helps investors and learners analyze stocks using machine learning and AI-generated financial recommendations.

---

## ✨ Features

### 📊 Technical Analysis

Calculate and visualize:

* RSI (Relative Strength Index)
* MACD
* SMA 20
* SMA 50
* EMA 20
* Bollinger Bands
* ATR
* OBV

---

### 🤖 AI Financial Analyst

Powered by Groq LLM.

Generates:

* BUY / HOLD / SELL Recommendations
* Confidence Scores
* Bullish Factors
* Bearish Factors
* Risk Analysis
* Final Investment Verdict

---

### 📈 LSTM Forecast Engine

Deep Learning model built using TensorFlow.

Features:

* Multi-layer LSTM Network
* Dropout Regularization
* Early Stopping
* Learning Rate Scheduling
* Next-Day Price Prediction

---

### 📉 Model Evaluation

Performance Metrics:

* RMSE
* MAE
* MAPE

---

### 📊 Interactive Dashboards

Built with Plotly:

* Historical Price Trends
* Technical Indicators
* Actual vs Predicted Prices
* Future Forecast Charts

---

## 🏗️ Project Architecture

```text
Stock AI Analyst
│
├── app.py
│
├── data_fetcher.py
│   └── Yahoo Finance Integration
│
├── features.py
│   └── Technical Indicators
│
├── preprocessor.py
│   └── Data Scaling & Sequence Creation
│
├── model.py
│   └── LSTM Neural Network
│
├── predictor.py
│   └── Forecasting & Metrics
│
├── analyst.py
│   └── Groq AI Analysis
│
├── requirements.txt
└── runtime.txt
```

---

## 🧠 Machine Learning Pipeline

```text
Stock Data
    ↓
Technical Indicators
    ↓
Data Preprocessing
    ↓
MinMax Scaling
    ↓
LSTM Training
    ↓
Prediction
    ↓
Evaluation
    ↓
AI Financial Analysis
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/KishanG02/Stock-AI-Analyst.git
cd Stock-AI-Analyst
```

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=YOUR_GROQ_API_KEY
```

---

## ▶️ Run Locally

```bash
streamlit run app.py
```

Application will launch at:

```text
http://localhost:8501
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit
* Plotly

### Machine Learning

* TensorFlow
* Scikit-Learn
* NumPy
* Pandas

### Financial Data

* Yahoo Finance (yfinance)

### AI

* Groq API
* Llama Models

### Technical Analysis

* ta Library

---

## 📈 Example Analysis

Input:

```text
Ticker: AAPL
```

Output:

```text
Current Price: $210.12
Predicted Price: $214.85

Recommendation:
BUY

Confidence:
Medium

Bullish Factors:
• Positive MACD
• RSI in healthy range
• Forecasted upward trend
```

---

## 🎯 Learning Outcomes

This project demonstrates:

* Time Series Forecasting
* Deep Learning with LSTM
* Financial Data Engineering
* Feature Engineering
* AI-Powered Recommendations
* Streamlit Deployment
* Model Evaluation
* End-to-End ML Workflow

---

## 🚀 Future Enhancements

* Multi-Day Forecasting
* News Sentiment Analysis
* Portfolio Analyzer
* RAG-based Financial Research
* Multi-Agent Investment Advisor
* PDF Investment Reports
* Earnings Call Analysis

---

## 👨‍💻 Author

**Krishna Gupta**

Aspiring AI Engineer

* Python
* AWS
* Machine Learning
* Generative AI
* LLM Applications

---

## ⭐ Support

If you found this project useful:

⭐ Star the repository

🍴 Fork the repository

🛠️ Contribute improvements

---

### Disclaimer

This project is for educational and research purposes only.

It does not constitute financial or investment advice.

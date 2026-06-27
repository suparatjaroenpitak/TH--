import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from cachetools import TTLCache
import streamlit as st

cache = TTLCache(maxsize=100, ttl=1800)

@st.cache_data(ttl=1800, show_spinner="กำลังโหลดข้อมูลหุ้น...")
def fetch_stock_data(tickers, period="2y"):
    data = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            if not hist.empty:
                info = stock.info
                data[ticker] = {
                    "hist": hist,
                    "name": info.get("longName", ticker),
                    "sector": info.get("sector", "N/A"),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE", None),
                    "dividend_yield": info.get("dividendYield", 0),
                }
        except Exception as e:
            st.warning(f"ไม่สามารถโหลดข้อมูล {ticker}: {e}")
    return data

@st.cache_data(ttl=1800, show_spinner="กำลังคำนวณผลตอบแทน...")
def calculate_returns(data):
    returns_df = pd.DataFrame()
    prices_df = pd.DataFrame()
    for ticker, info in data.items():
        prices_df[ticker] = info["hist"]["Close"]
        returns_df[ticker] = info["hist"]["Close"].pct_change()
    return returns_df.dropna(), prices_df

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="1mo")
        current_price = hist["Close"].iloc[-1] if not hist.empty else 0
        return {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": current_price,
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE", None),
            "dividend_yield": info.get("dividendYield", 0),
            "beta": info.get("beta", None),
            "52w_high": info.get("fiftyTwoWeekHigh", 0),
            "52w_low": info.get("fiftyTwoWeekLow", 0),
        }
    except:
        return None

def search_stocks(query):
    try:
        stock = yf.Ticker(query)
        info = stock.info
        if info and info.get("longName"):
            hist = stock.history(period="5d")
            if not hist.empty:
                return {query: info.get("longName", query)}
    except:
        pass
    return {}

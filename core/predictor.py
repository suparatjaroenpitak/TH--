import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class StockPredictor:
    def __init__(self, hist_data):
        self.hist = hist_data
        self.models = {}

    def _create_features(self, df):
        df = df.copy()
        df["returns_1"] = df["Close"].pct_change(1)
        df["returns_5"] = df["Close"].pct_change(5)
        df["returns_20"] = df["Close"].pct_change(20)
        df["sma_5"] = df["Close"].rolling(5).mean()
        df["sma_20"] = df["Close"].rolling(20).mean()
        df["volatility_5"] = df["returns_1"].rolling(5).std()
        df["volatility_20"] = df["returns_1"].rolling(20).std()
        df["volume_ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
        df["high_low_pct"] = (df["High"] - df["Low"]) / df["Close"]
        df["close_open_pct"] = (df["Close"] - df["Open"]) / df["Open"]
        return df.dropna()

    def linear_prediction(self, ticker):
        if ticker not in self.hist:
            return None

        df = self._create_features(self.hist[ticker])
        if len(df) < 30:
            return None

        feature_cols = ["returns_1", "returns_5", "sma_5", "volatility_5",
                        "volume_ratio", "high_low_pct"]
        available = [c for c in feature_cols if c in df.columns]
        if len(available) < 3:
            return None

        X = df[available].values
        y = df["Close"].values

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        try:
            model = LinearRegression()
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)

            last_features = df[available].iloc[-1:].values
            next_price = model.predict(last_features)[0]
            current_price = df["Close"].iloc[-1]

            direction = "UP" if next_price > current_price else "DOWN"
            pct_change = abs((next_price - current_price) / current_price) * 100

            return {
                "current_price": current_price,
                "predicted_price": next_price,
                "change_pct": pct_change,
                "direction": direction,
                "confidence": max(0, min(100, score * 100)),
                "r2_score": score,
            }
        except:
            return None

    def momentum_score(self, ticker):
        if ticker not in self.hist:
            return 0

        df = self.hist[ticker]
        if len(df) < 20:
            return 0

        returns_1m = (df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) if len(df) >= 21 else 0
        returns_3m = (df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1) if len(df) >= 63 else 0
        returns_6m = (df["Close"].iloc[-1] / df["Close"].iloc[-126] - 1) if len(df) >= 126 else 0

        r1 = max(-1, min(1, returns_1m))
        r3 = max(-1, min(1, returns_3m))
        r6 = max(-1, min(1, returns_6m))

        score = (r1 * 0.5 + r3 * 0.3 + r6 * 0.2) * 100
        return score

    def volatility_regime(self, ticker):
        if ticker not in self.hist:
            return "normal"

        df = self.hist[ticker]
        if len(df) < 20:
            return "normal"

        recent_vol = df["Close"].pct_change().tail(20).std()
        hist_vol = df["Close"].pct_change().std()

        if hist_vol == 0:
            return "normal"

        ratio = recent_vol / hist_vol
        if ratio > 1.5:
            return "high_volatility"
        elif ratio > 1.2:
            return "elevated"
        elif ratio < 0.7:
            return "low_volatility"
        else:
            return "normal"

    def predict_multi_ticker(self, tickers):
        predictions = {}
        for t in tickers:
            if t in self.hist:
                pred = self.linear_prediction(t)
                momentum = self.momentum_score(t)
                regime = self.volatility_regime(t)
                predictions[t] = {
                    "prediction": pred,
                    "momentum": momentum,
                    "regime": regime,
                }
        return predictions

    def get_ai_rating(self, ticker):
        """AI rating: 1-100, higher = better buy"""
        score = 50

        pred = self.linear_prediction(ticker)
        if pred:
            if pred["direction"] == "UP" and pred["confidence"] > 30:
                score += pred["confidence"] * 0.3
            else:
                score -= 15

        momentum = self.momentum_score(ticker)
        score += momentum * 0.2

        regime = self.volatility_regime(ticker)
        if regime == "low_volatility": score += 10
        elif regime == "high_volatility": score -= 10
        elif regime == "elevated": score -= 5

        return max(1, min(100, score))

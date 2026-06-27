import numpy as np
import pandas as pd

class RiskManager:
    def __init__(self, returns_df, prices_df):
        self.returns = returns_df
        self.prices = prices_df

    def calculate_var(self, weights, confidence=0.95):
        portfolio_returns = self.returns.dot(weights)
        return np.percentile(portfolio_returns, (1 - confidence) * 100)

    def calculate_cvar(self, weights, confidence=0.95):
        portfolio_returns = self.returns.dot(weights)
        var = self.calculate_var(weights, confidence)
        return portfolio_returns[portfolio_returns <= var].mean()

    def calculate_max_drawdown(self, weights):
        portfolio_price = (1 + self.returns).cumprod().dot(weights)
        running_max = portfolio_price.cummax()
        drawdown = (portfolio_price - running_max) / running_max
        return drawdown.min()

    def calculate_beta(self, weights, market_returns):
        portfolio_returns = self.returns.dot(weights)
        cov = np.cov(portfolio_returns, market_returns)[0, 1]
        var = np.var(market_returns)
        return cov / var if var > 0 else 0

    def risk_report(self, weights):
        report = {}

        portfolio_returns = self.returns.dot(weights)
        report["expected_return"] = float(portfolio_returns.mean() * 252)
        report["volatility"] = float(portfolio_returns.std() * np.sqrt(252))
        report["sharpe_ratio"] = float(
            report["expected_return"] / report["volatility"]
            if report["volatility"] > 0 else 0
        )

        report["var_95"] = float(self.calculate_var(weights, 0.95))
        report["var_99"] = float(self.calculate_var(weights, 0.99))
        report["cvar_95"] = float(self.calculate_cvar(weights, 0.95))
        report["max_drawdown"] = float(self.calculate_max_drawdown(weights))

        neg_returns = portfolio_returns[portfolio_returns < 0]
        report["downside_risk"] = float(neg_returns.std() * np.sqrt(252)) if len(neg_returns) > 0 else 0
        report["win_rate"] = float((portfolio_returns > 0).mean())
        report["best_day"] = float(portfolio_returns.max())
        report["worst_day"] = float(portfolio_returns.min())

        positive = portfolio_returns[portfolio_returns > 0]
        if len(positive) > 0:
            report["avg_gain"] = float(positive.mean())
        else:
            report["avg_gain"] = 0.0
        negative = portfolio_returns[portfolio_returns < 0]
        if len(negative) > 0:
            report["avg_loss"] = float(negative.mean())
        else:
            report["avg_loss"] = 0.0

        return report

    def concentration_risk(self, weights, sector_map):
        sectors = {}
        for i, ticker in enumerate(self.returns.columns):
            sector = sector_map.get(ticker, "Unknown")
            sectors[sector] = sectors.get(sector, 0) + weights[i]

        max_sector = max(sectors.values()) if sectors else 0
        hhi = sum(w ** 2 for w in weights)

        return {
            "sector_exposure": sectors,
            "max_sector_weight": max_sector,
            "hhi_concentration": hhi,
            "is_diversified": hhi < 0.3,
        }

    def stress_test(self, weights, shock_pct=-0.15):
        shock_returns = self.returns.copy()
        shock_returns *= (1 + shock_pct)
        portfolio_shock = shock_returns.dot(weights)
        return {
            "shock_return": float(portfolio_shock.mean()),
            "shock_vol": float(portfolio_shock.std() * np.sqrt(252)),
            "max_loss_shock": float(portfolio_shock.min()),
        }

    def portfolio_health_score(self, weights):
        score = 0
        report = self.risk_report(weights)

        if report["sharpe_ratio"] > 0.5: score += 20
        if report["sharpe_ratio"] > 1.0: score += 15
        if report["sharpe_ratio"] > 1.5: score += 15

        if report["max_drawdown"] > -0.1: score += 15
        elif report["max_drawdown"] > -0.2: score += 10

        if report["var_95"] > -0.02: score += 15
        elif report["var_95"] > -0.03: score += 10

        if report["win_rate"] > 0.55: score += 10

        if report["volatility"] < 0.2: score += 10
        elif report["volatility"] < 0.3: score += 5

        return min(score, 100)

    def get_stop_loss_levels(self, weights, tickers, multiplier=2):
        levels = {}
        portfolio_returns = self.returns.dot(weights)
        daily_vol = portfolio_returns.std()

        for i, ticker in enumerate(tickers):
            stock_vol = self.returns[ticker].std()
            if stock_vol > 0 and daily_vol > 0:
                contribution = weights[i] * stock_vol
                stop_loss = contribution * multiplier
                levels[ticker] = {
                    "stop_loss_pct": float(stop_loss),
                    "daily_vol": float(stock_vol),
                    "contribution": float(contribution),
                }

        return levels

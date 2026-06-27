import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import streamlit as st

class PortfolioOptimizer:
    def __init__(self, returns_df, risk_free_rate=0.025):
        self.returns = returns_df
        self.risk_free_rate = risk_free_rate
        self.n_assets = returns_df.shape[1]
        self.tickers = returns_df.columns.tolist()

    def _estimate_covariance(self):
        lw = LedoitWolf().fit(self.returns.values)
        return lw.covariance_

    def _portfolio_stats(self, weights):
        ret = np.sum(self.returns.mean() * weights) * 252
        cov = self._estimate_covariance()
        vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)) * 252)
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0
        return ret, vol, sharpe

    def _neg_sharpe(self, weights):
        return -self._portfolio_stats(weights)[2]

    def _portfolio_volatility(self, weights):
        return self._portfolio_stats(weights)[1]

    def _portfolio_return(self, weights):
        return self._portfolio_stats(weights)[0]

    def max_sharpe_optimization(self):
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.02, 0.35) for _ in range(self.n_assets))
        init = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            self._neg_sharpe, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            result = minimize(
                self._neg_sharpe, init,
                method="trust-constr",
                bounds=bounds,
                constraints=constraints,
            )

        weights = result.x / np.sum(result.x)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return weights, ret, vol, sharpe

    def min_volatility_optimization(self):
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.02, 0.35) for _ in range(self.n_assets))
        init = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            self._portfolio_volatility, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        weights = result.x / np.sum(result.x)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return weights, ret, vol, sharpe

    def balanced_optimization(self, risk_weight=0.5):
        def objective(weights):
            ret, vol, sharpe = self._portfolio_stats(weights)
            return -(risk_weight * sharpe - (1 - risk_weight) * vol)

        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0.02, 0.35) for _ in range(self.n_assets))
        init = np.array([1.0 / self.n_assets] * self.n_assets)

        result = minimize(
            objective, init,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        weights = result.x / np.sum(result.x)
        ret, vol, sharpe = self._portfolio_stats(weights)
        return weights, ret, vol, sharpe

    def efficient_frontier(self, n_points=20):
        min_vol, max_vol = 0.05, 0.5
        targets = np.linspace(min_vol, max_vol, n_points)
        frontiers = []

        for target in targets:
            constraints = [
                {"type": "eq", "fun": lambda x: np.sum(x) - 1},
                {"type": "eq", "fun": lambda x: self._portfolio_volatility(x) - target},
            ]
            bounds = tuple((0, 1) for _ in range(self.n_assets))
            init = np.array([1.0 / self.n_assets] * self.n_assets)

            result = minimize(
                lambda x: -self._portfolio_stats(x)[0],
                init, method="SLSQP",
                bounds=bounds, constraints=constraints,
                options={"maxiter": 1000},
            )

            if result.success:
                w = result.x / np.sum(result.x)
                r, v, s = self._portfolio_stats(w)
                frontiers.append({"vol": v, "ret": r, "sharpe": s})

        return pd.DataFrame(frontiers)

    def get_portfolio_allocations(self, weights, total_investment):
        latest_prices = self.returns.iloc[-1] * 0
        for t in self.tickers:
            col = self.returns[t].dropna()
            if len(col) > 0:
                latest_prices[t] = self.returns[t].iloc[-1]

        last_prices = {}
        for t in self.tickers:
            series = self.returns[t].dropna()
            reference_price = 0
            for val in self.returns[t].iloc[-5:].values:
                if not np.isnan(val) and val != 0:
                    reference_price = val
                    break
            last_prices[t] = max(reference_price, 1)

        allocations = []
        for i, ticker in enumerate(self.tickers):
            weight = weights[i]
            amount = total_investment * weight
            price = last_prices[ticker]

            shares = int(amount / price) if price > 0 else 0
            actual_amount = shares * price

            allocations.append({
                "ticker": ticker,
                "weight": weight,
                "amount": actual_amount,
                "shares": shares,
                "price": price,
            })

        total_actual = sum(a["amount"] for a in allocations)
        remaining = total_investment - total_actual

        if remaining > 0 and allocations:
            max_idx = max(range(len(allocations)), key=lambda i: allocations[i]["weight"])
            price = allocations[max_idx]["price"]
            extra_shares = int(remaining / price)
            if extra_shares > 0:
                allocations[max_idx]["shares"] += extra_shares
                allocations[max_idx]["amount"] += extra_shares * price

        return allocations

    def monte_carlo_simulation(self, n_simulations=1000):
        results = np.zeros((3, n_simulations))
        for i in range(n_simulations):
            weights = np.random.random(self.n_assets)
            weights = weights / np.sum(weights)
            ret, vol, sharpe = self._portfolio_stats(weights)
            results[0, i] = ret
            results[1, i] = vol
            results[2, i] = sharpe
        return results

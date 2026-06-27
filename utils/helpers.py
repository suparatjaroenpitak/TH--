import pandas as pd
import numpy as np

def format_currency(value, currency="฿"):
    if value >= 1_000_000_000:
        return f"{currency}{value/1_000_000_000:.2f}B"
    elif value >= 1_000_000:
        return f"{currency}{value/1_000_000:.2f}M"
    elif value >= 1_000:
        return f"{currency}{value:,.2f}"
    return f"{currency}{value:.2f}"

def format_percent(value):
    if isinstance(value, (int, float)):
        return f"{value * 100 if abs(value) < 1 else value:.2f}%"
    return str(value)

def format_ratio(value):
    return f"{value:.2f}"

def color_for_change(value):
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    return "gray"

def interpret_sharpe(sharpe):
    if sharpe >= 2: return "ดีเยี่ยม"
    if sharpe >= 1.5: return "ดีมาก"
    if sharpe >= 1: return "ดี"
    if sharpe >= 0.5: return "พอใช้"
    return "ควรปรับปรุง"

def interpret_var(var):
    if var > -0.01: return "ความเสี่ยงต่ำ"
    if var > -0.02: return "ความเสี่ยงปานกลาง"
    if var > -0.03: return "ความเสี่ยงสูง"
    return "ความเสี่ยงสูงมาก"

def interpret_health(score):
    if score >= 80: return "แข็งแรงมาก"
    if score >= 60: return "แข็งแรง"
    if score >= 40: return "ปานกลาง"
    if score >= 20: return "อ่อนแอ"
    return "เสี่ยง"

def calculate_required_amount(allocations):
    return sum(a["amount"] for a in allocations)

def suggest_monthly_investment(target_amount, months=12, current_amount=0):
    remaining = target_amount - current_amount
    if remaining <= 0:
        return 0
    return remaining / months

def filter_stocks_by_budget(allocations, budget):
    affordable = []
    for a in allocations:
        if a["amount"] <= budget:
            affordable.append(a)
    return affordable

def get_market_summary(data):
    summary = {
        "total_stocks": len(data),
        "avg_pe": [],
        "avg_div_yield": [],
        "sectors": {},
    }

    for ticker, info in data.items():
        if info.get("pe_ratio") and info["pe_ratio"] > 0:
            summary["avg_pe"].append(info["pe_ratio"])
        if info.get("dividend_yield"):
            summary["avg_div_yield"].append(info["dividend_yield"])

        sector = info.get("sector", "N/A")
        summary["sectors"][sector] = summary["sectors"].get(sector, 0) + 1

    summary["avg_pe"] = np.mean(summary["avg_pe"]) if summary["avg_pe"] else 0
    summary["avg_div_yield"] = np.mean(summary["avg_div_yield"]) if summary["avg_div_yield"] else 0

    return summary

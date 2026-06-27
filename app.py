import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="AI จัดพอร์ตลงทุน",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import SET_TICKERS, US_TICKERS, RISK_FREE_RATE, DEFAULT_INVESTMENT
from core.data_fetcher import fetch_stock_data, calculate_returns, get_stock_info
from core.portfolio_optimizer import PortfolioOptimizer
from core.risk_manager import RiskManager
from core.predictor import StockPredictor
from utils.helpers import (
    format_currency, format_percent, format_ratio, color_for_change,
    interpret_sharpe, interpret_var, interpret_health,
    get_market_summary, suggest_monthly_investment,
)
from core.alpaca_broker import AlpacaBroker, ALPACA_AVAILABLE

if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}
if "selected_tickers" not in st.session_state:
    st.session_state.selected_tickers = []
if "optimization_done" not in st.session_state:
    st.session_state.optimization_done = False
if "alpaca_broker" not in st.session_state:
    st.session_state.alpaca_broker = None
if "alpaca_connected" not in st.session_state:
    st.session_state.alpaca_connected = False
if "executed_orders" not in st.session_state:
    st.session_state.executed_orders = []

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1a5276; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.4rem; font-weight: 600; color: #2c3e50; margin-bottom: 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem; border-radius: 12px; color: white; text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card-green { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .metric-card-red { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .metric-card-blue { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .metric-card-gold { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; opacity: 0.9; }
    .health-bar {
        height: 10px; border-radius: 5px; background: #e0e0e0; margin: 8px 0;
    }
    .health-fill { height: 10px; border-radius: 5px; background: linear-gradient(90deg, #00b09b, #96c93d); }
    .info-card { padding: 1rem; border-radius: 10px; background: #f8f9fa; border-left: 4px solid #667eea; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🤖 AI จัดพอร์ตลงทุนอัตโนมัติ</div>', unsafe_allow_html=True)
st.markdown("ระบบ AI จัดสรรพอร์ตการลงทุนให้มีกำไรดีแต่ขาดทุนน้อย เพียงใส่จำนวนเงินที่ต้องการลงทุน")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 จัดพอร์ตอัตโนมัติ",
    "📋 รายละเอียดพอร์ต",
    "📈 วิเคราะห์ความเสี่ยง",
    "🔮 AI พยากรณ์",
    "💰 ซื้อขายจริง (Alpaca)",
    "⚙️ ตั้งค่า",
])

with st.sidebar:
    st.markdown("### 🎯 ตั้งค่าการลงทุน")

    market = st.radio("เลือกตลาดหุ้น", ["SET (ไทย)", "US (อเมริกา)", "ผสมทั้งสองตลาด"], index=0)

    if market == "SET (ไทย)":
        all_tickers = SET_TICKERS
    elif market == "US (อเมริกา)":
        all_tickers = US_TICKERS
    else:
        all_tickers = {**SET_TICKERS, **US_TICKERS}

    ticker_options = list(all_tickers.keys())

    selected = st.multiselect(
        "เลือกหุ้นที่สนใจ (3-15 ตัว)",
        ticker_options,
        default=ticker_options[:8] if len(ticker_options) >= 8 else ticker_options,
        max_selections=15,
    )

    investment = st.number_input(
        "จำนวนเงินลงทุน (บาท)",
        min_value=1000,
        max_value=100_000_000,
        value=DEFAULT_INVESTMENT,
        step=10000,
        format="%d",
    )

    risk_level = st.select_slider(
        "ระดับความเสี่ยง",
        options=["ต่ำมาก", "ต่ำ", "ปานกลาง", "สูง", "สูงมาก"],
        value="ปานกลาง",
    )

    risk_map = {"ต่ำมาก": 0.15, "ต่ำ": 0.3, "ปานกลาง": 0.5, "สูง": 0.7, "สูงมาก": 0.85}

    optimize_btn = st.button("🚀 จัดพอร์ตอัตโนมัติ", type="primary", use_container_width=True)

    st.divider()
    st.markdown("### 🔑 เชื่อมต่อ Alpaca")

    if not ALPACA_AVAILABLE:
        st.warning("⚠️ ยังไม่ได้ติดตั้ง `alpaca-trade-api` รัน: `pip install alpaca-trade-api`")
    else:
        alpaca_api_key = st.text_input("API Key ID", type="password", placeholder="PK...")
        alpaca_secret_key = st.text_input("Secret Key", type="password", placeholder="...")
        use_paper = st.checkbox("💰 โหมด Paper Trading (ทดลอง)", value=True)

        if st.button("เชื่อมต่อ", use_container_width=True):
            if not alpaca_api_key or not alpaca_secret_key:
                st.error("กรุณาใส่ API Key และ Secret Key")
            else:
                with st.spinner("กำลังเชื่อมต่อ Alpaca..."):
                    broker = AlpacaBroker(alpaca_api_key, alpaca_secret_key, paper=use_paper)
                    if broker.connected:
                        st.session_state.alpaca_broker = broker
                        st.session_state.alpaca_connected = True
                        st.success("✅ เชื่อมต่อสำเร็จ!")
                        st.rerun()
                    else:
                        st.error("❌ เชื่อมต่อไม่สำเร็จ ตรวจสอบ API Key")

        if st.session_state.alpaca_connected:
            st.success("✅ เชื่อมต่ออยู่" + (" (Paper)" if st.session_state.alpaca_broker.paper else " (Real)"))
            if st.button("ตัดการเชื่อมต่อ", use_container_width=True):
                st.session_state.alpaca_broker = None
                st.session_state.alpaca_connected = False
                st.rerun()

    st.divider()
    st.markdown("### 💡 Tips")
    st.info(
        "🔹 เลือกหุ้น 8-12 ตัวเพื่อการกระจายความเสี่ยงที่ดี\n\n"
        "🔹 ลงทุนอย่างน้อย 50,000 บาทเพื่อการจัดพอร์ตที่มีประสิทธิภาพ\n\n"
        "🔹 AI จะจัดสรรน้ำหนักหุ้นให้เหมาะสมกับระดับความเสี่ยง\n\n"
        "🔹 เชื่อมต่อ Alpaca เพื่อเทรดจริง (Paper/Live)"
    )

if optimize_btn and selected:
    st.session_state.selected_tickers = selected

    with st.spinner("🔄 กำลังวิเคราะห์ข้อมูล... (ใช้เวลา 1-2 นาที)"):
        progress_bar = st.progress(0, text="กำลังดาวน์โหลดข้อมูลหุ้น...")

        tickers_map = {v: k for k, v in all_tickers.items() if k in selected}
        actual_tickers = [all_tickers[t] for t in selected]

        progress_bar.progress(20, text="กำลังวิเคราะห์ผลตอบแทน...")
        stock_data = fetch_stock_data(actual_tickers)

        if len(stock_data) < 3:
            st.error(f"โหลดข้อมูลได้เพียง {len(stock_data)} ตัว กรุณาเลือกหุ้นใหม่อีกครั้ง")
            st.stop()

        progress_bar.progress(40, text="กำลังคำนวณค่าสถิติ...")
        returns_df, prices_df = calculate_returns(stock_data)

        valid_tickers = list(returns_df.columns)
        valid_names = [tickers_map.get(t, t) for t in valid_tickers]

        progress_bar.progress(60, text="AI กำลังหาพอร์ตที่เหมาะสมที่สุด...")
        optimizer = PortfolioOptimizer(returns_df, RISK_FREE_RATE)
        risk_w = risk_map[risk_level]

        if risk_w <= 0.3:
            weights, ret, vol, sharpe = optimizer.min_volatility_optimization()
            strategy = "เน้นความปลอดภัย (Min Volatility)"
        elif risk_w >= 0.7:
            weights, ret, vol, sharpe = optimizer.max_sharpe_optimization()
            strategy = "เน้นผลตอบแทน (Max Sharpe)"
        else:
            weights, ret, vol, sharpe = optimizer.balanced_optimization(risk_w)
            strategy = f"สมดุล (Balanced) - น้ำหนักความเสี่ยง {risk_level}"

        progress_bar.progress(80, text="กำลังวิเคราะห์ความเสี่ยง...")
        risk_manager = RiskManager(returns_df, prices_df)
        risk_report = risk_manager.risk_report(weights)

        sector_map = {}
        for t in valid_tickers:
            if t in stock_data:
                sector_map[t] = stock_data[t].get("sector", "Unknown")
        concentration = risk_manager.concentration_risk(weights, sector_map)

        progress_bar.progress(90, text="AI กำลังพยากรณ์แนวโน้ม...")
        hist_data = {t: stock_data[t]["hist"] for t in valid_tickers}
        predictor = StockPredictor(hist_data)
        predictions = predictor.predict_multi_ticker(valid_tickers)

        allocations = optimizer.get_portfolio_allocations(weights, investment)
        health_score = risk_manager.portfolio_health_score(weights)
        stop_losses = risk_manager.get_stop_loss_levels(weights, valid_tickers)

        progress_bar.progress(100, text="✅ เสร็จสิ้น!")

        st.session_state.update({
            "optimization_done": True,
            "weights": weights,
            "ret": ret,
            "vol": vol,
            "sharpe": sharpe,
            "allocations": allocations,
            "risk_report": risk_report,
            "concentration": concentration,
            "predictions": predictions,
            "strategy": strategy,
            "health_score": health_score,
            "stock_data": stock_data,
            "returns_df": returns_df,
            "prices_df": prices_df,
            "valid_tickers": valid_tickers,
            "valid_names": valid_names,
            "optimizer": optimizer,
            "risk_manager": risk_manager,
            "predictor": predictor,
            "stop_losses": stop_losses,
            "investment": investment,
        })

        st.rerun()

with tab1:
    if not st.session_state.optimization_done:
        st.info("👈 เลือกหุ้นและจำนวนเงินที่ต้องการลงทุน จากนั้นกด **'จัดพอร์ตอัตโนมัติ'**")

        st.markdown("### ✨ ฟีเจอร์ของระบบ")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #667eea, #764ba2);">
                <div style="font-size:2.5rem;">📊</div>
                <div class="metric-value">AI Optimization</div>
                <div class="metric-label">ใช้ Mean-Variance Optimization</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #11998e, #38ef7d);">
                <div style="font-size:2.5rem;">🛡️</div>
                <div class="metric-value">Risk Management</div>
                <div class="metric-label">VaR, CVaR, Drawdown</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe, #00f2fe);">
                <div style="font-size:2.5rem;">🔮</div>
                <div class="metric-value">AI Prediction</div>
                <div class="metric-label">Linear Regression + Momentum</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown("""
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb, #f5576c);">
                <div style="font-size:2.5rem;">🔄</div>
                <div class="metric-value">Auto Rebalance</div>
                <div class="metric-label">จัดสรรน้ำหนักอัตโนมัติ</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🎯 วิธีใช้งาน")
        st.markdown("""
        1. **เลือกตลาดหุ้น** - SET (ไทย) หรือ US (อเมริกา)
        2. **เลือกหุ้น** - อย่างน้อย 3 ตัว เพื่อกระจายความเสี่ยง
        3. **กำหนดเงินลงทุน** - จำนวนเงินที่ต้องการลงทุน
        4. **เลือกระดับความเสี่ยง** - ต่ำมาก ถึง สูงมาก
        5. **กด จัดพอร์ตอัตโนมัติ** - AI จัดสรรให้อัตโนมัติ
        """)
    else:
        s = st.session_state

        st.markdown(f'<div class="sub-header">📍 พอร์ตการลงทุน — {s["strategy"]}</div>', unsafe_allow_html=True)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="metric-card metric-card-green">
                <div class="metric-label">ผลตอบแทนคาดหวัง (ต่อปี)</div>
                <div class="metric-value">{s["ret"]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card metric-card-red">
                <div class="metric-label">ความผันผวน (ต่อปี)</div>
                <div class="metric-value">{s["vol"]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card metric-card-blue">
                <div class="metric-label">Sharpe Ratio</div>
                <div class="metric-value">{s["sharpe"]:.2f}</div>
                <div class="metric-label">{interpret_sharpe(s["sharpe"])}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="metric-card metric-card-gold">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value">{s["risk_report"]["max_drawdown"]*100:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            score = s["health_score"]
            health_text = interpret_health(score)
            bar_color = "green" if score >= 60 else "orange" if score >= 40 else "red"
            st.markdown(f"""
            <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #3498db);">
                <div class="metric-label">Health Score</div>
                <div class="metric-value">{score}/100</div>
                <div class="health-bar"><div class="health-fill" style="width:{score}%;background:{bar_color};"></div></div>
                <div class="metric-label">{health_text}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("### 📋 ตารางจัดสรรเงินลงทุน")
        alloc_df = pd.DataFrame(s["allocations"])
        alloc_df["ticker"] = s["valid_names"]
        alloc_df["weight"] = alloc_df["weight"].apply(lambda x: f"{x*100:.2f}%")
        alloc_df["amount"] = alloc_df["amount"].apply(lambda x: f"฿{x:,.2f}")
        alloc_df["shares"] = alloc_df["shares"].astype(int)
        alloc_df["price"] = alloc_df["price"].apply(lambda x: f"฿{x:,.2f}")
        alloc_df.columns = ["หุ้น", "% น้ำหนัก", "จำนวนเงิน", "จำนวนหุ้น", "ราคาล่าสุด"]

        st.dataframe(alloc_df, use_container_width=True, hide_index=True)

        total_allocated = sum(a["amount"] for a in s["allocations"])
        remaining = s["investment"] - total_allocated
        st.caption(f"💰 ลงทุนรวม: ฿{total_allocated:,.2f} | เงินคงเหลือ: ฿{remaining:,.2f}")

        st.markdown("---")

        fig_pie = px.pie(
            names=s["valid_names"],
            values=[w * 100 for w in s["weights"]],
            title="สัดส่วนการลงทุน",
            hole=0.4,
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                              marker=dict(line=dict(color="white", width=2)))
        fig_pie.update_layout(height=400)

        fig_bar = go.Figure()
        colors = ["#2ecc71" if w > 1/len(s["weights"]) else "#e74c3c" for w in s["weights"]]
        fig_bar.add_trace(go.Bar(
            x=s["valid_names"],
            y=s["weights"] * 100,
            marker_color=colors,
            text=[f"{w*100:.1f}%" for w in s["weights"]],
            textposition="outside",
        ))
        fig_bar.update_layout(
            title="น้ำหนักการลงทุนแต่ละตัว",
            xaxis_title="หุ้น",
            yaxis_title="น้ำหนัก (%)",
            height=400,
            yaxis=dict(range=[0, max(s["weights"]) * 100 * 1.2]),
        )

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    if not st.session_state.optimization_done:
        st.info("👈 กรุณาจัดพอร์ตก่อนในแท็บ 'จัดพอร์ตอัตโนมัติ'")
    else:
        s = st.session_state

        st.markdown("### 📋 รายละเอียดพอร์ตทั้งหมด")

        st.markdown("#### ข้อมูลหุ้นในพอร์ต")
        stock_info_rows = []
        for i, ticker in enumerate(s["valid_tickers"]):
            if ticker in s["stock_data"]:
                info = s["stock_data"][ticker]
                weight_pct = s["weights"][i] * 100
                price_df = info["hist"]["Close"]
                change_1m = (price_df.iloc[-1] / price_df.iloc[-21] - 1) * 100 if len(price_df) >= 21 else 0

                pred_info = s["predictions"].get(ticker, {})
                ai_rating = s["predictor"].get_ai_rating(ticker) if hasattr(s["predictor"], "get_ai_rating") else 50
                direction = pred_info.get("prediction", {}).get("direction", "-") if pred_info else "-"

                stock_info_rows.append({
                    "หุ้น": s["valid_names"][i],
                    "น้ำหนัก": f"{weight_pct:.2f}%",
                    "ราคาล่าสุด": f"฿{price_df.iloc[-1]:.2f}",
                    "P/E": f"{info.get('pe_ratio', '-'):.2f}" if info.get("pe_ratio") else "-",
                    "Div Yield": f"{info.get('dividend_yield', 0)*100:.2f}%" if info.get("dividend_yield") else "-",
                    "เปลี่ยน 1 เดือน": f"{change_1m:+.2f}%",
                    "AI Rating": f"{ai_rating:.0f}/100",
                    "AI ทิศทาง": direction,
                })

        if stock_info_rows:
            st.dataframe(pd.DataFrame(stock_info_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Cumulative Return Simulation")

        portfolio_returns = s["returns_df"].dot(s["weights"])
        cumulative = (1 + portfolio_returns).cumprod()
        cumulative.index = pd.to_datetime(cumulative.index)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative.index,
            y=cumulative.values,
            mode="lines",
            name="พอร์ต AI",
            line=dict(color="#2ecc71", width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 204, 113, 0.1)",
        ))

        equal_weight = s["returns_df"].dot(np.array([1/len(s["valid_tickers"])] * len(s["valid_tickers"])))
        eq_cumulative = (1 + equal_weight).cumprod()
        eq_cumulative.index = pd.to_datetime(eq_cumulative.index)
        fig.add_trace(go.Scatter(
            x=eq_cumulative.index,
            y=eq_cumulative.values,
            mode="lines",
            name="ถัวเฉลี่ยเท่ากัน",
            line=dict(color="#3498db", width=2, dash="dash"),
        ))

        fig.update_layout(
            title="เปรียบเทียบผลการดำเนินงาน",
            xaxis_title="วันที่",
            yaxis_title="มูลค่า (倍数)",
            height=450,
            hovermode="x unified",
        )
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📈 Monte Carlo Simulation")
        mc_results = s["optimizer"].monte_carlo_simulation(500)

        fig_mc = go.Figure()
        fig_mc.add_trace(go.Scatter(
            x=mc_results[1], y=mc_results[0],
            mode="markers",
            marker=dict(color=mc_results[2], colorscale="Viridis", size=4, showscale=True,
                        colorbar=dict(title="Sharpe")),
            name="จำลอง 500 พอร์ต",
        ))
        fig_mc.add_trace(go.Scatter(
            x=[s["vol"]], y=[s["ret"]],
            mode="markers",
            marker=dict(color="red", size=15, symbol="star"),
            name="พอร์ตที่เลือก",
        ))

        fig_mc.update_layout(
            title="Monte Carlo Simulation (500 พอร์ต)",
            xaxis_title="ความเสี่ยง (Volatility)",
            yaxis_title="ผลตอบแทน (Return)",
            height=500,
        )
        st.plotly_chart(fig_mc, use_container_width=True)

with tab3:
    if not st.session_state.optimization_done:
        st.info("👈 กรุณาจัดพอร์ตก่อนในแท็บ 'จัดพอร์ตอัตโนมัติ'")
    else:
        s = st.session_state
        r = s["risk_report"]

        st.markdown("### 🛡️ รายงานความเสี่ยง")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#e74c3c;">
                <b>Value at Risk (95%)</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['var_95']*100:.2f}%</span><br>
                <small>{interpret_var(r['var_95'])}</small>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#e67e22;">
                <b>Conditional VaR (95%)</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['cvar_95']*100:.2f}%</span><br>
                <small>ขาดทุนเฉลี่ยในวันที่แย่ที่สุด</small>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#3498db;">
                <b>Win Rate</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['win_rate']*100:.2f}%</span><br>
                <small>โอกาสทำกำไรในแต่ละวัน</small>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#9b59b6;">
                <b>Downside Risk</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['downside_risk']*100:.2f}%</span><br>
                <small>ความเสี่ยงด้านลบ</small>
            </div>
            """, unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#2ecc71;">
                <b>ค่าเฉลี่ยกำไร (วัน)</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['avg_gain']*100:+.4f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#e74c3c;">
                <b>ค่าเฉลี่ยขาดทุน (วัน)</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['avg_loss']*100:+.4f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#f39c12;">
                <b>วันที่ดีที่สุด</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['best_day']*100:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            st.markdown(f"""
            <div class="info-card" style="border-left-color:#c0392b;">
                <b>วันที่แย่ที่สุด</b><br>
                <span style="font-size:1.3rem;font-weight:bold;">{r['worst_day']*100:+.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        st.markdown("#### 🏢 การกระจายตัวตาม Sector")
        sector_data = s["concentration"]["sector_exposure"]
        if sector_data:
            sector_df = pd.DataFrame([
                {"Sector": k, "น้ำหนัก": v * 100}
                for k, v in sector_data.items()
            ])
            fig_sector = px.bar(
                sector_df, x="Sector", y="น้ำหนัก",
                color="น้ำหนัก", color_continuous_scale="Blues",
                text=sector_df["น้ำหนัก"].apply(lambda x: f"{x:.1f}%"),
            )
            fig_sector.update_layout(height=350)
            st.plotly_chart(fig_sector, use_container_width=True)

            conc_text = "✅ กระจายตัวดี" if s["concentration"]["is_diversified"] else "⚠️ กระจุกตัวสูง"
            st.write(f"HHI Concentration: {s['concentration']['hhi_concentration']:.3f} — {conc_text}")

        st.markdown("---")
        st.markdown("#### 🧪 Stress Test Simulation")
        shock_col1, shock_col2 = st.columns(2)
        with shock_col1:
            shock_pct = st.slider("จำลองวิกฤต: ตลาดตก", -5, -50, -15, 5, format="%d%%")
        with shock_col2:
            pass

        shock_result = s["risk_manager"].stress_test(s["weights"], shock_pct / 100)
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("ผลกระทบรวม", f"{shock_result['shock_return']*100:.2f}%")
        with col_b:
            st.metric("ความผันผวนในวิกฤต", f"{shock_result['shock_vol']*100:.2f}%")
        with col_c:
            st.metric("ขาดทุนสูงสุด", f"{shock_result['max_loss_shock']*100:.2f}%")

with tab4:
    if not st.session_state.optimization_done:
        st.info("👈 กรุณาจัดพอร์ตก่อนในแท็บ 'จัดพอร์ตอัตโนมัติ'")
    else:
        s = st.session_state

        st.markdown("### 🔮 AI พยากรณ์แนวโน้มหุ้น")

        st.markdown("ระบบใช้ **Linear Regression** ร่วมกับ **Momentum Analysis** "
                    "เพื่อพยากรณ์ทิศทางราคา")

        pred_rows = []
        for i, ticker in enumerate(s["valid_tickers"]):
            pred_info = s["predictions"].get(ticker, {})
            momentum = pred_info.get("momentum", 0) if pred_info else 0
            regime = pred_info.get("regime", "normal") if pred_info else "normal"
            direction = "-"
            confidence = 0
            change_pct = 0

            if pred_info and pred_info.get("prediction"):
                p = pred_info["prediction"]
                direction = p["direction"]
                confidence = p["confidence"]
                change_pct = p["change_pct"]

            regime_emoji = {"high_volatility": "🔴", "elevated": "🟡", "normal": "🟢", "low_volatility": "🔵"}.get(regime, "⚪")
            regime_th = {"high_volatility": "ผันผวนสูง", "elevated": "ผันผวนปานกลาง",
                         "normal": "ปกติ", "low_volatility": "ผันผวนต่ำ"}.get(regime, "-")

            dir_emoji = "🟢" if direction == "UP" else "🔴" if direction == "DOWN" else "⚪"
            ai_rating = s["predictor"].get_ai_rating(ticker)

            pred_rows.append({
                "หุ้น": s["valid_names"][i],
                "AI เรตติ้ง": f"{ai_rating:.0f}/100",
                "ทิศทาง": f"{dir_emoji} {direction}",
                "ความมั่นใจ": f"{confidence:.1f}%",
                "เปลี่ยนแปลง": f"{change_pct:.2f}%",
                "Momentum": f"{momentum:+.2f}",
                "สถานะตลาด": f"{regime_emoji} {regime_th}",
            })

        if pred_rows:
            st.dataframe(pd.DataFrame(pred_rows), use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### 📊 Momentum Score")
        momentum_scores = []
        for i, ticker in enumerate(s["valid_tickers"]):
            score = s["predictor"].momentum_score(ticker)
            momentum_scores.append({"หุ้น": s["valid_names"][i], "Momentum Score": score})

        if momentum_scores:
            mom_df = pd.DataFrame(momentum_scores)
            fig_mom = px.bar(
                mom_df.sort_values("Momentum Score"),
                x="หุ้น", y="Momentum Score",
                color="Momentum Score",
                color_continuous_scale="RdYlGn",
                text=mom_df.sort_values("Momentum Score")["Momentum Score"].apply(lambda x: f"{x:.1f}"),
            )
            fig_mom.update_layout(height=400)
            st.plotly_chart(fig_mom, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📋 คำแนะนำ AI รวม")
        st.info("💡 **กลยุทธ์:** AI แนะนำให้ลงทุนในหุ้นที่มี Momentum เป็นบวก "
                "และอยู่ในสภาวะตลาดปกติ ลดน้ำหนักหุ้นที่มีความผันผวนสูง")

        strong_buy = sum(1 for t in s["valid_tickers"]
                        if s["predictor"].get_ai_rating(t) >= 70)
        buy = sum(1 for t in s["valid_tickers"]
                 if 50 <= s["predictor"].get_ai_rating(t) < 70)
        hold = sum(1 for t in s["valid_tickers"]
                  if 30 <= s["predictor"].get_ai_rating(t) < 50)
        sell = sum(1 for t in s["valid_tickers"]
                  if s["predictor"].get_ai_rating(t) < 30)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 Strong Buy", strong_buy)
        with col2:
            st.metric("🔵 Buy", buy)
        with col3:
            st.metric("🟡 Hold", hold)
        with col4:
            st.metric("🔴 Sell", sell)

with tab5:
    if not st.session_state.alpaca_connected or not st.session_state.alpaca_broker:
        st.info("🔑 กรุณาเชื่อมต่อ Alpaca API ที่เมนูด้านซ้ายก่อน")
    elif not st.session_state.optimization_done:
        st.info("👈 กรุณาจัดพอร์ตก่อนในแท็บ 'จัดพอร์ตอัตโนมัติ'")
    else:
        s = st.session_state
        broker = s["alpaca_broker"]

        st.markdown("### 💰 ซื้อขายจริงผ่าน Alpaca")

        col1, col2, col3, col4 = st.columns(4)
        account = broker.get_account_info()
        if account:
            with col1:
                st.markdown(f"""
                <div class="metric-card metric-card-green">
                    <div class="metric-label">เงินในบัญชี</div>
                    <div class="metric-value">${account['cash']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card metric-card-blue">
                    <div class="metric-label">มูลค่าพอร์ต</div>
                    <div class="metric-value">${account['portfolio_value']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div class="metric-card metric-card-gold">
                    <div class="metric-label">Equity</div>
                    <div class="metric-value">${account['equity']:,.2f}</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                mode = "📄 Paper" if account["is_paper"] else "🔴 Live"
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #2c3e50, #3498db);">
                    <div class="metric-label">โหมด</div>
                    <div class="metric-value">{mode}</div>
                </div>
                """, unsafe_allow_html=True)

        clock = broker.get_clock()
        if clock:
            if clock["is_open"]:
                st.success(f"🟢 ตลาดเปิดอยู่ — จะปิด {clock['next_close'].strftime('%H:%M')}")
            else:
                st.warning(f"🔴 ตลาดปิดอยู่ — จะเปิด {clock['next_open'].strftime('%d/%m %H:%M')}")

        st.markdown("---")
        st.markdown("#### 📋 พอร์ตปัจจุบัน (จาก Alpaca)")

        positions = broker.get_positions()
        if positions:
            pos_df = pd.DataFrame(positions)
            pos_df["ticker"] = pos_df["ticker"]
            pos_df["qty"] = pos_df["qty"].astype(int)
            pos_df["avg_entry_price"] = pos_df["avg_entry_price"].apply(lambda x: f"${x:.2f}")
            pos_df["current_price"] = pos_df["current_price"].apply(lambda x: f"${x:.2f}")
            pos_df["market_value"] = pos_df["market_value"].apply(lambda x: f"${x:,.2f}")
            pos_df["unrealized_pl"] = pos_df["unrealized_pl"].apply(lambda x: f"${x:+,.2f}")
            pos_df["unrealized_plpc"] = pos_df["unrealized_plpc"].apply(lambda x: f"{x*100:+.2f}%")
            display_pos = pos_df[["ticker", "qty", "avg_entry_price", "current_price",
                                  "market_value", "unrealized_pl", "unrealized_plpc"]]
            display_pos.columns = ["หุ้น", "จำนวน", "ราคาเฉลี่ย", "ราคาปัจจุบัน",
                                   "มูลค่า", "กำไร/ขาดทุน", "% เปลี่ยนแปลง"]
            st.dataframe(display_pos, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีพอร์ตใน Alpaca")

        st.markdown("---")
        st.markdown("#### 🚀 สั่งซื้อตามที่ AI จัดพอร์ต")

        st.caption(f"AI แนะนำให้ลงทุน ${s['investment']:,.2f}")
        st.caption(f"เงินในบัญชี: ${account['cash']:,.2f}" if account else "")

        alloc_df = pd.DataFrame(s["allocations"])
        alloc_df["ticker"] = s["valid_names"]

        col_a, col_b = st.columns([1, 1])
        with col_a:
            rebalance_btn = st.button("🔄 Rebalance อัตโนมัติ (ขายของเก่า + ซื้อของใหม่)",
                                      type="primary", use_container_width=True)
        with col_b:
            buy_only_btn = st.button("➕ ซื้อเพิ่มอย่างเดียว", use_container_width=True)

        if rebalance_btn:
            if not clock or not clock["is_open"]:
                st.error("❌ ตลาดปิดอยู่ ไม่สามารถส่งคำสั่งซื้อขายได้")
            else:
                with st.spinner("🔄 กำลัง Rebalance พอร์ต..."):
                    orders = broker.rebalance_portfolio(s["allocations"], s["investment"])
                    if orders:
                        s["executed_orders"].extend(orders)
                        st.success(f"✅ ส่งคำสั่งซื้อขายสำเร็จ {len(orders)} รายการ")
                        for o in orders:
                            st.info(f"{o['side'].upper()} {o['qty']} {o['symbol']} — {o['status']}")
                        st.rerun()
                    else:
                        st.warning("ไม่มีการเปลี่ยนแปลงหรือเกิดข้อผิดพลาด")

        if buy_only_btn:
            if not clock or not clock["is_open"]:
                st.error("❌ ตลาดปิดอยู่ ไม่สามารถส่งคำสั่งซื้อได้")
            else:
                existing = {p["ticker"]: p["qty"] for p in positions}
                with st.spinner("กำลังซื้อหุ้นตาม AI จัดสรร..."):
                    orders = []
                    for alloc in s["allocations"]:
                        ticker = alloc["ticker"]
                        target_qty = alloc["shares"]
                        current_qty = existing.get(ticker, 0)
                        diff = target_qty - current_qty

                        if diff > 0 and account and account["cash"] > 0:
                            order = broker.place_market_order(ticker, diff, "buy")
                            if order:
                                orders.append(order)
                                st.success(f"✅ ซื้อ {diff} {ticker}")
                    if orders:
                        s["executed_orders"].extend(orders)
                        st.rerun()
                    else:
                        st.info("ไม่มีหุ้นที่ต้องซื้อเพิ่ม หรือเงินไม่พอ")

        st.markdown("---")
        st.markdown("#### 📜 ประวัติคำสั่งล่าสุด")

        alpaca_orders = broker.get_orders(status="all", limit=20) if st.session_state.alpaca_connected else []
        if alpaca_orders:
            ord_df = pd.DataFrame(alpaca_orders)
            ord_df = ord_df[["symbol", "side", "qty", "filled_qty", "type", "status", "created_at"]]
            ord_df.columns = ["หุ้น", "ซื้อ/ขาย", "จำนวน", "จำนวนที่เติม", "ประเภท", "สถานะ", "เวลา"]
            ord_df["เวลา"] = pd.to_datetime(ord_df["เวลา"]).dt.strftime("%d/%m %H:%M")
            st.dataframe(ord_df, use_container_width=True, hide_index=True)
        else:
            st.info("ยังไม่มีประวัติคำสั่งซื้อขาย")

        st.markdown("---")
        with st.expander("⚠️ คำสั่งฉุกเฉิน"):
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                if st.button("🛑 ยกเลิกคำสั่งทั้งหมด", type="secondary", use_container_width=True):
                    if broker.cancel_all_orders():
                        st.success("ยกเลิกคำสั่งทั้งหมดแล้ว")
                        st.rerun()
            with col_c2:
                if st.button("🔴 ขายทั้งหมด (Liquidate)", type="secondary", use_container_width=True):
                    if broker.close_all_positions():
                        st.success("ขายทั้งหมดแล้ว")
                        st.rerun()

with tab6:
    st.markdown("### ⚙️ ตั้งค่าระบบ")

    st.markdown("#### วิธีการทำงานของ AI")
    st.markdown("""
    ระบบ AI จัดพอร์ตนี้ทำงานด้วยขั้นตอนอัจฉริยะดังนี้:

    1. **📥 ดึงข้อมูล** — ดึงข้อมูลราคาหุ้นย้อนหลัง 2 ปีจาก Yahoo Finance
    2. **📊 วิเคราะห์สถิติ** — คำนวณผลตอบแทน ความผันผวน ความสัมพันธ์ของหุ้นแต่ละตัว
    3. **🧮 AI Optimization** — ใช้เทคนิค Modern Portfolio Theory ร่วมกับ Ledoit-Wolf shrinkage
       เพื่อหาสัดส่วนการลงทุนที่เหมาะสมที่สุด
    4. **🛡️ Risk Check** — ตรวจสอบ VaR, CVaR, Max Drawdown, Stress Test
    5. **🔮 พยากรณ์** — ใช้ Linear Regression + Momentum Analysis
    6. **✅ จัดสรรเงิน** — คำนวณจำนวนหุ้นที่ต้องซื้อตามงบประมาณ

    #### ระดับความเสี่ยง
    - **ต่ำมาก** → เน้นลดความผันผวน (Min Volatility)
    - **ต่ำ** → น้ำหนักความเสี่ยง 30%
    - **ปานกลาง** → สมดุลระหว่างผลตอบแทนและความเสี่ยง
    - **สูง** → น้ำหนักความเสี่ยง 70%
    - **สูงมาก** → เน้น Sharpe Ratio สูงสุด (Max Sharpe)
    """)

    st.markdown("---")
    st.markdown("#### 🔄 การ Deploy ด้วย Render")

    with st.expander("📘 คู่มือ Deploy ไปยัง Render"):
        st.markdown("""
        ### ขั้นตอนการ Deploy ไปยัง Render

        #### 1. อัปโหลดโค้ดไปที่ GitHub
        ```bash
        git init
        git add .
        git commit -m "Initial commit"
        # สร้าง repo บน GitHub แล้วรัน:
        git remote add origin https://github.com/yourusername/your-repo.git
        git push -u origin main
        ```

        #### 2. เชื่อมต่อ Render
        1. ไปที่ [https://dashboard.render.com](https://dashboard.render.com)
        2. กด **New +** → **Web Service**
        3. Connect GitHub repo
        4. ตั้งค่า:
        - **Name**: `ai-investment-portfolio`
        - **Runtime**: `Python 3`
        - **Build Command**: `pip install -r requirements.txt`
        - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
        - **Plan**: Free ($0/month)
        5. กด **Create Web Service**

        #### 3. เสร็จ!
        Render จะ Deploy ให้อัตโนมัติ ใช้เวลาประมาณ 3-5 นาที
        """)

    st.markdown("---")
    st.markdown(f"#### ข้อมูลเวอร์ชัน")

    ver_col1, ver_col2, ver_col3 = st.columns(3)
    with ver_col1:
        st.metric("Python", "3.12")
    with ver_col2:
        st.metric("Streamlit", "1.40")
    with ver_col3:
        st.metric("AI Engine", "Mean-Variance + Ledoit-Wolf")

st.markdown("---")
st.caption("⚠️ **คำเตือน:** การลงทุนมีความเสี่ยง ผลการดำเนินงานในอดีตไม่ได้รับประกันผลในอนาคต "
           "แอพนี้ใช้สำหรับการศึกษาและวิเคราะห์เท่านั้น ไม่ใช่คำแนะนำทางการเงิน")

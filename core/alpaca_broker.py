import os
import time
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, GetOrdersRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

class AlpacaBroker:
    def __init__(self, api_key=None, secret_key=None, paper=True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.paper = paper
        self.trading_client = None
        self.data_client = None
        self.connected = False

        if not ALPACA_AVAILABLE:
            return

        if api_key and secret_key:
            try:
                self.trading_client = TradingClient(api_key, secret_key, paper=paper)
                account = self.trading_client.get_account()
                self.connected = account.status == "ACTIVE"
                if self.connected:
                    try:
                        self.data_client = StockHistoricalDataClient(api_key, secret_key)
                    except:
                        self.data_client = None
            except Exception as e:
                st.error(f"เชื่อมต่อ Alpaca ไม่สำเร็จ: {e}")
                self.connected = False

    def get_account_info(self):
        if not self.connected or not self.trading_client:
            return None
        try:
            account = self.trading_client.get_account()
            return {
                "status": account.status,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "day_trade_count": int(account.day_trade_count),
                "is_paper": self.paper,
            }
        except Exception as e:
            st.error(f"ดึงข้อมูลบัญชีไม่สำเร็จ: {e}")
            return None

    def get_positions(self):
        if not self.connected or not self.trading_client:
            return []
        try:
            positions = self.trading_client.get_all_positions()
            result = []
            for p in positions:
                result.append({
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "cost_basis": float(p.cost_basis),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                    "day_pl": getattr(p, 'day_pl', 0),
                    "day_plpc": getattr(p, 'day_plpc', 0),
                })
            return result
        except Exception as e:
            st.error(f"ดึงข้อมูลพอร์ตไม่สำเร็จ: {e}")
            return []

    def get_total_position_value(self):
        positions = self.get_positions()
        return sum(p["market_value"] for p in positions)

    def place_market_order(self, ticker, qty, side="buy"):
        if not self.connected or not self.trading_client:
            return None
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            order_data = MarketOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self.trading_client.submit_order(order_data)
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status,
                "type": order.type.value,
                "created_at": order.created_at,
            }
        except Exception as e:
            st.error(f"ส่งคำสั่งซื้อขายไม่สำเร็จ ({ticker}): {e}")
            return None

    def place_limit_order(self, ticker, qty, limit_price, side="buy"):
        if not self.connected or not self.trading_client:
            return None
        try:
            order_side = OrderSide.BUY if side == "buy" else OrderSide.SELL
            order_data = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=order_side,
                time_in_force=TimeInForce.DAY,
                limit_price=limit_price,
            )
            order = self.trading_client.submit_order(order_data)
            return {
                "id": order.id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "side": order.side.value,
                "status": order.status,
                "type": order.type.value,
                "limit_price": float(limit_price),
                "created_at": order.created_at,
            }
        except Exception as e:
            st.error(f"ส่งคำสั่ง Limit ไม่สำเร็จ ({ticker}): {e}")
            return None

    def get_orders(self, status="all", limit=50):
        if not self.connected or not self.trading_client:
            return []
        try:
            status_filter = None if status == "all" else status
            request = GetOrdersRequest(status=status_filter, limit=limit)
            orders = self.trading_client.get_orders(request)
            result = []
            for o in orders:
                result.append({
                    "id": o.id,
                    "symbol": o.symbol,
                    "qty": float(o.qty),
                    "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
                    "side": o.side.value,
                    "type": o.type.value,
                    "status": o.status,
                    "limit_price": float(o.limit_price) if o.limit_price else None,
                    "filled_avg_price": float(o.filled_avg_price) if o.filled_avg_price else None,
                    "created_at": o.created_at,
                    "updated_at": o.updated_at,
                })
            return result
        except Exception as e:
            return []

    def cancel_all_orders(self):
        if not self.connected or not self.trading_client:
            return False
        try:
            self.trading_client.cancel_orders()
            return True
        except:
            return False

    def close_all_positions(self):
        if not self.connected or not self.trading_client:
            return False
        try:
            self.trading_client.close_all_positions()
            return True
        except:
            return False

    def get_clock(self):
        if not self.connected or not self.trading_client:
            return None
        try:
            clock = self.trading_client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
                "timestamp": clock.timestamp,
            }
        except:
            return None

    def rebalance_portfolio(self, target_allocations, total_investment):
        if not self.connected or not self.trading_client:
            return []

        positions = self.get_positions()
        current_holdings = {p["ticker"]: p["qty"] for p in positions}
        current_values = {p["ticker"]: p["market_value"] for p in positions}
        account_info = self.get_account_info()
        if not account_info:
            return []
        available_cash = account_info["cash"]

        orders_placed = []

        target_map = {}
        for alloc in target_allocations:
            ticker = alloc["ticker"]
            target_qty = alloc["shares"]
            target_map[ticker] = target_qty

        sell_orders = []
        buy_orders = []

        for ticker, target_qty in target_map.items():
            current_qty = current_holdings.get(ticker, 0)
            diff = target_qty - current_qty

            if diff > 0:
                buy_orders.append((ticker, diff))
            elif diff < 0:
                sell_orders.append((ticker, abs(diff)))

        for ticker, qty in sell_orders:
            order = self.place_market_order(ticker, qty, "sell")
            if order:
                orders_placed.append(order)
                time.sleep(0.1)

        time.sleep(0.5)

        for ticker, qty in buy_orders:
            if available_cash <= 0:
                break
            order = self.place_market_order(ticker, qty, "buy")
            if order:
                orders_placed.append(order)
                time.sleep(0.1)

        return orders_placed

    def get_recent_trades(self, ticker, days=5):
        if not self.data_client:
            return None
        try:
            end = datetime.now()
            start = end - timedelta(days=days)
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=start,
                end=end,
            )
            bars = self.data_client.get_stock_bars(request)
            return bars.df if bars.df is not None else None
        except:
            return None

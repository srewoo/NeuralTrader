"""
Unit Tests for Paper Trading
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestPaperTradingEngine:
    """Test Paper Trading Engine"""

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_engine_initialization(self, mock_db):
        """Test engine initializes with default portfolio"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine

        engine = PaperTradingEngine(initial_capital=100000)

        assert engine.initial_capital == 100000
        assert engine.cash == 100000
        assert engine.positions == {}

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_engine_initialization_with_custom_capital(self, mock_db):
        """Test engine with custom starting capital"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine

        engine = PaperTradingEngine(initial_capital=500000)

        assert engine.initial_capital == 500000
        assert engine.cash == 500000

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_place_buy_order(self, mock_db):
        """Test placing a buy order"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine, OrderSide, OrderType

        engine = PaperTradingEngine(initial_capital=100000)

        order = engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            current_price=2500.00
        )

        assert order is not None
        assert order.symbol == "RELIANCE"
        assert order.side == OrderSide.BUY
        assert order.quantity == 10

        # Cash should be reduced
        assert engine.cash < 100000

        # Position should exist
        assert "RELIANCE" in engine.positions

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_place_sell_order(self, mock_db):
        """Test placing a sell order"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine, OrderSide, OrderType

        engine = PaperTradingEngine(initial_capital=100000)

        # First buy
        engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            current_price=2500.00
        )

        # Then sell
        order = engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=5,
            order_type=OrderType.MARKET,
            current_price=2600.00
        )

        assert order is not None
        assert order.side == OrderSide.SELL
        assert engine.positions["RELIANCE"].quantity == 5

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_cannot_sell_without_position(self, mock_db):
        """Test selling without position fails"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine, OrderSide, OrderType, OrderStatus

        engine = PaperTradingEngine(initial_capital=100000)

        order = engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=10,
            order_type=OrderType.MARKET,
            current_price=2500.00
        )

        # Should be rejected
        assert order.status == OrderStatus.REJECTED

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_cannot_buy_without_sufficient_funds(self, mock_db):
        """Test buying without sufficient funds fails"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine, OrderSide, OrderType, OrderStatus

        engine = PaperTradingEngine(initial_capital=10000)

        order = engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=100,  # Would cost 250,000
            order_type=OrderType.MARKET,
            current_price=2500.00
        )

        # Should be rejected
        assert order.status == OrderStatus.REJECTED

    @patch('portfolio.paper_trading.paper_trading_db')
    def test_get_portfolio_value(self, mock_db):
        """Test portfolio value calculation"""
        mock_db.load_portfolio_state.return_value = None
        mock_db.load_positions.return_value = []
        mock_db.load_orders.return_value = []
        mock_db.load_trades.return_value = []

        from portfolio.paper_trading import PaperTradingEngine, OrderSide, OrderType

        engine = PaperTradingEngine(initial_capital=100000)

        engine.place_order(
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            order_type=OrderType.MARKET,
            current_price=2500.00
        )

        # Simulate price increase
        current_prices = {"RELIANCE": 2600.00}
        portfolio_value = engine.get_portfolio_value(current_prices)

        # Should have positions + cash
        assert portfolio_value > 0


class TestPosition:
    """Test Position class"""

    def test_position_add_quantity(self):
        """Test adding to position"""
        from portfolio.paper_trading import Position

        position = Position("RELIANCE")
        position.add_quantity(10, 2500.00)

        assert position.quantity == 10
        assert position.average_price == 2500.00

    def test_position_average_price(self):
        """Test average price calculation"""
        from portfolio.paper_trading import Position

        position = Position("RELIANCE")
        position.add_quantity(10, 2500.00)
        position.add_quantity(10, 2600.00)

        assert position.quantity == 20
        assert position.average_price == 2550.00

    def test_position_reduce_quantity(self):
        """Test reducing position"""
        from portfolio.paper_trading import Position

        position = Position("RELIANCE")
        position.add_quantity(10, 2500.00)
        pnl = position.reduce_quantity(5, 2600.00)

        assert position.quantity == 5
        assert pnl == 500  # 5 shares * 100 profit

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation"""
        from portfolio.paper_trading import Position

        position = Position("RELIANCE")
        position.add_quantity(10, 2500.00)

        pnl = position.get_unrealized_pnl(2600.00)

        assert pnl == 1000  # 10 shares * 100 profit


class TestTrade:
    """Test Trade class"""

    def test_trade_total_cost_buy(self):
        """Test trade total cost calculation for buy"""
        from portfolio.paper_trading import Trade, OrderSide

        trade = Trade(
            trade_id="TRD_000001",
            order_id="ORD_000001",
            symbol="RELIANCE",
            side=OrderSide.BUY,
            quantity=10,
            price=2500.00,
            commission=7.50,
            slippage=25.00
        )

        # Buy: base_cost + commission + slippage
        # = 25000 + 7.50 + 25 = 25032.50
        total = trade.get_total_cost()
        assert total == 25000 + 7.50 + 25.00

    def test_trade_total_cost_sell(self):
        """Test trade total cost calculation for sell"""
        from portfolio.paper_trading import Trade, OrderSide

        trade = Trade(
            trade_id="TRD_000001",
            order_id="ORD_000001",
            symbol="RELIANCE",
            side=OrderSide.SELL,
            quantity=10,
            price=2600.00,
            commission=7.80,
            slippage=26.00
        )

        # Sell: base_cost - commission - slippage
        # = 26000 - 7.80 - 26 = 25966.20
        total = trade.get_total_cost()
        assert total == 26000 - 7.80 - 26.00


class TestOrder:
    """Test Order class"""

    def test_order_to_dict(self):
        """Test order to_dict method"""
        from portfolio.paper_trading import Order, OrderSide, OrderType

        order = Order(
            order_id="ORD_000001",
            symbol="RELIANCE",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=10,
            price=2500.00
        )

        data = order.to_dict()

        assert data["order_id"] == "ORD_000001"
        assert data["symbol"] == "RELIANCE"
        assert data["side"] == "BUY"
        assert data["order_type"] == "MARKET"
        assert data["quantity"] == 10


class TestEnums:
    """Test Paper Trading Enums"""

    def test_order_type_values(self):
        """Test OrderType enum values"""
        from portfolio.paper_trading import OrderType

        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP_LOSS.value == "STOP_LOSS"

    def test_order_side_values(self):
        """Test OrderSide enum values"""
        from portfolio.paper_trading import OrderSide

        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_status_values(self):
        """Test OrderStatus enum values"""
        from portfolio.paper_trading import OrderStatus

        assert OrderStatus.PENDING.value == "PENDING"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.CANCELLED.value == "CANCELLED"

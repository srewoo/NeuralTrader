"""
Paper Trading Engine
Simulates real trading with virtual portfolio
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import logging

from . import paper_trading_db

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order:
    """Represents a trading order"""

    def __init__(
        self,
        order_id: str,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: int,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.quantity = quantity
        self.price = price
        self.stop_price = stop_price
        self.filled_quantity = 0
        self.average_price = 0.0
        self.status = OrderStatus.PENDING
        self.timestamp = timestamp or datetime.now(timezone.utc)
        self.filled_timestamp: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "order_type": self.order_type.value,
            "quantity": self.quantity,
            "price": self.price,
            "stop_price": self.stop_price,
            "filled_quantity": self.filled_quantity,
            "average_price": self.average_price,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "filled_timestamp": self.filled_timestamp.isoformat() if self.filled_timestamp else None
        }


class Trade:
    """Represents a completed trade"""

    def __init__(
        self,
        trade_id: str,
        order_id: str,
        symbol: str,
        side: OrderSide,
        quantity: int,
        price: float,
        commission: float,
        slippage: float,
        timestamp: Optional[datetime] = None
    ):
        self.trade_id = trade_id
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.price = price
        self.commission = commission
        self.slippage = slippage
        self.timestamp = timestamp or datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        total_cost = self.get_total_cost()
        return {
            "trade_id": self.trade_id,
            "order_id": self.order_id,
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": self.quantity,
            "price": self.price,
            "execution_price": self.price,  # Alias for frontend compatibility
            "commission": self.commission,
            "slippage": self.slippage,
            "total_cost": total_cost,
            "total_value": total_cost,  # Alias for frontend compatibility
            "timestamp": self.timestamp.isoformat()
        }

    def get_total_cost(self) -> float:
        """Calculate total cost including commission and slippage"""
        base_cost = self.quantity * self.price
        if self.side == OrderSide.BUY:
            return base_cost + self.commission + (self.quantity * self.slippage)
        else:
            return base_cost - self.commission - (self.quantity * self.slippage)


class Position:
    """Represents a stock position"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.quantity = 0
        self.average_price = 0.0
        self.realized_pnl = 0.0

    def add_quantity(self, quantity: int, price: float):
        """Add to position (buy)"""
        total_cost = (self.quantity * self.average_price) + (quantity * price)
        self.quantity += quantity
        if self.quantity > 0:
            self.average_price = total_cost / self.quantity

    def reduce_quantity(self, quantity: int, price: float) -> float:
        """Reduce position (sell) and calculate realized P&L"""
        if quantity > self.quantity:
            raise ValueError(f"Cannot sell {quantity} shares, only {self.quantity} available")

        pnl = quantity * (price - self.average_price)
        self.realized_pnl += pnl
        self.quantity -= quantity

        if self.quantity == 0:
            self.average_price = 0.0

        return pnl

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L"""
        return self.quantity * (current_price - self.average_price)

    def to_dict(self, current_price: float = 0.0) -> Dict[str, Any]:
        unrealized_pnl = self.get_unrealized_pnl(current_price)
        cost_basis = self.quantity * self.average_price
        return_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else 0.0

        return {
            "symbol": self.symbol,
            "quantity": self.quantity,
            "average_price": round(self.average_price, 2),
            "current_price": round(current_price, 2),
            "market_value": round(self.quantity * current_price, 2),
            "cost_basis": round(cost_basis, 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "realized_pnl": round(self.realized_pnl, 2),
            "return_pct": round(return_pct, 2)
        }


class PaperTradingEngine:
    """
    Paper Trading Engine
    Simulates real trading with virtual portfolio
    Includes realistic slippage and commission costs
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0003,  # 0.03% brokerage
        slippage_bps: int = 10,  # 10 basis points (0.1%)
        max_position_size: float = 0.20,  # 20% max per position
        user_id: Optional[str] = None
    ):
        self.user_id = user_id or "default"
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps
        self.max_position_size = max_position_size

        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trades: List[Trade] = []
        self.order_counter = 0
        self.trade_counter = 0

        # Load from database
        self._load_from_db()

        logger.info(f"Paper trading engine initialized with ${self.cash:,.2f} cash")

    def _load_from_db(self):
        """Load portfolio state from database"""
        try:
            # Load portfolio state
            state = paper_trading_db.load_portfolio_state(self.user_id)
            if state:
                self.initial_capital = state['initial_capital']
                self.cash = state['cash']
                self.commission_rate = state['commission_rate']
                self.slippage_bps = state['slippage_bps']
                self.max_position_size = state['max_position_size']
                self.order_counter = state['order_counter']
                self.trade_counter = state['trade_counter']
                logger.info(f"Loaded portfolio state for user {self.user_id}")

            # Load positions
            positions_data = paper_trading_db.load_positions(self.user_id)
            for pos_data in positions_data:
                position = Position(pos_data['symbol'])
                position.quantity = pos_data['quantity']
                position.average_price = pos_data['average_price']
                position.realized_pnl = pos_data['realized_pnl']
                self.positions[pos_data['symbol']] = position

            # Load orders
            orders_data = paper_trading_db.load_orders(self.user_id)
            for order_data in orders_data:
                order = Order(
                    order_id=order_data['order_id'],
                    symbol=order_data['symbol'],
                    side=OrderSide(order_data['side']),
                    order_type=OrderType(order_data['order_type']),
                    quantity=order_data['quantity'],
                    price=order_data['price'],
                    stop_price=order_data['stop_price'],
                    timestamp=datetime.fromisoformat(order_data['timestamp'])
                )
                order.filled_quantity = order_data['filled_quantity']
                order.average_price = order_data['average_price']
                order.status = OrderStatus(order_data['status'])
                if order_data['filled_timestamp']:
                    order.filled_timestamp = datetime.fromisoformat(order_data['filled_timestamp'])
                self.orders[order_data['order_id']] = order

            # Load trades
            trades_data = paper_trading_db.load_trades(self.user_id)
            for trade_data in trades_data:
                trade = Trade(
                    trade_id=trade_data['trade_id'],
                    order_id=trade_data['order_id'],
                    symbol=trade_data['symbol'],
                    side=OrderSide(trade_data['side']),
                    quantity=trade_data['quantity'],
                    price=trade_data['price'],
                    commission=trade_data['commission'],
                    slippage=trade_data['slippage'],
                    timestamp=datetime.fromisoformat(trade_data['timestamp'])
                )
                self.trades.append(trade)

            logger.info(f"Loaded {len(self.positions)} positions, {len(self.orders)} orders, {len(self.trades)} trades")
        except Exception as e:
            logger.error(f"Failed to load portfolio from database: {e}")

    def _save_state_to_db(self):
        """Save portfolio state to database"""
        try:
            state = {
                'initial_capital': self.initial_capital,
                'cash': self.cash,
                'commission_rate': self.commission_rate,
                'slippage_bps': self.slippage_bps,
                'max_position_size': self.max_position_size,
                'order_counter': self.order_counter,
                'trade_counter': self.trade_counter
            }
            paper_trading_db.save_portfolio_state(self.user_id, state)
        except Exception as e:
            logger.error(f"Failed to save portfolio state: {e}")

    def _save_position_to_db(self, position: Position):
        """Save position to database"""
        try:
            pos_data = {
                'symbol': position.symbol,
                'quantity': position.quantity,
                'average_price': position.average_price,
                'realized_pnl': position.realized_pnl
            }
            paper_trading_db.save_position(self.user_id, pos_data)
        except Exception as e:
            logger.error(f"Failed to save position: {e}")

    def _delete_position_from_db(self, symbol: str):
        """Delete position from database"""
        try:
            paper_trading_db.delete_position(self.user_id, symbol)
        except Exception as e:
            logger.error(f"Failed to delete position: {e}")

    def _save_order_to_db(self, order: Order):
        """Save order to database"""
        try:
            paper_trading_db.save_order(self.user_id, order.to_dict())
        except Exception as e:
            logger.error(f"Failed to save order: {e}")

    def _save_trade_to_db(self, trade: Trade):
        """Save trade to database"""
        try:
            paper_trading_db.save_trade(self.user_id, trade.to_dict())
        except Exception as e:
            logger.error(f"Failed to save trade: {e}")

    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"ORD_{self.order_counter:06d}"

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID"""
        self.trade_counter += 1
        return f"TRD_{self.trade_counter:06d}"

    def _calculate_slippage(self, price: float, side: OrderSide) -> float:
        """Calculate slippage based on order side"""
        slippage_amount = price * (self.slippage_bps / 10000)
        # Buy orders pay higher, sell orders receive lower
        return slippage_amount if side == OrderSide.BUY else -slippage_amount

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """Calculate brokerage commission"""
        order_value = quantity * price
        return order_value * self.commission_rate

    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        current_price: Optional[float] = None
    ) -> Order:
        """
        Place a paper trading order

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Number of shares
            order_type: MARKET, LIMIT, STOP_LOSS, etc.
            limit_price: Price for limit orders
            stop_price: Trigger price for stop orders
            current_price: Current market price (required for market orders)

        Returns:
            Order object
        """
        order_id = self._generate_order_id()

        # Create order
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=limit_price,
            stop_price=stop_price
        )

        # For market orders, execute immediately
        if order_type == OrderType.MARKET:
            if current_price is None:
                order.status = OrderStatus.REJECTED
                logger.error(f"Market order rejected: current_price required")
                return order

            # Check if we have enough capital/shares
            if side == OrderSide.BUY:
                execution_price = current_price + self._calculate_slippage(current_price, side)
                commission = self._calculate_commission(quantity, execution_price)
                total_cost = (quantity * execution_price) + commission

                # Check max position size
                portfolio_value = self.get_portfolio_value({symbol: current_price})
                if total_cost > portfolio_value * self.max_position_size:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Order rejected: exceeds max position size ({self.max_position_size*100}%)")
                    return order

                if total_cost > self.cash:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Order rejected: insufficient funds (need ${total_cost:,.2f}, have ${self.cash:,.2f})")
                    return order

            else:  # SELL
                position = self.positions.get(symbol)
                if not position or position.quantity < quantity:
                    order.status = OrderStatus.REJECTED
                    available = position.quantity if position else 0
                    logger.warning(f"Order rejected: insufficient shares (need {quantity}, have {available})")
                    return order

            # Execute the order
            self._execute_order(order, current_price)

        self.orders[order_id] = order
        self._save_order_to_db(order)
        self._save_state_to_db()
        return order

    def _execute_order(self, order: Order, execution_price: float):
        """Execute an order"""
        slippage = self._calculate_slippage(execution_price, order.side)
        final_price = execution_price + slippage
        commission = self._calculate_commission(order.quantity, final_price)

        # Create trade
        trade = Trade(
            trade_id=self._generate_trade_id(),
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=final_price,
            commission=commission,
            slippage=slippage * order.quantity
        )

        # Update position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(order.symbol)

        position = self.positions[order.symbol]

        if order.side == OrderSide.BUY:
            position.add_quantity(order.quantity, final_price)
            self.cash -= trade.get_total_cost()
        else:
            realized_pnl = position.reduce_quantity(order.quantity, final_price)
            self.cash += trade.get_total_cost()
            logger.info(f"Realized P&L: ${realized_pnl:,.2f}")

        # Update order status
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = final_price
        order.filled_timestamp = datetime.now(timezone.utc)

        self.trades.append(trade)

        # Save to database
        self._save_trade_to_db(trade)
        self._save_position_to_db(position)

        # Delete position from DB if quantity is zero
        if position.quantity == 0:
            self._delete_position_from_db(position.symbol)

        logger.info(f"Order {order.order_id} executed: {order.side.value} {order.quantity} {order.symbol} @ ${final_price:.2f}")
        logger.info(f"Commission: ${commission:.2f}, Slippage: ${slippage * order.quantity:.2f}")
        logger.info(f"Cash remaining: ${self.cash:,.2f}")

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order"""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        if order.status != OrderStatus.PENDING:
            return False

        order.status = OrderStatus.CANCELLED
        logger.info(f"Order {order_id} cancelled")
        return True

    def get_position(self, symbol: str, current_price: float = 0.0) -> Optional[Dict[str, Any]]:
        """Get position for a symbol"""
        if symbol not in self.positions:
            return None
        return self.positions[symbol].to_dict(current_price)

    def get_all_positions(self, current_prices: Dict[str, float]) -> List[Dict[str, Any]]:
        """Get all positions with current prices"""
        return [
            pos.to_dict(current_prices.get(symbol, 0.0))
            for symbol, pos in self.positions.items()
            if pos.quantity > 0
        ]

    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value"""
        positions_value = sum(
            pos.quantity * current_prices.get(symbol, 0.0)
            for symbol, pos in self.positions.items()
        )
        return self.cash + positions_value

    def get_portfolio_summary(self, current_prices: Dict[str, float]) -> Dict[str, Any]:
        """Get comprehensive portfolio summary"""
        positions = self.get_all_positions(current_prices)
        portfolio_value = self.get_portfolio_value(current_prices)

        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions)
        total_realized_pnl = sum(pos['realized_pnl'] for pos in positions)
        total_pnl = total_unrealized_pnl + total_realized_pnl

        total_return_pct = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

        return {
            "initial_capital": round(self.initial_capital, 2),
            "cash": round(self.cash, 2),
            "positions_value": round(portfolio_value - self.cash, 2),
            "portfolio_value": round(portfolio_value, 2),
            "total_value": round(portfolio_value, 2),  # Alias for frontend compatibility
            "total_realized_pnl": round(total_realized_pnl, 2),
            "total_unrealized_pnl": round(total_unrealized_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct, 2),
            "return_pct": round(total_return_pct, 2),  # Alias for frontend compatibility
            "positions_count": len([p for p in positions if p['quantity'] > 0]),
            "total_trades": len(self.trades)
        }

    def get_trade_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent trade history"""
        recent_trades = sorted(self.trades, key=lambda t: t.timestamp, reverse=True)[:limit]
        return [trade.to_dict() for trade in recent_trades]

    def get_order_history(self, status: Optional[OrderStatus] = None, limit: int = 50) -> List[Dict[str, Any]]:
        """Get order history"""
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o.status == status]
        orders = sorted(orders, key=lambda o: o.timestamp, reverse=True)[:limit]
        return [order.to_dict() for order in orders]

    def reset(self):
        """Reset paper trading account"""
        self.cash = self.initial_capital
        self.positions.clear()
        self.orders.clear()
        self.trades.clear()
        self.order_counter = 0
        self.trade_counter = 0

        # Clear database
        paper_trading_db.reset_user_portfolio(self.user_id)

        logger.info(f"Paper trading account reset to ${self.initial_capital:,.2f}")


# Singleton instance per user
_paper_trading_engines: Dict[str, PaperTradingEngine] = {}


def get_paper_trading_engine(user_id: str = "default") -> PaperTradingEngine:
    """Get or create paper trading engine for user"""
    if user_id not in _paper_trading_engines:
        _paper_trading_engines[user_id] = PaperTradingEngine(user_id=user_id)
    return _paper_trading_engines[user_id]

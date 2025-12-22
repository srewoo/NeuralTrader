"""
SQLite persistence layer for paper trading
"""

import sqlite3
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = Path(__file__).parent.parent / "paper_trading.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the paper trading database"""
    conn = get_connection()
    cursor = conn.cursor()

    # Portfolio state table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_state (
            user_id TEXT PRIMARY KEY,
            initial_capital REAL NOT NULL,
            cash REAL NOT NULL,
            commission_rate REAL NOT NULL,
            slippage_bps INTEGER NOT NULL,
            max_position_size REAL NOT NULL,
            order_counter INTEGER NOT NULL DEFAULT 0,
            trade_counter INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL
        )
    """)

    # Positions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            average_price REAL NOT NULL,
            realized_pnl REAL NOT NULL DEFAULT 0.0,
            updated_at TEXT NOT NULL,
            UNIQUE(user_id, symbol)
        )
    """)

    # Orders table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            order_type TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL,
            stop_price REAL,
            filled_quantity INTEGER NOT NULL DEFAULT 0,
            average_price REAL NOT NULL DEFAULT 0.0,
            status TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            filled_timestamp TEXT
        )
    """)

    # Trades table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            trade_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            order_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            price REAL NOT NULL,
            commission REAL NOT NULL,
            slippage REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_positions_user ON positions(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_orders_user ON orders(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_user ON trades(user_id)")

    conn.commit()
    conn.close()
    logger.info(f"Paper trading database initialized at {DB_PATH}")


def save_portfolio_state(user_id: str, state: Dict[str, Any]) -> bool:
    """Save portfolio state"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO portfolio_state
            (user_id, initial_capital, cash, commission_rate, slippage_bps,
             max_position_size, order_counter, trade_counter, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            user_id,
            state['initial_capital'],
            state['cash'],
            state['commission_rate'],
            state['slippage_bps'],
            state['max_position_size'],
            state['order_counter'],
            state['trade_counter']
        ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save portfolio state: {e}")
        return False


def load_portfolio_state(user_id: str) -> Optional[Dict[str, Any]]:
    """Load portfolio state"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM portfolio_state WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return {
            'initial_capital': row['initial_capital'],
            'cash': row['cash'],
            'commission_rate': row['commission_rate'],
            'slippage_bps': row['slippage_bps'],
            'max_position_size': row['max_position_size'],
            'order_counter': row['order_counter'],
            'trade_counter': row['trade_counter']
        }
    except Exception as e:
        logger.error(f"Failed to load portfolio state: {e}")
        return None


def save_position(user_id: str, position: Dict[str, Any]) -> bool:
    """Save or update position"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO positions
            (user_id, symbol, quantity, average_price, realized_pnl, updated_at)
            VALUES (?, ?, ?, ?, ?, datetime('now'))
        """, (
            user_id,
            position['symbol'],
            position['quantity'],
            position['average_price'],
            position['realized_pnl']
        ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save position: {e}")
        return False


def load_positions(user_id: str) -> List[Dict[str, Any]]:
    """Load all positions for a user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM positions WHERE user_id = ? AND quantity > 0", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        return [{
            'symbol': row['symbol'],
            'quantity': row['quantity'],
            'average_price': row['average_price'],
            'realized_pnl': row['realized_pnl']
        } for row in rows]
    except Exception as e:
        logger.error(f"Failed to load positions: {e}")
        return []


def delete_position(user_id: str, symbol: str) -> bool:
    """Delete a position"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM positions WHERE user_id = ? AND symbol = ?", (user_id, symbol))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to delete position: {e}")
        return False


def save_order(user_id: str, order: Dict[str, Any]) -> bool:
    """Save order"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO orders
            (order_id, user_id, symbol, side, order_type, quantity, price, stop_price,
             filled_quantity, average_price, status, timestamp, filled_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            order['order_id'],
            user_id,
            order['symbol'],
            order['side'],
            order['order_type'],
            order['quantity'],
            order.get('price'),
            order.get('stop_price'),
            order['filled_quantity'],
            order['average_price'],
            order['status'],
            order['timestamp'],
            order.get('filled_timestamp')
        ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save order: {e}")
        return False


def load_orders(user_id: str) -> List[Dict[str, Any]]:
    """Load all orders for a user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM orders WHERE user_id = ? ORDER BY timestamp DESC", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to load orders: {e}")
        return []


def save_trade(user_id: str, trade: Dict[str, Any]) -> bool:
    """Save trade"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO trades
            (trade_id, user_id, order_id, symbol, side, quantity, price, commission, slippage, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade['trade_id'],
            user_id,
            trade['order_id'],
            trade['symbol'],
            trade['side'],
            trade['quantity'],
            trade['price'],
            trade['commission'],
            trade['slippage'],
            trade['timestamp']
        ))

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to save trade: {e}")
        return False


def load_trades(user_id: str, limit: int = 1000) -> List[Dict[str, Any]]:
    """Load all trades for a user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM trades WHERE user_id = ? ORDER BY timestamp DESC LIMIT ?",
            (user_id, limit)
        )
        rows = cursor.fetchall()
        conn.close()

        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to load trades: {e}")
        return []


def reset_user_portfolio(user_id: str) -> bool:
    """Reset all paper trading data for a user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("DELETE FROM positions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM orders WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM trades WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM portfolio_state WHERE user_id = ?", (user_id,))

        conn.commit()
        conn.close()
        logger.info(f"Reset paper trading portfolio for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to reset portfolio: {e}")
        return False


# Initialize database on module load
init_db()

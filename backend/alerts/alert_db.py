"""
SQLite persistence layer for alerts
"""

import sqlite3
import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = Path(__file__).parent.parent / "alerts.db"


def get_connection() -> sqlite3.Connection:
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initialize the alerts database"""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            alert_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            condition TEXT NOT NULL,
            message TEXT NOT NULL,
            delivery_channels TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE',
            created_at TEXT NOT NULL,
            triggered_at TEXT,
            expires_at TEXT,
            metadata TEXT
        )
    """)

    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user_id ON alerts(user_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON alerts(symbol)")

    conn.commit()
    conn.close()
    logger.info(f"Alerts database initialized at {DB_PATH}")


def save_alert(alert_data: Dict[str, Any]) -> bool:
    """Save alert to database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO alerts
            (alert_id, user_id, alert_type, symbol, condition, message,
             delivery_channels, status, created_at, triggered_at, expires_at, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert_data['alert_id'],
            alert_data['user_id'],
            alert_data['alert_type'],
            alert_data['symbol'],
            json.dumps(alert_data['condition']),
            alert_data['message'],
            json.dumps(alert_data['delivery_channels']),
            alert_data['status'],
            alert_data['created_at'],
            alert_data.get('triggered_at'),
            alert_data.get('expires_at'),
            json.dumps(alert_data.get('metadata')) if alert_data.get('metadata') else None
        ))

        conn.commit()
        conn.close()
        logger.debug(f"Alert saved to database: {alert_data['alert_id']}")
        return True
    except Exception as e:
        logger.error(f"Failed to save alert to database: {e}")
        return False


def delete_alert(alert_id: str) -> bool:
    """Delete alert from database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM alerts WHERE alert_id = ?", (alert_id,))
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        if deleted:
            logger.debug(f"Alert deleted from database: {alert_id}")
        return deleted
    except Exception as e:
        logger.error(f"Failed to delete alert from database: {e}")
        return False


def update_alert_status(alert_id: str, status: str, triggered_at: Optional[str] = None) -> bool:
    """Update alert status in database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if triggered_at:
            cursor.execute(
                "UPDATE alerts SET status = ?, triggered_at = ? WHERE alert_id = ?",
                (status, triggered_at, alert_id)
            )
        else:
            cursor.execute(
                "UPDATE alerts SET status = ? WHERE alert_id = ?",
                (status, alert_id)
            )

        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Failed to update alert status: {e}")
        return False


def load_all_alerts() -> List[Dict[str, Any]]:
    """Load all alerts from database"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM alerts")
        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert_data = {
                'alert_id': row['alert_id'],
                'user_id': row['user_id'],
                'alert_type': row['alert_type'],
                'symbol': row['symbol'],
                'condition': json.loads(row['condition']),
                'message': row['message'],
                'delivery_channels': json.loads(row['delivery_channels']),
                'status': row['status'],
                'created_at': row['created_at'],
                'triggered_at': row['triggered_at'],
                'expires_at': row['expires_at'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else None
            }
            alerts.append(alert_data)

        logger.info(f"Loaded {len(alerts)} alerts from database")
        return alerts
    except Exception as e:
        logger.error(f"Failed to load alerts from database: {e}")
        return []


def load_user_alerts(user_id: str, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load alerts for a specific user"""
    try:
        conn = get_connection()
        cursor = conn.cursor()

        if status:
            cursor.execute(
                "SELECT * FROM alerts WHERE user_id = ? AND status = ? ORDER BY created_at DESC",
                (user_id, status)
            )
        else:
            cursor.execute(
                "SELECT * FROM alerts WHERE user_id = ? ORDER BY created_at DESC",
                (user_id,)
            )

        rows = cursor.fetchall()
        conn.close()

        alerts = []
        for row in rows:
            alert_data = {
                'alert_id': row['alert_id'],
                'user_id': row['user_id'],
                'alert_type': row['alert_type'],
                'symbol': row['symbol'],
                'condition': json.loads(row['condition']),
                'message': row['message'],
                'delivery_channels': json.loads(row['delivery_channels']),
                'status': row['status'],
                'created_at': row['created_at'],
                'triggered_at': row['triggered_at'],
                'expires_at': row['expires_at'],
                'metadata': json.loads(row['metadata']) if row['metadata'] else None
            }
            alerts.append(alert_data)

        return alerts
    except Exception as e:
        logger.error(f"Failed to load user alerts: {e}")
        return []


def get_max_alert_counter() -> int:
    """Get the maximum alert counter from existing alerts"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT alert_id FROM alerts ORDER BY alert_id DESC LIMIT 1")
        row = cursor.fetchone()
        conn.close()

        if row and row['alert_id']:
            # Extract counter from alert_id like "ALERT_00000001"
            try:
                counter = int(row['alert_id'].split('_')[1])
                return counter
            except (IndexError, ValueError):
                return 0
        return 0
    except Exception as e:
        logger.error(f"Failed to get max alert counter: {e}")
        return 0


# Initialize database on module load
init_db()

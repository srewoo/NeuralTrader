"""
Alert Management System (Simplified)
Price alerts with Telegram + Email delivery
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio
from dataclasses import dataclass, asdict
import requests

from . import alert_db

logger = logging.getLogger(__name__)


class AlertType(str, Enum):
    PRICE = "PRICE"


class AlertStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TRIGGERED = "TRIGGERED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


class PriceCondition(str, Enum):
    ABOVE = "ABOVE"
    BELOW = "BELOW"
    CROSSES_ABOVE = "CROSSES_ABOVE"
    CROSSES_BELOW = "CROSSES_BELOW"
    PERCENT_CHANGE_ABOVE = "PERCENT_CHANGE_ABOVE"
    PERCENT_CHANGE_BELOW = "PERCENT_CHANGE_BELOW"


class DeliveryChannel(str, Enum):
    EMAIL = "EMAIL"
    TELEGRAM = "TELEGRAM"


@dataclass
class Alert:
    """Represents a price alert"""
    alert_id: str
    user_id: str
    alert_type: AlertType
    symbol: str
    condition: Dict[str, Any]
    message: str
    delivery_channels: List[DeliveryChannel]
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: datetime = None
    triggered_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['status'] = self.status.value
        data['delivery_channels'] = [ch.value for ch in self.delivery_channels]
        data['created_at'] = self.created_at.isoformat()
        data['triggered_at'] = self.triggered_at.isoformat() if self.triggered_at else None
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        return data


class TelegramNotifier:
    """Send notifications via Telegram Bot"""

    def __init__(self, bot_token: Optional[str] = None, chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token and chat_id)

    def send_message(self, message: str) -> bool:
        """Send message via Telegram"""
        if not self.enabled:
            logger.warning("Telegram notifier not configured")
            return False

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            payload = {
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": "HTML"
            }
            response = requests.post(url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Telegram notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False


class EmailNotifier:
    """Send notifications via Email (SMTP)"""

    def __init__(self, smtp_config: Optional[Dict[str, str]] = None):
        self.smtp_config = smtp_config
        self.enabled = bool(smtp_config)

    def send_email(self, to_email: str, subject: str, body: str) -> bool:
        """Send email notification"""
        if not self.enabled:
            logger.warning("Email notifier not configured")
            return False

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            msg = MIMEMultipart()
            msg['From'] = self.smtp_config['from_email']
            msg['To'] = to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'html'))

            with smtplib.SMTP(self.smtp_config['smtp_host'], self.smtp_config['smtp_port']) as server:
                server.starttls()
                server.login(self.smtp_config['smtp_user'], self.smtp_config['smtp_password'])
                server.send_message(msg)

            logger.info(f"Email sent to {to_email}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class AlertManager:
    """
    Price Alert Management System
    Monitors prices and sends Telegram + Email notifications
    """

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        smtp_config: Optional[Dict[str, str]] = None,
    ):
        self.alerts: Dict[str, Alert] = {}
        self.alert_counter = 0

        # Initialize notifiers
        self.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)
        self.email = EmailNotifier(smtp_config)

        # Price tracking for cross alerts
        self.price_history: Dict[str, List[float]] = {}

        # Load alerts from database
        self._load_alerts_from_db()

    def _load_alerts_from_db(self):
        """Load alerts from database on startup"""
        try:
            alert_data_list = alert_db.load_all_alerts()
            for alert_data in alert_data_list:
                alert = Alert(
                    alert_id=alert_data['alert_id'],
                    user_id=alert_data['user_id'],
                    alert_type=AlertType(alert_data['alert_type']),
                    symbol=alert_data['symbol'],
                    condition=alert_data['condition'],
                    message=alert_data['message'],
                    delivery_channels=[DeliveryChannel(ch) for ch in alert_data['delivery_channels'] if ch in ('EMAIL', 'TELEGRAM')],
                    status=AlertStatus(alert_data['status']),
                    created_at=datetime.fromisoformat(alert_data['created_at']),
                    triggered_at=datetime.fromisoformat(alert_data['triggered_at']) if alert_data.get('triggered_at') else None,
                    expires_at=datetime.fromisoformat(alert_data['expires_at']) if alert_data.get('expires_at') else None,
                    metadata=alert_data.get('metadata')
                )
                self.alerts[alert.alert_id] = alert

            self.alert_counter = alert_db.get_max_alert_counter()
            logger.info(f"Loaded {len(self.alerts)} alerts from database")
        except Exception as e:
            logger.error(f"Failed to load alerts from database: {e}")

    def _save_alert_to_db(self, alert: Alert):
        """Save alert to database"""
        alert_db.save_alert(alert.to_dict())

    def _generate_alert_id(self) -> str:
        """Generate unique alert ID"""
        self.alert_counter += 1
        return f"ALERT_{self.alert_counter:08d}"

    def create_price_alert(
        self,
        user_id: str,
        symbol: str,
        condition: PriceCondition,
        target_price: float,
        delivery_channels: List[DeliveryChannel],
        percent_change: Optional[float] = None,
        expires_at: Optional[datetime] = None
    ) -> Alert:
        """Create a price alert"""
        condition_dict = {
            "type": condition.value,
            "target_price": target_price,
            "percent_change": percent_change
        }

        messages = {
            PriceCondition.PERCENT_CHANGE_ABOVE: f"🚀 {symbol} gained {percent_change}%! Current price: ₹{target_price}",
            PriceCondition.PERCENT_CHANGE_BELOW: f"📉 {symbol} dropped {abs(percent_change) if percent_change else 0}%! Current price: ₹{target_price}",
            PriceCondition.ABOVE: f"📊 {symbol} is now above ₹{target_price}",
            PriceCondition.BELOW: f"📊 {symbol} is now below ₹{target_price}",
            PriceCondition.CROSSES_ABOVE: f"📈 {symbol} crossed above ₹{target_price}",
            PriceCondition.CROSSES_BELOW: f"📉 {symbol} crossed below ₹{target_price}",
        }
        message = messages.get(condition, f"📊 {symbol} alert triggered at ₹{target_price}")

        alert = Alert(
            alert_id=self._generate_alert_id(),
            user_id=user_id,
            alert_type=AlertType.PRICE,
            symbol=symbol,
            condition=condition_dict,
            message=message,
            delivery_channels=delivery_channels,
            expires_at=expires_at
        )

        self.alerts[alert.alert_id] = alert
        self._save_alert_to_db(alert)
        logger.info(f"Price alert created: {symbol} {condition.value} ₹{target_price}")
        return alert

    def check_price_alert(self, alert: Alert, current_price: float, previous_price: Optional[float] = None) -> bool:
        """Check if price alert should trigger"""
        condition = alert.condition
        target = condition.get('target_price')
        condition_type = PriceCondition(condition.get('type'))

        if condition_type == PriceCondition.ABOVE:
            return current_price > target
        elif condition_type == PriceCondition.BELOW:
            return current_price < target
        elif condition_type == PriceCondition.CROSSES_ABOVE:
            if previous_price is None:
                return False
            return previous_price <= target and current_price > target
        elif condition_type == PriceCondition.CROSSES_BELOW:
            if previous_price is None:
                return False
            return previous_price >= target and current_price < target
        elif condition_type == PriceCondition.PERCENT_CHANGE_ABOVE:
            if previous_price is None:
                return False
            pct_change = ((current_price - previous_price) / previous_price) * 100
            return pct_change >= condition.get('percent_change', 0)
        elif condition_type == PriceCondition.PERCENT_CHANGE_BELOW:
            if previous_price is None:
                return False
            pct_change = ((current_price - previous_price) / previous_price) * 100
            return pct_change <= -condition.get('percent_change', 0)

        return False

    async def trigger_alert(self, alert: Alert):
        """Trigger an alert and send notifications"""
        alert.status = AlertStatus.TRIGGERED
        alert.triggered_at = datetime.now(timezone.utc)

        alert_db.update_alert_status(
            alert.alert_id,
            AlertStatus.TRIGGERED.value,
            alert.triggered_at.isoformat()
        )

        logger.info(f"Alert triggered: {alert.alert_id} - {alert.message}")

        for channel in alert.delivery_channels:
            try:
                if channel == DeliveryChannel.TELEGRAM:
                    self.telegram.send_message(f"<b>🔔 NeuralTrader Alert</b>\n\n{alert.message}")
                elif channel == DeliveryChannel.EMAIL:
                    # Email delivery (needs user email from settings)
                    pass
            except Exception as e:
                logger.error(f"Failed to send alert via {channel}: {e}")

    def cancel_alert(self, alert_id: str) -> bool:
        """Cancel an alert"""
        if alert_id not in self.alerts:
            return False
        alert = self.alerts[alert_id]
        if alert.status != AlertStatus.ACTIVE:
            return False
        alert.status = AlertStatus.CANCELLED
        alert_db.update_alert_status(alert_id, AlertStatus.CANCELLED.value)
        return True

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert permanently"""
        if alert_id not in self.alerts:
            return False
        del self.alerts[alert_id]
        alert_db.delete_alert(alert_id)
        return True

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID"""
        return self.alerts.get(alert_id)

    def get_user_alerts(self, user_id: str, status: Optional[AlertStatus] = None) -> List[Alert]:
        """Get all alerts for a user"""
        user_alerts = [a for a in self.alerts.values() if a.user_id == user_id]
        if status:
            user_alerts = [a for a in user_alerts if a.status == status]
        return sorted(user_alerts, key=lambda a: a.created_at, reverse=True)

    def cleanup_expired_alerts(self):
        """Remove expired alerts"""
        now = datetime.now(timezone.utc)
        expired_ids = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.expires_at and alert.expires_at < now
        ]
        for alert_id in expired_ids:
            del self.alerts[alert_id]
            alert_db.delete_alert(alert_id)
        if expired_ids:
            logger.info(f"Cleaned up {len(expired_ids)} expired alerts")


# Singleton instance
_alert_manager: Optional[AlertManager] = None


def get_alert_manager(
    telegram_bot_token: Optional[str] = None,
    telegram_chat_id: Optional[str] = None,
    smtp_config: Optional[Dict[str, str]] = None,
) -> AlertManager:
    """Get or create alert manager singleton"""
    global _alert_manager

    if _alert_manager is None:
        _alert_manager = AlertManager(telegram_bot_token, telegram_chat_id, smtp_config)
    else:
        if telegram_bot_token or telegram_chat_id:
            _alert_manager.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)
        if smtp_config:
            _alert_manager.email = EmailNotifier(smtp_config)

    return _alert_manager

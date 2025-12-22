"""
Comprehensive Alert Management System
Price alerts, pattern alerts, news alerts, portfolio alerts
"""

from typing import Dict, List, Optional, Any, Callable
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
    PATTERN = "PATTERN"
    NEWS = "NEWS"
    PORTFOLIO = "PORTFOLIO"
    TECHNICAL = "TECHNICAL"


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
    SLACK = "SLACK"
    WHATSAPP = "WHATSAPP"
    PUSH = "PUSH"
    WEBHOOK = "WEBHOOK"


@dataclass
class Alert:
    """Represents a trading alert"""
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
    """Send notifications via Email (requires SMTP configuration)"""

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


class WebhookNotifier:
    """Send notifications to custom webhook"""

    def send_webhook(self, webhook_url: str, payload: Dict[str, Any]) -> bool:
        """Send webhook notification"""
        try:
            response = requests.post(webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info(f"Webhook sent to {webhook_url}")
            return True
        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")
            return False


class SlackNotifier:
    """
    Send notifications via Slack Incoming Webhooks

    Setup:
    1. Go to https://api.slack.com/messaging/webhooks
    2. Create an Incoming Webhook for your workspace
    3. Get the webhook URL (looks like: https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXX)
    4. Add to Settings
    """

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send_message(self, message: str, alert_type: str = "INFO") -> bool:
        """Send message to Slack channel"""
        if not self.enabled:
            logger.warning("Slack notifier not configured")
            return False

        try:
            # Color coding based on alert type
            colors = {
                "SUCCESS": "#28a745",
                "WARNING": "#ffc107",
                "DANGER": "#dc3545",
                "INFO": "#17a2b8"
            }

            payload = {
                "text": f"NeuralTrader Alert",
                "attachments": [{
                    "color": colors.get(alert_type, "#17a2b8"),
                    "text": message,
                    "footer": "NeuralTrader Alert System",
                    "footer_icon": "https://platform.slack-edge.com/img/default_application_icon.png",
                    "ts": int(datetime.now(timezone.utc).timestamp())
                }]
            }

            response = requests.post(self.webhook_url, json=payload, timeout=5)
            response.raise_for_status()
            logger.info("Slack notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False


class WhatsAppNotifier:
    """
    Send notifications via WhatsApp Business API

    Options:
    1. Twilio WhatsApp API (Recommended - Easy)
       - Sign up: https://www.twilio.com/whatsapp
       - Get Account SID, Auth Token, WhatsApp number
       - Free trial: $15 credit

    2. WhatsApp Business API (Official - Complex)
       - Requires business verification
       - Enterprise solution

    3. Unofficial Libraries (Not Recommended)
       - Can get banned
       - Against WhatsApp ToS
    """

    def __init__(
        self,
        twilio_account_sid: Optional[str] = None,
        twilio_auth_token: Optional[str] = None,
        twilio_whatsapp_number: Optional[str] = None,
        user_whatsapp_number: Optional[str] = None
    ):
        self.account_sid = twilio_account_sid
        self.auth_token = twilio_auth_token
        self.from_number = twilio_whatsapp_number  # Format: whatsapp:+14155238886
        self.to_number = user_whatsapp_number      # Format: whatsapp:+919876543210
        self.enabled = all([twilio_account_sid, twilio_auth_token, twilio_whatsapp_number, user_whatsapp_number])

    def send_message(self, message: str) -> bool:
        """Send WhatsApp message via Twilio"""
        if not self.enabled:
            logger.warning("WhatsApp notifier not configured")
            return False

        try:
            from twilio.rest import Client

            client = Client(self.account_sid, self.auth_token)

            twilio_message = client.messages.create(
                body=message,
                from_=self.from_number,
                to=self.to_number
            )

            logger.info(f"WhatsApp message sent: {twilio_message.sid}")
            return True
        except ImportError:
            logger.error("Twilio library not installed. Install with: pip install twilio")
            return False
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")
            return False


class AlertManager:
    """
    Comprehensive Alert Management System
    Monitors prices, patterns, news, and portfolio metrics
    """

    def __init__(
        self,
        telegram_bot_token: Optional[str] = None,
        telegram_chat_id: Optional[str] = None,
        smtp_config: Optional[Dict[str, str]] = None,
        slack_webhook_url: Optional[str] = None,
        twilio_config: Optional[Dict[str, str]] = None
    ):
        self.alerts: Dict[str, Alert] = {}
        self.alert_counter = 0
        self.monitoring_task: Optional[asyncio.Task] = None

        # Initialize notifiers
        self.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)
        self.email = EmailNotifier(smtp_config)
        self.webhook = WebhookNotifier()
        self.slack = SlackNotifier(slack_webhook_url)
        self.whatsapp = WhatsAppNotifier(
            twilio_account_sid=twilio_config.get('account_sid') if twilio_config else None,
            twilio_auth_token=twilio_config.get('auth_token') if twilio_config else None,
            twilio_whatsapp_number=twilio_config.get('whatsapp_number') if twilio_config else None,
            user_whatsapp_number=twilio_config.get('user_whatsapp_number') if twilio_config else None
        )

        # Price tracking for cross alerts
        self.price_history: Dict[str, List[float]] = {}

        # Load alerts from database
        self._load_alerts_from_db()

    def _load_alerts_from_db(self):
        """Load alerts from database on startup"""
        try:
            alert_data_list = alert_db.load_all_alerts()
            for alert_data in alert_data_list:
                # Reconstruct Alert object from database data
                alert = Alert(
                    alert_id=alert_data['alert_id'],
                    user_id=alert_data['user_id'],
                    alert_type=AlertType(alert_data['alert_type']),
                    symbol=alert_data['symbol'],
                    condition=alert_data['condition'],
                    message=alert_data['message'],
                    delivery_channels=[DeliveryChannel(ch) for ch in alert_data['delivery_channels']],
                    status=AlertStatus(alert_data['status']),
                    created_at=datetime.fromisoformat(alert_data['created_at']),
                    triggered_at=datetime.fromisoformat(alert_data['triggered_at']) if alert_data.get('triggered_at') else None,
                    expires_at=datetime.fromisoformat(alert_data['expires_at']) if alert_data.get('expires_at') else None,
                    metadata=alert_data.get('metadata')
                )
                self.alerts[alert.alert_id] = alert

            # Set counter to max existing ID
            self.alert_counter = alert_db.get_max_alert_counter()
            logger.info(f"Loaded {len(self.alerts)} alerts from database, counter at {self.alert_counter}")
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
        """
        Create price alert

        Examples:
            - Alert when RELIANCE crosses above 2500
            - Alert when INFY drops below 1400
            - Alert when TCS changes by +5%
        """
        condition_dict = {
            "type": condition.value,
            "target_price": target_price,
            "percent_change": percent_change
        }

        if condition == PriceCondition.PERCENT_CHANGE_ABOVE:
            message = f"🚀 {symbol} gained {percent_change}%! Current price: ₹{target_price}"
        elif condition == PriceCondition.PERCENT_CHANGE_BELOW:
            message = f"📉 {symbol} dropped {abs(percent_change)}%! Current price: ₹{target_price}"
        elif condition == PriceCondition.ABOVE:
            message = f"📊 {symbol} is now above ₹{target_price}"
        elif condition == PriceCondition.BELOW:
            message = f"📊 {symbol} is now below ₹{target_price}"
        elif condition == PriceCondition.CROSSES_ABOVE:
            message = f"📈 {symbol} crossed above ₹{target_price}"
        else:
            message = f"📉 {symbol} crossed below ₹{target_price}"

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

    def create_pattern_alert(
        self,
        user_id: str,
        symbol: str,
        pattern_types: List[str],
        delivery_channels: List[DeliveryChannel],
        expires_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create pattern alert

        Examples:
            - Alert on Hammer or Engulfing pattern
            - Alert on Morning Star formation
        """
        condition_dict = {
            "pattern_types": pattern_types
        }

        patterns_str = ", ".join(pattern_types)
        message = f"🔔 Pattern detected on {symbol}: {patterns_str}"

        alert = Alert(
            alert_id=self._generate_alert_id(),
            user_id=user_id,
            alert_type=AlertType.PATTERN,
            symbol=symbol,
            condition=condition_dict,
            message=message,
            delivery_channels=delivery_channels,
            expires_at=expires_at
        )

        self.alerts[alert.alert_id] = alert
        self._save_alert_to_db(alert)
        logger.info(f"Pattern alert created: {symbol} watching for {patterns_str}")
        return alert

    def create_news_alert(
        self,
        user_id: str,
        keywords: List[str],
        symbols: Optional[List[str]],
        delivery_channels: List[DeliveryChannel],
        expires_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create news alert

        Examples:
            - Alert on news containing "acquisition" or "merger"
            - Alert on earnings announcements
        """
        condition_dict = {
            "keywords": keywords,
            "symbols": symbols or []
        }

        keywords_str = ", ".join(keywords)
        message = f"📰 News alert: Keywords matched - {keywords_str}"

        alert = Alert(
            alert_id=self._generate_alert_id(),
            user_id=user_id,
            alert_type=AlertType.NEWS,
            symbol=symbols[0] if symbols else "MARKET",
            condition=condition_dict,
            message=message,
            delivery_channels=delivery_channels,
            expires_at=expires_at
        )

        self.alerts[alert.alert_id] = alert
        self._save_alert_to_db(alert)
        logger.info(f"News alert created: watching for {keywords_str}")
        return alert

    def create_portfolio_alert(
        self,
        user_id: str,
        metric: str,
        threshold: float,
        condition: str,  # "above" or "below"
        delivery_channels: List[DeliveryChannel],
        expires_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create portfolio alert

        Examples:
            - Alert when portfolio drawdown exceeds 5%
            - Alert when total P&L crosses 10%
        """
        condition_dict = {
            "metric": metric,
            "threshold": threshold,
            "condition": condition
        }

        message = f"💼 Portfolio alert: {metric} is {condition} {threshold}%"

        alert = Alert(
            alert_id=self._generate_alert_id(),
            user_id=user_id,
            alert_type=AlertType.PORTFOLIO,
            symbol="PORTFOLIO",
            condition=condition_dict,
            message=message,
            delivery_channels=delivery_channels,
            expires_at=expires_at
        )

        self.alerts[alert.alert_id] = alert
        self._save_alert_to_db(alert)
        logger.info(f"Portfolio alert created: {metric} {condition} {threshold}%")
        return alert

    def create_technical_alert(
        self,
        user_id: str,
        symbol: str,
        indicator: str,
        condition: str,
        threshold: float,
        delivery_channels: List[DeliveryChannel],
        expires_at: Optional[datetime] = None
    ) -> Alert:
        """
        Create technical indicator alert

        Examples:
            - Alert when RSI crosses below 30 (oversold)
            - Alert when MACD crosses above signal line
        """
        condition_dict = {
            "indicator": indicator,
            "condition": condition,
            "threshold": threshold
        }

        message = f"📊 {symbol}: {indicator} {condition} {threshold}"

        alert = Alert(
            alert_id=self._generate_alert_id(),
            user_id=user_id,
            alert_type=AlertType.TECHNICAL,
            symbol=symbol,
            condition=condition_dict,
            message=message,
            delivery_channels=delivery_channels,
            expires_at=expires_at
        )

        self.alerts[alert.alert_id] = alert
        self._save_alert_to_db(alert)
        logger.info(f"Technical alert created: {symbol} {indicator} {condition} {threshold}")
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

        # Update database
        alert_db.update_alert_status(
            alert.alert_id,
            AlertStatus.TRIGGERED.value,
            alert.triggered_at.isoformat()
        )

        logger.info(f"Alert triggered: {alert.alert_id} - {alert.message}")

        # Send notifications based on delivery channels
        for channel in alert.delivery_channels:
            try:
                if channel == DeliveryChannel.TELEGRAM:
                    self.telegram.send_message(f"<b>🔔 NeuralTrader Alert</b>\n\n{alert.message}")

                elif channel == DeliveryChannel.SLACK:
                    # Determine alert type for color coding
                    alert_type = "INFO"
                    if "🚀" in alert.message or "gained" in alert.message:
                        alert_type = "SUCCESS"
                    elif "📉" in alert.message or "dropped" in alert.message:
                        alert_type = "DANGER"
                    elif "⚠️" in alert.message:
                        alert_type = "WARNING"

                    self.slack.send_message(alert.message, alert_type)

                elif channel == DeliveryChannel.WHATSAPP:
                    self.whatsapp.send_message(f"🔔 NeuralTrader Alert\n\n{alert.message}")

                elif channel == DeliveryChannel.EMAIL:
                    # Would need user email from database
                    pass

                elif channel == DeliveryChannel.WEBHOOK:
                    # Would need webhook URL from metadata
                    webhook_url = alert.metadata.get('webhook_url') if alert.metadata else None
                    if webhook_url:
                        self.webhook.send_webhook(webhook_url, alert.to_dict())

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
        logger.info(f"Alert cancelled: {alert_id}")
        return True

    def delete_alert(self, alert_id: str) -> bool:
        """Delete an alert permanently"""
        if alert_id not in self.alerts:
            return False

        del self.alerts[alert_id]
        alert_db.delete_alert(alert_id)
        logger.info(f"Alert deleted: {alert_id}")
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
        expired_ids = []

        for alert_id, alert in self.alerts.items():
            if alert.expires_at and alert.expires_at < now:
                alert.status = AlertStatus.EXPIRED
                expired_ids.append(alert_id)

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
    slack_webhook_url: Optional[str] = None,
    twilio_config: Optional[Dict[str, str]] = None
) -> AlertManager:
    """
    Get or create alert manager singleton

    Note: If config parameters are provided and differ from existing instance,
    the notifiers will be updated to use the new configuration.
    """
    global _alert_manager

    if _alert_manager is None:
        _alert_manager = AlertManager(
            telegram_bot_token,
            telegram_chat_id,
            smtp_config,
            slack_webhook_url,
            twilio_config
        )
    else:
        # Update notifiers if new config is provided
        if telegram_bot_token or telegram_chat_id:
            _alert_manager.telegram = TelegramNotifier(telegram_bot_token, telegram_chat_id)
        if smtp_config:
            _alert_manager.email = EmailNotifier(smtp_config)
        if slack_webhook_url:
            _alert_manager.slack = SlackNotifier(slack_webhook_url)
        if twilio_config:
            _alert_manager.whatsapp = WhatsAppNotifier(
                twilio_account_sid=twilio_config.get('account_sid'),
                twilio_auth_token=twilio_config.get('auth_token'),
                twilio_whatsapp_number=twilio_config.get('whatsapp_number'),
                user_whatsapp_number=twilio_config.get('user_whatsapp_number')
            )

    return _alert_manager

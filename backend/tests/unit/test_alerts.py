"""
Unit Tests for Alerts System
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestAlertManager:
    """Test Alert Manager"""

    @patch('alerts.alert_manager.alert_db')
    def test_manager_initialization(self, mock_db):
        """Test alert manager initializes correctly"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager

        manager = AlertManager()

        assert manager is not None
        assert manager.alerts == {}

    @patch('alerts.alert_manager.alert_db')
    def test_create_price_alert(self, mock_db):
        """Test creating a price alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        assert alert is not None
        assert alert.symbol == "RELIANCE"
        mock_db.save_alert.assert_called_once()

    @patch('alerts.alert_manager.alert_db')
    def test_get_user_alerts(self, mock_db):
        """Test getting user's alerts"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        # Create some alerts
        manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )
        manager.create_price_alert(
            user_id="test_user",
            symbol="TCS",
            condition=PriceCondition.BELOW,
            target_price=3400.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        alerts = manager.get_user_alerts("test_user")

        assert len(alerts) == 2

    @patch('alerts.alert_manager.alert_db')
    def test_delete_alert(self, mock_db):
        """Test deleting an alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        result = manager.delete_alert(alert.alert_id)

        assert result == True
        assert alert.alert_id not in manager.alerts

    @patch('alerts.alert_manager.alert_db')
    def test_cancel_alert(self, mock_db):
        """Test cancelling an alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel, AlertStatus

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        result = manager.cancel_alert(alert.alert_id)

        assert result == True
        assert alert.status == AlertStatus.CANCELLED


class TestAlertConditions:
    """Test Alert Condition Checking"""

    @patch('alerts.alert_manager.alert_db')
    def test_check_price_above(self, mock_db):
        """Test price above condition"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        # Should trigger
        assert manager.check_price_alert(alert, 2650.00) == True

        # Should not trigger
        assert manager.check_price_alert(alert, 2550.00) == False

    @patch('alerts.alert_manager.alert_db')
    def test_check_price_below(self, mock_db):
        """Test price below condition"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.BELOW,
            target_price=2500.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        # Should trigger
        assert manager.check_price_alert(alert, 2450.00) == True

        # Should not trigger
        assert manager.check_price_alert(alert, 2550.00) == False

    @patch('alerts.alert_manager.alert_db')
    def test_check_crosses_above(self, mock_db):
        """Test crosses above condition"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, PriceCondition, DeliveryChannel

        manager = AlertManager()

        alert = manager.create_price_alert(
            user_id="test_user",
            symbol="RELIANCE",
            condition=PriceCondition.CROSSES_ABOVE,
            target_price=2600.00,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        # Should trigger (crossed from below to above)
        assert manager.check_price_alert(alert, 2650.00, previous_price=2550.00) == True

        # Should not trigger (was already above)
        assert manager.check_price_alert(alert, 2650.00, previous_price=2620.00) == False


class TestNotifiers:
    """Test Notifier Classes"""

    def test_telegram_notifier_initialization(self):
        """Test Telegram notifier initialization"""
        from alerts.alert_manager import TelegramNotifier

        # Without config
        notifier = TelegramNotifier()
        assert notifier.enabled == False

        # With config
        notifier = TelegramNotifier(bot_token="test", chat_id="123")
        assert notifier.enabled == True

    def test_slack_notifier_initialization(self):
        """Test Slack notifier initialization"""
        from alerts.alert_manager import SlackNotifier

        # Without config
        notifier = SlackNotifier()
        assert notifier.enabled == False

        # With config
        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        assert notifier.enabled == True

    def test_email_notifier_initialization(self):
        """Test Email notifier initialization"""
        from alerts.alert_manager import EmailNotifier

        # Without config
        notifier = EmailNotifier()
        assert notifier.enabled == False

        # With config
        notifier = EmailNotifier(smtp_config={
            "smtp_host": "smtp.test.com",
            "smtp_port": 587
        })
        assert notifier.enabled == True


class TestAlertTypes:
    """Test Different Alert Types"""

    @patch('alerts.alert_manager.alert_db')
    def test_create_pattern_alert(self, mock_db):
        """Test creating pattern alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, DeliveryChannel, AlertType

        manager = AlertManager()

        alert = manager.create_pattern_alert(
            user_id="test_user",
            symbol="RELIANCE",
            pattern_types=["HAMMER", "ENGULFING"],
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PATTERN

    @patch('alerts.alert_manager.alert_db')
    def test_create_technical_alert(self, mock_db):
        """Test creating technical indicator alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, DeliveryChannel, AlertType

        manager = AlertManager()

        alert = manager.create_technical_alert(
            user_id="test_user",
            symbol="RELIANCE",
            indicator="RSI",
            condition="below",
            threshold=30,
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        assert alert is not None
        assert alert.alert_type == AlertType.TECHNICAL

    @patch('alerts.alert_manager.alert_db')
    def test_create_portfolio_alert(self, mock_db):
        """Test creating portfolio alert"""
        mock_db.load_all_alerts.return_value = []
        mock_db.get_max_alert_counter.return_value = 0

        from alerts.alert_manager import AlertManager, DeliveryChannel, AlertType

        manager = AlertManager()

        alert = manager.create_portfolio_alert(
            user_id="test_user",
            metric="drawdown",
            threshold=5.0,
            condition="above",
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        assert alert is not None
        assert alert.alert_type == AlertType.PORTFOLIO


class TestAlertDataclass:
    """Test Alert Dataclass"""

    def test_alert_to_dict(self):
        """Test Alert to_dict method"""
        from alerts.alert_manager import Alert, AlertType, DeliveryChannel, AlertStatus

        alert = Alert(
            alert_id="ALERT_00000001",
            user_id="test_user",
            alert_type=AlertType.PRICE,
            symbol="RELIANCE",
            condition={"type": "ABOVE", "target_price": 2600},
            message="Test alert",
            delivery_channels=[DeliveryChannel.TELEGRAM]
        )

        data = alert.to_dict()

        assert data["alert_id"] == "ALERT_00000001"
        assert data["user_id"] == "test_user"
        assert data["alert_type"] == "PRICE"
        assert data["symbol"] == "RELIANCE"
        assert "TELEGRAM" in data["delivery_channels"]

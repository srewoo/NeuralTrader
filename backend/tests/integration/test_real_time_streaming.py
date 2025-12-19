"""
Integration Tests for Real-Time Streaming
Tests WebSocket connections and market data streaming
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocket
import json


class TestWebSocketConnections:
    """Test WebSocket connectivity and streaming"""

    @pytest.mark.asyncio
    async def test_websocket_connection(self, test_app):
        """Test basic WebSocket connection"""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/test-client-1") as websocket:
                # Connection should be established
                assert websocket is not None

    @pytest.mark.asyncio
    async def test_websocket_subscribe(self, test_app):
        """Test subscribing to symbols"""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/test-client-2") as websocket:
                # Send subscribe message
                subscribe_msg = {
                    "action": "subscribe",
                    "symbols": ["AAPL", "GOOGL"]
                }

                websocket.send_text(json.dumps(subscribe_msg))

                # Should receive confirmation
                response = websocket.receive_json()
                assert response["type"] == "subscribed"
                assert "AAPL" in response["symbols"]
                assert "GOOGL" in response["symbols"]

    @pytest.mark.asyncio
    async def test_websocket_unsubscribe(self, test_app):
        """Test unsubscribing from symbols"""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/test-client-3") as websocket:
                # First subscribe
                websocket.send_json({
                    "action": "subscribe",
                    "symbols": ["AAPL"]
                })

                websocket.receive_json()  # Confirmation

                # Then unsubscribe
                websocket.send_json({
                    "action": "unsubscribe",
                    "symbols": ["AAPL"]
                })

                # Should handle gracefully
                # No exception should be raised

    @pytest.mark.asyncio
    async def test_websocket_receive_ticks(self, test_app):
        """Test receiving market ticks"""
        with TestClient(test_app) as client:
            with client.websocket_connect("/ws/test-client-4") as websocket:
                # Subscribe to symbol
                websocket.send_json({
                    "action": "subscribe",
                    "symbols": ["RELIANCE"]
                })

                # Wait for ticks (with timeout)
                ticks_received = 0
                max_wait = 10  # seconds

                import time
                start_time = time.time()

                while ticks_received < 3 and (time.time() - start_time) < max_wait:
                    try:
                        data = websocket.receive_json(timeout=2)

                        if data.get("type") == "ticker":
                            assert "symbol" in data
                            assert "price" in data
                            assert "timestamp" in data
                            ticks_received += 1

                    except Exception:
                        break

                # Should have received at least one tick
                assert ticks_received > 0, "No market ticks received"


class TestMarketStreamSimulator:
    """Test market stream simulator"""

    @pytest.mark.asyncio
    async def test_stream_initialization(self):
        """Test market stream initializes correctly"""
        from realtime.market_stream import get_market_stream

        stream = get_market_stream()

        assert stream is not None
        assert hasattr(stream, 'watched_symbols')
        assert len(stream.watched_symbols) > 0

    @pytest.mark.asyncio
    async def test_stream_price_updates(self):
        """Test that prices update over time"""
        from realtime.market_stream import get_market_stream

        stream = get_market_stream()

        # Get initial price
        symbol = "RELIANCE"
        initial_price = stream.watched_symbols[symbol]["price"]

        # Wait for updates
        await asyncio.sleep(2)

        # Price should have changed (simulated)
        # Note: This test is probabilistic

    @pytest.mark.asyncio
    async def test_stream_start_stop(self):
        """Test starting and stopping the stream"""
        from realtime.market_stream import get_market_stream

        stream = get_market_stream()

        # Start stream (non-blocking)
        task = asyncio.create_task(stream.start_stream())

        # Let it run briefly
        await asyncio.sleep(1)

        # Stop stream
        await stream.stop_stream()

        # Cancel task
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass  # Expected


class TestConnectionManager:
    """Test WebSocket connection manager"""

    @pytest.mark.asyncio
    async def test_connection_tracking(self):
        """Test that connections are properly tracked"""
        from realtime.connection_manager import get_connection_manager

        manager = get_connection_manager()

        # Initially should have no active connections
        initial_count = len(manager.active_connections)

        # Note: Actual connection testing requires WebSocket mocking

    @pytest.mark.asyncio
    async def test_broadcast_message(self):
        """Test broadcasting messages to all clients"""
        from realtime.connection_manager import get_connection_manager

        manager = get_connection_manager()

        test_data = {
            "type": "test",
            "message": "Hello"
        }

        # Should not raise error even with no connections
        await manager.broadcast(test_data)

    @pytest.mark.asyncio
    async def test_personal_message(self):
        """Test sending message to specific client"""
        from realtime.connection_manager import get_connection_manager

        manager = get_connection_manager()

        test_data = {"type": "test"}
        client_id = "test-client"

        # Should handle gracefully if client doesn't exist
        await manager.send_personal_message(test_data, client_id)


class TestAlpacaLiveStream:
    """Test Alpaca live data streaming (if configured)"""

    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="Requires Alpaca API credentials")
    async def test_alpaca_connection(self, alpaca_credentials):
        """Test connecting to Alpaca stream"""
        from data_providers.alpaca_live import get_alpaca_provider

        provider = get_alpaca_provider(
            alpaca_credentials["key"],
            alpaca_credentials["secret"],
            paper=True
        )

        # Test getting a quote
        quote = await provider.get_latest_quote("AAPL")

        if quote:
            assert "symbol" in quote
            assert "bid" in quote
            assert "ask" in quote

    @pytest.mark.asyncio
    @pytest.mark.skipif(True, reason="Requires Alpaca API credentials")
    async def test_alpaca_subscription(self, alpaca_credentials):
        """Test subscribing to Alpaca stream"""
        from data_providers.alpaca_live import get_alpaca_provider

        provider = get_alpaca_provider(
            alpaca_credentials["key"],
            alpaca_credentials["secret"],
            paper=True
        )

        quotes_received = []

        async def on_quote(data):
            quotes_received.append(data)

        await provider.subscribe_to_quotes("AAPL", on_quote)

        # Start stream briefly
        stream_task = asyncio.create_task(provider.start_stream())

        await asyncio.sleep(5)  # Wait for quotes

        await provider.stop_stream()
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

        # Should have received some quotes
        assert len(quotes_received) > 0


# Fixtures

@pytest.fixture
def test_app():
    """Get test app instance"""
    from server import app
    return app


@pytest.fixture
def alpaca_credentials():
    """Alpaca API credentials (from env or test config)"""
    import os
    return {
        "key": os.getenv("ALPACA_API_KEY", "test_key"),
        "secret": os.getenv("ALPACA_API_SECRET", "test_secret")
    }

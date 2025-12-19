"""
WebSocket Connection Manager
Handles active WebSocket connections and broadcasting.
"""

from typing import List, Dict, Set
from fastapi import WebSocket
import logging
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """
    Manages WebSocket connections and channel subscriptions.
    """
    def __init__(self):
        # Active connections: client_id -> WebSocket
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Subscriptions: symbol -> Set[client_id]
        self.subscriptions: Dict[str, Set[str]] = {}
        
        # Broadcast groups: group_name -> Set[client_id]
        # e.g., "alerts", "portfolio_updates"
        self.groups: Dict[str, Set[str]] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a new connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total: {len(self.active_connections)}")

    def disconnect(self, client_id: str):
        """Remove a connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
        # Clean up subscriptions
        for symbol in list(self.subscriptions.keys()):
            if client_id in self.subscriptions[symbol]:
                self.subscriptions[symbol].discard(client_id)
                if not self.subscriptions[symbol]:
                    del self.subscriptions[symbol]
                    
        logger.info(f"Client {client_id} disconnected")

    async def subscribe(self, client_id: str, symbols: List[str]):
        """Subscribe client to specific symbols"""
        for symbol in symbols:
            if symbol not in self.subscriptions:
                self.subscriptions[symbol] = set()
            self.subscriptions[symbol].add(client_id)
        logger.info(f"Client {client_id} subscribed to {symbols}")

    async def unsubscribe(self, client_id: str, symbols: List[str]):
        """Unsubscribe client from symbols"""
        for symbol in symbols:
            if symbol in self.subscriptions and client_id in self.subscriptions[symbol]:
                self.subscriptions[symbol].discard(client_id)
                if not self.subscriptions[symbol]:
                    del self.subscriptions[symbol]

    async def broadcast_ticker(self, symbol: str, data: Dict):
        """Send market data to subscribers of a symbol"""
        if symbol in self.subscriptions:
            message = {
                "type": "ticker",
                "symbol": symbol,
                "data": data
            }
            subscribers = list(self.subscriptions[symbol])
            for client_id in subscribers:
                await self.send_personal_message(message, client_id)

    async def broadcast_alert(self, data: Dict):
        """Broadcast alert to all connected clients"""
        message = {
            "type": "alert",
            "data": data
        }
        for client_id in self.active_connections:
            await self.send_personal_message(message, client_id)

    async def send_personal_message(self, message: Dict, client_id: str):
        """Send a message to a specific client"""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)

# Global instance
manager = ConnectionManager()

def get_connection_manager() -> ConnectionManager:
    return manager

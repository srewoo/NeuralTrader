
import asyncio
import websockets
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("WebSocketClient")

async def test_websocket():
    uri = "ws://localhost:8005/ws/test_client_1"
    
    logger.info(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            logger.info("Connected!")
            
            # Subscribe to simulated symbols
            subscribe_msg = {
                "action": "subscribe",
                "symbols": ["RELIANCE", "TCS"]
            }
            await websocket.send(json.dumps(subscribe_msg))
            logger.info(f"Sent subscription: {subscribe_msg}")
            
            # Listen for messages
            for i in range(5):
                response = await websocket.recv()
                data = json.loads(response)
                logger.info(f"Received: {data}")
                
                if data.get("type") == "ticker":
                    logger.info(f"✅ Ticker received for {data['symbol']}: {data['data']['price']}")
            
            logger.info("Test complete - closing connection.")
            
    except Exception as e:
        logger.error(f"WebSocket connection failed: {e}")
        logger.error("Make sure the backend server is running!")

if __name__ == "__main__":
    asyncio.run(test_websocket())

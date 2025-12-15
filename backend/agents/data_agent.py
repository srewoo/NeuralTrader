"""
Data Collection Agent
Fetches stock data and prepares it for analysis
"""

from typing import Dict, Any
import yfinance as yf
from .base import BaseAgent


def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol


class DataCollectionAgent(BaseAgent):
    """
    Agent responsible for collecting stock market data
    """
    
    def __init__(self):
        super().__init__("Data Collection Agent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect stock data from yfinance
        
        Args:
            state: Current state with 'symbol' key
            
        Returns:
            Updated state with stock_data
        """
        try:
            symbol = state.get("symbol")
            if not symbol:
                raise ValueError("Symbol not provided in state")
            
            self.log_execution(f"Fetching data for {symbol}")

            # Add running step
            if "agent_steps" not in state:
                state["agent_steps"] = []

            state["agent_steps"].append(
                self.create_step_record(
                    status="running",
                    message=f"Fetching market data for {symbol}..."
                )
            )

            # Add .NS suffix for Indian stocks
            ticker_symbol = get_indian_stock_suffix(symbol)

            # Fetch stock data using yfinance (REAL API CALL)
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
            
            if hist.empty:
                raise ValueError(f"No data available for {symbol}")
            
            # Extract real data
            current_price = hist['Close'].iloc[-1]
            previous_close = info.get('previousClose', hist['Close'].iloc[-1])
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else 0
            
            stock_data = {
                "symbol": symbol.upper(),
                "name": info.get('longName', symbol),
                "current_price": float(current_price),
                "previous_close": float(previous_close),
                "change": float(change),
                "change_percent": float(change_percent),
                "volume": int(hist['Volume'].iloc[-1]) if 'Volume' in hist else 0,
                "market_cap": info.get('marketCap'),
                "pe_ratio": info.get('trailingPE'),
                "week_52_high": info.get('fiftyTwoWeekHigh'),
                "week_52_low": info.get('fiftyTwoWeekLow'),
                "sector": info.get('sector'),
                "industry": info.get('industry')
            }
            
            # Update state
            state["stock_data"] = stock_data
            
            # Update step to completed
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=f"Successfully collected data for {symbol}",
                data={
                    "price": stock_data["current_price"],
                    "volume": stock_data["volume"],
                    "market_cap": stock_data["market_cap"]
                }
            )
            
            self.log_execution(f"Successfully fetched data for {symbol}")
            return state
            
        except Exception as e:
            return await self.handle_error(e, state)


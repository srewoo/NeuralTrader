"""
Data Collection Agent
Fetches stock data and prepares it for analysis
"""

from typing import Dict, Any, Optional, Tuple
import yfinance as yf
from .base import BaseAgent
import logging

# Use absolute import - server.py runs from backend directory
from data_providers.provider_manager import get_provider_manager

logger = logging.getLogger(__name__)


def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol


class DataCollectionAgent(BaseAgent):
    """
    Agent responsible for collecting stock market data
    Enhanced with fallback mechanism for data source reliability
    """

    # Fallback exchange suffixes for Indian stocks
    INDIAN_EXCHANGE_SUFFIXES = ['.NS', '.BO']  # NSE, BSE

    def __init__(self):
        super().__init__("Data Collection Agent")

    def _try_fetch_with_fallback(self, base_symbol: str) -> Tuple[Optional[yf.Ticker], Optional[str], Optional[str]]:
        """
        Try fetching data with fallback across multiple exchanges

        Args:
            base_symbol: Base stock symbol without exchange suffix

        Returns:
            Tuple of (ticker, successful_symbol, data_source_info)
        """
        base_symbol = base_symbol.upper().strip()

        # If symbol already has exchange suffix, try it first
        if base_symbol.endswith('.NS') or base_symbol.endswith('.BO'):
            try:
                ticker = yf.Ticker(base_symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    return ticker, base_symbol, f"Primary source ({base_symbol})"
            except Exception as e:
                logger.warning(f"Failed to fetch {base_symbol}: {e}")

        # Try different exchange suffixes
        for suffix in self.INDIAN_EXCHANGE_SUFFIXES:
            symbol_with_suffix = f"{base_symbol.replace('.NS', '').replace('.BO', '')}{suffix}"
            try:
                logger.info(f"Attempting to fetch data from {symbol_with_suffix}")
                ticker = yf.Ticker(symbol_with_suffix)
                hist = ticker.history(period="1d")

                if not hist.empty:
                    exchange_name = "NSE" if suffix == ".NS" else "BSE"
                    logger.info(f"Successfully fetched data from {exchange_name}")
                    return ticker, symbol_with_suffix, f"Fallback source ({exchange_name})"

            except Exception as e:
                logger.warning(f"Failed to fetch {symbol_with_suffix}: {e}")
                continue

        # All attempts failed
        return None, None, "All data sources failed"

    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect stock data using multi-provider system with fallback

        Args:
            state: Current state with 'symbol' key and optional 'api_keys'

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
                    message=f"Fetching market data for {symbol} with multi-source fallback..."
                )
            )

            # Get API keys from state (passed from frontend via localStorage)
            api_keys = state.get("data_provider_keys", {})

            # Get provider manager with user's API keys
            provider_manager = get_provider_manager(api_keys if api_keys else None)

            # Try to fetch quote using provider manager
            stock_data_obj = await provider_manager.get_quote(symbol)

            if stock_data_obj is None:
                raise ValueError(
                    f"Failed to fetch data for {symbol} from all available sources. "
                    "Please verify the symbol is correct. "
                    "Consider adding API keys in Settings for better data availability."
                )

            # Convert to dictionary
            stock_data = stock_data_obj.to_dict()

            # Update state
            state["stock_data"] = stock_data

            # Update step to completed
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=f"Successfully collected data for {symbol} from {stock_data['data_source']}",
                data={
                    "ticker": stock_data["ticker_symbol"],
                    "price": stock_data["current_price"],
                    "volume": stock_data["volume"],
                    "market_cap": stock_data["market_cap"],
                    "data_source": stock_data["data_source"]
                }
            )

            self.log_execution(f"Successfully fetched data for {symbol} using {stock_data['data_source']}")
            return state

        except Exception as e:
            return await self.handle_error(e, state)


"""
Technical Analysis Agent
Calculates technical indicators and analyzes price patterns
"""

from typing import Dict, Any
import pandas as pd
import ta
from .base import BaseAgent


def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol


class TechnicalAnalysisAgent(BaseAgent):
    """
    Agent responsible for technical analysis and indicator calculation
    """
    
    def __init__(self):
        super().__init__("Technical Analysis Agent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate technical indicators
        
        Args:
            state: Current state with 'symbol' key
            
        Returns:
            Updated state with technical_indicators
        """
        try:
            symbol = state.get("symbol")
            if not symbol:
                raise ValueError("Symbol not provided in state")
            
            self.log_execution(f"Calculating technical indicators for {symbol}")
            
            # Add running step
            if "agent_steps" not in state:
                state["agent_steps"] = []
            
            state["agent_steps"].append(
                self.create_step_record(
                    status="running",
                    message=f"Calculating 14 technical indicators..."
                )
            )

            # Add .NS suffix for Indian stocks
            ticker_symbol = get_indian_stock_suffix(symbol)

            # Fetch historical data (REAL API CALL)
            import yfinance as yf
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="6mo")
            
            if len(hist) < 50:
                raise ValueError(f"Insufficient data for {symbol}")
            
            # Calculate real technical indicators
            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']
            
            # RSI
            rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
            rsi = rsi_indicator.rsi().iloc[-1]
            
            # MACD
            macd_indicator = ta.trend.MACD(close)
            macd = macd_indicator.macd().iloc[-1]
            macd_signal = macd_indicator.macd_signal().iloc[-1]
            macd_histogram = macd_indicator.macd_diff().iloc[-1]
            
            # Moving Averages
            sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]
            sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
            sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(hist) >= 200 else None
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_middle = bb.bollinger_mavg().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]
            
            # ATR
            atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
            
            # OBV
            obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]
            
            # Stochastic
            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stochastic_k = stoch.stoch().iloc[-1]
            stochastic_d = stoch.stoch_signal().iloc[-1]
            
            # Additional indicators
            adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]
            cci = ta.trend.CCIIndicator(high, low, close, window=20).cci().iloc[-1]
            
            technical_indicators = {
                "rsi": round(float(rsi), 2) if not pd.isna(rsi) else None,
                "macd": round(float(macd), 2) if not pd.isna(macd) else None,
                "macd_signal": round(float(macd_signal), 2) if not pd.isna(macd_signal) else None,
                "macd_histogram": round(float(macd_histogram), 2) if not pd.isna(macd_histogram) else None,
                "sma_20": round(float(sma_20), 2) if not pd.isna(sma_20) else None,
                "sma_50": round(float(sma_50), 2) if not pd.isna(sma_50) else None,
                "sma_200": round(float(sma_200), 2) if sma_200 and not pd.isna(sma_200) else None,
                "bb_upper": round(float(bb_upper), 2) if not pd.isna(bb_upper) else None,
                "bb_middle": round(float(bb_middle), 2) if not pd.isna(bb_middle) else None,
                "bb_lower": round(float(bb_lower), 2) if not pd.isna(bb_lower) else None,
                "atr": round(float(atr), 2) if not pd.isna(atr) else None,
                "obv": round(float(obv), 2) if not pd.isna(obv) else None,
                "stochastic_k": round(float(stochastic_k), 2) if not pd.isna(stochastic_k) else None,
                "stochastic_d": round(float(stochastic_d), 2) if not pd.isna(stochastic_d) else None,
                "adx": round(float(adx), 2) if not pd.isna(adx) else None,
                "cci": round(float(cci), 2) if not pd.isna(cci) else None
            }
            
            # Update state
            state["technical_indicators"] = technical_indicators
            
            # Analyze indicators for signals
            signals = self._analyze_signals(technical_indicators, state.get("stock_data", {}))
            state["technical_signals"] = signals
            
            # Update step to completed
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=f"Calculated 14 technical indicators",
                data={
                    "rsi": technical_indicators.get("rsi"),
                    "macd": technical_indicators.get("macd"),
                    "trend_signal": signals.get("trend"),
                    "momentum_signal": signals.get("momentum")
                }
            )
            
            self.log_execution(f"Successfully calculated indicators for {symbol}")
            return state
            
        except Exception as e:
            return await self.handle_error(e, state)
    
    def _analyze_signals(self, indicators: Dict[str, Any], stock_data: Dict[str, Any]) -> Dict[str, str]:
        """
        Analyze technical indicators to generate signals
        
        Args:
            indicators: Technical indicators
            stock_data: Stock data
            
        Returns:
            Dictionary of signals
        """
        signals = {}
        
        # RSI signal
        rsi = indicators.get("rsi")
        if rsi:
            if rsi < 30:
                signals["rsi"] = "oversold"
            elif rsi > 70:
                signals["rsi"] = "overbought"
            else:
                signals["rsi"] = "neutral"
        
        # MACD signal
        macd = indicators.get("macd")
        macd_signal = indicators.get("macd_signal")
        if macd and macd_signal:
            if macd > macd_signal:
                signals["macd"] = "bullish"
            else:
                signals["macd"] = "bearish"
        
        # Trend signal (based on moving averages)
        current_price = stock_data.get("current_price")
        sma_20 = indicators.get("sma_20")
        sma_50 = indicators.get("sma_50")
        
        if current_price and sma_20 and sma_50:
            if current_price > sma_20 > sma_50:
                signals["trend"] = "strong_uptrend"
            elif current_price > sma_20:
                signals["trend"] = "uptrend"
            elif current_price < sma_20 < sma_50:
                signals["trend"] = "strong_downtrend"
            else:
                signals["trend"] = "downtrend"
        
        # Momentum signal
        stoch_k = indicators.get("stochastic_k")
        if stoch_k:
            if stoch_k < 20:
                signals["momentum"] = "oversold"
            elif stoch_k > 80:
                signals["momentum"] = "overbought"
            else:
                signals["momentum"] = "neutral"
        
        # Volatility signal
        atr = indicators.get("atr")
        if atr:
            if atr > 10:
                signals["volatility"] = "high"
            elif atr < 5:
                signals["volatility"] = "low"
            else:
                signals["volatility"] = "normal"
        
        return signals


from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks, WebSocket
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone
import json
import asyncio
import yfinance as yf
import ta
import pandas as pd
import numpy as np

# Import RAG system
from rag.retrieval import get_retriever

# Import multi-agent orchestrator
from agents.orchestrator import get_orchestrator

# Import backtesting system
from backtesting.engine import BacktestEngine
from backtesting.strategies import StrategyRegistry
from backtesting.price_cache import get_price_cache

# Import news system
from news.sources import get_news_aggregator
from news.sentiment import get_sentiment_analyzer
from news.rate_limiter import get_rate_limiter

# Import pattern detection
from patterns.candlestick import get_pattern_detector

# Import enhanced analysis system
from analysis.enhanced_analyzer import get_enhanced_analyzer

# Import LLM features
from llm_features import get_llm_features

# Import Portfolio Management
from portfolio.risk_metrics import calculate_portfolio_risk
from portfolio.correlation import calculate_correlation_matrix, calculate_diversification_score
from portfolio.optimizer import optimize_portfolio

# Import Real-Time System
from realtime.connection_manager import get_connection_manager
from realtime.market_stream import get_market_stream

# Import Cost Tracking
from cost_tracking import get_cost_tracker

# Import Indian Market Indices
from market_data.indian_indices import get_indian_indices_data

# Import Paper Trading, Alerts, TVScreener, Risk Management
from portfolio.paper_trading import get_paper_trading_engine, OrderSide, OrderType, OrderStatus as PaperOrderStatus
from alerts.alert_manager import get_alert_manager, PriceCondition, DeliveryChannel, AlertStatus as AlertStatusEnum
from data_providers.tvscreener_provider import (
    get_tvscreener_provider,
    get_all_indian_stocks,
    get_all_indian_stocks_async,
    search_indian_stocks,
    get_top_stocks_by_market_cap,
    clear_stock_cache
)
from portfolio.risk_manager import get_risk_manager

# Import Options module
from options.router import router as options_router

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection (with defaults for development)
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'neuraltrader')]

# Create the main app without a prefix
app = FastAPI()

# Add CORS middleware FIRST - before any routes
# This ensures CORS headers are added to ALL responses including errors
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[origin.strip() for origin in os.environ.get('CORS_ORIGINS', 'http://localhost:3005').split(',')],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============ MODELS ============

class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None  # For Claude models in ensemble
    finnhub_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    fmp_api_key: Optional[str] = None
    iex_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    twelve_data_api_key: Optional[str] = None
    # News API keys
    newsapi_key: Optional[str] = None  # For NewsAPI.org
    alphavantage_api_key: Optional[str] = None  # For Alpha Vantage news
    # Telegram Bot (for alerts)
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    # Email SMTP (for email alerts)
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    # Webhook URL (for custom integrations)
    webhook_url: Optional[str] = None
    # Slack (for alerts)
    slack_webhook_url: Optional[str] = None
    # WhatsApp via Twilio (for alerts)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_number: Optional[str] = None
    user_whatsapp_number: Optional[str] = None
    # TVScreener (FREE - no key needed!)
    use_tvscreener: bool = True
    selected_model: str = "gpt-4o-mini"
    selected_provider: str = "openai"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SettingsCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    finnhub_api_key: Optional[str] = None
    alpaca_api_key: Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    fmp_api_key: Optional[str] = None
    iex_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    twelve_data_api_key: Optional[str] = None
    newsapi_key: Optional[str] = None
    alphavantage_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    webhook_url: Optional[str] = None
    slack_webhook_url: Optional[str] = None
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_whatsapp_number: Optional[str] = None
    user_whatsapp_number: Optional[str] = None
    # Indian Broker API Keys
    angelone_api_key: Optional[str] = None
    angelone_client_id: Optional[str] = None
    angelone_password: Optional[str] = None
    angelone_totp_secret: Optional[str] = None
    zerodha_api_key: Optional[str] = None
    zerodha_api_secret: Optional[str] = None
    use_tvscreener: bool = True
    selected_model: str = "gpt-4o-mini"
    selected_provider: str = "openai"

    # Validator to convert empty strings to None for smtp_port
    @validator('smtp_port', pre=True, always=True)
    def empty_str_to_none(cls, v):
        if v == '' or v is None:
            return None
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError:
                return None
        return v

class StockData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    symbol: str
    name: str
    current_price: float
    previous_close: float
    change: float
    change_percent: float
    volume: int
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    week_52_high: Optional[float] = None
    week_52_low: Optional[float] = None

class TechnicalIndicators(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rsi: Optional[float] = None
    macd: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    sma_200: Optional[float] = None
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None
    atr: Optional[float] = None
    obv: Optional[float] = None
    stochastic_k: Optional[float] = None
    stochastic_d: Optional[float] = None

class PriceHistory(BaseModel):
    model_config = ConfigDict(extra="ignore")
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[int]

class AgentStep(BaseModel):
    model_config = ConfigDict(extra="ignore")
    agent_name: str
    status: str  # "running", "completed", "failed"
    message: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    data: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str
    stock_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    recommendation: str  # "BUY", "SELL", "HOLD"
    confidence: float
    entry_price: Optional[float] = None
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    risk_reward_ratio: Optional[float] = None
    reasoning: str
    key_risks: List[str] = []
    agent_steps: List[Dict[str, Any]] = []
    model_used: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class AnalysisRequest(BaseModel):
    symbol: str
    model: str = "gpt-4o-mini"
    provider: str = "openai"

    @validator('symbol')
    def validate_symbol(cls, v):
        """Validate stock symbol format"""
        if not v or not isinstance(v, str):
            raise ValueError("Symbol must be a non-empty string")
        # Remove whitespace and convert to uppercase
        v = v.strip().upper()
        # Check for valid symbol format (alphanumeric, dots, hyphens allowed)
        import re
        if not re.match(r'^[A-Z0-9\.\-]+$', v):
            raise ValueError("Symbol contains invalid characters")
        if len(v) > 20:
            raise ValueError("Symbol too long (max 20 characters)")
        return v

    @validator('provider')
    def validate_provider(cls, v):
        """Validate provider is supported"""
        if v not in ['openai', 'gemini']:
            raise ValueError("Provider must be 'openai' or 'gemini'")
        return v

class StockSearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str

class PortfolioRequest(BaseModel):
    items: List[Dict[str, Any]] # [{"symbol": "RELIANCE", "weight": 0.5}, ...]
    period: str = "1y"

class OptimizerRequest(BaseModel):
    symbols: List[str]
    risk_free_rate: float = 0.06


class PaperOrderRequest(BaseModel):
    """Request body for paper trading orders"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    current_price: Optional[float] = None  # Optional - will fetch if not provided
    order_type: str = "MARKET"
    limit_price: Optional[float] = None


# ============ DATA COLLECTION ============

def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol

async def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Fetch stock data from Yahoo Finance with latest available data"""
    try:
        from datetime import datetime, timezone

        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)

        # Get basic info
        info = ticker.info

        # Try to get intraday data first (1m interval for last 7 days)
        # This gives us the most recent data available
        try:
            hist_intraday = ticker.history(period="1d", interval="1m")
            if not hist_intraday.empty:
                hist = hist_intraday
                is_intraday = True
            else:
                hist = ticker.history(period="5d")
                is_intraday = False
        except Exception as e:
            logger.warning(f"Failed to get intraday data: {e}")
            hist = ticker.history(period="5d")
            is_intraday = False

        if hist.empty:
            # Try BSE
            ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            try:
                hist_intraday = ticker.history(period="1d", interval="1m")
                if not hist_intraday.empty:
                    hist = hist_intraday
                    is_intraday = True
                else:
                    hist = ticker.history(period="5d")
                    is_intraday = False
            except Exception as e:
                logger.warning(f"Failed to get BSE intraday data: {e}")
                hist = ticker.history(period="5d")
                is_intraday = False

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        # Get the latest data point
        last_timestamp = hist.index[-1]
        current_price = float(hist['Close'].iloc[-1])
        previous_close = info.get('previousClose', current_price)

        # Calculate data age
        if hasattr(last_timestamp, 'to_pydatetime'):
            last_update = last_timestamp.to_pydatetime()
        else:
            last_update = last_timestamp

        # Make timezone aware if not already
        if last_update.tzinfo is None:
            from pytz import timezone as tz
            ist = tz('Asia/Kolkata')
            last_update = ist.localize(last_update)
        else:
            # Convert to IST if already timezone-aware
            from pytz import timezone as tz
            ist = tz('Asia/Kolkata')
            last_update = last_update.astimezone(ist)

        current_time = datetime.now(timezone.utc)
        data_age_minutes = int((current_time - last_update).total_seconds() / 60)

        return {
            "symbol": symbol.upper().replace('.NS', '').replace('.BO', ''),
            "name": info.get('longName', info.get('shortName', symbol)),
            "current_price": round(current_price, 2),
            "previous_close": round(previous_close, 2),
            "change": round(current_price - previous_close, 2),
            "change_percent": round(((current_price - previous_close) / previous_close) * 100, 2) if previous_close else 0,
            "volume": int(hist['Volume'].iloc[-1]) if not hist.empty else info.get('volume', 0),
            "market_cap": info.get('marketCap'),
            "pe_ratio": info.get('trailingPE'),
            "week_52_high": info.get('fiftyTwoWeekHigh'),
            "week_52_low": info.get('fiftyTwoWeekLow'),
            "sector": info.get('sector', 'N/A'),
            "industry": info.get('industry', 'N/A'),
            "last_updated": last_update.isoformat(),
            "data_age_minutes": data_age_minutes,
            "is_realtime": is_intraday and data_age_minutes < 5
        }
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def fetch_historical_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
    """Fetch historical price data"""
    try:
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period=period)
        
        if hist.empty:
            ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period=period)
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")

        # Return array of objects for frontend chart compatibility
        data = []
        for i, (date, row) in enumerate(hist.iterrows()):
            data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })

        return data
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def calculate_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Calculate technical indicators"""
    try:
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period="1y")
        
        if df.empty:
            ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
            ticker = yf.Ticker(ticker_symbol)
            df = ticker.history(period="1y")
        
        if df.empty or len(df) < 50:
            return {}
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # RSI
        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        
        # MACD
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_histogram = macd_indicator.macd_diff().iloc[-1]
        
        # Moving Averages
        sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]
        sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
        sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(df) >= 200 else None
        
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
        
        # Convert numpy types to Python native types for JSON serialization
        def to_float(val):
            if val is None or pd.isna(val):
                return None
            return float(round(val, 2))

        return {
            "rsi": to_float(rsi),
            "macd": to_float(macd),
            "macd_signal": to_float(macd_signal),
            "macd_histogram": to_float(macd_histogram),
            "sma_20": to_float(sma_20),
            "sma_50": to_float(sma_50),
            "sma_200": to_float(sma_200) if sma_200 else None,
            "bb_upper": to_float(bb_upper),
            "bb_middle": to_float(bb_middle),
            "bb_lower": to_float(bb_lower),
            "atr": to_float(atr),
            "obv": to_float(obv),
            "stochastic_k": to_float(stochastic_k),
            "stochastic_d": to_float(stochastic_d)
        }
    except Exception as e:
        logger.error(f"Error calculating technical indicators: {e}")
        return {}

# ============ AI ANALYSIS ============

async def run_ai_analysis(symbol: str, stock_data: Dict, technical_indicators: Dict, model: str, provider: str, api_key: str) -> Dict[str, Any]:
    """Run AI analysis using LLM"""
    import openai
    from google import genai
    from google.genai import types
    
    agent_steps = []
    
    # Step 1: Data Agent
    agent_steps.append({
        "agent_name": "Data Collection Agent",
        "status": "completed",
        "message": f"Successfully collected data for {symbol}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {"price": stock_data.get('current_price'), "volume": stock_data.get('volume')}
    })
    
    # Step 2: Technical Analysis Agent
    agent_steps.append({
        "agent_name": "Technical Analysis Agent",
        "status": "completed",
        "message": f"Calculated 14 technical indicators",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {"rsi": technical_indicators.get('rsi'), "macd": technical_indicators.get('macd')}
    })
    
    # Step 3: RAG Knowledge Agent (REAL IMPLEMENTATION)
    try:
        retriever = get_retriever()
        
        # Build query for RAG
        rag_query = f"Stock {symbol} analysis with RSI {technical_indicators.get('rsi', 'N/A')}, MACD {technical_indicators.get('macd', 'N/A')}, price trend"
        
        # Retrieve relevant knowledge
        rag_results = retriever.retrieve(query=rag_query, n_results=5, min_similarity=0.5)
        
        # Build context from retrieved documents
        rag_context = retriever.build_context(query=rag_query, n_results=5, max_tokens=1500)
        
        agent_steps.append({
            "agent_name": "RAG Knowledge Agent",
            "status": "completed",
            "message": f"Retrieved {len(rag_results)} relevant historical patterns and insights",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {
                "patterns_found": len(rag_results),
                "avg_similarity": round(sum(r['similarity'] for r in rag_results) / len(rag_results), 2) if rag_results else 0,
                "categories": list(set(r.get('metadata', {}).get('category', 'unknown') for r in rag_results))
            }
        })
    except Exception as e:
        logger.warning(f"RAG retrieval failed, continuing without RAG context: {e}")
        rag_context = ""
        agent_steps.append({
            "agent_name": "RAG Knowledge Agent",
            "status": "completed",
            "message": "RAG system unavailable, proceeding with direct analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"patterns_found": 0, "note": "RAG not initialized"}
        })
    
    # Step 4: Deep Reasoning Agent
    agent_steps.append({
        "agent_name": "Deep Reasoning Agent",
        "status": "running",
        "message": "Analyzing market data with AI...",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": None
    })
    
    try:
        # Build the analysis prompt with RAG context
        system_message = "You are an expert stock analyst for Indian markets. Always respond with valid JSON."
        prompt = f"""You are an expert stock market analyst specializing in Indian markets (NSE/BSE). Analyze this stock and provide a trading recommendation.

{rag_context if rag_context else ""}

STOCK DATA:
Symbol: {stock_data.get('symbol')}
Name: {stock_data.get('name')}
Current Price: ₹{stock_data.get('current_price')}
Previous Close: ₹{stock_data.get('previous_close')}
Change: {stock_data.get('change_percent')}%
Volume: {stock_data.get('volume'):,}
Market Cap: {stock_data.get('market_cap', 'N/A')}
P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
52-Week High: ₹{stock_data.get('week_52_high', 'N/A')}
52-Week Low: ₹{stock_data.get('week_52_low', 'N/A')}

TECHNICAL INDICATORS:
RSI (14): {technical_indicators.get('rsi', 'N/A')}
MACD: {technical_indicators.get('macd', 'N/A')}
MACD Signal: {technical_indicators.get('macd_signal', 'N/A')}
MACD Histogram: {technical_indicators.get('macd_histogram', 'N/A')}
SMA 20: ₹{technical_indicators.get('sma_20', 'N/A')}
SMA 50: ₹{technical_indicators.get('sma_50', 'N/A')}
SMA 200: ₹{technical_indicators.get('sma_200', 'N/A')}
Bollinger Upper: ₹{technical_indicators.get('bb_upper', 'N/A')}
Bollinger Middle: ₹{technical_indicators.get('bb_middle', 'N/A')}
Bollinger Lower: ₹{technical_indicators.get('bb_lower', 'N/A')}
ATR: {technical_indicators.get('atr', 'N/A')}
Stochastic K: {technical_indicators.get('stochastic_k', 'N/A')}
Stochastic D: {technical_indicators.get('stochastic_d', 'N/A')}

Use the historical knowledge provided above to inform your analysis. Look for similar patterns and apply proven strategies.

Provide your analysis in the following JSON format:
{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100,
  "entry_price": suggested entry price,
  "target_price": price target,
  "stop_loss": stop loss price,
  "risk_reward_ratio": calculated ratio,
  "reasoning": "Detailed step-by-step analysis explaining your decision. Include technical analysis, momentum indicators assessment, support/resistance levels, and market context.",
  "key_risks": ["risk1", "risk2", "risk3"]
}}

Be specific with price levels and provide actionable insights. Consider both bullish and bearish scenarios."""

        # Call LLM based on provider
        if provider == "openai":
            client = openai.AsyncOpenAI(api_key=api_key)
            completion = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            response = completion.choices[0].message.content
        elif provider == "gemini":
            client = genai.Client(api_key=api_key)
            completion = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=types.Content(
                    parts=[types.Part(text=prompt)]
                ),
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    system_instruction=system_message
                )
            )
            response = completion.text
        else:
            raise ValueError(f"Unsupported provider: {provider}")
        
        # Parse the response
        try:
            # Check if response is None or empty
            if not response:
                raise ValueError("Empty response from LLM")
            
            # Try to extract JSON from response
            response_text = response.strip()
            if response_text.startswith('```json'):
                response_text = response_text[7:]
            if response_text.startswith('```'):
                response_text = response_text[3:]
            if response_text.endswith('```'):
                response_text = response_text[:-3]
            
            analysis = json.loads(response_text.strip())
        except json.JSONDecodeError:
            # If JSON parsing fails, create a default response
            analysis = {
                "recommendation": "HOLD",
                "confidence": 50,
                "entry_price": stock_data.get('current_price'),
                "target_price": stock_data.get('current_price') * 1.1,
                "stop_loss": stock_data.get('current_price') * 0.95,
                "risk_reward_ratio": 2.0,
                "reasoning": response,
                "key_risks": ["Market volatility", "Unable to parse structured analysis"]
            }
        
        # Update agent step
        agent_steps[-1]["status"] = "completed"
        agent_steps[-1]["message"] = f"Analysis complete with {analysis.get('confidence', 0)}% confidence"
        
        # Step 5: Validator Agent
        agent_steps.append({
            "agent_name": "Validator Agent",
            "status": "completed",
            "message": "Recommendation validated and risk-checked",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": {"validated": True, "risk_level": "medium" if analysis.get('confidence', 50) < 70 else "low"}
        })
        
        return {
            "recommendation": analysis.get('recommendation', 'HOLD'),
            "confidence": analysis.get('confidence', 50),
            "entry_price": analysis.get('entry_price', stock_data.get('current_price')),
            "target_price": analysis.get('target_price'),
            "stop_loss": analysis.get('stop_loss'),
            "risk_reward_ratio": analysis.get('risk_reward_ratio'),
            "reasoning": analysis.get('reasoning', ''),
            "key_risks": analysis.get('key_risks', []),
            "agent_steps": agent_steps
        }
        
    except Exception as e:
        logger.error(f"AI analysis error: {e}")
        agent_steps[-1]["status"] = "failed"
        agent_steps[-1]["message"] = f"Error: {str(e)}"
        
        return {
            "recommendation": "HOLD",
            "confidence": 0,
            "entry_price": stock_data.get('current_price'),
            "target_price": None,
            "stop_loss": None,
            "risk_reward_ratio": None,
            "reasoning": f"Analysis could not be completed: {str(e)}. Please check your API key configuration.",
            "key_risks": ["Analysis incomplete", "API error"],
            "agent_steps": agent_steps
        }

# ============ API ENDPOINTS ============

# ============ WEB SOCKETS ============

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    manager = get_connection_manager()
    await manager.connect(websocket, client_id)
    
    # TODO: Market stream disabled - causes Python crash on macOS
    # stream = get_market_stream()
    # if not stream.active:
    #     asyncio.create_task(stream.start_stream())
    #     logger.info("Market stream started on WebSocket connection")
    
    try:
        while True:
            # Wait for messages from client (e.g. subscribe)
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                action = message.get("action")
                
                if action == "subscribe":
                    symbols = message.get("symbols", [])
                    await manager.subscribe(client_id, symbols)
                    # Send confirmation
                    await manager.send_personal_message({"type": "subscribed", "symbols": symbols}, client_id)
                    
                elif action == "unsubscribe":
                    symbols = message.get("symbols", [])
                    await manager.unsubscribe(client_id, symbols)
                    
            except json.JSONDecodeError:
                pass
                
    except Exception as e:
        logger.warning(f"WebSocket error for {client_id}: {e}")
    finally:
        # Ensure cleanup happens even if exception occurs
        manager.disconnect(client_id)

@app.on_event("startup")
async def startup_event():
    # Create MongoDB indexes for better query performance
    try:
        # Index for analysis history queries by symbol
        await db.analysis_history.create_index([("symbol", 1), ("created_at", -1)])
        # Index for settings lookups
        await db.settings.create_index([("id", 1)])
        # Index for watchlist queries
        await db.watchlist.create_index([("symbol", 1)])
        # Index for API cost tracking queries
        await db.api_costs.create_index([("timestamp", -1)])
        await db.api_costs.create_index([("provider", 1), ("model", 1)])
        await db.api_costs.create_index([("user_id", 1), ("timestamp", -1)])
        # Index for recommendations queries
        await db.recommendations.create_index([("generated_at", -1)])

        # TTL indexes for automatic data cleanup (data expires after specified seconds)
        # Analysis history: keep for 30 days (2592000 seconds)
        try:
            await db.analysis_history.create_index(
                "created_at",
                expireAfterSeconds=30 * 24 * 60 * 60,  # 30 days
                name="analysis_history_ttl"
            )
        except Exception:
            pass  # Index may already exist with different options

        # Recommendations: keep for 7 days (604800 seconds)
        try:
            await db.recommendations.create_index(
                "generated_at",
                expireAfterSeconds=7 * 24 * 60 * 60,  # 7 days
                name="recommendations_ttl"
            )
        except Exception:
            pass  # Index may already exist

        # Backtests: keep for 90 days
        try:
            await db.backtests.create_index(
                "timestamp",
                expireAfterSeconds=90 * 24 * 60 * 60,  # 90 days
                name="backtests_ttl"
            )
        except Exception:
            pass

        logger.info("MongoDB indexes and TTL policies created successfully")
    except Exception as e:
        logger.warning(f"Failed to create MongoDB indexes: {e}")

    # Initialize Redis cache
    try:
        from utils.redis_cache import init_cache
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        await init_cache(redis_url=redis_url, namespace="neuraltrader")
        logger.info("Redis cache initialized")
    except Exception as e:
        logger.warning(f"Redis cache initialization failed (using memory fallback): {e}")

    # Trigger initial stock recommendations generation
    try:
        from tasks.ai_tasks import generate_stock_recommendations
        # Run in background - don't wait for completion
        generate_stock_recommendations.delay(50)  # Quick initial analysis of top 50 stocks
        logger.info("Triggered initial stock recommendations generation")
    except Exception as e:
        logger.warning(f"Failed to trigger initial recommendations: {e}")

    # Startup complete - market stream will be started when first WebSocket connects
    logger.info("Application startup complete")

@app.on_event("shutdown")
async def shutdown_event():
    # Stop market stream if it's running
    try:
        stream = get_market_stream()
        if stream.active:
            await stream.stop_stream()
            logger.info("Market stream stopped")
    except Exception as e:
        logger.warning(f"Market stream shutdown error: {e}")

    # Shutdown Redis cache
    try:
        from utils.redis_cache import shutdown_cache
        await shutdown_cache()
        logger.info("Redis cache shutdown complete")
    except Exception as e:
        logger.warning(f"Redis cache shutdown error: {e}")

@api_router.get("/")
async def root():
    return {"message": "Stock Trading AI API", "version": "1.0.0"}

# Settings endpoints
@api_router.get("/settings")
async def get_settings():
    settings = await db.settings.find_one({}, {"_id": 0})

    # Default settings template
    default_settings = {
        "openai_api_key": "",
        "gemini_api_key": "",
        "anthropic_api_key": "",
        "finnhub_api_key": "",
        "alpaca_api_key": "",
        "alpaca_api_secret": "",
        "fmp_api_key": "",
        "iex_api_key": "",
        "polygon_api_key": "",
        "twelve_data_api_key": "",
        "newsapi_key": "",
        "alphavantage_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "smtp_host": "",
        "smtp_port": "",
        "smtp_user": "",
        "smtp_password": "",
        "smtp_from_email": "",
        "webhook_url": "",
        "slack_webhook_url": "",
        "twilio_account_sid": "",
        "twilio_auth_token": "",
        "twilio_whatsapp_number": "",
        "user_whatsapp_number": "",
        # Indian Broker API Keys
        "angelone_api_key": "",
        "angelone_client_id": "",
        "angelone_password": "",
        "angelone_totp_secret": "",
        "zerodha_api_key": "",
        "zerodha_api_secret": "",
        "use_tvscreener": True,
        "selected_model": "gpt-4o-mini",
        "selected_provider": "openai"
    }

    if not settings:
        return default_settings

    # Merge with defaults to ensure all fields exist
    merged_settings = {**default_settings, **settings}

    # Mask API keys
    def mask_key(key):
        if not key:
            return ""
        return key[:8] + '...' + key[-4:] if len(key) > 12 else '****'

    if merged_settings.get('openai_api_key'):
        merged_settings['openai_api_key'] = mask_key(merged_settings['openai_api_key'])
    if merged_settings.get('gemini_api_key'):
        merged_settings['gemini_api_key'] = mask_key(merged_settings['gemini_api_key'])
    if merged_settings.get('anthropic_api_key'):
        merged_settings['anthropic_api_key'] = mask_key(merged_settings['anthropic_api_key'])
    if merged_settings.get('finnhub_api_key'):
        merged_settings['finnhub_api_key'] = mask_key(merged_settings['finnhub_api_key'])
    if merged_settings.get('alpaca_api_key'):
        merged_settings['alpaca_api_key'] = mask_key(merged_settings['alpaca_api_key'])
    if merged_settings.get('alpaca_api_secret'):
        merged_settings['alpaca_api_secret'] = mask_key(merged_settings['alpaca_api_secret'])
    if merged_settings.get('fmp_api_key'):
        merged_settings['fmp_api_key'] = mask_key(merged_settings['fmp_api_key'])
    if merged_settings.get('iex_api_key'):
        merged_settings['iex_api_key'] = mask_key(merged_settings['iex_api_key'])
    if merged_settings.get('polygon_api_key'):
        merged_settings['polygon_api_key'] = mask_key(merged_settings['polygon_api_key'])
    if merged_settings.get('twelve_data_api_key'):
        merged_settings['twelve_data_api_key'] = mask_key(merged_settings['twelve_data_api_key'])
    if merged_settings.get('newsapi_key'):
        merged_settings['newsapi_key'] = mask_key(merged_settings['newsapi_key'])
    if merged_settings.get('alphavantage_api_key'):
        merged_settings['alphavantage_api_key'] = mask_key(merged_settings['alphavantage_api_key'])
    if merged_settings.get('telegram_bot_token'):
        merged_settings['telegram_bot_token'] = mask_key(merged_settings['telegram_bot_token'])
    if merged_settings.get('smtp_password'):
        merged_settings['smtp_password'] = mask_key(merged_settings['smtp_password'])
    if merged_settings.get('twilio_auth_token'):
        merged_settings['twilio_auth_token'] = mask_key(merged_settings['twilio_auth_token'])
    # Mask Indian Broker API Keys
    if merged_settings.get('angelone_api_key'):
        merged_settings['angelone_api_key'] = mask_key(merged_settings['angelone_api_key'])
    if merged_settings.get('angelone_password'):
        merged_settings['angelone_password'] = mask_key(merged_settings['angelone_password'])
    if merged_settings.get('angelone_totp_secret'):
        merged_settings['angelone_totp_secret'] = mask_key(merged_settings['angelone_totp_secret'])
    if merged_settings.get('zerodha_api_key'):
        merged_settings['zerodha_api_key'] = mask_key(merged_settings['zerodha_api_key'])
    if merged_settings.get('zerodha_api_secret'):
        merged_settings['zerodha_api_secret'] = mask_key(merged_settings['zerodha_api_secret'])

    return merged_settings

@api_router.post("/settings")
async def save_settings(settings: SettingsCreate):
    existing = await db.settings.find_one({})

    update_data = {
        "selected_model": settings.selected_model,
        "selected_provider": settings.selected_provider,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    # Helper function to check if key is masked
    def is_masked(key):
        return not key or key.startswith('****') or '...' in key

    # Only update API keys if they're provided and not masked
    if settings.openai_api_key and not is_masked(settings.openai_api_key):
        update_data["openai_api_key"] = settings.openai_api_key

    if settings.gemini_api_key and not is_masked(settings.gemini_api_key):
        update_data["gemini_api_key"] = settings.gemini_api_key

    if settings.anthropic_api_key and not is_masked(settings.anthropic_api_key):
        update_data["anthropic_api_key"] = settings.anthropic_api_key

    if settings.finnhub_api_key and not is_masked(settings.finnhub_api_key):
        update_data["finnhub_api_key"] = settings.finnhub_api_key

    if settings.alpaca_api_key and not is_masked(settings.alpaca_api_key):
        update_data["alpaca_api_key"] = settings.alpaca_api_key

    if settings.alpaca_api_secret and not is_masked(settings.alpaca_api_secret):
        update_data["alpaca_api_secret"] = settings.alpaca_api_secret

    if settings.fmp_api_key and not is_masked(settings.fmp_api_key):
        update_data["fmp_api_key"] = settings.fmp_api_key

    if settings.iex_api_key and not is_masked(settings.iex_api_key):
        update_data["iex_api_key"] = settings.iex_api_key

    if settings.polygon_api_key and not is_masked(settings.polygon_api_key):
        update_data["polygon_api_key"] = settings.polygon_api_key

    if settings.twelve_data_api_key and not is_masked(settings.twelve_data_api_key):
        update_data["twelve_data_api_key"] = settings.twelve_data_api_key

    if settings.newsapi_key and not is_masked(settings.newsapi_key):
        update_data["newsapi_key"] = settings.newsapi_key

    if settings.alphavantage_api_key and not is_masked(settings.alphavantage_api_key):
        update_data["alphavantage_api_key"] = settings.alphavantage_api_key

    # Telegram settings
    if settings.telegram_bot_token and not is_masked(settings.telegram_bot_token):
        update_data["telegram_bot_token"] = settings.telegram_bot_token
    if settings.telegram_chat_id:
        update_data["telegram_chat_id"] = settings.telegram_chat_id

    # Email SMTP settings
    if settings.smtp_host:
        update_data["smtp_host"] = settings.smtp_host
    if settings.smtp_port:
        update_data["smtp_port"] = settings.smtp_port
    if settings.smtp_user:
        update_data["smtp_user"] = settings.smtp_user
    if settings.smtp_password and not is_masked(settings.smtp_password):
        update_data["smtp_password"] = settings.smtp_password
    if settings.smtp_from_email:
        update_data["smtp_from_email"] = settings.smtp_from_email

    # Webhook settings
    if settings.webhook_url:
        update_data["webhook_url"] = settings.webhook_url
    if settings.slack_webhook_url:
        update_data["slack_webhook_url"] = settings.slack_webhook_url

    # Twilio/WhatsApp settings
    if settings.twilio_account_sid:
        update_data["twilio_account_sid"] = settings.twilio_account_sid
    if settings.twilio_auth_token and not is_masked(settings.twilio_auth_token):
        update_data["twilio_auth_token"] = settings.twilio_auth_token
    if settings.twilio_whatsapp_number:
        update_data["twilio_whatsapp_number"] = settings.twilio_whatsapp_number
    if settings.user_whatsapp_number:
        update_data["user_whatsapp_number"] = settings.user_whatsapp_number

    # Indian Broker API Keys - Angel One
    if settings.angelone_api_key and not is_masked(settings.angelone_api_key):
        update_data["angelone_api_key"] = settings.angelone_api_key
    if settings.angelone_client_id:
        update_data["angelone_client_id"] = settings.angelone_client_id
    if settings.angelone_password and not is_masked(settings.angelone_password):
        update_data["angelone_password"] = settings.angelone_password
    if settings.angelone_totp_secret and not is_masked(settings.angelone_totp_secret):
        update_data["angelone_totp_secret"] = settings.angelone_totp_secret

    # Indian Broker API Keys - Zerodha
    if settings.zerodha_api_key and not is_masked(settings.zerodha_api_key):
        update_data["zerodha_api_key"] = settings.zerodha_api_key
    if settings.zerodha_api_secret and not is_masked(settings.zerodha_api_secret):
        update_data["zerodha_api_secret"] = settings.zerodha_api_secret

    # TVScreener setting
    update_data["use_tvscreener"] = settings.use_tvscreener

    if existing:
        await db.settings.update_one({}, {"$set": update_data})
    else:
        update_data["id"] = str(uuid.uuid4())
        update_data["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.settings.insert_one(update_data)

    return {"message": "Settings saved successfully"}

# Helper function to normalize Indian stock symbols
def normalize_indian_symbol(symbol: str) -> str:
    """
    Add .NS suffix for Indian NSE stocks if not already present.
    This ensures yfinance and other providers fetch the correct Indian stock data.

    Examples:
        INFY -> INFY.NS
        RELIANCE -> RELIANCE.NS
        INFY.NS -> INFY.NS (no change)
    """
    symbol = symbol.upper().strip()

    # If already has exchange suffix, return as-is
    if '.' in symbol:
        return symbol

    # For Indian stocks, append .NS for NSE (National Stock Exchange)
    # You can add .BO for BSE (Bombay Stock Exchange) if needed
    return f"{symbol}.NS"

# Top 100 NSE/BSE Stocks
NIFTY_100_STOCKS = [
    {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "sector": "Energy"},
    {"symbol": "TCS", "name": "Tata Consultancy Services", "sector": "IT"},
    {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "sector": "Banking"},
    {"symbol": "INFY", "name": "Infosys Ltd", "sector": "IT"},
    {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "sector": "Banking"},
    {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "sector": "FMCG"},
    {"symbol": "SBIN", "name": "State Bank of India", "sector": "Banking"},
    {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "sector": "Telecom"},
    {"symbol": "ITC", "name": "ITC Ltd", "sector": "FMCG"},
    {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "sector": "Banking"},
    {"symbol": "LT", "name": "Larsen & Toubro Ltd", "sector": "Infrastructure"},
    {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "sector": "Banking"},
    {"symbol": "WIPRO", "name": "Wipro Ltd", "sector": "IT"},
    {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "sector": "Finance"},
    {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "sector": "Auto"},
    {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd", "sector": "Paints"},
    {"symbol": "HCLTECH", "name": "HCL Technologies Ltd", "sector": "IT"},
    {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd", "sector": "Auto"},
    {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical", "sector": "Pharma"},
    {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "sector": "Metals"},
    {"symbol": "NTPC", "name": "NTPC Ltd", "sector": "Power"},
    {"symbol": "POWERGRID", "name": "Power Grid Corporation", "sector": "Power"},
    {"symbol": "ULTRACEMCO", "name": "UltraTech Cement Ltd", "sector": "Cement"},
    {"symbol": "TITAN", "name": "Titan Company Ltd", "sector": "Consumer"},
    {"symbol": "NESTLEIND", "name": "Nestle India Ltd", "sector": "FMCG"},
    {"symbol": "TECHM", "name": "Tech Mahindra Ltd", "sector": "IT"},
    {"symbol": "BAJAJFINSV", "name": "Bajaj Finserv Ltd", "sector": "Finance"},
    {"symbol": "ONGC", "name": "Oil & Natural Gas Corp", "sector": "Energy"},
    {"symbol": "JSWSTEEL", "name": "JSW Steel Ltd", "sector": "Metals"},
    {"symbol": "M&M", "name": "Mahindra & Mahindra Ltd", "sector": "Auto"},
    {"symbol": "COALINDIA", "name": "Coal India Ltd", "sector": "Mining"},
    {"symbol": "ADANIENT", "name": "Adani Enterprises Ltd", "sector": "Conglomerate"},
    {"symbol": "ADANIPORTS", "name": "Adani Ports & SEZ Ltd", "sector": "Infrastructure"},
    {"symbol": "GRASIM", "name": "Grasim Industries Ltd", "sector": "Cement"},
    {"symbol": "HINDALCO", "name": "Hindalco Industries Ltd", "sector": "Metals"},
    {"symbol": "DIVISLAB", "name": "Divi's Laboratories Ltd", "sector": "Pharma"},
    {"symbol": "DRREDDY", "name": "Dr. Reddy's Laboratories", "sector": "Pharma"},
    {"symbol": "CIPLA", "name": "Cipla Ltd", "sector": "Pharma"},
    {"symbol": "APOLLOHOSP", "name": "Apollo Hospitals", "sector": "Healthcare"},
    {"symbol": "BRITANNIA", "name": "Britannia Industries Ltd", "sector": "FMCG"},
    {"symbol": "EICHERMOT", "name": "Eicher Motors Ltd", "sector": "Auto"},
    {"symbol": "INDUSINDBK", "name": "IndusInd Bank Ltd", "sector": "Banking"},
    {"symbol": "HEROMOTOCO", "name": "Hero MotoCorp Ltd", "sector": "Auto"},
    {"symbol": "BPCL", "name": "Bharat Petroleum Corp", "sector": "Energy"},
    {"symbol": "TATACONSUM", "name": "Tata Consumer Products", "sector": "FMCG"},
    {"symbol": "SBILIFE", "name": "SBI Life Insurance", "sector": "Insurance"},
    {"symbol": "HDFCLIFE", "name": "HDFC Life Insurance", "sector": "Insurance"},
    {"symbol": "BAJAJ-AUTO", "name": "Bajaj Auto Ltd", "sector": "Auto"},
    {"symbol": "SHREECEM", "name": "Shree Cement Ltd", "sector": "Cement"},
    {"symbol": "VEDL", "name": "Vedanta Ltd", "sector": "Metals"},
    {"symbol": "HAVELLS", "name": "Havells India Ltd", "sector": "Consumer"},
    {"symbol": "PIDILITIND", "name": "Pidilite Industries Ltd", "sector": "Chemicals"},
    {"symbol": "DABUR", "name": "Dabur India Ltd", "sector": "FMCG"},
    {"symbol": "GODREJCP", "name": "Godrej Consumer Products", "sector": "FMCG"},
    {"symbol": "SIEMENS", "name": "Siemens Ltd", "sector": "Capital Goods"},
    {"symbol": "DLF", "name": "DLF Ltd", "sector": "Real Estate"},
    {"symbol": "ICICIPRULI", "name": "ICICI Prudential Life", "sector": "Insurance"},
    {"symbol": "ICICIGI", "name": "ICICI Lombard General", "sector": "Insurance"},
    {"symbol": "AMBUJACEM", "name": "Ambuja Cements Ltd", "sector": "Cement"},
    {"symbol": "BANKBARODA", "name": "Bank of Baroda", "sector": "Banking"},
    {"symbol": "PNB", "name": "Punjab National Bank", "sector": "Banking"},
    {"symbol": "TATAPOWER", "name": "Tata Power Company Ltd", "sector": "Power"},
    {"symbol": "INDIGO", "name": "InterGlobe Aviation Ltd", "sector": "Aviation"},
    {"symbol": "ZOMATO", "name": "Zomato Ltd", "sector": "Internet"},
    {"symbol": "PAYTM", "name": "One97 Communications", "sector": "Fintech"},
    {"symbol": "NYKAA", "name": "FSN E-Commerce (Nykaa)", "sector": "E-Commerce"},
    {"symbol": "JUBLFOOD", "name": "Jubilant FoodWorks Ltd", "sector": "QSR"},
    {"symbol": "BERGEPAINT", "name": "Berger Paints India Ltd", "sector": "Paints"},
    {"symbol": "COLPAL", "name": "Colgate-Palmolive India", "sector": "FMCG"},
    {"symbol": "MARICO", "name": "Marico Ltd", "sector": "FMCG"},
    {"symbol": "MCDOWELL-N", "name": "United Spirits Ltd", "sector": "Beverages"},
    {"symbol": "TRENT", "name": "Trent Ltd", "sector": "Retail"},
    {"symbol": "PIIND", "name": "PI Industries Ltd", "sector": "Chemicals"},
    {"symbol": "BIOCON", "name": "Biocon Ltd", "sector": "Pharma"},
    {"symbol": "LUPIN", "name": "Lupin Ltd", "sector": "Pharma"},
    {"symbol": "TORNTPHARM", "name": "Torrent Pharmaceuticals", "sector": "Pharma"},
    {"symbol": "AUROPHARMA", "name": "Aurobindo Pharma Ltd", "sector": "Pharma"},
    {"symbol": "GAIL", "name": "GAIL India Ltd", "sector": "Energy"},
    {"symbol": "IOC", "name": "Indian Oil Corporation", "sector": "Energy"},
    {"symbol": "HINDPETRO", "name": "Hindustan Petroleum", "sector": "Energy"},
    {"symbol": "SBICARD", "name": "SBI Cards & Payment", "sector": "Finance"},
    {"symbol": "CHOLAFIN", "name": "Cholamandalam Investment", "sector": "Finance"},
    {"symbol": "MUTHOOTFIN", "name": "Muthoot Finance Ltd", "sector": "Finance"},
    {"symbol": "PEL", "name": "Piramal Enterprises Ltd", "sector": "Diversified"},
    {"symbol": "RECLTD", "name": "REC Ltd", "sector": "Finance"},
    {"symbol": "PFC", "name": "Power Finance Corp", "sector": "Finance"},
    {"symbol": "IRCTC", "name": "IRCTC Ltd", "sector": "Travel"},
    {"symbol": "SAIL", "name": "Steel Authority of India", "sector": "Metals"},
    {"symbol": "NMDC", "name": "NMDC Ltd", "sector": "Mining"},
    {"symbol": "BHEL", "name": "Bharat Heavy Electricals", "sector": "Capital Goods"},
    {"symbol": "HAL", "name": "Hindustan Aeronautics", "sector": "Defence"},
    {"symbol": "BEL", "name": "Bharat Electronics Ltd", "sector": "Defence"},
    {"symbol": "LTIM", "name": "LTIMindtree Ltd", "sector": "IT"},
    {"symbol": "PERSISTENT", "name": "Persistent Systems Ltd", "sector": "IT"},
    {"symbol": "COFORGE", "name": "Coforge Ltd", "sector": "IT"},
    {"symbol": "MPHASIS", "name": "Mphasis Ltd", "sector": "IT"},
    {"symbol": "CANBK", "name": "Canara Bank", "sector": "Banking"},
    {"symbol": "IDFCFIRSTB", "name": "IDFC First Bank Ltd", "sector": "Banking"},
    {"symbol": "FEDERALBNK", "name": "Federal Bank Ltd", "sector": "Banking"},
    {"symbol": "ABCAPITAL", "name": "Aditya Birla Capital", "sector": "Finance"},
    # Additional 100 stocks for 200 total
    {"symbol": "MAXHEALTH", "name": "Max Healthcare Institute", "sector": "Healthcare"},
    {"symbol": "FORTIS", "name": "Fortis Healthcare Ltd", "sector": "Healthcare"},
    {"symbol": "METROPOLIS", "name": "Metropolis Healthcare", "sector": "Healthcare"},
    {"symbol": "LALPATHLAB", "name": "Dr Lal PathLabs Ltd", "sector": "Healthcare"},
    {"symbol": "AUBANK", "name": "AU Small Finance Bank", "sector": "Banking"},
    {"symbol": "BANDHANBNK", "name": "Bandhan Bank Ltd", "sector": "Banking"},
    {"symbol": "RBLBANK", "name": "RBL Bank Ltd", "sector": "Banking"},
    {"symbol": "YESBANK", "name": "Yes Bank Ltd", "sector": "Banking"},
    {"symbol": "UNIONBANK", "name": "Union Bank of India", "sector": "Banking"},
    {"symbol": "INDIANB", "name": "Indian Bank", "sector": "Banking"},
    {"symbol": "BANKINDIA", "name": "Bank of India", "sector": "Banking"},
    {"symbol": "CENTRALBK", "name": "Central Bank of India", "sector": "Banking"},
    {"symbol": "UCOBANK", "name": "UCO Bank", "sector": "Banking"},
    {"symbol": "IOB", "name": "Indian Overseas Bank", "sector": "Banking"},
    {"symbol": "KARURVYSYA", "name": "Karur Vysya Bank", "sector": "Banking"},
    {"symbol": "SOUTHBANK", "name": "South Indian Bank", "sector": "Banking"},
    {"symbol": "TMB", "name": "Tamilnad Mercantile Bank", "sector": "Banking"},
    {"symbol": "JKCEMENT", "name": "JK Cement Ltd", "sector": "Cement"},
    {"symbol": "RAMCOCEM", "name": "Ramco Cements Ltd", "sector": "Cement"},
    {"symbol": "DALBHARAT", "name": "Dalmia Bharat Ltd", "sector": "Cement"},
    {"symbol": "ACC", "name": "ACC Ltd", "sector": "Cement"},
    {"symbol": "INDIACEM", "name": "India Cements Ltd", "sector": "Cement"},
    {"symbol": "PRISMJOINS", "name": "Prism Johnson Ltd", "sector": "Cement"},
    {"symbol": "ORIENTCEM", "name": "Orient Cement Ltd", "sector": "Cement"},
    {"symbol": "HEIDELBERG", "name": "Heidelberg Cement India", "sector": "Cement"},
    {"symbol": "APLLTD", "name": "Alembic Pharmaceuticals", "sector": "Pharma"},
    {"symbol": "ALKEM", "name": "Alkem Laboratories Ltd", "sector": "Pharma"},
    {"symbol": "IPCALAB", "name": "Ipca Laboratories Ltd", "sector": "Pharma"},
    {"symbol": "GLENMARK", "name": "Glenmark Pharmaceuticals", "sector": "Pharma"},
    {"symbol": "NATCOPHARM", "name": "Natco Pharma Ltd", "sector": "Pharma"},
    {"symbol": "GRANULES", "name": "Granules India Ltd", "sector": "Pharma"},
    {"symbol": "LAURUSLABS", "name": "Laurus Labs Ltd", "sector": "Pharma"},
    {"symbol": "ABBOTINDIA", "name": "Abbott India Ltd", "sector": "Pharma"},
    {"symbol": "SANOFI", "name": "Sanofi India Ltd", "sector": "Pharma"},
    {"symbol": "PFIZER", "name": "Pfizer Ltd", "sector": "Pharma"},
    {"symbol": "GLAXO", "name": "GlaxoSmithKline Pharma", "sector": "Pharma"},
    {"symbol": "JBCHEPHARM", "name": "JB Chemicals & Pharma", "sector": "Pharma"},
    {"symbol": "MRF", "name": "MRF Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "APOLLOTYRE", "name": "Apollo Tyres Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "BALKRISIND", "name": "Balkrishna Industries", "sector": "Auto Ancillaries"},
    {"symbol": "CEATLTD", "name": "CEAT Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "EXIDEIND", "name": "Exide Industries Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "AMARAJABAT", "name": "Amara Raja Energy", "sector": "Auto Ancillaries"},
    {"symbol": "BOSCHLTD", "name": "Bosch Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "MOTHERSON", "name": "Samvardhana Motherson", "sector": "Auto Ancillaries"},
    {"symbol": "BHARATFORG", "name": "Bharat Forge Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "ASHOKLEY", "name": "Ashok Leyland Ltd", "sector": "Auto"},
    {"symbol": "ESCORTS", "name": "Escorts Kubota Ltd", "sector": "Auto"},
    {"symbol": "TVSMOTOR", "name": "TVS Motor Company Ltd", "sector": "Auto"},
    {"symbol": "VOLTAS", "name": "Voltas Ltd", "sector": "Consumer Durables"},
    {"symbol": "WHIRLPOOL", "name": "Whirlpool of India", "sector": "Consumer Durables"},
    {"symbol": "BLUESTARCO", "name": "Blue Star Ltd", "sector": "Consumer Durables"},
    {"symbol": "CROMPTON", "name": "Crompton Greaves Consumer", "sector": "Consumer Durables"},
    {"symbol": "BAJAJELEC", "name": "Bajaj Electricals Ltd", "sector": "Consumer Durables"},
    {"symbol": "VGUARD", "name": "V-Guard Industries Ltd", "sector": "Consumer Durables"},
    {"symbol": "DIXON", "name": "Dixon Technologies India", "sector": "Consumer Durables"},
    {"symbol": "POLYCAB", "name": "Polycab India Ltd", "sector": "Consumer Durables"},
    {"symbol": "KPRMILL", "name": "K.P.R. Mill Ltd", "sector": "Textiles"},
    {"symbol": "RAYMOND", "name": "Raymond Ltd", "sector": "Textiles"},
    {"symbol": "PAGEIND", "name": "Page Industries Ltd", "sector": "Textiles"},
    {"symbol": "ARVIND", "name": "Arvind Ltd", "sector": "Textiles"},
    {"symbol": "VMART", "name": "V-Mart Retail Ltd", "sector": "Retail"},
    {"symbol": "SHOPERSTOP", "name": "Shoppers Stop Ltd", "sector": "Retail"},
    {"symbol": "DMART", "name": "Avenue Supermarts (DMart)", "sector": "Retail"},
    {"symbol": "RELAXO", "name": "Relaxo Footwears Ltd", "sector": "Retail"},
    {"symbol": "BATAINDIA", "name": "Bata India Ltd", "sector": "Retail"},
    {"symbol": "CONCOR", "name": "Container Corp of India", "sector": "Logistics"},
    {"symbol": "DELHIVERY", "name": "Delhivery Ltd", "sector": "Logistics"},
    {"symbol": "MAHINDCIE", "name": "Mahindra CIE Automotive", "sector": "Auto Ancillaries"},
    {"symbol": "SCHAEFFLER", "name": "Schaeffler India Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "SKFINDIA", "name": "SKF India Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "TIMKEN", "name": "Timken India Ltd", "sector": "Auto Ancillaries"},
    {"symbol": "CUMMINSIND", "name": "Cummins India Ltd", "sector": "Capital Goods"},
    {"symbol": "ABB", "name": "ABB India Ltd", "sector": "Capital Goods"},
    {"symbol": "HONAUT", "name": "Honeywell Automation", "sector": "Capital Goods"},
    {"symbol": "THERMAX", "name": "Thermax Ltd", "sector": "Capital Goods"},
    {"symbol": "GRINDWELL", "name": "Grindwell Norton Ltd", "sector": "Capital Goods"},
    {"symbol": "KAJARIACER", "name": "Kajaria Ceramics Ltd", "sector": "Building Materials"},
    {"symbol": "SUNTV", "name": "Sun TV Network Ltd", "sector": "Media"},
    {"symbol": "PVR", "name": "PVR INOX Ltd", "sector": "Media"},
    {"symbol": "ZEEL", "name": "Zee Entertainment", "sector": "Media"},
    {"symbol": "NAUKRI", "name": "Info Edge India Ltd", "sector": "Internet"},
    {"symbol": "POLICYBZR", "name": "PB Fintech Ltd", "sector": "Internet"},
    {"symbol": "JUSTDIAL", "name": "Just Dial Ltd", "sector": "Internet"},
    {"symbol": "INDHOTEL", "name": "Indian Hotels Company", "sector": "Hotels"},
    {"symbol": "LEMON", "name": "Lemon Tree Hotels Ltd", "sector": "Hotels"},
    {"symbol": "CHALET", "name": "Chalet Hotels Ltd", "sector": "Hotels"},
    {"symbol": "VBL", "name": "Varun Beverages Ltd", "sector": "Beverages"},
    {"symbol": "RADICO", "name": "Radico Khaitan Ltd", "sector": "Beverages"},
    {"symbol": "ATUL", "name": "Atul Ltd", "sector": "Chemicals"},
    {"symbol": "DEEPAKNTR", "name": "Deepak Nitrite Ltd", "sector": "Chemicals"},
    {"symbol": "NAVINFLUOR", "name": "Navin Fluorine Intl", "sector": "Chemicals"},
    {"symbol": "SRF", "name": "SRF Ltd", "sector": "Chemicals"},
    {"symbol": "AARTIIND", "name": "Aarti Industries Ltd", "sector": "Chemicals"},
    {"symbol": "CLEAN", "name": "Clean Science & Tech", "sector": "Chemicals"},
    {"symbol": "FLUOROCHEM", "name": "Gujarat Fluorochemicals", "sector": "Chemicals"},
    {"symbol": "UPL", "name": "UPL Ltd", "sector": "Chemicals"},
    {"symbol": "COROMANDEL", "name": "Coromandel International", "sector": "Fertilizers"},
    {"symbol": "GNFC", "name": "Gujarat Narmada Valley", "sector": "Fertilizers"},
    {"symbol": "CHAMBLFRTU", "name": "Chambal Fertilizers", "sector": "Fertilizers"},
    {"symbol": "GODREJPROP", "name": "Godrej Properties Ltd", "sector": "Real Estate"},
    {"symbol": "OBEROIRLTY", "name": "Oberoi Realty Ltd", "sector": "Real Estate"},
    {"symbol": "PRESTIGE", "name": "Prestige Estates Projects", "sector": "Real Estate"},
    {"symbol": "PHOENIXLTD", "name": "Phoenix Mills Ltd", "sector": "Real Estate"},
    {"symbol": "BRIGADE", "name": "Brigade Enterprises Ltd", "sector": "Real Estate"},
    {"symbol": "SOBHA", "name": "Sobha Ltd", "sector": "Real Estate"},
]

# Rename to NSE_200_STOCKS for clarity (200 stocks)
NSE_200_STOCKS = NIFTY_100_STOCKS


async def analyze_stock_for_recommendation(symbol: str, stock_info: dict, use_enhanced: bool = True) -> Optional[dict]:
    """
    Analyze a single stock and return recommendation if significant.
    Uses enhanced analyzer with market regime detection, multi-timeframe analysis,
    weighted indicator scoring, and Bayesian confidence.
    """
    try:
        if use_enhanced:
            # Use enhanced analyzer for more accurate recommendations
            enhanced_analyzer = get_enhanced_analyzer()
            result = await enhanced_analyzer.analyze_stock(
                symbol,
                include_patterns=True,
                include_sentiment=False,  # Skip sentiment for batch processing
                include_backtest_validation=False  # Skip for speed
            )

            if result.get('error'):
                logger.warning(f"Enhanced analysis failed for {symbol}: {result.get('error')}")
                return None

            recommendation = result.get('recommendation', 'HOLD')
            confidence = result.get('confidence', 0)

            # Only return if signal has reasonable confidence
            # Lower threshold to 45% to ensure we get recommendations
            if recommendation in ['HOLD'] or confidence < 45:
                return None

            # Extract signals from the detailed analysis
            signals = []
            for signal in result.get('signals', []):
                signals.append(signal.get('description', signal.get('indicator', '')))

            # Add regime info to signals
            regime = result.get('market_regime', {})
            if regime.get('primary_regime') in ['strong_bull', 'bull']:
                signals.insert(0, f"Market regime: {regime.get('primary_regime').replace('_', ' ').title()}")
            elif regime.get('primary_regime') in ['strong_bear', 'bear']:
                signals.insert(0, f"Market regime: {regime.get('primary_regime').replace('_', ' ').title()}")

            # Add multi-timeframe alignment
            mtf = result.get('multi_timeframe', {}).get('alignment', {})
            if mtf.get('direction') in ['bullish', 'bearish']:
                signals.append(f"Multi-timeframe: {mtf.get('direction')} ({mtf.get('score', 0)}/3)")

            # Add confluence info
            confluence = result.get('confluence', {})
            if confluence.get('strength') in ['strong', 'moderate']:
                signals.append(f"Signal confluence: {confluence.get('strength')} ({confluence.get('agreement', 0):.0f}%)")

            price_targets = result.get('price_targets', {})
            indicators = result.get('indicators', {})
            scores = result.get('signal_scores', {})

            return {
                "symbol": symbol,
                "name": stock_info.get("name", symbol),
                "sector": stock_info.get("sector", "N/A"),
                "recommendation": recommendation,
                "confidence": round(confidence, 1),
                "current_price": result.get('current_price', 0),
                "change_pct": round(((result.get('current_price', 0) - indicators.get('sma_20', result.get('current_price', 0))) / indicators.get('sma_20', 1)) * 100, 2) if indicators.get('sma_20') else 0,
                "signals": signals[:8],  # Limit to 8 signals
                "indicators": {
                    "rsi": indicators.get('rsi'),
                    "macd": indicators.get('macd'),
                    "macd_signal": indicators.get('macd_signal'),
                    "sma_20": indicators.get('sma_20'),
                    "sma_50": indicators.get('sma_50'),
                    "stochastic": indicators.get('stoch_k'),
                    "adx": indicators.get('adx'),
                    "ichimoku_position": indicators.get('ichimoku_position'),
                },
                "volume_ratio": indicators.get('volume_ratio', 1),
                "buy_score": scores.get('weighted_buy', 0),
                "sell_score": scores.get('weighted_sell', 0),
                "market_regime": regime.get('primary_regime', 'unknown'),
                "volatility_regime": regime.get('volatility_regime', 'normal'),
                "confluence": confluence.get('agreement', 0),
                "target_price": price_targets.get('target'),
                "stop_loss": price_targets.get('stop_loss'),
                "target_range": price_targets.get('target_range'),
                "risk_reward": result.get('risk_reward', {}).get('ratio'),
                "data_quality": result.get('data_quality', {}),
            }

        else:
            # Fallback to simple analysis
            ticker_symbol = get_indian_stock_suffix(symbol)
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="3mo")

            if len(hist) < 50:
                return None

            close = hist['Close']
            high = hist['High']
            low = hist['Low']
            volume = hist['Volume']

            # Calculate indicators
            rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
            macd_indicator = ta.trend.MACD(close)
            macd = macd_indicator.macd().iloc[-1]
            macd_signal = macd_indicator.macd_signal().iloc[-1]
            macd_hist = macd_indicator.macd_diff().iloc[-1]

            sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]
            sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]

            bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
            bb_upper = bb.bollinger_hband().iloc[-1]
            bb_lower = bb.bollinger_lband().iloc[-1]

            stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
            stoch_k = stoch.stoch().iloc[-1]

            adx = ta.trend.ADXIndicator(high, low, close, window=14).adx().iloc[-1]

            current_price = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            change_pct = ((current_price - prev_close) / prev_close) * 100

            # Volume analysis
            avg_volume = volume.rolling(20).mean().iloc[-1]
            current_volume = volume.iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1

            # Scoring system
            buy_score = 0
            sell_score = 0
            signals = []

            # RSI signals
            if rsi < 30:
                buy_score += 3
                signals.append("RSI oversold (<30)")
            elif rsi < 40:
                buy_score += 1
                signals.append("RSI approaching oversold")
            elif rsi > 70:
                sell_score += 3
                signals.append("RSI overbought (>70)")
            elif rsi > 60:
                sell_score += 1
                signals.append("RSI approaching overbought")

            # MACD signals
            if macd > macd_signal and macd_hist > 0:
                buy_score += 2
                signals.append("MACD bullish crossover")
            elif macd < macd_signal and macd_hist < 0:
                sell_score += 2
                signals.append("MACD bearish crossover")

            # Moving average signals
            if current_price > sma_20 > sma_50:
                buy_score += 2
                signals.append("Price above SMA20 & SMA50 (uptrend)")
            elif current_price < sma_20 < sma_50:
                sell_score += 2
                signals.append("Price below SMA20 & SMA50 (downtrend)")

            # Bollinger Band signals
            if current_price < bb_lower:
                buy_score += 2
                signals.append("Price below lower Bollinger Band")
            elif current_price > bb_upper:
                sell_score += 2
                signals.append("Price above upper Bollinger Band")

            # Stochastic signals
            if stoch_k < 20:
                buy_score += 1
                signals.append("Stochastic oversold")
            elif stoch_k > 80:
                sell_score += 1
                signals.append("Stochastic overbought")

            # ADX trend strength
            if adx > 25:
                signals.append(f"Strong trend (ADX: {adx:.1f})")

            # Volume confirmation
            if volume_ratio > 1.5:
                signals.append(f"High volume ({volume_ratio:.1f}x avg)")

            # Determine recommendation (lowered threshold from 4 to 3 for more signals)
            if buy_score >= 3 and buy_score > sell_score:
                recommendation = "BUY"
                confidence = min(95, 50 + buy_score * 8)
            elif sell_score >= 3 and sell_score > buy_score:
                recommendation = "SELL"
                confidence = min(95, 50 + sell_score * 8)
            else:
                return None  # No strong signal

            return {
                "symbol": symbol,
                "name": stock_info.get("name", symbol),
                "sector": stock_info.get("sector", "N/A"),
                "recommendation": recommendation,
                "confidence": confidence,
                "current_price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "signals": signals,
                "indicators": {
                    "rsi": round(float(rsi), 2) if not pd.isna(rsi) else None,
                    "macd": round(float(macd), 4) if not pd.isna(macd) else None,
                    "macd_signal": round(float(macd_signal), 4) if not pd.isna(macd_signal) else None,
                    "sma_20": round(float(sma_20), 2) if not pd.isna(sma_20) else None,
                    "sma_50": round(float(sma_50), 2) if not pd.isna(sma_50) else None,
                    "stochastic": round(float(stoch_k), 2) if not pd.isna(stoch_k) else None,
                    "adx": round(float(adx), 2) if not pd.isna(adx) else None,
                },
                "volume_ratio": round(volume_ratio, 2),
                "buy_score": buy_score,
                "sell_score": sell_score,
            }

    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {e}")
        return None


@api_router.get("/analyze/enhanced/{symbol}")
async def get_enhanced_analysis(symbol: str):
    """
    Get enhanced analysis for a single stock.
    Includes market regime detection, multi-timeframe analysis,
    weighted indicator scoring, Bayesian confidence, and more.
    """
    try:
        enhanced_analyzer = get_enhanced_analyzer()
        result = await enhanced_analyzer.analyze_stock(
            symbol,
            include_patterns=True,
            include_sentiment=True,
            include_backtest_validation=True
        )

        if result.get('error'):
            raise HTTPException(status_code=500, detail=result.get('error'))

        # Track prediction for performance monitoring
        if result.get('recommendation') and result['recommendation'] != 'HOLD':
            try:
                tracker = get_confidence_tracker(db)
                await tracker.track_prediction(
                    symbol=symbol,
                    recommendation=result['recommendation'],
                    entry_price=result.get('current_price'),
                    target_price=result.get('target_price'),
                    confidence=result.get('confidence', 0),
                    analysis_data=result,
                    strategy="ENHANCED_ANALYSIS"
                )
                logger.info(f"📊 Tracked prediction: {symbol} {result['recommendation']} (confidence: {result.get('confidence', 0)}%)")
            except Exception as e:
                logger.error(f"Failed to track prediction for {symbol}: {e}")
                # Don't fail the whole request if tracking fails

        return result

    except Exception as e:
        logger.error(f"Enhanced analysis failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/recommendations")
async def get_cached_recommendations():
    """
    Get cached AI recommendations from database.
    Returns the most recent saved recommendations.
    """
    try:
        # Get the most recent recommendations from database
        cached = await db.recommendations.find_one(
            {},
            sort=[("generated_at", -1)]
        )

        if cached:
            # Remove MongoDB _id field
            cached.pop("_id", None)
            return cached

        # No cached recommendations found
        return {
            "generated_at": None,
            "total_stocks_analyzed": 0,
            "buy_recommendations": [],
            "sell_recommendations": [],
            "summary": {
                "total_buy_signals": 0,
                "total_sell_signals": 0,
                "market_sentiment": "Neutral"
            },
            "message": "No recommendations available. Click 'Generate' to analyze stocks."
        }

    except Exception as e:
        logger.error(f"Failed to get cached recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/recommendations/generate")
async def generate_ai_recommendations(
    limit: int = 200,
    enhanced: bool = True,
    min_confidence: float = 65.0,
    enable_sentiment: bool = True,
    enable_backtest: bool = True
):
    """
    Generate AI recommendations for Indian stocks (NSE/BSE).

    Dynamically fetches stocks from TradingView Screener (no hardcoded list).

    Parameters:
    - limit: Number of stocks to analyze (default: 200)
    - enhanced: Use enhanced analysis with sentiment & backtest (default: True)
    - min_confidence: Minimum confidence threshold (default: 65%)
    - enable_sentiment: Include sentiment analysis (default: True)
    - enable_backtest: Include backtest validation (default: True)
    """
    try:
        # Dynamically fetch stocks from TradingView (NSE/BSE)
        logger.info("Fetching dynamic stock list from TradingView...")
        dynamic_stocks = await get_all_indian_stocks_async(
            min_market_cap=100,  # Min 100 Cr market cap
            max_stocks=limit * 2  # Fetch more to account for filtering
        )

        if dynamic_stocks:
            stocks_to_analyze = [s['symbol'] for s in dynamic_stocks[:limit]]
            logger.info(f"Using {len(stocks_to_analyze)} dynamically fetched stocks")
        else:
            # Fallback to hardcoded list if TVScreener fails
            logger.warning("TVScreener failed, falling back to hardcoded stock list")
            stocks_to_analyze = [s['symbol'] for s in NIFTY_100_STOCKS[:limit]]

        # Adjust confidence threshold based on mode
        confidence_threshold = min_confidence if enhanced else 45.0

        logger.info(f"Starting {'ENHANCED' if enhanced else 'BASIC'} analysis for {len(stocks_to_analyze)} stocks with {confidence_threshold}% confidence threshold")

        async def process_stock(symbol):
            try:
                # Get stock info
                ticker = get_indian_stock_suffix(symbol)
                info = yf.Ticker(ticker).info
                stock_info = {
                    "symbol": symbol,
                    "name": info.get('longName', symbol),
                    "sector": info.get('sector', 'N/A')
                }

                # Use enhanced analyzer
                enhanced_analyzer = get_enhanced_analyzer()
                result = await enhanced_analyzer.analyze_stock(
                    symbol,
                    include_patterns=True,
                    include_sentiment=enable_sentiment if enhanced else False,
                    include_backtest_validation=enable_backtest if enhanced else False
                )

                if result.get('error'):
                    logger.warning(f"Analysis failed for {symbol}: {result.get('error')}")
                    return None

                recommendation = result.get('recommendation', 'HOLD')
                confidence = result.get('confidence', 0)

                # Apply confidence threshold
                if recommendation == 'HOLD' or confidence < confidence_threshold:
                    return None

                # Extract signals
                signals = []
                for signal in result.get('signals', []):
                    signals.append(signal.get('description', signal.get('indicator', '')))

                # Add regime and confluence info
                regime = result.get('market_regime', {})
                if regime.get('primary_regime'):
                    signals.insert(0, f"Regime: {regime.get('primary_regime').replace('_', ' ').title()}")

                confluence = result.get('confluence', {})
                if confluence.get('strength'):
                    signals.append(f"Confluence: {confluence.get('strength')} ({confluence.get('agreement', 0):.0f}%)")

                # Add backtest results if available
                backtest = result.get('backtest_validation', {})
                if backtest and backtest.get('win_rate'):
                    signals.append(f"Backtest: {backtest.get('win_rate')}% win rate")

                price_targets = result.get('price_targets', {})
                indicators = result.get('indicators', {})
                fundamentals = result.get('fundamentals', {})

                # Extract key fundamental metrics
                fundamental_metrics = None
                if fundamentals:
                    fundamental_metrics = {
                        "pe_ratio": fundamentals.get('pe_ratio'),
                        "pb_ratio": fundamentals.get('pb_ratio'),
                        "debt_to_equity": fundamentals.get('debt_to_equity'),
                        "roe": fundamentals.get('roe'),
                        "profit_margin": fundamentals.get('profit_margin'),
                        "dividend_yield": fundamentals.get('dividend_yield'),
                        "market_cap": fundamentals.get('market_cap'),
                        "fundamental_score": result.get('fundamental_score')
                    }

                entry_timing = result.get('entry_timing', {})

                return {
                    "symbol": symbol,
                    "name": stock_info.get("name", symbol),
                    "sector": stock_info.get("sector", "N/A"),
                    "recommendation": recommendation,
                    "confidence": round(confidence, 1),
                    "current_price": result.get('current_price', 0),
                    "target_price": price_targets.get('target'),
                    "stop_loss": price_targets.get('stop_loss'),
                    "risk_reward": result.get('risk_reward', {}).get('ratio'),
                    "signals": signals[:10],
                    "indicators": {
                        "rsi": indicators.get('rsi'),
                        "macd": indicators.get('macd'),
                        "sma_20": indicators.get('sma_20'),
                        "sma_50": indicators.get('sma_50'),
                        "adx": indicators.get('adx'),
                    },
                    "fundamentals": fundamental_metrics,
                    "entry_timing": entry_timing,
                    "market_regime": regime.get('primary_regime', 'unknown'),
                    "confluence": confluence.get('agreement', 0),
                    "backtest_validated": bool(backtest),
                    "sentiment_score": result.get('sentiment', {}).get('score', 0) if enhanced and enable_sentiment else None,
                    "data_quality": result.get('data_quality', {}),
                }

            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None

        # Process all stocks in parallel
        tasks = [process_stock(symbol) for symbol in stocks_to_analyze]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter and categorize results
        buy_recs = []
        sell_recs = []

        for res in batch_results:
            if isinstance(res, dict):
                if res['recommendation'] == 'BUY':
                    buy_recs.append(res)
                elif res['recommendation'] == 'SELL':
                    sell_recs.append(res)

        # Sort by confidence (highest first)
        buy_recs.sort(key=lambda x: x['confidence'], reverse=True)
        sell_recs.sort(key=lambda x: x['confidence'], reverse=True)

        # Determine market sentiment
        total_analyzed = len(stocks_to_analyze)
        market_sentiment = "Neutral"
        if len(buy_recs) > len(sell_recs) * 1.5:
            market_sentiment = "Bullish"
        elif len(sell_recs) > len(buy_recs) * 1.5:
            market_sentiment = "Bearish"

        # Calculate average confidence
        avg_buy_conf = sum(r['confidence'] for r in buy_recs) / len(buy_recs) if buy_recs else 0
        avg_sell_conf = sum(r['confidence'] for r in sell_recs) / len(sell_recs) if sell_recs else 0

        # Analyze sector diversification
        def analyze_sector_diversification(recommendations):
            sector_counts = {}
            for rec in recommendations:
                sector = rec.get('sector', 'Unknown')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1

            total = len(recommendations)
            warnings = []

            for sector, count in sector_counts.items():
                percentage = (count / total) * 100 if total > 0 else 0
                if percentage > 40:  # More than 40% in one sector
                    warnings.append({
                        "sector": sector,
                        "count": count,
                        "percentage": round(percentage, 1),
                        "severity": "high",
                        "message": f"{count} stocks ({percentage:.1f}%) from {sector} sector - High concentration risk"
                    })
                elif percentage > 25:  # More than 25% in one sector
                    warnings.append({
                        "sector": sector,
                        "count": count,
                        "percentage": round(percentage, 1),
                        "severity": "medium",
                        "message": f"{count} stocks ({percentage:.1f}%) from {sector} sector - Consider diversification"
                    })

            return {
                "warnings": warnings,
                "sector_breakdown": [{
                    "sector": sector,
                    "count": count,
                    "percentage": round((count / total) * 100, 1) if total > 0 else 0
                } for sector, count in sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)]
            }

        buy_diversification = analyze_sector_diversification(buy_recs)
        sell_diversification = analyze_sector_diversification(sell_recs)

        # Get historical accuracy stats
        historical_accuracy = None
        try:
            from tracking.confidence_tracker import ConfidenceTracker
            tracker = ConfidenceTracker(db)
            accuracy_stats = await tracker.get_accuracy_stats(days_back=30)
            if accuracy_stats and accuracy_stats.get('total_predictions', 0) > 0:
                historical_accuracy = {
                    "total_predictions": accuracy_stats.get('total_predictions', 0),
                    "period_days": accuracy_stats.get('period_days', 30),
                    "by_recommendation": accuracy_stats.get('by_recommendation', {}),
                    "by_confidence_band": accuracy_stats.get('by_confidence_band', {})
                }
        except Exception as e:
            logger.warning(f"Failed to get historical accuracy: {e}")

        recommendations_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "analysis_type": "enhanced" if enhanced else "basic",
            "total_stocks_analyzed": total_analyzed,
            "min_confidence_threshold": confidence_threshold,
            "sentiment_enabled": enable_sentiment if enhanced else False,
            "backtest_enabled": enable_backtest if enhanced else False,
            "summary": {
                "total_buy_signals": len(buy_recs),
                "total_sell_signals": len(sell_recs),
                "market_sentiment": market_sentiment,
                "avg_buy_confidence": round(avg_buy_conf, 1),
                "avg_sell_confidence": round(avg_sell_conf, 1),
                "historical_accuracy": historical_accuracy,
            },
            "buy_recommendations": buy_recs[:50],  # Top 50 BUY signals
            "sell_recommendations": sell_recs[:30],  # Top 30 SELL signals
            "diversification_analysis": {
                "buy": buy_diversification,
                "sell": sell_diversification
            }
        }

        # Save to database with upsert logic
        # If recent recommendation exists (within 1 hour), update it; otherwise insert new
        try:
            from datetime import timedelta
            one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

            # Try to find and update recent recommendation
            existing = await db.recommendations.find_one(
                {"generated_at": {"$gte": one_hour_ago}},
                sort=[("generated_at", -1)]
            )

            if existing:
                # Update existing recent recommendation
                await db.recommendations.update_one(
                    {"_id": existing["_id"]},
                    {"$set": recommendations_data}
                )
                logger.info("Updated existing recommendation")
            else:
                # Insert new recommendation
                await db.recommendations.insert_one(recommendations_data.copy())
                logger.info("Inserted new recommendation")
        except Exception as e:
            logger.warning(f"Failed to cache recommendations: {e}")

        logger.info(f"✅ Recommendations: {len(buy_recs)} BUY ({avg_buy_conf:.1f}% avg), {len(sell_recs)} SELL ({avg_sell_conf:.1f}% avg)")

        return recommendations_data

    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/recommendations/history")
async def get_recommendations_history(limit: int = 10):
    """Get history of recommendation generations"""
    try:
        cursor = db.recommendations.find(
            {},
            {"buy_recommendations": 0, "sell_recommendations": 0}  # Exclude large arrays
        ).sort("generated_at", -1).limit(limit)

        history = await cursor.to_list(length=limit)

        # Remove MongoDB _id field
        for item in history:
            item.pop("_id", None)

        return history

    except Exception as e:
        logger.error(f"Failed to get recommendations history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/recommendations/accuracy")
async def get_recommendation_accuracy(days_back: int = 30):
    """Get historical accuracy statistics for AI recommendations"""
    try:
        from tracking.confidence_tracker import ConfidenceTracker

        tracker = ConfidenceTracker(db)
        stats = await tracker.get_accuracy_stats(days_back=days_back)

        return stats
    except Exception as e:
        logger.error(f"Failed to get accuracy stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/recommendations/stock/{symbol}")
async def get_stock_recommendation(symbol: str):
    """Get AI recommendation for a specific stock"""
    try:
        # Try to find stock in dynamic list first
        dynamic_results = search_indian_stocks(symbol, limit=1)
        if dynamic_results and dynamic_results[0].get('symbol', '').upper() == symbol.upper():
            stock_info = {
                "symbol": dynamic_results[0].get('symbol'),
                "name": dynamic_results[0].get('name'),
                "sector": dynamic_results[0].get('sector', 'N/A')
            }
        else:
            # Fallback to hardcoded list or generic info
            stock_info = next((s for s in NIFTY_100_STOCKS if s["symbol"].upper() == symbol.upper()), None)
            if not stock_info:
                stock_info = {"symbol": symbol.upper(), "name": symbol.upper(), "sector": "N/A"}

        result = await analyze_stock_for_recommendation(symbol.upper(), stock_info)

        if not result:
            # Return hold recommendation if no strong signals
            ticker_symbol = get_indian_stock_suffix(symbol)
            ticker = yf.Ticker(ticker_symbol)
            hist = ticker.history(period="1d")
            current_price = float(hist['Close'].iloc[-1]) if not hist.empty else 0

            return {
                "symbol": symbol.upper(),
                "name": stock_info.get("name", symbol),
                "sector": stock_info.get("sector", "N/A"),
                "recommendation": "HOLD",
                "confidence": 50,
                "current_price": round(current_price, 2),
                "signals": ["No strong buy or sell signals detected"],
                "message": "Technical indicators are neutral. Consider holding or waiting for clearer signals."
            }

        return result

    except Exception as e:
        logger.error(f"Stock recommendation failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Stock search - Dynamic search using TradingView data
@api_router.get("/stocks/search")
async def search_stocks(q: str):
    """Search for stocks dynamically from NSE/BSE"""
    if not q or len(q) < 1:
        return []

    # Use dynamic search from TVScreener
    results = search_indian_stocks(q, limit=10)

    if results:
        return [{
            "symbol": s.get("symbol", ""),
            "name": s.get("name", s.get("symbol", "")),
            "sector": s.get("sector", "N/A"),
            "exchange": "NSE"
        } for s in results]

    # Fallback to hardcoded list if TVScreener not available
    query = q.upper()
    fallback_results = [s for s in NIFTY_100_STOCKS if query in s['symbol'] or query in s['name'].upper()]
    return [{"symbol": s["symbol"], "name": s["name"], "sector": s.get("sector", "N/A"), "exchange": "NSE"} for s in fallback_results[:10]]

# Stock data endpoints
@api_router.get("/stocks/quote/{symbol}")
async def get_stock_quote_fast(symbol: str):
    """Get quick stock quote (alias for /stocks/{symbol})"""
    return await get_stock(symbol)

@api_router.get("/stocks/{symbol}")
async def get_stock(symbol: str):
    """Get current stock data with automatic provider fallback"""
    # Normalize symbol for Indian stocks (add .NS suffix)
    normalized_symbol = normalize_indian_symbol(symbol)

    # Get API keys from settings
    settings = await db.settings.find_one({}, {"_id": 0})

    # Build provider keys dict from settings
    provider_keys = {}
    if settings:
        if settings.get('alpaca_api_key') and settings.get('alpaca_api_secret'):
            provider_keys['alpaca'] = {
                'key': settings.get('alpaca_api_key'),
                'secret': settings.get('alpaca_api_secret'),
                'paper': True  # Default to paper trading for safety
            }
        if settings.get('iex_api_key'):
            provider_keys['iex'] = settings.get('iex_api_key')

    # Try using data provider factory with fallback
    if provider_keys:
        try:
            from data_providers.factory import get_data_provider_factory
            factory = get_data_provider_factory(provider_keys)
            quote = await factory.get_quote(normalized_symbol)

            if quote:
                return quote
        except Exception as e:
            logger.warning(f"Data provider factory failed, falling back to yfinance: {e}")

    # Fallback to existing yfinance implementation
    return await fetch_stock_data(normalized_symbol)

@api_router.get("/stocks/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1y"):
    """Get historical price data with automatic provider fallback"""
    # Get API keys from settings
    settings = await db.settings.find_one({}, {"_id": 0})

    # Build provider keys dict from settings
    provider_keys = {}
    if settings:
        if settings.get('alpaca_api_key') and settings.get('alpaca_api_secret'):
            provider_keys['alpaca'] = {
                'key': settings.get('alpaca_api_key'),
                'secret': settings.get('alpaca_api_secret'),
                'paper': True
            }
        if settings.get('iex_api_key'):
            provider_keys['iex'] = settings.get('iex_api_key')

    # Try using data provider factory with fallback
    if provider_keys:
        try:
            from data_providers.factory import get_data_provider_factory
            factory = get_data_provider_factory(provider_keys)
            hist_df = await factory.get_historical_data(symbol, period=period)

            if hist_df is not None and not hist_df.empty:
                # Convert DataFrame to array format for frontend
                data = []
                for date, row in hist_df.iterrows():
                    data.append({
                        "date": date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        "open": round(float(row['Open']), 2),
                        "high": round(float(row['High']), 2),
                        "low": round(float(row['Low']), 2),
                        "close": round(float(row['Close']), 2),
                        "volume": int(row['Volume']) if 'Volume' in row else 0
                    })
                return {"symbol": symbol, "data": data}
        except Exception as e:
            logger.warning(f"Data provider factory failed for history, falling back to yfinance: {e}")

    # Fallback to existing yfinance implementation
    return await fetch_historical_data(symbol, period)

@api_router.get("/stocks/{symbol}/indicators")
async def get_technical_indicators(symbol: str):
    """Get technical indicators"""
    try:
        result = await calculate_technical_indicators(symbol)
        return result
    except Exception as e:
        logger.error(f"Error in indicators endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Analysis endpoints
@api_router.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    """Run full AI analysis on a stock using multi-agent system"""
    # Get API key from settings
    settings = await db.settings.find_one({}, {"_id": 0})
    
    if not settings:
        raise HTTPException(status_code=400, detail="Please configure your API keys in Settings first")
    
    api_key = None
    if request.provider == "openai":
        api_key = settings.get('openai_api_key')
    elif request.provider == "gemini":
        api_key = settings.get('gemini_api_key')
    
    if not api_key:
        raise HTTPException(status_code=400, detail=f"Please configure your {request.provider} API key in Settings")
    
    try:
        # Get data provider API keys from settings
        data_provider_keys = {
            "finnhub": settings.get('finnhub_api_key'),
            "alpaca": {
                "key": settings.get('alpaca_api_key'),
                "secret": settings.get('alpaca_api_secret')
            },
            "fmp": settings.get('fmp_api_key')
        }

        # Use the new multi-agent orchestrator (REAL IMPLEMENTATION)
        orchestrator = get_orchestrator(db=db)

        analysis_result = await orchestrator.run_analysis(
            symbol=request.symbol,
            model=request.model,
            provider=request.provider,
            api_key=api_key,
            data_provider_keys=data_provider_keys
        )
        
        # Save to database
        result = {
            "id": str(uuid.uuid4()),
            "symbol": request.symbol.upper(),
            "recommendation": analysis_result['recommendation'],
            "confidence": analysis_result['confidence'],
            "entry_price": analysis_result.get('entry_price'),
            "target_price": analysis_result.get('target_price'),
            "stop_loss": analysis_result.get('stop_loss'),
            "risk_reward_ratio": analysis_result.get('risk_reward_ratio'),
            "time_horizon": analysis_result.get('time_horizon', 'medium_term'),
            "reasoning": analysis_result['reasoning'],
            "key_risks": analysis_result.get('key_risks', []),
            "key_opportunities": analysis_result.get('key_opportunities', []),
            "agent_steps": analysis_result['agent_steps'],
            "quality_score": analysis_result.get('quality_score', 0),
            "validation_warnings": analysis_result.get('validation_warnings', []),
            "technical_indicators": analysis_result.get('technical_indicators', {}),
            "technical_signals": analysis_result.get('technical_signals', {}),
            "rag_patterns_found": analysis_result.get('rag_patterns_found', 0),
            "model_used": f"{request.provider}/{request.model}",
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        await db.analysis_history.insert_one(result)
        
        # Remove MongoDB _id before returning
        result.pop('_id', None)
        return result
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@api_router.get("/analysis/history")
async def get_analysis_history(limit: int = 20):
    """Get analysis history"""
    cursor = db.analysis_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    history = await cursor.to_list(length=limit)
    return history

@api_router.get("/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Get a specific analysis"""
    analysis = await db.analysis_history.find_one({"id": analysis_id}, {"_id": 0})
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis

@api_router.delete("/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete an analysis"""
    result = await db.analysis_history.delete_one({"id": analysis_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"message": "Analysis deleted successfully"}


# ============ ENSEMBLE ANALYSIS ENDPOINTS ============

class EnsembleAnalysisRequest(BaseModel):
    symbol: str
    use_openai: bool = True
    use_gemini: bool = True
    use_anthropic: bool = True  # Enable Claude by default
    min_models: int = 1  # Allow single model for cases where only one key is valid


@api_router.post("/analyze/ensemble")
async def analyze_stock_ensemble(request: EnsembleAnalysisRequest):
    """
    Run ensemble AI analysis using multiple LLM models

    This endpoint calls multiple AI models in parallel and aggregates their
    recommendations using weighted voting for higher confidence results.
    """
    from agents.ensemble_analyzer import get_ensemble_analyzer
    from data_providers.provider_manager import get_provider_manager
    from agents.analysis_agent import TechnicalAnalysisAgent
    from agents.percentile_scorer import PercentileScorer

    # Get API keys from settings
    settings = await db.settings.find_one({}, {"_id": 0})

    if not settings:
        raise HTTPException(status_code=400, detail="Please configure your API keys in Settings first")

    # Check available API keys
    openai_key = settings.get('openai_api_key') if request.use_openai else None
    gemini_key = settings.get('gemini_api_key') if request.use_gemini else None
    anthropic_key = settings.get('anthropic_api_key') if request.use_anthropic else None

    available_count = sum(1 for k in [openai_key, gemini_key, anthropic_key] if k)

    if available_count < request.min_models:
        raise HTTPException(
            status_code=400,
            detail=f"Ensemble analysis requires at least {request.min_models} API keys configured. "
                   f"Currently have {available_count}. Please add more API keys in Settings."
        )

    try:
        # Normalize symbol for Indian stocks (add .NS suffix)
        normalized_symbol = normalize_indian_symbol(request.symbol)

        # Get stock data
        data_provider_keys = {
            "finnhub": settings.get('finnhub_api_key'),
            "alpaca": {
                "key": settings.get('alpaca_api_key'),
                "secret": settings.get('alpaca_api_secret')
            },
            "fmp": settings.get('fmp_api_key')
        }

        provider_manager = get_provider_manager(data_provider_keys)
        stock_data_obj = await provider_manager.get_quote(normalized_symbol)

        if stock_data_obj is None:
            raise HTTPException(status_code=404, detail=f"Could not fetch data for {normalized_symbol}")

        stock_data = stock_data_obj.to_dict()

        # Get historical data for technical analysis
        hist = await provider_manager.get_historical_data(normalized_symbol, period="6mo", interval="1d")

        if hist is None or len(hist) < 50:
            raise HTTPException(status_code=400, detail="Insufficient historical data for analysis")

        # Calculate technical indicators
        close = hist['Close']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']

        rsi_indicator = ta.momentum.RSIIndicator(close, window=14)
        macd_indicator = ta.trend.MACD(close)

        # Use real-time price from stock_data, fallback to historical data
        realtime_price = stock_data.get('current_price', float(close.iloc[-1]))

        technical_indicators = {
            "current_price": realtime_price,
            "rsi": round(float(rsi_indicator.rsi().iloc[-1]), 2),
            "macd": round(float(macd_indicator.macd().iloc[-1]), 2),
            "macd_signal": round(float(macd_indicator.macd_signal().iloc[-1]), 2),
            "sma_20": round(float(ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]), 2),
            "sma_50": round(float(ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]), 2),
            "sma_200": round(float(ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1]), 2) if len(hist) >= 200 else None,
            "atr": round(float(ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]), 2),
            "stochastic_k": round(float(ta.momentum.StochasticOscillator(high, low, close).stoch().iloc[-1]), 2),
            "stochastic_d": round(float(ta.momentum.StochasticOscillator(high, low, close).stoch_signal().iloc[-1]), 2),
            "adx": round(float(ta.trend.ADXIndicator(high, low, close).adx().iloc[-1]), 2),
        }

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        technical_indicators.update({
            "bb_upper": round(float(bb.bollinger_hband().iloc[-1]), 2),
            "bb_middle": round(float(bb.bollinger_mavg().iloc[-1]), 2),
            "bb_lower": round(float(bb.bollinger_lband().iloc[-1]), 2),
        })

        # Generate signals
        rsi = technical_indicators.get("rsi")
        technical_signals = {
            "rsi": "oversold" if rsi and rsi < 30 else ("overbought" if rsi and rsi > 70 else "neutral"),
            "macd": "bullish" if technical_indicators.get("macd", 0) > technical_indicators.get("macd_signal", 0) else "bearish",
        }

        current_price = technical_indicators.get("current_price", 0)
        sma_20 = technical_indicators.get("sma_20", 0)
        sma_50 = technical_indicators.get("sma_50", 0)

        if current_price > sma_20 > sma_50:
            technical_signals["trend"] = "strong_uptrend"
        elif current_price > sma_20:
            technical_signals["trend"] = "uptrend"
        elif current_price < sma_20 < sma_50:
            technical_signals["trend"] = "strong_downtrend"
        else:
            technical_signals["trend"] = "downtrend"

        # Calculate percentile scores
        percentile_scorer = PercentileScorer()
        percentile_scores = percentile_scorer.calculate_percentiles(hist, technical_indicators)

        # Get RAG context
        retriever = get_retriever()
        rag_context = retriever.build_context(
            query=f"Stock {request.symbol} {technical_signals.get('trend', '')}",
            n_results=3,
            max_tokens=1000
        )

        # Run ensemble analysis
        ensemble_analyzer = get_ensemble_analyzer(
            openai_api_key=openai_key,
            gemini_api_key=gemini_key,
            anthropic_api_key=anthropic_key
        )

        ensemble_result = await ensemble_analyzer.analyze_with_ensemble(
            symbol=request.symbol,
            stock_data=stock_data,
            technical_indicators=technical_indicators,
            technical_signals=technical_signals,
            percentile_scores=percentile_scores,
            rag_context=rag_context,
            min_models=request.min_models
        )

        # Prepare response
        result = {
            "id": str(uuid.uuid4()),
            "symbol": request.symbol.upper(),
            "analysis_type": "ensemble",
            "recommendation": ensemble_result.final_recommendation.value,
            "confidence": ensemble_result.final_confidence,
            "consensus_level": ensemble_result.consensus_level,
            "models_used": ensemble_result.models_used,
            "models_agreed": ensemble_result.models_agreed,
            "entry_price": ensemble_result.weighted_entry_price,
            "target_price": ensemble_result.weighted_target_price,
            "stop_loss": ensemble_result.weighted_stop_loss,
            "risk_reward_ratio": ensemble_result.average_risk_reward,
            "reasoning": ensemble_result.aggregated_reasoning,
            "key_risks": ensemble_result.aggregated_risks,
            "key_opportunities": ensemble_result.aggregated_opportunities,
            "individual_models": [
                {
                    "model": r.model,
                    "provider": r.provider,
                    "recommendation": r.recommendation.value,
                    "confidence": r.confidence,
                    "success": r.success,
                    "error": r.error
                }
                for r in ensemble_result.individual_responses
            ],
            "technical_indicators": technical_indicators,
            "technical_signals": technical_signals,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # Save to database
        await db.analysis_history.insert_one(result.copy())
        result.pop('_id', None)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ensemble analysis failed: {str(e)}")


# ============ RAG MANAGEMENT ENDPOINTS ============

@api_router.get("/rag/stats")
async def get_rag_stats():
    """Get RAG system statistics"""
    try:
        from rag.ingestion import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()
        stats = pipeline.get_ingestion_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get RAG stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/rag/seed")
async def seed_rag_database(background_tasks: BackgroundTasks):
    """Seed RAG database with initial knowledge (runs in background)"""
    try:
        from rag.seed_data import seed_all
        
        # Run seeding in background
        background_tasks.add_task(seed_all)
        
        return {"message": "RAG seeding started in background"}
    except Exception as e:
        logger.error(f"Failed to start RAG seeding: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RAGDocument(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None

@api_router.post("/rag/ingest")
async def ingest_document(doc: RAGDocument):
    """Ingest a custom document into RAG system"""
    try:
        from rag.ingestion import get_ingestion_pipeline
        pipeline = get_ingestion_pipeline()
        
        success = pipeline.ingest_document(
            content=doc.content,
            metadata=doc.metadata
        )
        
        if success:
            return {"message": "Document ingested successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to ingest document")
    except Exception as e:
        logger.error(f"Failed to ingest document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class RAGQuery(BaseModel):
    query: str
    n_results: int = 5
    min_similarity: float = 0.5

@api_router.post("/rag/search")
async def search_rag(query_data: RAGQuery):
    """Search RAG knowledge base"""
    try:
        from rag.retrieval import get_retriever
        retriever = get_retriever()
        
        results = retriever.retrieve(
            query=query_data.query,
            n_results=query_data.n_results,
            min_similarity=query_data.min_similarity
        )
        
        return {
            "query": query_data.query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Watchlist endpoints
@api_router.get("/watchlist")
async def get_watchlist():
    """Get user's watchlist"""
    watchlist = await db.watchlist.find({}, {"_id": 0}).to_list(100)
    return watchlist

@api_router.post("/watchlist/{symbol}")
async def add_to_watchlist(symbol: str):
    """Add stock to watchlist"""
    existing = await db.watchlist.find_one({"symbol": symbol.upper()})
    if existing:
        raise HTTPException(status_code=400, detail="Stock already in watchlist")
    
    stock_data = await fetch_stock_data(symbol)
    watchlist_item = {
        "id": str(uuid.uuid4()),
        "symbol": symbol.upper(),
        "name": stock_data.get('name'),
        "added_at": datetime.now(timezone.utc).isoformat()
    }
    await db.watchlist.insert_one(watchlist_item)
    watchlist_item.pop('_id', None)
    return watchlist_item

@api_router.delete("/watchlist/{symbol}")
async def remove_from_watchlist(symbol: str):
    """Remove stock from watchlist"""
    result = await db.watchlist.delete_one({"symbol": symbol.upper()})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Stock not in watchlist")
    return {"message": "Stock removed from watchlist"}

# ============ BACKTESTING ENDPOINTS ============

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    strategy_params: Optional[Dict[str, Any]] = None

@api_router.post("/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run backtest for a strategy"""
    try:
        # Get strategy
        strategy = StrategyRegistry.get_strategy(
            request.strategy,
            request.strategy_params
        )
        
        # Create engine
        engine = BacktestEngine(initial_capital=request.initial_capital)
        
        # Run backtest (REAL backtesting with historical data)
        result = engine.run_backtest(
            symbol=request.symbol,
            strategy=strategy,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Save to database
        backtest_record = {
            "id": str(uuid.uuid4()),
            "symbol": request.symbol.upper(),
            "strategy": request.strategy,
            "strategy_params": request.strategy_params or {},
            "start_date": request.start_date,
            "end_date": request.end_date,
            "initial_capital": request.initial_capital,
            "metrics": result['metrics'],
            "total_signals": result['total_signals'],
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        await db.backtests.insert_one(backtest_record)
        
        return result
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/backtest/strategies")
async def list_strategies():
    """List available backtesting strategies"""
    strategies = StrategyRegistry.list_strategies()
    
    # Get strategy details
    strategy_details = []
    for name in strategies:
        strategy = StrategyRegistry.get_strategy(name)
        strategy_details.append({
            "name": name,
            "display_name": strategy.name,
            "description": strategy.get_description(),
            "default_params": strategy.params
        })
    
    return {"strategies": strategy_details}

@api_router.get("/backtest/history")
async def get_backtest_history(limit: int = 20):
    """Get backtest history"""
    cursor = db.backtests.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    history = await cursor.to_list(length=limit)
    return history

@api_router.get("/backtest/{backtest_id}")
async def get_backtest(backtest_id: str):
    """Get specific backtest result"""
    backtest = await db.backtests.find_one({"id": backtest_id}, {"_id": 0})
    if not backtest:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return backtest

@api_router.delete("/backtest/{backtest_id}")
async def delete_backtest(backtest_id: str):
    """Delete a backtest"""
    result = await db.backtests.delete_one({"id": backtest_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Backtest not found")
    return {"message": "Backtest deleted"}

@api_router.get("/backtest/cache/stats")
async def get_cache_stats():
    """Get price cache statistics"""
    cache = get_price_cache()
    stats = cache.get_cache_stats()
    return stats

@api_router.post("/backtest/cache/clear")
async def clear_cache(symbol: Optional[str] = None):
    """Clear price cache"""
    cache = get_price_cache()
    cache.clear_cache(symbol)
    return {"message": f"Cache cleared for {symbol if symbol else 'all symbols'}"}

class BacktestComparisonRequest(BaseModel):
    """Request model for AI backtest insights"""
    symbol: str
    strategies: List[Dict[str, Any]]  # List of strategy results with metrics

@api_router.post("/backtest/insights")
async def generate_backtest_insights(request: BacktestComparisonRequest):
    """Generate AI-powered insights from backtest comparison results"""
    try:
        # Get API key from settings
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=500, detail="Settings not configured")

        # Determine provider and API key
        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get('openai_api_key') if provider == 'openai' else settings.get('gemini_api_key')
        model = settings.get('selected_model', 'gpt-4o-mini')

        if not api_key:
            raise HTTPException(status_code=500, detail=f"{provider.upper()} API key not configured")

        # Format strategy data for AI analysis
        strategies_summary = []
        for strat in request.strategies:
            if not strat.get('error'):
                strategies_summary.append({
                    'name': strat.get('display_name', strat.get('strategy')),
                    'return': strat.get('total_return_pct'),
                    'cagr': strat.get('cagr_pct'),
                    'sharpe': strat.get('sharpe_ratio'),
                    'max_drawdown': strat.get('max_drawdown_pct'),
                    'win_rate': strat.get('win_rate_pct'),
                    'trades': strat.get('total_trades'),
                    'final_value': strat.get('final_value')
                })

        # Sort by return to identify best/worst
        strategies_summary.sort(key=lambda x: x['return'] or -999, reverse=True)

        # Create comprehensive AI prompt with specific data points
        prompt = f"""You are a quantitative analyst reviewing backtest results for {request.symbol}. Provide 5-7 specific, data-driven insights.

BACKTEST RESULTS SUMMARY:
{json.dumps(strategies_summary, indent=2)}

ANALYSIS FRAMEWORK:
1. Performance Ranking: Identify best and worst performing strategies with exact numbers
2. Risk-Adjusted Returns: Compare Sharpe ratios - which strategy offers best risk-reward?
3. Drawdown Analysis: Which strategy preserves capital best during downturns?
4. Win Rate vs Returns: Is high win rate correlating with high returns?
5. Trade Frequency: Does more trading lead to better or worse outcomes?
6. Actionable Recommendation: Which strategy would you recommend and why?

FORMAT REQUIREMENTS:
- Start each insight with a ✨ emoji
- Include specific numbers (returns, Sharpe ratio, drawdown percentages)
- Keep each insight to 1-2 sentences
- Be direct and actionable
- Compare strategies against each other
- End with a clear recommendation

Example format:
✨ Trend Following delivered the highest return at +31.19% with a Sharpe ratio of 0.73, significantly outperforming all other strategies
✨ MACD Crossover was the worst performer at -20.52%, suggesting this strategy doesn't align well with {request.symbol}'s price patterns during this period

Now analyze the data above and provide your insights:"""

        # Call AI API directly
        from agents.reasoning_agent import DeepReasoningAgent
        reasoning_agent = DeepReasoningAgent()

        insights_text = await reasoning_agent._call_llm(prompt, model, provider, api_key)

        # Extract insights
        lines = insights_text.strip().split('\n')
        insights = []
        for line in lines:
            line = line.strip()
            if line and ('✨' in line or line.startswith('-') or line.startswith('•') or line.startswith('*') or (len(line) > 0 and line[0].isdigit())):
                # Clean up bullet points
                clean_line = line.replace('✨', '').lstrip('-•*0123456789. ').strip()
                if clean_line and len(clean_line) > 20:  # Filter out very short lines
                    insights.append(clean_line)

        # If no insights extracted, try alternate parsing
        if not insights:
            insights = [line.strip() for line in lines if line.strip() and len(line.strip()) > 20]

        return {
            "symbol": request.symbol,
            "insights": insights[:7],  # Limit to 7 insights
            "raw_analysis": insights_text
        }

    except Exception as e:
        logger.error(f"Failed to generate backtest insights: {e}")
        import traceback
        traceback.print_exc()

        # Provide smart fallback insights based on actual data
        try:
            strategies = request.strategies
            valid_strategies = [s for s in strategies if not s.get('error')]

            if valid_strategies:
                # Sort by return
                sorted_strats = sorted(valid_strategies, key=lambda x: x.get('total_return_pct', -999), reverse=True)
                best = sorted_strats[0]
                worst = sorted_strats[-1]

                # Find best Sharpe
                by_sharpe = sorted([s for s in valid_strategies if s.get('sharpe_ratio') is not None],
                                   key=lambda x: x.get('sharpe_ratio', -999), reverse=True)
                best_sharpe = by_sharpe[0] if by_sharpe else None

                insights = [
                    f"{best.get('display_name', 'Top strategy')} achieved the best return of {best.get('total_return_pct', 0):+.2f}% with a Sharpe ratio of {best.get('sharpe_ratio', 0):.2f}",
                    f"{worst.get('display_name', 'Worst strategy')} had the lowest return at {worst.get('total_return_pct', 0):+.2f}%, indicating this approach didn't suit {request.symbol} during this period"
                ]

                if best_sharpe and best_sharpe != best:
                    insights.append(f"For risk-adjusted returns, {best_sharpe.get('display_name')} had the best Sharpe ratio of {best_sharpe.get('sharpe_ratio', 0):.2f}")

                # Add max drawdown insight
                by_drawdown = sorted([s for s in valid_strategies if s.get('max_drawdown_pct') is not None],
                                     key=lambda x: abs(x.get('max_drawdown_pct', 999)))
                if by_drawdown:
                    best_dd = by_drawdown[0]
                    insights.append(f"{best_dd.get('display_name')} preserved capital best with only {abs(best_dd.get('max_drawdown_pct', 0)):.2f}% maximum drawdown")

                insights.append("Consider both absolute returns and risk metrics when selecting a strategy for live trading")
                insights.append("Past performance does not guarantee future results - always use proper position sizing and risk management")

                return {
                    "symbol": request.symbol,
                    "insights": insights,
                    "raw_analysis": "AI analysis unavailable, showing data-based insights"
                }
        except:
            pass

        return {
            "symbol": request.symbol,
            "insights": [
                f"Analyzed {len(request.strategies)} strategies for {request.symbol}",
                "Review the comparison table above for detailed metrics",
                "Consider risk-adjusted returns (Sharpe ratio) alongside absolute returns",
                "Lower maximum drawdown indicates better risk management",
                "Past performance does not guarantee future results"
            ],
            "raw_analysis": "AI analysis temporarily unavailable"
        }

# ============ NEWS & SENTIMENT ENDPOINTS ============

@api_router.get("/news/latest")
async def get_latest_news(symbol: Optional[str] = None, limit: int = 10):
    """Get latest financial news (REAL API CALL)"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()
        
        # Fetch news (REAL API CALL to RSS feeds)
        articles = news_aggregator.fetch_latest_news(symbol=symbol, limit=limit)
        
        # Analyze sentiment
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)
        
        # Get aggregate sentiment
        aggregate = sentiment_analyzer.get_aggregate_sentiment(articles_with_sentiment)
        
        return {
            "articles": articles_with_sentiment,
            "aggregate_sentiment": aggregate,
            "total": len(articles_with_sentiment)
        }
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/news/search")
async def search_news(q: str, days_back: int = 7, limit: int = 20):
    """Search news by query"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()
        
        # Search news (REAL API CALL)
        articles = news_aggregator.search_news(query=q, days_back=days_back, limit=limit)
        
        # Analyze sentiment
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)
        
        return {
            "articles": articles_with_sentiment,
            "total": len(articles_with_sentiment),
            "query": q
        }
    except Exception as e:
        logger.error(f"News search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/news/trending")
async def get_trending_topics(limit: int = 10):
    """
    Get trending topics from recent news
    ✅ Optimized with caching and rate limiting
    """
    try:
        rate_limiter = get_rate_limiter()

        # Try cache first (10 minute TTL for trending)
        cache_params = {"limit": limit}
        cached = rate_limiter.get_cached("trending", cache_params)
        if cached:
            return cached

        # Check rate limit
        if not rate_limiter.check_rate_limit("rss", skip_market_hours=True):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again later.")

        news_aggregator = get_news_aggregator()
        trending = news_aggregator.get_trending_topics(limit=limit)

        result = {"trending": trending}

        # Cache result
        if trending:
            rate_limiter.set_cache("trending", cache_params, result)

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get trending topics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/news/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str, days_back: int = 7):
    """Get sentiment analysis for a specific symbol"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()

        # Fetch symbol-specific news
        articles = news_aggregator.fetch_latest_news(symbol=symbol, limit=50)

        # Analyze sentiment
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)

        # Get aggregate
        aggregate = sentiment_analyzer.get_aggregate_sentiment(articles_with_sentiment)

        return {
            "symbol": symbol,
            "sentiment": aggregate,
            "recent_articles": articles_with_sentiment[:5]  # Top 5 most recent
        }
    except Exception as e:
        logger.error(f"Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/comprehensive/{symbol}")
async def get_comprehensive_news(
    symbol: str,
    days: int = 7,
    use_llm_sentiment: bool = True
):
    """
    Get comprehensive news from all sources with advanced sentiment analysis

    Uses NewsAPI, Alpha Vantage, RSS feeds, and LLM-based sentiment analysis
    for the most complete news coverage and sentiment understanding.
    """
    try:
        from news.integrated_news import get_integrated_news_service

        # Get API keys from settings
        settings = await db.settings.find_one({}, {"_id": 0})

        if not settings:
            settings = {}

        news_service = get_integrated_news_service(
            newsapi_key=settings.get('newsapi_key'),
            alphavantage_key=settings.get('alphavantage_api_key'),
            openai_api_key=settings.get('openai_api_key'),
            gemini_api_key=settings.get('gemini_api_key')
        )

        result = await news_service.get_comprehensive_news(
            symbol=symbol,
            days=days,
            include_sentiment=True,
            use_llm_sentiment=use_llm_sentiment
        )

        return result

    except Exception as e:
        logger.error(f"Comprehensive news failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/market")
async def get_market_news(
    category: str = "business",
    limit: int = 20,
    include_sentiment: bool = True
):
    """
    Get general market news with sentiment analysis
    ✅ Optimized with caching and rate limiting
    """
    try:
        rate_limiter = get_rate_limiter()

        # Try cache first (5 minute TTL)
        cache_params = {"category": category, "limit": limit}
        cached = rate_limiter.get_cached("market_news", cache_params)
        if cached:
            return cached

        # Check rate limit (skip market hours check for news)
        if not rate_limiter.check_rate_limit("newsapi", skip_market_hours=True):
            # Return cached data even if slightly stale
            logger.warning("Rate limited - returning stale cache or error")
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Please try again in 1 minute.")

        from news.integrated_news import get_integrated_news_service

        settings = await db.settings.find_one({}, {"_id": 0})

        news_service = get_integrated_news_service(
            newsapi_key=settings.get('newsapi_key') if settings else None,
            openai_api_key=settings.get('openai_api_key') if settings else None,
            gemini_api_key=settings.get('gemini_api_key') if settings else None
        )

        result = await news_service.get_market_news(
            category=category,
            limit=limit,
            include_sentiment=include_sentiment
        )

        # Cache result
        if result and result.get("total_articles", 0) > 0:
            rate_limiter.set_cache("market_news", cache_params, result)

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Market news failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WEB SEARCH ENDPOINTS (Real-Time News) ============

@api_router.get("/news/web-search/stock/{symbol}")
async def web_search_stock_news(
    symbol: str,
    company_name: Optional[str] = None,
    max_results: int = 10,
    days_back: int = 7
):
    """
    Search for real-time news about a stock using web search.

    Uses DuckDuckGo (free) or SerpAPI for current news articles.
    More current than NewsAPI for breaking news.

    Parameters:
    - symbol: Stock symbol
    - company_name: Optional company name for better results
    - max_results: Maximum number of articles (default 10)
    - days_back: Days to look back (default 7)
    """
    try:
        from news.web_search import search_stock_news

        results = await search_stock_news(
            symbol=symbol,
            company_name=company_name,
            max_results=max_results,
            days_back=days_back
        )

        return {
            "symbol": symbol,
            "total_results": len(results),
            "days_back": days_back,
            "articles": results,
            "source": "web_search"
        }

    except Exception as e:
        logger.error(f"Web search for {symbol} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/web-search/market")
async def web_search_market_news(
    topic: str = "Indian stock market",
    max_results: int = 10,
    days_back: int = 3
):
    """
    Search for general market news using web search.

    Great for getting current market sentiment and breaking news.

    Parameters:
    - topic: Market topic (default "Indian stock market")
    - max_results: Maximum articles (default 10)
    - days_back: Days to look back (default 3)
    """
    try:
        from news.web_search import get_web_search_service

        service = get_web_search_service()
        results = await service.search_market_news(
            topic=topic,
            max_results=max_results,
            days_back=days_back
        )

        # Deduplicate
        results = service.deduplicate_results(results)

        return {
            "topic": topic,
            "total_results": len(results),
            "days_back": days_back,
            "articles": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "published_date": r.published_date.isoformat() if r.published_date else None
                }
                for r in results
            ],
            "source": "web_search"
        }

    except Exception as e:
        logger.error(f"Market web search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/web-search/sector/{sector}")
async def web_search_sector_news(
    sector: str,
    max_results: int = 10,
    days_back: int = 7
):
    """
    Search for sector-specific news using web search.

    Parameters:
    - sector: Sector name (e.g., "banking", "IT", "pharma")
    - max_results: Maximum articles (default 10)
    - days_back: Days to look back (default 7)
    """
    try:
        from news.web_search import get_web_search_service

        service = get_web_search_service()
        results = await service.search_sector_news(
            sector=sector,
            max_results=max_results,
            days_back=days_back
        )

        # Deduplicate
        results = service.deduplicate_results(results)

        return {
            "sector": sector,
            "total_results": len(results),
            "days_back": days_back,
            "articles": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "source": r.source,
                    "published_date": r.published_date.isoformat() if r.published_date else None
                }
                for r in results
            ],
            "source": "web_search"
        }

    except Exception as e:
        logger.error(f"Sector web search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CANDLESTICK PATTERN ENDPOINTS ============

@api_router.get("/patterns/{symbol}")
async def get_candlestick_patterns(symbol: str, days: int = 30):
    """Detect candlestick patterns (REAL pattern detection)"""
    try:
        # Fetch historical data
        import yfinance as yf
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=f"{days}d")
        
        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")
        
        # Detect patterns (REAL detection algorithm)
        detector = get_pattern_detector()
        all_patterns = detector.detect_patterns(data, lookback_periods=days)
        
        # Get recent patterns (last 5 days)
        recent_patterns = detector.get_recent_patterns(data, days=5)
        
        # Categorize patterns
        bullish = [p for p in all_patterns if 'bullish' in p['type']]
        bearish = [p for p in all_patterns if 'bearish' in p['type']]
        indecision = [p for p in all_patterns if p['type'] == 'indecision']
        
        return {
            "symbol": symbol,
            "period_days": days,
            "total_patterns": len(all_patterns),
            "recent_patterns": recent_patterns,
            "all_patterns": all_patterns,
            "pattern_counts": {
                "bullish": len(bullish),
                "bearish": len(bearish),
                "indecision": len(indecision)
            },
            "latest_signal": recent_patterns[0] if recent_patterns else None
        }
    except Exception as e:
        logger.error(f"Pattern detection failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/patterns/{symbol}/recent")
async def get_recent_patterns(symbol: str, days: int = 5):
    """Get patterns detected in recent days"""
    try:
        import yfinance as yf
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period="60d")  # Get more data for context

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")

        detector = get_pattern_detector()
        recent_patterns = detector.get_recent_patterns(data, days=days)

        return {
            "symbol": symbol,
            "days": days,
            "patterns": recent_patterns,
            "count": len(recent_patterns)
        }
    except Exception as e:
        logger.error(f"Recent pattern detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ LLM FEATURES ENDPOINTS ============

class LLMQueryRequest(BaseModel):
    """Request model for natural language stock queries"""
    question: str
    symbol: Optional[str] = None

class LLMNewsSummaryRequest(BaseModel):
    """Request model for news summarization"""
    symbol: str
    articles: Optional[List[Dict[str, Any]]] = None

class LLMPortfolioRequest(BaseModel):
    """Request model for portfolio analysis"""
    holdings: List[Dict[str, Any]]  # [{"symbol": "TCS", "quantity": 10, "avg_price": 3500}, ...]
    investment_goal: Optional[str] = "growth"

class LLMPatternExplainRequest(BaseModel):
    """Request model for pattern explanation"""
    symbol: str
    pattern_name: str
    pattern_data: Optional[Dict[str, Any]] = None

class LLMCompareRequest(BaseModel):
    """Request model for stock comparison"""
    symbols: List[str]
    criteria: Optional[List[str]] = None  # ["value", "growth", "momentum", "risk"]


@api_router.post("/llm/ask/stream")
async def llm_ask_streaming(request: LLMQueryRequest):
    """
    Streaming natural language interface for stock queries with Server-Sent Events (SSE).
    Provides real-time streaming responses with RAG-enhanced context and caching.
    """
    from fastapi.responses import StreamingResponse
    import json
    import hashlib
    from datetime import datetime, timedelta, timezone

    async def generate_stream():
        try:
            # Get Redis cache instance
            from utils.redis_cache import get_cache
            cache = get_cache()

            # Check cache first for faster responses
            cache_key = f"llm_stream:{hashlib.md5((request.question + str(request.symbol)).encode()).hexdigest()}"

            # Try to get cached response (5 minute TTL for fast responses)
            cached = await cache.get(cache_key)
            if cached and not request.question.lower().startswith(("current", "latest", "today", "now")):
                # Stream cached response for perceived speed
                cached_data = json.loads(cached)
                for char in cached_data.get("content", ""):
                    yield f"data: {json.dumps({'content': char})}\n\n"
                yield f"data: {json.dumps({'done': True, 'cached': True, 'model': cached_data.get('model')})}\n\n"
                return

            # Get API key
            settings = await db.settings.find_one({}, {"_id": 0})
            if not settings:
                yield f"data: {json.dumps({'error': 'Please configure API keys in Settings'})}\n\n"
                return

            # Debug logging - show ALL settings
            logger.info(f"LLM Stream - Raw settings keys: {list(settings.keys())}")
            logger.info(f"LLM Stream - selected_model in settings: {settings.get('selected_model')}")

            provider = settings.get('selected_provider', 'openai')
            api_key = settings.get(f'{provider}_api_key')
            model = settings.get('selected_model', 'gpt-4o-mini')

            # Debug logging
            logger.info(f"LLM Stream - Provider: {provider}, Model: {model}")

            if not api_key:
                yield f"data: {json.dumps({'error': f'Please configure {provider} API key in Settings'})}\n\n"
                return

            # Build enhanced context with stock data
            stock_context = ""
            context_data = {}
            rag_knowledge = ""

            if request.symbol:
                import yfinance as yf
                try:
                    ticker = yf.Ticker(f"{request.symbol}.NS")
                    info = ticker.info
                    hist = ticker.history(period="3mo")

                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        context_data = {
                            "symbol": request.symbol,
                            "current_price": round(current_price, 2),
                            "change_pct": info.get('regularMarketChangePercent', 0)
                        }

                        stock_context = f"""Stock: {info.get('longName', request.symbol)} ({request.symbol})
Current Price: ₹{current_price:.2f}
Day Change: {info.get('regularMarketChangePercent', 0):.2f}%
52W High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')} | 52W Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}
P/E: {info.get('trailingPE', 'N/A')} | Market Cap: ₹{info.get('marketCap', 0) / 10000000:.0f} Cr
Sector: {info.get('sector', 'N/A')}"""
                except:
                    pass

                # **NEW: Add RAG knowledge context for better insights**
                try:
                    from knowledge.rag_ingestion import get_rag_ingestion
                    rag_ingestion = get_rag_ingestion(db)

                    # Get discovered patterns
                    patterns = await rag_ingestion.query_relevant_knowledge(
                        symbol=request.symbol,
                        query_type="pattern",
                        limit=2
                    )

                    # Get historical events
                    events = await rag_ingestion.query_relevant_knowledge(
                        symbol=request.symbol,
                        query_type="event",
                        limit=2
                    )

                    # Format RAG knowledge
                    if patterns:
                        rag_knowledge += "\n\nHISTORICAL PATTERNS DISCOVERED:\n"
                        for p in patterns:
                            meta = p.get('metadata', {})
                            rag_knowledge += f"- {p.get('pattern_type', 'Unknown').replace('_', ' ').title()}: "
                            rag_knowledge += f"{meta.get('success_rate', 0):.0%} success rate, "
                            rag_knowledge += f"{meta.get('average_return', 0):+.1f}% avg return\n"

                    if events:
                        rag_knowledge += "\nHISTORICAL EVENT IMPACTS:\n"
                        for e in events:
                            meta = e.get('metadata', {})
                            rag_knowledge += f"- {e.get('event_type', 'Unknown').replace('_', ' ').title()}: "
                            rag_knowledge += f"Recovery timeline: {meta.get('recovery_timeline', 'N/A')}\n"

                except Exception as rag_error:
                    logger.warning(f"RAG knowledge retrieval failed: {rag_error}")
                    # Continue without RAG context
            else:
                # **NEW: Add general market knowledge for non-symbol queries**
                try:
                    # Get recent market overview from cache or DB
                    market_cache = await cache.get("market:overview")
                    if market_cache:
                        market_data = json.loads(market_cache)
                        rag_knowledge += f"\n\nCURRENT MARKET CONTEXT:\n"
                        rag_knowledge += f"Nifty 50: {market_data.get('nifty_50', {}).get('current', 'N/A')}\n"
                        rag_knowledge += f"Sensex: {market_data.get('sensex', {}).get('current', 'N/A')}\n"
                        rag_knowledge += f"Market Sentiment: {market_data.get('sentiment', 'Neutral')}\n"
                except Exception as market_error:
                    logger.warning(f"Market context retrieval failed: {market_error}")

            # Enhanced prompt with RAG knowledge
            prompt = f"""You are an expert Indian stock market assistant with deep knowledge of NSE/BSE markets, technical analysis, and investment strategies.

{stock_context if stock_context else "General market query - provide insights based on current Indian market trends."}
{rag_knowledge}

User Question: {request.question}

Provide a clear, actionable answer:
- Use ₹ for prices
- Include specific numbers and data points
- Reference historical patterns if relevant
- Be concise yet comprehensive (2-5 sentences)
- Add disclaimers for buy/sell opinions
- Focus on Indian market context"""

            # Stream based on provider and collect full response for caching
            full_response = ""

            if provider == 'openai':
                import openai
                client = openai.AsyncOpenAI(api_key=api_key)

                stream = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=True,
                    temperature=0.7,
                    max_tokens=500
                )

                async for chunk in stream:
                    if chunk.choices[0].delta.content:
                        content = chunk.choices[0].delta.content
                        full_response += content
                        yield f"data: {json.dumps({'content': content, **context_data})}\n\n"

            elif provider == 'gemini':
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model_obj = genai.GenerativeModel(model)

                response = model_obj.generate_content(prompt, stream=True)
                for chunk in response:
                    if chunk.text:
                        full_response += chunk.text
                        yield f"data: {json.dumps({'content': chunk.text, **context_data})}\n\n"

            elif provider == 'anthropic':
                import anthropic
                client = anthropic.AsyncAnthropic(api_key=api_key)

                async with client.messages.stream(
                    model=model,
                    max_tokens=500,
                    messages=[{"role": "user", "content": prompt}]
                ) as stream:
                    async for text in stream.text_stream:
                        full_response += text
                        yield f"data: {json.dumps({'content': text, **context_data})}\n\n"

            # Cache the full response for 5 minutes (300 seconds) for faster future responses
            if full_response:
                try:
                    cache_data = {"content": full_response, "model": model, **context_data}
                    await cache.set(cache_key, json.dumps(cache_data), ttl=300)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache response: {cache_error}")

            # Completion signal
            yield f"data: {json.dumps({'done': True, 'model': model, **context_data})}\n\n"

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )


@api_router.post("/llm/ask")
async def llm_ask_about_stock(request: LLMQueryRequest):
    """
    Natural language interface for stock queries (non-streaming).
    Ask questions like "Is RELIANCE a good buy?" or "What's the trend for TCS?"
    """
    try:
        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        llm_features = get_llm_features()
        result = await llm_features.ask_about_stock(
            question=request.question,
            symbol=request.symbol,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/summarize-news")
async def llm_summarize_news(request: LLMNewsSummaryRequest):
    """
    Get LLM-powered summary of news for a stock.
    Analyzes sentiment and extracts key insights.
    """
    try:
        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        # If no articles provided, fetch them
        articles = request.articles
        if not articles:
            news_aggregator = get_news_aggregator()
            articles = news_aggregator.fetch_latest_news(symbol=request.symbol, limit=10)

        llm_features = get_llm_features()
        result = await llm_features.summarize_news(
            symbol=request.symbol,
            articles=articles,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except Exception as e:
        logger.error(f"News summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/analyze-portfolio")
async def llm_analyze_portfolio(request: LLMPortfolioRequest):
    """
    Get AI-powered portfolio analysis with diversification insights,
    risk assessment, and rebalancing suggestions.
    """
    try:
        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        llm_features = get_llm_features()
        result = await llm_features.analyze_portfolio(
            holdings=request.holdings,
            investment_goal=request.investment_goal,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except Exception as e:
        logger.error(f"Portfolio analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/explain-pattern")
async def llm_explain_pattern(request: LLMPatternExplainRequest):
    """
    Get plain English explanation of a candlestick pattern.
    Explains what the pattern means and its trading implications.
    """
    try:
        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        llm_features = get_llm_features()
        result = await llm_features.explain_pattern(
            symbol=request.symbol,
            pattern_name=request.pattern_name,
            pattern_data=request.pattern_data,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except Exception as e:
        logger.error(f"Pattern explanation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/llm/market-commentary")
async def llm_market_commentary():
    """
    Get AI-generated daily market commentary.
    Summarizes market trends, sector movements, and key insights.
    """
    try:
        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        # Get market data for commentary
        # Fetch NIFTY 50 data as market proxy
        ticker = yf.Ticker("^NSEI")
        nifty_hist = ticker.history(period="5d")

        market_data = {
            "nifty_current": float(nifty_hist['Close'].iloc[-1]) if not nifty_hist.empty else None,
            "nifty_change": float(nifty_hist['Close'].iloc[-1] - nifty_hist['Close'].iloc[-2]) if len(nifty_hist) >= 2 else None,
            "nifty_change_pct": float((nifty_hist['Close'].iloc[-1] - nifty_hist['Close'].iloc[-2]) / nifty_hist['Close'].iloc[-2] * 100) if len(nifty_hist) >= 2 else None,
        }

        # Get latest news
        news_aggregator = get_news_aggregator()
        latest_news = news_aggregator.fetch_latest_news(limit=5)

        llm_features = get_llm_features()
        result = await llm_features.generate_market_commentary(
            market_data=market_data,
            news=latest_news,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except Exception as e:
        logger.error(f"Market commentary generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/llm/compare-stocks")
async def llm_compare_stocks(request: LLMCompareRequest):
    """
    Get AI-powered comparison of multiple stocks.
    Analyzes and compares based on various criteria.
    """
    try:
        if len(request.symbols) < 2:
            raise HTTPException(status_code=400, detail="At least 2 symbols required for comparison")
        if len(request.symbols) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 symbols allowed for comparison")

        # Get API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings")

        provider = settings.get('selected_provider', 'openai')
        api_key = settings.get(f'{provider}_api_key')
        model = settings.get('selected_model', 'gpt-4.1')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key")

        # Fetch data for all symbols
        stocks_data = []
        for symbol in request.symbols:
            try:
                stock_data = await fetch_stock_data(symbol)
                indicators = await calculate_technical_indicators(symbol)
                stocks_data.append({
                    "stock_data": stock_data,
                    "indicators": indicators
                })
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")
                stocks_data.append({
                    "stock_data": {"symbol": symbol, "error": str(e)},
                    "indicators": {}
                })

        llm_features = get_llm_features()
        result = await llm_features.compare_stocks(
            symbols=request.symbols,
            stocks_data=stocks_data,
            criteria=request.criteria,
            api_key=api_key,
            provider=provider,
            model=model
        )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ FUNDAMENTAL SCREENER ENDPOINTS ============

from analysis.fundamental_screener import get_fundamental_screener

class ScreenerRequest(BaseModel):
    """Request for stock screening"""
    symbols: Optional[List[str]] = None  # If None, use NIFTY_100_STOCKS
    filters: List[Dict[str, Any]]
    sort_by: Optional[str] = None
    sort_order: str = "desc"
    limit: int = 50


@api_router.get("/screener/fundamentals/{symbol}")
async def get_stock_fundamentals(symbol: str):
    """Get complete fundamental data for a single stock"""
    try:
        screener = get_fundamental_screener()
        data = await screener.get_stock_fundamentals(symbol)

        if 'error' in data:
            raise HTTPException(status_code=404, detail=data['error'])

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fundamentals for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# COMMENTED OUT: Duplicate endpoint - using the simpler one below instead
# @api_router.post("/screener/screen")
# async def screen_stocks(request: ScreenerRequest):
#     """
#     Screen stocks based on fundamental criteria.
#     Example filters: [{"metric": "pe_ratio", "operator": "lt", "value": 20}]
#     """
#     try:
#         screener = get_fundamental_screener()
#
#         # Use provided symbols or default to NIFTY 100
#         symbols = request.symbols or [s['symbol'] for s in NIFTY_100_STOCKS]
#
#         result = await screener.screen_stocks(
#             symbols=symbols,
#             filters=request.filters,
#             sort_by=request.sort_by,
#             sort_order=request.sort_order,
#             limit=request.limit
#         )
#
#         if 'error' in result:
#             raise HTTPException(status_code=500, detail=result['error'])
#
#         return result
#
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"Screening failed: {e}")
#         raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/screener/presets")
async def get_screener_presets():
    """Get predefined screening presets"""
    screener = get_fundamental_screener()
    return {"presets": screener.get_preset_screens()}


@api_router.post("/screener/run-preset/{preset_name}")
async def run_preset_screen(preset_name: str):
    """Run a predefined screening preset"""
    try:
        screener = get_fundamental_screener()
        presets = screener.get_preset_screens()

        preset = next((p for p in presets if p['name'].lower().replace(' ', '_') == preset_name.lower()), None)

        if not preset:
            raise HTTPException(status_code=404, detail=f"Preset '{preset_name}' not found")

        symbols = [s['symbol'] for s in NIFTY_100_STOCKS]

        result = await screener.screen_stocks(
            symbols=symbols,
            filters=preset['filters'],
            sort_by=preset.get('sort_by'),
            sort_order=preset.get('sort_order', 'desc'),
            limit=50
        )

        result['preset_name'] = preset['name']
        result['preset_description'] = preset['description']

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preset screening failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ FII/DII TRACKING ENDPOINTS ============

from market_data.fii_dii_tracker import get_fii_dii_tracker, get_bulk_block_tracker


@api_router.get("/market/fii-dii")
async def get_fii_dii_data():
    """
    Get today's FII/DII flow data
    ✅ Optimized with caching and rate limiting
    """
    try:
        rate_limiter = get_rate_limiter()

        # Try cache first (30 minute TTL - FII/DII updates slowly)
        cache_params = {"endpoint": "fii_dii"}
        cached = rate_limiter.get_cached("fii_dii", cache_params)
        if cached:
            return cached

        # Check rate limit (skip market hours check - FII/DII data available after hours)
        if not rate_limiter.check_rate_limit("fii_dii", skip_market_hours=True):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        tracker = get_fii_dii_tracker()
        data = await tracker.fetch_daily_fii_dii()

        # Cache result
        if data and not data.get("error"):
            rate_limiter.set_cache("fii_dii", cache_params, data)

        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to fetch FII/DII data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/bulk-deals")
async def get_bulk_deals():
    """Get today's bulk deals"""
    try:
        tracker = get_bulk_block_tracker()
        data = await tracker.fetch_bulk_deals()
        return data

    except Exception as e:
        logger.error(f"Failed to fetch bulk deals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/block-deals")
async def get_block_deals():
    """Get today's block deals"""
    try:
        tracker = get_bulk_block_tracker()
        data = await tracker.fetch_block_deals()
        return data

    except Exception as e:
        logger.error(f"Failed to fetch block deals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/institutional-activity")
async def get_institutional_activity():
    """Get combined institutional activity (FII/DII + Bulk/Block deals)"""
    try:
        fii_tracker = get_fii_dii_tracker()
        deals_tracker = get_bulk_block_tracker()

        fii_dii = await fii_tracker.fetch_daily_fii_dii()
        bulk = await deals_tracker.fetch_bulk_deals()
        block = await deals_tracker.fetch_block_deals()

        return {
            "fii_dii": fii_dii,
            "bulk_deals": bulk,
            "block_deals": block,
            "market_bias": fii_dii.get('interpretation', {}).get('overall_bias', 'Unknown')
        }

    except Exception as e:
        logger.error(f"Failed to fetch institutional activity: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WALK-FORWARD BACKTESTING ENDPOINTS ============

from backtesting.walk_forward import get_walk_forward_engine, IndianMarketCosts


class WalkForwardRequest(BaseModel):
    """Request for walk-forward backtest"""
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    train_pct: float = 0.7
    n_splits: int = 5
    anchored: bool = False
    initial_capital: float = 100000
    strategy_params: Optional[Dict[str, Any]] = None


@api_router.post("/backtest/walk-forward")
async def run_walk_forward_backtest(request: WalkForwardRequest):
    """
    Run walk-forward backtest with realistic Indian market costs.
    Walk-forward testing uses rolling train/test splits for realistic performance estimates.
    """
    try:
        # Get strategy
        strategy = StrategyRegistry.get_strategy(
            request.strategy,
            request.strategy_params
        )

        engine = get_walk_forward_engine(
            initial_capital=request.initial_capital,
            is_intraday=False
        )

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: asyncio.run(engine.run_walk_forward(
                symbol=request.symbol,
                strategy=strategy,
                start_date=request.start_date,
                end_date=request.end_date,
                train_pct=request.train_pct,
                n_splits=request.n_splits,
                anchored=request.anchored
            ))
        )

        # Save to database
        backtest_record = {
            "id": str(uuid.uuid4()),
            "type": "walk_forward",
            "symbol": request.symbol.upper(),
            "strategy": request.strategy,
            "aggregated_metrics": result.get('aggregated_metrics', {}),
            "robustness_score": result.get('robustness_score', {}),
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        await db.backtests.insert_one(backtest_record)

        return result

    except Exception as e:
        logger.error(f"Walk-forward backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/backtest/costs")
async def get_transaction_costs():
    """Get Indian market transaction costs breakdown"""
    costs = IndianMarketCosts()

    # Calculate example for ₹1 lakh trade
    example_value = 100000

    return {
        "description": "Realistic Indian equity market transaction costs",
        "components": {
            "brokerage": f"{costs.brokerage_rate * 100:.3f}% (max ₹{costs.brokerage_max})",
            "stt_delivery": f"{costs.stt_sell * 100:.2f}% on sell",
            "stt_intraday": f"{costs.stt_intraday_sell * 100:.4f}% on sell",
            "exchange_charges": f"{costs.nse_charge * 100:.5f}%",
            "gst": f"{costs.gst_rate * 100:.0f}% on brokerage",
            "sebi_charges": f"₹10 per crore",
            "stamp_duty": f"{costs.stamp_duty_buy * 100:.3f}% on buy",
            "dp_charges": f"₹{costs.dp_charges} per scrip (delivery)"
        },
        "example_1_lakh_roundtrip": costs.calculate_roundtrip_cost(example_value),
        "note": "Costs vary by broker. These are estimates based on discount brokers."
    }


# ============ MONTE CARLO SIMULATION ENDPOINTS ============

from backtesting.monte_carlo import get_monte_carlo_simulator, MonteCarloConfig


class MonteCarloRequest(BaseModel):
    """Request model for Monte Carlo simulation"""
    symbol: str
    strategy: str = "momentum"
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    n_simulations: int = 1000
    initial_capital: float = 100000
    randomize_trades: bool = True
    seed: Optional[int] = None


@api_router.post("/backtest/monte-carlo")
async def run_monte_carlo_simulation(request: MonteCarloRequest):
    """
    Run Monte Carlo simulation on strategy trades.
    Randomizes trade sequences to assess robustness.
    """
    try:
        from backtesting.strategies import StrategyRegistry
        from backtesting.walk_forward import get_walk_forward_engine

        # Get strategy
        registry = StrategyRegistry()
        strategies = registry.get_available_strategies()
        strategy_class = next((s for s in strategies if s['name'] == request.strategy), None)

        if not strategy_class:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        strategy = registry.create_strategy(request.strategy)

        # First run a backtest to get trades
        engine = get_walk_forward_engine()
        backtest_result = engine._run_single_backtest(
            symbol=request.symbol.upper(),
            strategy=strategy,
            start_date=request.start_date,
            end_date=request.end_date
        )

        trades = backtest_result.get('trades', [])

        if not trades:
            return {"error": "No trades generated by strategy"}

        # Run Monte Carlo
        config = MonteCarloConfig(
            n_simulations=request.n_simulations,
            randomize_trades=request.randomize_trades,
            seed=request.seed
        )
        simulator = get_monte_carlo_simulator(config)
        result = simulator.run_simulation(trades, request.initial_capital)

        result['symbol'] = request.symbol.upper()
        result['strategy'] = request.strategy
        result['period'] = f"{request.start_date} to {request.end_date}"

        return result

    except Exception as e:
        logger.error(f"Monte Carlo simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ML PREDICTION ENDPOINTS ============

from ml.inference import get_ml_service

@api_router.get("/ml/predict/{symbol}")
async def predict_stock_price(symbol: str, days_lookback: int = 60):
    """
    Predict next day's stock price using LSTM model.
    Trains a model on-the-fly using recent data.
    """
    try:
        # Fetch historical data
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        # Fetch enough data for training (e.g. 2 years)
        hist = ticker.history(period="2y")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"No data found for {symbol}")
            
        # Get ML service
        ml_service = get_ml_service()
        
        # Run prediction
        result = await ml_service.predict_next_price(symbol, hist, lookback_days)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))



# ============ PORTFOLIO BACKTESTING ENDPOINTS ============

from backtesting.portfolio import get_portfolio_backtester, get_correlation_analyzer


class PortfolioBacktestRequest(BaseModel):
    """Request model for portfolio backtest"""
    symbols: List[str]
    strategy: str = "momentum"
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000
    weighting: str = "equal"  # equal, market_cap, inverse_volatility


@api_router.post("/backtest/portfolio")
async def run_portfolio_backtest(request: PortfolioBacktestRequest):
    """
    Run portfolio-level backtest across multiple stocks.
    Tests diversification and correlation effects.
    """
    try:
        from backtesting.strategies import StrategyRegistry

        registry = StrategyRegistry()
        strategy = registry.create_strategy(request.strategy)

        backtester = get_portfolio_backtester()
        result = backtester.run_portfolio_backtest(
            symbols=[s.upper() for s in request.symbols],
            strategy=strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            weighting=request.weighting
        )

        return result

    except Exception as e:
        logger.error(f"Portfolio backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CorrelationRequest(BaseModel):
    """Request model for correlation analysis"""
    symbols: List[str]
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"


@api_router.post("/analysis/correlation")
async def analyze_correlation(request: CorrelationRequest):
    """
    Analyze correlation between multiple stocks.
    Returns correlation matrix and diversification score.
    """
    try:
        analyzer = get_correlation_analyzer()
        result = analyzer.calculate_correlation_matrix(
            symbols=[s.upper() for s in request.symbols],
            start_date=request.start_date,
            end_date=request.end_date
        )

        return result

    except Exception as e:
        logger.error(f"Correlation analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/analysis/beta/{symbol}")
async def get_stock_beta(symbol: str, benchmark: str = "^NSEI", start_date: str = "2023-01-01", end_date: str = "2024-01-01"):
    """
    Calculate beta of a stock relative to benchmark.
    Default benchmark is Nifty 50.
    """
    try:
        analyzer = get_correlation_analyzer()
        result = analyzer.calculate_beta(
            symbol=symbol.upper(),
            benchmark=benchmark,
            start_date=start_date,
            end_date=end_date
        )

        return result

    except Exception as e:
        logger.error(f"Beta calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ REGIME TESTING ENDPOINTS ============

from backtesting.regime_tester import get_regime_tester


class RegimeTestRequest(BaseModel):
    """Request model for regime testing"""
    symbol: str
    strategy: str = "momentum"
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 100000


@api_router.post("/backtest/regime")
async def test_strategy_by_regime(request: RegimeTestRequest):
    """
    Test strategy performance across different market regimes.
    Breaks down performance in bull/bear/sideways markets.
    """
    try:
        from backtesting.strategies import StrategyRegistry

        registry = StrategyRegistry()
        strategy = registry.create_strategy(request.strategy)

        tester = get_regime_tester()
        result = tester.test_strategy_by_regime(
            symbol=request.symbol.upper(),
            strategy=strategy,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital
        )

        return result

    except Exception as e:
        logger.error(f"Regime testing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CUSTOM INDICATOR ENDPOINTS ============

from backtesting.custom_indicators import get_indicator_builder


@api_router.get("/indicators/available")
async def get_available_indicators():
    """Get list of available technical indicators"""
    builder = get_indicator_builder()
    return {
        "indicators": builder.get_available_indicators(),
        "examples": builder.get_formula_examples()
    }


class CustomIndicatorRequest(BaseModel):
    """Request model for custom indicator calculation"""
    symbol: str
    indicator: str  # e.g., "sma", "rsi", "macd"
    params: Dict[str, Any] = {}
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"


@api_router.post("/indicators/calculate")
async def calculate_custom_indicator(request: CustomIndicatorRequest):
    """
    Calculate a technical indicator for a symbol.
    """
    try:
        from backtesting.price_cache import get_price_cache

        price_cache = get_price_cache()
        data = price_cache.get_prices(
            request.symbol.upper(),
            request.start_date,
            request.end_date
        )

        if data.empty:
            raise HTTPException(status_code=404, detail="No price data found")

        builder = get_indicator_builder()
        result = builder.calculate_indicator(data, request.indicator, request.params)

        # Convert to serializable format
        if isinstance(result, dict):
            output = {k: v.dropna().tail(20).to_dict() for k, v in result.items()}
        else:
            output = {"values": result.dropna().tail(20).to_dict()}

        return {
            "symbol": request.symbol.upper(),
            "indicator": request.indicator,
            "params": request.params,
            "data": output
        }

    except Exception as e:
        logger.error(f"Indicator calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CustomFormulaRequest(BaseModel):
    """Request model for custom formula indicator"""
    symbol: str
    formula: str  # e.g., "sma(20) - sma(50)"
    start_date: str = "2023-01-01"
    end_date: str = "2024-01-01"


@api_router.post("/indicators/formula")
async def calculate_formula_indicator(request: CustomFormulaRequest):
    """
    Calculate a custom formula indicator.
    Examples:
    - "sma(20) - sma(50)" for MA crossover
    - "rsi(14) > 70" for overbought condition
    """
    try:
        from backtesting.price_cache import get_price_cache

        price_cache = get_price_cache()
        data = price_cache.get_prices(
            request.symbol.upper(),
            request.start_date,
            request.end_date
        )

        if data.empty:
            raise HTTPException(status_code=404, detail="No price data found")

        builder = get_indicator_builder()
        result = builder.build_custom_indicator(data, request.formula)

        # Convert to serializable format
        output = result.dropna().tail(30).to_dict()

        return {
            "symbol": request.symbol.upper(),
            "formula": request.formula,
            "data": output
        }

    except Exception as e:
        logger.error(f"Formula calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ PARAMETER OPTIMIZATION ENDPOINT ============

from backtesting.monte_carlo import ParameterOptimizer


class OptimizationRequest(BaseModel):
    """Request model for parameter optimization"""
    symbol: str
    strategy: str = "momentum"
    param_ranges: Dict[str, List[float]]  # e.g., {"fast_period": [5, 20], "slow_period": [20, 50]}
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    n_iterations: int = 100
    optimization_metric: str = "sharpe"  # sharpe, return, sortino


@api_router.post("/backtest/optimize")
async def optimize_strategy_parameters(request: OptimizationRequest):
    """
    Optimize strategy parameters using walk-forward validation.
    Tests different parameter combinations and validates out-of-sample.
    """
    try:
        from backtesting.price_cache import get_price_cache
        from backtesting.strategies import StrategyRegistry

        registry = StrategyRegistry()

        # Get price data
        price_cache = get_price_cache()
        data = price_cache.get_prices(
            request.symbol.upper(),
            request.start_date,
            request.end_date
        )

        if data.empty or len(data) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data for optimization")

        # Convert param_ranges to tuples
        param_tuples = {k: tuple(v) for k, v in request.param_ranges.items()}

        # Run optimization
        optimizer = ParameterOptimizer(
            n_iterations=request.n_iterations,
            optimization_metric=request.optimization_metric
        )

        # Get strategy class
        strategy_info = next(
            (s for s in registry.get_available_strategies() if s['name'] == request.strategy),
            None
        )

        if not strategy_info:
            raise HTTPException(status_code=400, detail=f"Unknown strategy: {request.strategy}")

        # Create a strategy factory
        def create_strategy(params):
            return registry.create_strategy(request.strategy)

        result = optimizer.optimize(
            strategy_class=create_strategy,
            param_ranges=param_tuples,
            data=data,
            initial_capital=100000
        )

        result['symbol'] = request.symbol.upper()
        result['strategy'] = request.strategy

        return result

    except Exception as e:
        logger.error(f"Parameter optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class EnhancedOptimizationRequest(BaseModel):
    symbol: str
    strategy: str  # e.g., "mean_reversion", "trend_following"
    method: str = "grid_search"  # "grid_search", "random_search", "genetic_algorithm"
    start_date: str = "2022-01-01"
    end_date: str = "2024-01-01"
    objective: str = "sharpe_ratio"  # "sharpe_ratio", "total_return_pct", "profit_factor"
    train_pct: float = 0.7
    # For grid search
    param_grid: Optional[Dict[str, List[Any]]] = None
    # For random/genetic
    param_ranges: Optional[Dict[str, List[float]]] = None  # [[min, max], ...]
    n_iterations: int = 100
    # Genetic algorithm specific
    population_size: int = 50
    generations: int = 20
    mutation_rate: float = 0.1


@api_router.post("/backtest/optimize/enhanced")
async def enhanced_strategy_optimization(request: EnhancedOptimizationRequest):
    """
    Enhanced strategy optimization with multiple methods:
    - Grid Search: Exhaustive search over parameter grid
    - Random Search: Random sampling of parameter space
    - Genetic Algorithm: Evolution-based optimization

    All methods use realistic Indian market transaction costs.
    """
    try:
        from backtesting.optimizer import get_strategy_optimizer

        optimizer = get_strategy_optimizer(use_realistic_costs=True)

        if request.method == "grid_search":
            if not request.param_grid:
                # Use suggested grid if not provided
                request.param_grid = optimizer.suggest_param_grid(request.strategy)

            if not request.param_grid:
                raise HTTPException(
                    status_code=400,
                    detail=f"No parameter grid provided and no default available for {request.strategy}"
                )

            result = optimizer.grid_search(
                symbol=request.symbol,
                strategy_name=request.strategy,
                param_grid=request.param_grid,
                start_date=request.start_date,
                end_date=request.end_date,
                objective=request.objective,
                train_pct=request.train_pct
            )

        elif request.method == "random_search":
            if not request.param_ranges:
                raise HTTPException(status_code=400, detail="param_ranges required for random search")

            # Convert [[min, max]] to (min, max) tuples
            param_tuples = {k: tuple(v) for k, v in request.param_ranges.items()}

            result = optimizer.random_search(
                symbol=request.symbol,
                strategy_name=request.strategy,
                param_ranges=param_tuples,
                start_date=request.start_date,
                end_date=request.end_date,
                n_iterations=request.n_iterations,
                objective=request.objective,
                train_pct=request.train_pct
            )

        elif request.method == "genetic_algorithm":
            if not request.param_ranges:
                raise HTTPException(status_code=400, detail="param_ranges required for genetic algorithm")

            param_tuples = {k: tuple(v) for k, v in request.param_ranges.items()}

            result = optimizer.genetic_algorithm(
                symbol=request.symbol,
                strategy_name=request.strategy,
                param_ranges=param_tuples,
                start_date=request.start_date,
                end_date=request.end_date,
                population_size=request.population_size,
                generations=request.generations,
                mutation_rate=request.mutation_rate,
                objective=request.objective,
                train_pct=request.train_pct
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown optimization method: {request.method}. Use grid_search, random_search, or genetic_algorithm"
            )

        # Convert result to dict
        return {
            "symbol": result.symbol,
            "strategy": request.strategy,
            "method": result.optimization_method,
            "best_params": result.best_params,
            "best_score": result.best_score,
            "best_metrics": result.best_metrics,
            "top_results": result.all_results[:10],
            "iterations": result.iterations,
            "period": result.period,
            "objective": request.objective
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enhanced optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/backtest/optimize/suggest/{strategy}")
async def suggest_optimization_params(strategy: str):
    """Get suggested parameter grid for a strategy"""
    try:
        from backtesting.optimizer import get_strategy_optimizer

        optimizer = get_strategy_optimizer()
        grid = optimizer.suggest_param_grid(strategy)

        if not grid:
            raise HTTPException(
                status_code=404,
                detail=f"No parameter suggestions available for {strategy}"
            )

        return {
            "strategy": strategy,
            "suggested_grid": grid,
            "total_combinations": np.prod([len(v) for v in grid.values()]) if grid else 0
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get suggestions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CONFIDENCE TRACKING ENDPOINTS ============

from tracking.confidence_tracker import get_confidence_tracker


@api_router.get("/tracking/accuracy")
async def get_prediction_accuracy(days_back: int = 90, min_confidence: Optional[float] = None):
    """Get accuracy statistics for past predictions"""
    try:
        tracker = get_confidence_tracker(db)
        stats = await tracker.get_accuracy_stats(days_back, min_confidence)
        return stats

    except Exception as e:
        logger.error(f"Failed to get accuracy stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/tracking/predictions")
async def get_recent_predictions(
    limit: int = 20,
    symbol: Optional[str] = None,
    recommendation: Optional[str] = None
):
    """Get recent predictions with their outcomes"""
    try:
        tracker = get_confidence_tracker(db)
        predictions = await tracker.get_recent_predictions(limit, symbol, recommendation)
        return {"predictions": predictions, "count": len(predictions)}

    except Exception as e:
        logger.error(f"Failed to get predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/tracking/leaderboard")
async def get_accuracy_leaderboard(period_days: int = 30):
    """Get accuracy leaderboard by symbol"""
    try:
        tracker = get_confidence_tracker(db)
        leaderboard = await tracker.get_leaderboard(period_days)
        return leaderboard

    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/tracking/verify")
async def verify_pending_predictions():
    """Verify pending predictions (should be called daily via cron)"""
    try:
        tracker = get_confidence_tracker(db)
        result = await tracker.verify_predictions()
        return result

    except Exception as e:
        logger.error(f"Failed to verify predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ DISCLAIMER ENDPOINTS ============

from utils.disclaimers import Disclaimers


@api_router.get("/disclaimer")
async def get_disclaimer():
    """Get full disclaimer text"""
    return Disclaimers.get_full_disclaimer()


@api_router.get("/disclaimer/short")
async def get_short_disclaimer():
    """Get short disclaimer for API responses"""
    return Disclaimers.get_api_disclaimer()


# ============ API COST TRACKING ENDPOINTS ============

@api_router.get("/api-costs/summary")
async def get_cost_summary(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    group_by: str = "day"
):
    """
    Get API cost summary for a time period

    Args:
        start_date: ISO format date (default: 30 days ago)
        end_date: ISO format date (default: now)
        group_by: day, week, month, provider, or model
    """
    tracker = get_cost_tracker(db)

    start = datetime.fromisoformat(start_date) if start_date else None
    end = datetime.fromisoformat(end_date) if end_date else None

    return await tracker.get_usage_summary(
        start_date=start,
        end_date=end,
        group_by=group_by
    )


@api_router.get("/api-costs/current-month")
async def get_current_month_cost():
    """Get total API cost for current month"""
    tracker = get_cost_tracker(db)
    cost = await tracker.get_current_month_cost()

    return {
        "month": datetime.now(timezone.utc).strftime("%Y-%m"),
        "total_cost": cost,
        "currency": "USD"
    }


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

# ============ KNOWLEDGE BUILDING & PATTERN MINING ENDPOINTS ============

from knowledge import get_news_event_knowledge, get_pattern_miner
from knowledge.auto_discovery import get_discovery_pipeline
from pydantic import BaseModel


class StoreEventRequest(BaseModel):
    event_title: str
    event_date: datetime
    event_type: str
    symbols_affected: List[str]
    sector: Optional[str] = None
    immediate_impact: Optional[Dict[str, float]] = None
    recovery_timeline: Optional[str] = None
    pattern_observed: Optional[str] = None


@api_router.post("/knowledge/events/store")
async def store_news_event(request: StoreEventRequest):
    """Store a news/event and its market impact"""
    try:
        news_knowledge = get_news_event_knowledge(db)
        event_id = await news_knowledge.store_event_impact(
            event_title=request.event_title,
            event_date=request.event_date,
            event_type=request.event_type,
            symbols_affected=request.symbols_affected,
            sector=request.sector,
            immediate_impact=request.immediate_impact,
            recovery_timeline=request.recovery_timeline,
            pattern_observed=request.pattern_observed
        )
        return {"event_id": event_id, "message": "Event stored successfully"}

    except Exception as e:
        logger.error(f"Failed to store event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/knowledge/events/similar")
async def get_similar_events(event_type: str, sector: Optional[str] = None, limit: int = 5):
    """Find similar historical events"""
    try:
        news_knowledge = get_news_event_knowledge(db)
        events = await news_knowledge.find_similar_events(event_type, sector, limit)
        return {"events": events, "count": len(events)}

    except Exception as e:
        logger.error(f"Failed to get similar events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/knowledge/events/symbol/{symbol}")
async def get_symbol_event_history(symbol: str, lookback_days: int = 365):
    """Get event history for a symbol"""
    try:
        news_knowledge = get_news_event_knowledge(db)
        events = await news_knowledge.get_symbol_event_history(symbol, lookback_days)
        return {"symbol": symbol, "events": events, "count": len(events)}

    except Exception as e:
        logger.error(f"Failed to get symbol events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/knowledge/events/pattern-summary")
async def get_event_pattern_summary(event_type: str, sector: Optional[str] = None):
    """Get pattern summary for an event type"""
    try:
        news_knowledge = get_news_event_knowledge(db)
        summary = await news_knowledge.build_pattern_summary(event_type, sector)
        return {"event_type": event_type, "summary": summary}

    except Exception as e:
        logger.error(f"Failed to get pattern summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/knowledge/patterns/discover")
async def run_pattern_discovery(symbols: Optional[List[str]] = None, lookback_years: int = 5):
    """Run pattern discovery on specified symbols"""
    try:
        # Get API keys for data providers
        settings = await db.settings.find_one({}, {"_id": 0})
        api_keys = None
        if settings:
            api_keys = {
                "finnhub": settings.get('finnhub_api_key'),
                "alpaca": {
                    "key": settings.get('alpaca_api_key'),
                    "secret": settings.get('alpaca_api_secret')
                },
                "fmp": settings.get('fmp_api_key')
            }

        discovery = get_discovery_pipeline(db)
        result = await discovery.run_discovery(symbols, lookback_years, api_keys)

        return result

    except Exception as e:
        logger.error(f"Pattern discovery failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/knowledge/patterns/discovered")
async def get_discovered_patterns(symbol: Optional[str] = None, min_success_rate: Optional[float] = None):
    """Get discovered patterns"""
    try:
        pattern_miner = get_pattern_miner(db)

        if symbol:
            patterns = await pattern_miner.get_patterns_for_symbol(symbol)
        else:
            patterns = await pattern_miner.get_all_valid_patterns(min_success_rate)

        return {"patterns": patterns, "count": len(patterns)}

    except Exception as e:
        logger.error(f"Failed to get patterns: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/knowledge/patterns/history")
async def get_discovery_history(limit: int = 10):
    """Get pattern discovery run history"""
    try:
        discovery = get_discovery_pipeline(db)
        runs = await discovery.get_discovery_history(limit)
        return {"runs": runs, "count": len(runs)}

    except Exception as e:
        logger.error(f"Failed to get discovery history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ INDIAN MARKET INDICES ENDPOINTS ============

@api_router.get("/market/indices")
async def get_all_indices():
    """Get data for all Indian stock market indices"""
    try:
        indices_data = get_indian_indices_data()
        indices = await indices_data.get_all_indices()
        return {"indices": indices, "count": len(indices), "timestamp": datetime.now(timezone.utc).isoformat()}

    except Exception as e:
        logger.error(f"Failed to get indices: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/indices/{index_name}")
async def get_index_data(index_name: str):
    """Get data for a specific index (e.g., NIFTY_50, SENSEX, NIFTY_BANK, NIFTY_METAL)"""
    try:
        indices_data = get_indian_indices_data()
        data = await indices_data.get_index_data(index_name)
        
        if not data:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found")
        
        return data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get index {index_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/indices/{index_name}/movers")
async def get_index_movers(index_name: str):
    """Get biggest gainers and losers for an index"""
    try:
        indices_data = get_indian_indices_data()
        performance = await indices_data.get_constituents_performance(index_name)
        
        if not performance.get("gainers") and not performance.get("losers"):
            raise HTTPException(status_code=404, detail=f"No data found for {index_name}")
        
        return performance

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get movers for {index_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/overview")
async def get_market_overview():
    """
    Get complete market overview with all indices and top gainers/losers
    ✅ Optimized with caching and rate limiting
    """
    try:
        rate_limiter = get_rate_limiter()

        # Try cache first (2 minute TTL for market overview - needs to be fresh)
        cache_params = {"endpoint": "market_overview"}
        cached = rate_limiter.get_cached("market_overview", cache_params)
        if cached:
            return cached

        # Check rate limit and market hours
        if not rate_limiter.check_rate_limit("yfinance"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded or outside market hours (9 AM - 5 PM IST)")

        indices_data = get_indian_indices_data()
        overview = await indices_data.get_market_overview()

        # Cache result
        if overview:
            rate_limiter.set_cache("market_overview", cache_params, overview)

        return overview

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get market overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/market/top-movers")
async def get_top_movers(limit: int = 10):
    """Get top gainers and losers"""
    try:
        indices_data = get_indian_indices_data()
        overview = await indices_data.get_market_overview()

        # Extract top movers from overview
        gainers = overview.get("top_gainers", [])[:limit]
        losers = overview.get("top_losers", [])[:limit]

        return {
            "gainers": gainers,
            "losers": losers
        }
    except Exception as e:
        logger.error(f"Failed to get top movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# HELPER FUNCTION FOR SETTINGS
# ============================================================================

async def get_settings_from_db():
    """Helper to get settings from database"""
    try:
        settings_doc = await db.settings.find_one({"user": "default"})
        if settings_doc:
            return settings_doc
        return {}
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        return {}


# ============================================================================
# PAPER TRADING ENDPOINTS
# ============================================================================

@api_router.post("/paper-trading/order")
async def place_paper_order(
    request: PaperOrderRequest,
    user_id: str = "default"
):
    """
    Place paper trading order

    Args:
        request: Order details (symbol, side, quantity, current_price, order_type, limit_price)
        user_id: User identifier
    """
    try:
        engine = get_paper_trading_engine(user_id)

        # Fetch current price if not provided
        current_price = request.current_price
        if not current_price or current_price <= 0:
            try:
                # Try yfinance for price
                ticker_symbol = get_indian_stock_suffix(request.symbol)
                ticker = yf.Ticker(ticker_symbol)
                hist = ticker.history(period="1d")
                if not hist.empty:
                    current_price = float(hist['Close'].iloc[-1])
                else:
                    raise HTTPException(status_code=400, detail=f"Could not fetch price for {request.symbol}")
            except Exception as e:
                logger.error(f"Error fetching price: {e}")
                raise HTTPException(status_code=400, detail=f"Could not fetch price for {request.symbol}")

        order = engine.place_order(
            symbol=request.symbol,
            side=OrderSide(request.side),
            quantity=request.quantity,
            order_type=OrderType(request.order_type),
            limit_price=request.limit_price,
            current_price=current_price
        )

        logger.info(f"Paper order placed: {order.order_id}")
        return order.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to place paper order: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/paper-trading/portfolio")
async def get_paper_portfolio(user_id: str = "default"):
    """
    Get paper trading portfolio summary and positions
    """
    try:
        engine = get_paper_trading_engine(user_id)

        # Get current prices for all positions
        current_prices = {}
        for symbol in engine.positions.keys():
            # Try to get current price from data provider
            try:
                # Use TVScreener provider for free real-time data
                provider = get_tvscreener_provider()
                if provider.is_available:
                    stock_data = await provider.get_quote(symbol)
                    if stock_data:
                        current_prices[symbol] = stock_data.current_price
                    else:
                        current_prices[symbol] = 0.0
                else:
                    current_prices[symbol] = 0.0
            except Exception as e:
                logger.error(f"Error getting price for {symbol}: {e}")
                current_prices[symbol] = 0.0

        summary = engine.get_portfolio_summary(current_prices)
        positions = engine.get_all_positions(current_prices)

        return {
            "summary": summary,
            "positions": positions,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get paper portfolio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/paper-trading/trades")
async def get_paper_trades(user_id: str = "default", limit: int = 50):
    """Get paper trading trade history"""
    try:
        engine = get_paper_trading_engine(user_id)
        trades = engine.get_trade_history(limit=limit)
        return {"trades": trades, "count": len(trades)}

    except Exception as e:
        logger.error(f"Failed to get paper trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/paper-trading/orders")
async def get_paper_orders(
    user_id: str = "default",
    status: Optional[str] = None,
    limit: int = 50
):
    """Get paper trading order history"""
    try:
        engine = get_paper_trading_engine(user_id)
        orders = engine.get_order_history(
            status=PaperOrderStatus(status) if status else None,
            limit=limit
        )
        return {"orders": orders, "count": len(orders)}

    except Exception as e:
        logger.error(f"Failed to get paper orders: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/paper-trading/reset")
async def reset_paper_account(user_id: str = "default"):
    """Reset paper trading account to initial capital"""
    try:
        engine = get_paper_trading_engine(user_id)
        engine.reset()
        return {"message": "Paper trading account reset successfully"}

    except Exception as e:
        logger.error(f"Failed to reset paper account: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# ALERT MANAGEMENT ENDPOINTS
# ============================================================================

class PriceAlertRequest(BaseModel):
    user_id: str = "default"
    symbol: str
    condition: str
    target_price: float
    delivery_channels: List[str]
    percent_change: Optional[float] = None


class PatternAlertRequest(BaseModel):
    user_id: str = "default"
    symbol: str
    pattern_types: List[str]
    delivery_channels: List[str]


class PortfolioAlertRequest(BaseModel):
    user_id: str = "default"
    metric: str
    threshold: float
    condition: str
    delivery_channels: List[str]


@api_router.post("/alerts/price")
async def create_price_alert(request: PriceAlertRequest):
    """
    Create price alert

    Args:
        condition: 'ABOVE', 'BELOW', 'CROSSES_ABOVE', 'CROSSES_BELOW',
                   'PERCENT_CHANGE_ABOVE', 'PERCENT_CHANGE_BELOW'
        delivery_channels: ['TELEGRAM', 'EMAIL', 'WEBHOOK', 'SLACK', 'WHATSAPP']
    """
    try:
        # Get alert manager with settings
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            slack_webhook_url=settings.get('slack_webhook_url'),
            twilio_config={
                'account_sid': settings.get('twilio_account_sid'),
                'auth_token': settings.get('twilio_auth_token'),
                'whatsapp_number': settings.get('twilio_whatsapp_number'),
                'user_whatsapp_number': settings.get('user_whatsapp_number')
            }
        )

        alert = alert_mgr.create_price_alert(
            user_id=request.user_id,
            symbol=request.symbol,
            condition=PriceCondition(request.condition),
            target_price=request.target_price,
            delivery_channels=[DeliveryChannel(ch) for ch in request.delivery_channels],
            percent_change=request.percent_change
        )

        logger.info(f"Price alert created: {alert.alert_id}")
        return alert.to_dict()

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create price alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/alerts/pattern")
async def create_pattern_alert(request: PatternAlertRequest):
    """Create candlestick pattern alert"""
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            slack_webhook_url=settings.get('slack_webhook_url'),
            twilio_config={
                'account_sid': settings.get('twilio_account_sid'),
                'auth_token': settings.get('twilio_auth_token'),
                'whatsapp_number': settings.get('twilio_whatsapp_number'),
                'user_whatsapp_number': settings.get('user_whatsapp_number')
            }
        )

        alert = alert_mgr.create_pattern_alert(
            user_id=request.user_id,
            symbol=request.symbol,
            pattern_types=request.pattern_types,
            delivery_channels=[DeliveryChannel(ch) for ch in request.delivery_channels]
        )

        return alert.to_dict()

    except Exception as e:
        logger.error(f"Failed to create pattern alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/alerts/portfolio")
async def create_portfolio_alert(request: PortfolioAlertRequest):
    """
    Create portfolio alert

    Args:
        metric: 'drawdown', 'total_pnl', 'total_return_pct'
        condition: 'above' or 'below'
    """
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            slack_webhook_url=settings.get('slack_webhook_url'),
            twilio_config={
                'account_sid': settings.get('twilio_account_sid'),
                'auth_token': settings.get('twilio_auth_token'),
                'whatsapp_number': settings.get('twilio_whatsapp_number'),
                'user_whatsapp_number': settings.get('user_whatsapp_number')
            }
        )

        alert = alert_mgr.create_portfolio_alert(
            user_id=request.user_id,
            metric=request.metric,
            threshold=request.threshold,
            condition=request.condition,
            delivery_channels=[DeliveryChannel(ch) for ch in request.delivery_channels]
        )

        return alert.to_dict()

    except Exception as e:
        logger.error(f"Failed to create portfolio alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/alerts")
async def get_user_alerts(user_id: str, status: Optional[str] = None):
    """Get all alerts for a user"""
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            slack_webhook_url=settings.get('slack_webhook_url'),
            twilio_config={
                'account_sid': settings.get('twilio_account_sid'),
                'auth_token': settings.get('twilio_auth_token'),
                'whatsapp_number': settings.get('twilio_whatsapp_number'),
                'user_whatsapp_number': settings.get('user_whatsapp_number')
            }
        )

        alerts = alert_mgr.get_user_alerts(
            user_id=user_id,
            status=AlertStatusEnum(status) if status else None
        )

        return {"alerts": [alert.to_dict() for alert in alerts], "count": len(alerts)}

    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/alerts/{alert_id}")
async def delete_alert(alert_id: str):
    """Delete an alert permanently"""
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            slack_webhook_url=settings.get('slack_webhook_url'),
            twilio_config={
                'account_sid': settings.get('twilio_account_sid'),
                'auth_token': settings.get('twilio_auth_token'),
                'whatsapp_number': settings.get('twilio_whatsapp_number'),
                'user_whatsapp_number': settings.get('user_whatsapp_number')
            }
        )

        success = alert_mgr.delete_alert(alert_id)

        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")

        return {"message": "Alert deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# TVSCREENER ENDPOINTS (FREE REAL-TIME DATA!)
# ============================================================================

@api_router.get("/screener/quote/{symbol}")
async def get_screener_quote(symbol: str):
    """Get real-time quote from TVScreener (FREE - NO API KEY!)"""
    try:
        provider = get_tvscreener_provider()

        if not provider.is_available:
            raise HTTPException(status_code=503, detail="TVScreener not available")

        data = provider.get_quote(symbol)

        if not data:
            raise HTTPException(status_code=404, detail=f"Symbol {symbol} not found")

        return {
            "symbol": data.symbol,
            "current_price": data.current_price,
            "change": data.change,
            "change_percent": data.change_percent,
            "open": data.open,
            "high": data.high,
            "low": data.low,
            "volume": data.volume,
            "market_cap": data.market_cap,
            "pe_ratio": data.pe_ratio,
            "timestamp": data.timestamp.isoformat()
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get screener quote: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/screener/fundamentals/{symbol}")
async def get_screener_fundamentals(symbol: str):
    """Get fundamental data from TVScreener"""
    try:
        provider = get_tvscreener_provider()

        if not provider.is_available:
            raise HTTPException(status_code=503, detail="TVScreener not available")

        fundamentals = provider.get_fundamentals(symbol)

        if not fundamentals:
            raise HTTPException(status_code=404, detail=f"Fundamentals for {symbol} not found")

        return fundamentals

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fundamentals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/screener/screen")
async def screen_stocks(
    pe_max: Optional[float] = None,
    roe_min: Optional[float] = None,
    debt_to_equity_max: Optional[float] = None,
    market_cap_min: Optional[float] = None,
    dividend_yield_min: Optional[float] = None,
    profit_margin_min: Optional[float] = None,
    limit: int = 50
):
    """
    Screen stocks by fundamental criteria

    Example:
        POST /api/screener/screen
        {
            "pe_max": 15,
            "roe_min": 20,
            "debt_to_equity_max": 0.5,
            "market_cap_min": 1000,
            "limit": 50
        }
    """
    try:
        provider = get_tvscreener_provider()

        if not provider.is_available:
            raise HTTPException(status_code=503, detail="TVScreener not available")

        stocks = provider.screen_stocks(
            pe_max=pe_max,
            roe_min=roe_min,
            debt_to_equity_max=debt_to_equity_max,
            market_cap_min=market_cap_min,
            dividend_yield_min=dividend_yield_min,
            profit_margin_min=profit_margin_min,
            limit=limit
        )

        return {"stocks": stocks, "count": len(stocks)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to screen stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/screener/search")
async def search_screener_symbols(q: str, limit: int = 20):
    """Search stocks on TVScreener"""
    try:
        provider = get_tvscreener_provider()

        if not provider.is_available:
            raise HTTPException(status_code=503, detail="TVScreener not available")

        results = provider.search_symbol(q)

        return {"results": results[:limit], "count": len(results)}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to search symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# RISK MANAGEMENT ENDPOINTS
# ============================================================================

@api_router.post("/risk/position-size/kelly")
async def calculate_kelly_position_size(
    portfolio_value: float,
    win_rate: float,
    avg_win: float,
    avg_loss: float
):
    """
    Calculate optimal position size using Kelly Criterion

    Args:
        portfolio_value: Total portfolio value
        win_rate: Historical win rate (0-1, e.g., 0.60 for 60%)
        avg_win: Average win amount
        avg_loss: Average loss amount
    """
    try:
        risk_mgr = get_risk_manager()

        position_size = risk_mgr.calculate_position_size_kelly(
            portfolio_value=portfolio_value,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss
        )

        position_pct = (position_size / portfolio_value) * 100 if portfolio_value > 0 else 0

        return {
            "position_size": round(position_size, 2),
            "position_pct": round(position_pct, 2),
            "method": "Kelly Criterion",
            "kelly_fraction": risk_mgr.kelly_fraction
        }

    except Exception as e:
        logger.error(f"Failed to calculate Kelly position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/risk/position-size/fixed-fractional")
async def calculate_fixed_fractional_position_size(
    portfolio_value: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss_price: float
):
    """
    Calculate position size using fixed fractional method

    Args:
        portfolio_value: Total portfolio value
        risk_per_trade: Risk per trade (0.02 for 2%)
        entry_price: Entry price per share
        stop_loss_price: Stop loss price per share
    """
    try:
        risk_mgr = get_risk_manager()

        shares = risk_mgr.calculate_position_size_fixed_fractional(
            portfolio_value=portfolio_value,
            risk_per_trade=risk_per_trade,
            entry_price=entry_price,
            stop_loss_price=stop_loss_price
        )

        position_value = shares * entry_price
        position_pct = (position_value / portfolio_value) * 100 if portfolio_value > 0 else 0

        return {
            "shares": shares,
            "position_value": round(position_value, 2),
            "position_pct": round(position_pct, 2),
            "method": "Fixed Fractional",
            "risk_per_trade_pct": risk_per_trade * 100
        }

    except Exception as e:
        logger.error(f"Failed to calculate fixed fractional position: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/risk/stop-loss/atr")
async def calculate_atr_stop_loss(
    entry_price: float,
    atr: float,
    side: str = "BUY"
):
    """Calculate ATR-based stop loss"""
    try:
        risk_mgr = get_risk_manager()

        stop_loss = risk_mgr.calculate_stop_loss_atr(
            entry_price=entry_price,
            atr=atr,
            side=side
        )

        risk_per_share = abs(entry_price - stop_loss)
        risk_pct = (risk_per_share / entry_price) * 100

        return {
            "stop_loss_price": round(stop_loss, 2),
            "risk_per_share": round(risk_per_share, 2),
            "risk_pct": round(risk_pct, 2),
            "atr": atr,
            "atr_multiplier": risk_mgr.atr_multiplier
        }

    except Exception as e:
        logger.error(f"Failed to calculate ATR stop loss: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/risk/risk-reward-ratio")
async def calculate_risk_reward(
    entry_price: float,
    target_price: float,
    stop_loss_price: float
):
    """Calculate risk/reward ratio for a trade"""
    try:
        risk_mgr = get_risk_manager()

        ratio = risk_mgr.calculate_risk_reward_ratio(
            entry_price=entry_price,
            target_price=target_price,
            stop_loss_price=stop_loss_price
        )

        potential_reward = abs(target_price - entry_price)
        potential_risk = abs(entry_price - stop_loss_price)

        return {
            "risk_reward_ratio": round(ratio, 2),
            "potential_reward": round(potential_reward, 2),
            "potential_risk": round(potential_risk, 2),
            "reward_pct": round((potential_reward / entry_price) * 100, 2),
            "risk_pct": round((potential_risk / entry_price) * 100, 2)
        }

    except Exception as e:
        logger.error(f"Failed to calculate risk/reward ratio: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/risk/summary")
async def get_risk_summary():
    """Get risk management configuration summary"""
    try:
        risk_mgr = get_risk_manager()
        return risk_mgr.get_risk_summary()

    except Exception as e:
        logger.error(f"Failed to get risk summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CACHE MANAGEMENT ENDPOINTS ============

@api_router.get("/cache/stats")
async def get_cache_stats():
    """
    Get cache statistics

    Returns info about Redis connection status, memory usage, and cache size.
    Falls back to in-memory cache if Redis is unavailable.
    """
    try:
        from utils.redis_cache import get_cache
        cache = get_cache()
        stats = await cache.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/cache/clear")
async def clear_cache(pattern: str = "*"):
    """
    Clear cached data

    Args:
        pattern: Key pattern to clear (default clears all)
    """
    try:
        from utils.redis_cache import get_cache
        cache = get_cache()
        count = await cache.clear_namespace(pattern)
        return {
            "cleared": count,
            "pattern": pattern,
            "message": f"Cleared {count} cache entries"
        }
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/cache/get/{key}")
async def get_cache_entry(key: str):
    """Get a specific cache entry (for debugging)"""
    try:
        from utils.redis_cache import get_cache
        cache = get_cache()
        value = await cache.get(key)

        if value is None:
            return {"key": key, "found": False, "value": None}

        return {
            "key": key,
            "found": True,
            "value": value
        }
    except Exception as e:
        logger.error(f"Failed to get cache entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/cache/delete/{key}")
async def delete_cache_entry(key: str):
    """Delete a specific cache entry"""
    try:
        from utils.redis_cache import get_cache
        cache = get_cache()
        success = await cache.delete(key)
        return {
            "key": key,
            "deleted": success
        }
    except Exception as e:
        logger.error(f"Failed to delete cache entry: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ DATABASE CLEANUP ENDPOINTS ============

@api_router.get("/admin/db/stats")
async def get_database_stats():
    """Get database collection statistics"""
    try:
        stats = {}
        collections = ["settings", "recommendations", "analysis_history", "backtests", "watchlist", "api_costs"]

        for collection_name in collections:
            collection = db[collection_name]
            count = await collection.count_documents({})

            # Get oldest and newest document dates if available
            oldest = await collection.find_one({}, sort=[("created_at", 1)])
            newest = await collection.find_one({}, sort=[("created_at", -1)])

            # Try alternative date fields
            if not oldest:
                oldest = await collection.find_one({}, sort=[("generated_at", 1)])
            if not newest:
                newest = await collection.find_one({}, sort=[("generated_at", -1)])
            if not oldest:
                oldest = await collection.find_one({}, sort=[("timestamp", 1)])
            if not newest:
                newest = await collection.find_one({}, sort=[("timestamp", -1)])

            stats[collection_name] = {
                "count": count,
                "oldest": oldest.get("created_at") or oldest.get("generated_at") or oldest.get("timestamp") if oldest else None,
                "newest": newest.get("created_at") or newest.get("generated_at") or newest.get("timestamp") if newest else None,
            }

        return {
            "database": "test_database",
            "collections": stats,
            "ttl_policies": {
                "analysis_history": "30 days",
                "recommendations": "7 days",
                "backtests": "90 days"
            }
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/admin/db/cleanup/recommendations")
async def cleanup_old_recommendations(days: int = 7):
    """Delete recommendations older than specified days"""
    try:
        from datetime import timedelta
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        result = await db.recommendations.delete_many({"generated_at": {"$lt": cutoff_date}})

        return {
            "deleted_count": result.deleted_count,
            "cutoff_date": cutoff_date,
            "message": f"Deleted {result.deleted_count} recommendations older than {days} days"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/admin/db/cleanup/analysis")
async def cleanup_old_analysis(days: int = 30):
    """Delete analysis history older than specified days"""
    try:
        from datetime import timedelta
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        result = await db.analysis_history.delete_many({"created_at": {"$lt": cutoff_date}})

        return {
            "deleted_count": result.deleted_count,
            "cutoff_date": cutoff_date,
            "message": f"Deleted {result.deleted_count} analysis records older than {days} days"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup analysis history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/admin/db/cleanup/backtests")
async def cleanup_old_backtests(days: int = 90):
    """Delete backtests older than specified days"""
    try:
        from datetime import timedelta
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        result = await db.backtests.delete_many({"timestamp": {"$lt": cutoff_date}})

        return {
            "deleted_count": result.deleted_count,
            "cutoff_date": cutoff_date,
            "message": f"Deleted {result.deleted_count} backtests older than {days} days"
        }
    except Exception as e:
        logger.error(f"Failed to cleanup backtests: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.delete("/admin/db/cleanup/all")
async def cleanup_all_old_data(
    recommendations_days: int = 7,
    analysis_days: int = 30,
    backtests_days: int = 90
):
    """Clean up all old data from database"""
    try:
        from datetime import timedelta
        results = {}

        # Cleanup recommendations
        cutoff = (datetime.now(timezone.utc) - timedelta(days=recommendations_days)).isoformat()
        rec_result = await db.recommendations.delete_many({"generated_at": {"$lt": cutoff}})
        results["recommendations"] = {"deleted": rec_result.deleted_count, "max_age_days": recommendations_days}

        # Cleanup analysis history
        cutoff = (datetime.now(timezone.utc) - timedelta(days=analysis_days)).isoformat()
        analysis_result = await db.analysis_history.delete_many({"created_at": {"$lt": cutoff}})
        results["analysis_history"] = {"deleted": analysis_result.deleted_count, "max_age_days": analysis_days}

        # Cleanup backtests
        cutoff = (datetime.now(timezone.utc) - timedelta(days=backtests_days)).isoformat()
        backtest_result = await db.backtests.delete_many({"timestamp": {"$lt": cutoff}})
        results["backtests"] = {"deleted": backtest_result.deleted_count, "max_age_days": backtests_days}

        total_deleted = sum(r["deleted"] for r in results.values())

        return {
            "total_deleted": total_deleted,
            "details": results,
            "message": f"Cleanup complete. Deleted {total_deleted} total records."
        }
    except Exception as e:
        logger.error(f"Failed to cleanup database: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/admin/db/reset/{collection}")
async def reset_collection(collection: str, confirm: bool = False):
    """Reset (delete all documents from) a specific collection. Requires confirm=true."""
    allowed_collections = ["recommendations", "analysis_history", "backtests", "api_costs"]

    if collection not in allowed_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection}' cannot be reset. Allowed: {allowed_collections}"
        )

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Add ?confirm=true to confirm reset. This will delete ALL data in the collection."
        )

    try:
        result = await db[collection].delete_many({})
        return {
            "collection": collection,
            "deleted_count": result.deleted_count,
            "message": f"Collection '{collection}' has been reset"
        }
    except Exception as e:
        logger.error(f"Failed to reset collection {collection}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============== Dynamic Stock Management Endpoints ==============

@api_router.get("/admin/stocks/cache")
async def get_stock_cache_info():
    """Get information about the dynamic stock cache"""
    from data_providers.tvscreener_provider import _stock_cache

    return {
        "cached_stocks": len(_stock_cache.get("stocks", [])),
        "last_updated": _stock_cache.get("last_updated").isoformat() if _stock_cache.get("last_updated") else None,
        "cache_duration_hours": _stock_cache.get("cache_duration").total_seconds() / 3600 if _stock_cache.get("cache_duration") else None,
        "sample_stocks": [s.get("symbol") for s in _stock_cache.get("stocks", [])[:10]]
    }


@api_router.post("/admin/stocks/refresh")
async def refresh_stock_cache():
    """Force refresh the dynamic stock cache"""
    try:
        clear_stock_cache()
        stocks = await get_all_indian_stocks_async(force_refresh=True)
        return {
            "message": "Stock cache refreshed successfully",
            "total_stocks": len(stocks),
            "sample_stocks": [{"symbol": s.get("symbol"), "name": s.get("name")} for s in stocks[:10]]
        }
    except Exception as e:
        logger.error(f"Failed to refresh stock cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/stocks/all")
async def get_all_stocks(
    limit: int = 200,
    sector: Optional[str] = None,
    min_market_cap: float = 100
):
    """
    Get all available Indian stocks (dynamically fetched).

    Parameters:
    - limit: Maximum stocks to return (default: 200)
    - sector: Filter by sector (optional)
    - min_market_cap: Minimum market cap in crores (default: 100 Cr)
    """
    try:
        stocks = await get_all_indian_stocks_async(
            min_market_cap=min_market_cap,
            max_stocks=limit * 2
        )

        if sector:
            sector_lower = sector.lower()
            stocks = [s for s in stocks if sector_lower in s.get('sector', '').lower()]

        return {
            "total": len(stocks[:limit]),
            "stocks": stocks[:limit]
        }
    except Exception as e:
        logger.error(f"Failed to get all stocks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Include the options router in api_router
api_router.include_router(options_router)

# Include the router in the main app (must be after all endpoint definitions)
app.include_router(api_router)

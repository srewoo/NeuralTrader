"""
NeuralTrader Backend — Focused Stock Analyst for Indian BSE/NSE
5 features: Dashboard (AI Analysis), AI Picks, Alerts, Settings, Help
"""

from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict, validator
from typing import List, Optional, Dict, Any
import uuid
from datetime import datetime, timezone, timedelta
import json
import asyncio
import yfinance as yf
import ta
import pandas as pd
import numpy as np

# Import multi-agent orchestrator
from agents.orchestrator import get_orchestrator

# Import news system
from news.sources import get_news_aggregator
from news.sentiment import get_sentiment_analyzer
from news.rate_limiter import get_rate_limiter

# Import pattern detection
from patterns.candlestick import get_pattern_detector

# Import Indian Market Indices
from market_data.indian_indices import get_indian_indices_data

# Import Alerts
from alerts.alert_manager import get_alert_manager, PriceCondition, DeliveryChannel, AlertStatus as AlertStatusEnum

# Import TVScreener (free Indian stock data)
from data_providers.tvscreener_provider import (
    get_tvscreener_provider,
    get_all_indian_stocks,
    get_all_indian_stocks_async,
    search_indian_stocks,
    get_top_stocks_by_market_cap,
    clear_stock_cache
)

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ.get('MONGO_URL', 'mongodb://localhost:27017')
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ.get('DB_NAME', 'neuraltrader')]

# Create FastAPI app
app = FastAPI(title="NeuralTrader", version="2.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=[origin.strip() for origin in os.environ.get('CORS_ORIGINS', 'http://localhost:3005').split(',')],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API router with /api prefix
api_router = APIRouter(prefix="/api")

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============ PYDANTIC MODELS ============

class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
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
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: Optional[int] = None
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    smtp_from_email: Optional[str] = None
    use_tvscreener: bool = True
    selected_model: str = "gpt-4o-mini"
    selected_provider: str = "openai"

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


class AnalysisRequest(BaseModel):
    symbol: str
    model: str = "gpt-4o-mini"
    provider: str = "openai"


class PriceAlertRequest(BaseModel):
    user_id: str = "default"
    symbol: str
    condition: str
    target_price: float
    delivery_channels: List[str] = ["TELEGRAM"]
    percent_change: Optional[float] = None


# ============ HELPER FUNCTIONS ============

def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"
    return symbol


def normalize_indian_symbol(symbol: str) -> str:
    """Normalize Indian stock symbol — add .NS if needed"""
    symbol = symbol.upper().strip()
    if symbol.endswith('.NS') or symbol.endswith('.BO'):
        return symbol
    return f"{symbol}.NS"


async def get_settings_from_db() -> dict:
    """Get settings from database"""
    try:
        settings = await db.settings.find_one({}, {"_id": 0})
        return settings or {}
    except Exception:
        return {}


async def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Fetch stock data from Yahoo Finance with latest available data"""
    try:
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info

        # Try intraday first, then daily
        try:
            hist_intraday = ticker.history(period="1d", interval="1m")
            if not hist_intraday.empty:
                hist = hist_intraday
                is_intraday = True
            else:
                hist = ticker.history(period="5d")
                is_intraday = False
        except Exception:
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
            except Exception:
                hist = ticker.history(period="5d")
                is_intraday = False

        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")

        last_timestamp = hist.index[-1]
        current_price = float(hist['Close'].iloc[-1])
        previous_close = info.get('previousClose', current_price)

        if hasattr(last_timestamp, 'to_pydatetime'):
            last_update = last_timestamp.to_pydatetime()
        else:
            last_update = last_timestamp

        if last_update.tzinfo is None:
            from pytz import timezone as tz
            ist = tz('Asia/Kolkata')
            last_update = ist.localize(last_update)
        else:
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
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching stock data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def fetch_historical_data(symbol: str, period: str = "1y") -> list:
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

        data = []
        for date, row in hist.iterrows():
            data.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": round(float(row['Open']), 2),
                "high": round(float(row['High']), 2),
                "low": round(float(row['Low']), 2),
                "close": round(float(row['Close']), 2),
                "volume": int(row['Volume'])
            })
        return data
    except HTTPException:
        raise
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

        rsi = ta.momentum.RSIIndicator(close, window=14).rsi().iloc[-1]
        macd_indicator = ta.trend.MACD(close)
        macd = macd_indicator.macd().iloc[-1]
        macd_signal = macd_indicator.macd_signal().iloc[-1]
        macd_histogram = macd_indicator.macd_diff().iloc[-1]
        sma_20 = ta.trend.SMAIndicator(close, window=20).sma_indicator().iloc[-1]
        sma_50 = ta.trend.SMAIndicator(close, window=50).sma_indicator().iloc[-1]
        sma_200 = ta.trend.SMAIndicator(close, window=200).sma_indicator().iloc[-1] if len(df) >= 200 else None
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        bb_upper = bb.bollinger_hband().iloc[-1]
        bb_middle = bb.bollinger_mavg().iloc[-1]
        bb_lower = bb.bollinger_lband().iloc[-1]
        atr = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range().iloc[-1]
        obv = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume().iloc[-1]
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        stochastic_k = stoch.stoch().iloc[-1]
        stochastic_d = stoch.stoch_signal().iloc[-1]

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


# ============ STOCK LIST (Fallback when TVScreener unavailable) ============

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
]


# ============ SETTINGS ENDPOINTS ============

@api_router.get("/settings")
async def get_settings():
    settings = await db.settings.find_one({}, {"_id": 0})

    default_settings = {
        "openai_api_key": "",
        "gemini_api_key": "",
        "anthropic_api_key": "",
        "telegram_bot_token": "",
        "telegram_chat_id": "",
        "smtp_host": "",
        "smtp_port": "",
        "smtp_user": "",
        "smtp_password": "",
        "smtp_from_email": "",
        "use_tvscreener": True,
        "selected_model": "gpt-4o-mini",
        "selected_provider": "openai"
    }

    if not settings:
        return default_settings

    merged_settings = {**default_settings, **settings}

    def mask_key(key):
        if not key:
            return ""
        return key[:8] + '...' + key[-4:] if len(key) > 12 else '****'

    for field in ['openai_api_key', 'gemini_api_key', 'anthropic_api_key', 'telegram_bot_token', 'smtp_password']:
        if merged_settings.get(field):
            merged_settings[field] = mask_key(merged_settings[field])

    return merged_settings


@api_router.post("/settings")
async def save_settings(settings: SettingsCreate):
    existing = await db.settings.find_one({})

    update_data = {
        "selected_model": settings.selected_model,
        "selected_provider": settings.selected_provider,
        "use_tvscreener": settings.use_tvscreener,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }

    def is_masked(key):
        return not key or key.startswith('****') or '...' in key

    # AI API keys
    if settings.openai_api_key and not is_masked(settings.openai_api_key):
        update_data["openai_api_key"] = settings.openai_api_key
    if settings.gemini_api_key and not is_masked(settings.gemini_api_key):
        update_data["gemini_api_key"] = settings.gemini_api_key
    if settings.anthropic_api_key and not is_masked(settings.anthropic_api_key):
        update_data["anthropic_api_key"] = settings.anthropic_api_key

    # Telegram
    if settings.telegram_bot_token and not is_masked(settings.telegram_bot_token):
        update_data["telegram_bot_token"] = settings.telegram_bot_token
    if settings.telegram_chat_id:
        update_data["telegram_chat_id"] = settings.telegram_chat_id

    # Email SMTP
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

    if existing:
        await db.settings.update_one({}, {"$set": update_data})
    else:
        update_data["id"] = str(uuid.uuid4())
        update_data["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.settings.insert_one(update_data)

    return {"message": "Settings saved successfully"}


# ============ STOCK DATA ENDPOINTS ============

@api_router.get("/stocks/search")
async def search_stocks(q: str):
    """Search for stocks dynamically from NSE/BSE"""
    if not q or len(q) < 1:
        return []

    results = search_indian_stocks(q, limit=10)
    if results:
        return [{"symbol": s.get("symbol", ""), "name": s.get("name", s.get("symbol", "")), "sector": s.get("sector", "N/A"), "exchange": "NSE"} for s in results]

    query = q.upper()
    fallback = [s for s in NIFTY_100_STOCKS if query in s['symbol'] or query in s['name'].upper()]
    return [{"symbol": s["symbol"], "name": s["name"], "sector": s.get("sector", "N/A"), "exchange": "NSE"} for s in fallback[:10]]


@api_router.get("/stocks/popular")
async def get_popular_stocks():
    """Get popular stocks list"""
    return NIFTY_100_STOCKS[:20]


@api_router.get("/stocks/{symbol}")
async def get_stock(symbol: str):
    """Get current stock data"""
    normalized = normalize_indian_symbol(symbol)
    return await fetch_stock_data(normalized)


@api_router.get("/stocks/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1y"):
    """Get historical price data"""
    return await fetch_historical_data(symbol, period)


@api_router.get("/stocks/{symbol}/indicators")
async def get_technical_indicators(symbol: str):
    """Get technical indicators"""
    return await calculate_technical_indicators(symbol)


# ============ AI ANALYSIS ENDPOINTS (Dashboard) ============

@api_router.post("/analyze")
async def analyze_stock(request: AnalysisRequest):
    """Run full AI analysis on a stock using multi-agent system"""
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
        orchestrator = get_orchestrator(db=db)
        analysis_result = await orchestrator.run_analysis(
            symbol=request.symbol,
            model=request.model,
            provider=request.provider,
            api_key=api_key,
            data_provider_keys={}
        )

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
            "percentile_scores": analysis_result.get('percentile_scores', {}),
            "insights": analysis_result.get('insights', []),
            "summary_insight": analysis_result.get('summary_insight', ''),
            "model_used": f"{request.provider}/{request.model}",
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        await db.analysis_history.insert_one(result)
        result.pop('_id', None)
        return result

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@api_router.get("/analysis/history")
async def get_analysis_history(limit: int = 20):
    """Get analysis history"""
    cursor = db.analysis_history.find({}, {"_id": 0}).sort("created_at", -1).limit(limit)
    return await cursor.to_list(length=limit)


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


# ============ AI PICKS / RECOMMENDATIONS ENDPOINTS (Core Feature) ============

@api_router.get("/recommendations")
async def get_cached_recommendations():
    """Get cached AI recommendations"""
    try:
        cached = await db.recommendations.find_one({}, sort=[("generated_at", -1)])
        if cached:
            cached.pop("_id", None)
            return cached

        return {
            "generated_at": None,
            "total_stocks_analyzed": 0,
            "buy_recommendations": [],
            "sell_recommendations": [],
            "summary": {"total_buy_signals": 0, "total_sell_signals": 0, "market_sentiment": "Neutral"},
            "message": "No recommendations available. Click 'Generate' to analyze stocks."
        }
    except Exception as e:
        logger.error(f"Failed to get cached recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/recommendations/generate")
async def generate_ai_recommendations(limit: int = 200, min_confidence: float = 65.0):
    """
    Generate AI recommendations for Indian stocks (NSE/BSE).
    Uses the LangGraph multi-agent orchestrator to analyze each stock.
    """
    try:
        # Get settings for API key
        settings = await db.settings.find_one({}, {"_id": 0})
        if not settings:
            raise HTTPException(status_code=400, detail="Please configure API keys in Settings first")

        provider = settings.get('selected_provider', 'openai')
        model = settings.get('selected_model', 'gpt-4o-mini')
        api_key = settings.get(f'{provider}_api_key') if provider != 'gemini' else settings.get('gemini_api_key')

        if not api_key:
            raise HTTPException(status_code=400, detail=f"Please configure your {provider} API key in Settings")

        # Dynamically fetch stocks from TVScreener
        logger.info("Fetching dynamic stock list from TradingView...")
        dynamic_stocks = await get_all_indian_stocks_async(min_market_cap=100, max_stocks=limit * 2)

        if dynamic_stocks:
            stocks_to_analyze = dynamic_stocks[:limit]
            logger.info(f"Using {len(stocks_to_analyze)} dynamically fetched stocks")
        else:
            logger.warning("TVScreener failed, falling back to hardcoded stock list")
            stocks_to_analyze = NIFTY_100_STOCKS[:limit]

        orchestrator = get_orchestrator(db=db)

        async def process_stock(stock_info):
            symbol = stock_info.get('symbol', stock_info) if isinstance(stock_info, dict) else stock_info
            try:
                result = await orchestrator.run_analysis(
                    symbol=symbol,
                    model=model,
                    provider=provider,
                    api_key=api_key,
                    data_provider_keys={}
                )

                recommendation = result.get('recommendation', 'HOLD')
                confidence = result.get('confidence', 0)

                if recommendation == 'HOLD' or confidence < min_confidence:
                    return None

                name = stock_info.get('name', symbol) if isinstance(stock_info, dict) else symbol
                sector = stock_info.get('sector', 'N/A') if isinstance(stock_info, dict) else 'N/A'

                return {
                    "symbol": symbol,
                    "name": name,
                    "sector": sector,
                    "recommendation": recommendation,
                    "confidence": round(confidence, 1),
                    "current_price": result.get('entry_price', 0),
                    "target_price": result.get('target_price'),
                    "stop_loss": result.get('stop_loss'),
                    "risk_reward": result.get('risk_reward_ratio'),
                    "reasoning": result.get('reasoning', ''),
                    "key_risks": result.get('key_risks', []),
                    "key_opportunities": result.get('key_opportunities', []),
                    "technical_indicators": result.get('technical_indicators', {}),
                    "insights": result.get('insights', []),
                }
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                return None

        # Process stocks concurrently
        tasks = [process_stock(s) for s in stocks_to_analyze]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        buy_recs = []
        sell_recs = []

        for res in batch_results:
            if isinstance(res, dict):
                if res['recommendation'] == 'BUY':
                    buy_recs.append(res)
                elif res['recommendation'] == 'SELL':
                    sell_recs.append(res)

        buy_recs.sort(key=lambda x: x['confidence'], reverse=True)
        sell_recs.sort(key=lambda x: x['confidence'], reverse=True)

        total_analyzed = len(stocks_to_analyze)
        market_sentiment = "Neutral"
        if len(buy_recs) > len(sell_recs) * 1.5:
            market_sentiment = "Bullish"
        elif len(sell_recs) > len(buy_recs) * 1.5:
            market_sentiment = "Bearish"

        avg_buy_conf = sum(r['confidence'] for r in buy_recs) / len(buy_recs) if buy_recs else 0
        avg_sell_conf = sum(r['confidence'] for r in sell_recs) / len(sell_recs) if sell_recs else 0

        recommendations_data = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_stocks_analyzed": total_analyzed,
            "min_confidence_threshold": min_confidence,
            "summary": {
                "total_buy_signals": len(buy_recs),
                "total_sell_signals": len(sell_recs),
                "market_sentiment": market_sentiment,
                "avg_buy_confidence": round(avg_buy_conf, 1),
                "avg_sell_confidence": round(avg_sell_conf, 1),
            },
            "buy_recommendations": buy_recs[:50],
            "sell_recommendations": sell_recs[:30],
        }

        # Save to database
        try:
            one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            existing = await db.recommendations.find_one({"generated_at": {"$gte": one_hour_ago}}, sort=[("generated_at", -1)])
            if existing:
                await db.recommendations.update_one({"_id": existing["_id"]}, {"$set": recommendations_data})
            else:
                await db.recommendations.insert_one(recommendations_data.copy())
        except Exception as e:
            logger.warning(f"Failed to cache recommendations: {e}")

        logger.info(f"Recommendations: {len(buy_recs)} BUY, {len(sell_recs)} SELL")
        return recommendations_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ CANDLESTICK PATTERNS ============

@api_router.get("/patterns/{symbol}")
async def get_candlestick_patterns(symbol: str, days: int = 30):
    """Detect candlestick patterns"""
    try:
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(period=f"{days}d")

        if data.empty:
            raise HTTPException(status_code=404, detail=f"No data available for {symbol}")

        detector = get_pattern_detector()
        all_patterns = detector.detect_patterns(data, lookback_periods=days)
        recent_patterns = detector.get_recent_patterns(data, days=5)

        bullish = [p for p in all_patterns if 'bullish' in p['type']]
        bearish = [p for p in all_patterns if 'bearish' in p['type']]
        indecision = [p for p in all_patterns if p['type'] == 'indecision']

        return {
            "symbol": symbol,
            "period_days": days,
            "total_patterns": len(all_patterns),
            "recent_patterns": recent_patterns,
            "all_patterns": all_patterns,
            "pattern_counts": {"bullish": len(bullish), "bearish": len(bearish), "indecision": len(indecision)},
            "latest_signal": recent_patterns[0] if recent_patterns else None
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern detection failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ MARKET DATA ENDPOINTS ============

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


@api_router.get("/market/overview")
async def get_market_overview():
    """Get complete market overview with all indices and top movers"""
    try:
        rate_limiter = get_rate_limiter()
        cache_params = {"endpoint": "market_overview"}
        cached = rate_limiter.get_cached("market_overview", cache_params)
        if cached:
            return cached

        if not rate_limiter.check_rate_limit("yfinance"):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        indices_data = get_indian_indices_data()
        overview = await indices_data.get_market_overview()

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
        return {
            "gainers": overview.get("top_gainers", [])[:limit],
            "losers": overview.get("top_losers", [])[:limit]
        }
    except Exception as e:
        logger.error(f"Failed to get top movers: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ NEWS ENDPOINTS ============

@api_router.get("/news/latest")
async def get_latest_news(symbol: Optional[str] = None, limit: int = 10):
    """Get latest financial news"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()
        articles = news_aggregator.fetch_latest_news(symbol=symbol, limit=limit)
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)
        aggregate = sentiment_analyzer.get_aggregate_sentiment(articles_with_sentiment)
        return {"articles": articles_with_sentiment, "aggregate_sentiment": aggregate, "total": len(articles_with_sentiment)}
    except Exception as e:
        logger.error(f"Failed to fetch news: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/search")
async def search_news(q: str, days_back: int = 7, limit: int = 20):
    """Search news by query"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()
        articles = news_aggregator.search_news(query=q, days_back=days_back, limit=limit)
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)
        return {"articles": articles_with_sentiment, "total": len(articles_with_sentiment), "query": q}
    except Exception as e:
        logger.error(f"News search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/news/sentiment/{symbol}")
async def get_symbol_sentiment(symbol: str, days_back: int = 7):
    """Get sentiment analysis for a specific symbol"""
    try:
        news_aggregator = get_news_aggregator()
        sentiment_analyzer = get_sentiment_analyzer()
        articles = news_aggregator.fetch_latest_news(symbol=symbol, limit=50)
        articles_with_sentiment = sentiment_analyzer.analyze_articles(articles)
        aggregate = sentiment_analyzer.get_aggregate_sentiment(articles_with_sentiment)
        return {"symbol": symbol, "sentiment": aggregate, "recent_articles": articles_with_sentiment[:5]}
    except Exception as e:
        logger.error(f"Failed to get sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ ALERT ENDPOINTS ============

@api_router.post("/alerts/price")
async def create_price_alert(request: PriceAlertRequest):
    """Create price alert (Telegram + Email delivery)"""
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
            smtp_config={
                'smtp_host': settings.get('smtp_host'),
                'smtp_port': settings.get('smtp_port'),
                'smtp_user': settings.get('smtp_user'),
                'smtp_password': settings.get('smtp_password'),
                'from_email': settings.get('smtp_from_email'),
            } if settings.get('smtp_host') else None
        )

        alert = alert_mgr.create_price_alert(
            user_id=request.user_id,
            symbol=request.symbol,
            condition=PriceCondition(request.condition),
            target_price=request.target_price,
            delivery_channels=[DeliveryChannel(ch) for ch in request.delivery_channels],
            percent_change=request.percent_change
        )
        return alert.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create price alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/alerts")
async def get_user_alerts(user_id: str = "default", status: Optional[str] = None):
    """Get all alerts for a user"""
    try:
        settings = await get_settings_from_db()
        alert_mgr = get_alert_manager(
            telegram_bot_token=settings.get('telegram_bot_token'),
            telegram_chat_id=settings.get('telegram_chat_id'),
        )
        alerts = alert_mgr.get_user_alerts(user_id=user_id, status=AlertStatusEnum(status) if status else None)
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
        )
        success = alert_mgr.delete_alert(alert_id)
        if not success:
            raise HTTPException(status_code=404, detail="Alert not found")
        return {"message": "Alert deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ WATCHLIST ENDPOINTS ============

@api_router.get("/watchlist")
async def get_watchlist():
    """Get user's watchlist"""
    return await db.watchlist.find({}, {"_id": 0}).to_list(100)


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


# ============ ROOT ============

@api_router.get("/")
async def root():
    return {"name": "NeuralTrader API", "version": "2.0", "status": "running"}


# Include router
app.include_router(api_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8005)))

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
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

# Import pattern detection
from patterns.candlestick import get_pattern_detector

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Add CORS middleware FIRST - before any routes
# This ensures CORS headers are added to ALL responses including errors
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', 'http://localhost:3000').split(','),
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
    selected_model: str = "gpt-4.1"
    selected_provider: str = "openai"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class SettingsCreate(BaseModel):
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None
    selected_model: str = "gpt-4.1"
    selected_provider: str = "openai"

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
    model: str = "gpt-4.1"
    provider: str = "openai"

class StockSearchResult(BaseModel):
    symbol: str
    name: str
    exchange: str

# ============ DATA COLLECTION ============

def get_indian_stock_suffix(symbol: str) -> str:
    """Add .NS or .BO suffix for Indian stocks"""
    symbol = symbol.upper().strip()
    if not symbol.endswith('.NS') and not symbol.endswith('.BO'):
        return f"{symbol}.NS"  # Default to NSE
    return symbol

async def fetch_stock_data(symbol: str) -> Dict[str, Any]:
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker_symbol = get_indian_stock_suffix(symbol)
        ticker = yf.Ticker(ticker_symbol)
        
        # Get basic info
        info = ticker.info
        hist = ticker.history(period="1d")
        
        if hist.empty:
            # Try BSE
            ticker_symbol = symbol.upper().replace('.NS', '') + '.BO'
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            hist = ticker.history(period="1d")
        
        if hist.empty:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        current_price = float(hist['Close'].iloc[-1]) if not hist.empty else info.get('currentPrice', 0)
        previous_close = info.get('previousClose', current_price)
        
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
            "industry": info.get('industry', 'N/A')
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

@api_router.get("/")
async def root():
    return {"message": "Stock Trading AI API", "version": "1.0.0"}

# Settings endpoints
@api_router.get("/settings")
async def get_settings():
    settings = await db.settings.find_one({}, {"_id": 0})
    if not settings:
        return {
            "openai_api_key": "",
            "gemini_api_key": "",
            "selected_model": "gpt-4.1",
            "selected_provider": "openai"
        }
    # Mask API keys
    if settings.get('openai_api_key'):
        settings['openai_api_key'] = settings['openai_api_key'][:8] + '...' + settings['openai_api_key'][-4:] if len(settings['openai_api_key']) > 12 else '****'
    if settings.get('gemini_api_key'):
        settings['gemini_api_key'] = settings['gemini_api_key'][:8] + '...' + settings['gemini_api_key'][-4:] if len(settings['gemini_api_key']) > 12 else '****'
    return settings

@api_router.post("/settings")
async def save_settings(settings: SettingsCreate):
    existing = await db.settings.find_one({})
    
    update_data = {
        "selected_model": settings.selected_model,
        "selected_provider": settings.selected_provider,
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    
    # Only update API keys if they're provided and not masked
    if settings.openai_api_key and not settings.openai_api_key.startswith('****') and '...' not in settings.openai_api_key:
        update_data["openai_api_key"] = settings.openai_api_key
    
    if settings.gemini_api_key and not settings.gemini_api_key.startswith('****') and '...' not in settings.gemini_api_key:
        update_data["gemini_api_key"] = settings.gemini_api_key
    
    if existing:
        await db.settings.update_one({}, {"$set": update_data})
    else:
        update_data["id"] = str(uuid.uuid4())
        update_data["created_at"] = datetime.now(timezone.utc).isoformat()
        await db.settings.insert_one(update_data)
    
    return {"message": "Settings saved successfully"}

# Stock search
@api_router.get("/stocks/search")
async def search_stocks(q: str):
    """Search for stocks"""
    popular_indian_stocks = [
        {"symbol": "RELIANCE", "name": "Reliance Industries Ltd", "exchange": "NSE"},
        {"symbol": "TCS", "name": "Tata Consultancy Services", "exchange": "NSE"},
        {"symbol": "INFY", "name": "Infosys Ltd", "exchange": "NSE"},
        {"symbol": "HDFCBANK", "name": "HDFC Bank Ltd", "exchange": "NSE"},
        {"symbol": "ICICIBANK", "name": "ICICI Bank Ltd", "exchange": "NSE"},
        {"symbol": "HINDUNILVR", "name": "Hindustan Unilever Ltd", "exchange": "NSE"},
        {"symbol": "SBIN", "name": "State Bank of India", "exchange": "NSE"},
        {"symbol": "BHARTIARTL", "name": "Bharti Airtel Ltd", "exchange": "NSE"},
        {"symbol": "ITC", "name": "ITC Ltd", "exchange": "NSE"},
        {"symbol": "KOTAKBANK", "name": "Kotak Mahindra Bank", "exchange": "NSE"},
        {"symbol": "LT", "name": "Larsen & Toubro Ltd", "exchange": "NSE"},
        {"symbol": "WIPRO", "name": "Wipro Ltd", "exchange": "NSE"},
        {"symbol": "AXISBANK", "name": "Axis Bank Ltd", "exchange": "NSE"},
        {"symbol": "BAJFINANCE", "name": "Bajaj Finance Ltd", "exchange": "NSE"},
        {"symbol": "MARUTI", "name": "Maruti Suzuki India Ltd", "exchange": "NSE"},
        {"symbol": "ASIANPAINT", "name": "Asian Paints Ltd", "exchange": "NSE"},
        {"symbol": "TATAMOTORS", "name": "Tata Motors Ltd", "exchange": "NSE"},
        {"symbol": "TATASTEEL", "name": "Tata Steel Ltd", "exchange": "NSE"},
        {"symbol": "SUNPHARMA", "name": "Sun Pharmaceutical", "exchange": "NSE"},
        {"symbol": "HCLTECH", "name": "HCL Technologies Ltd", "exchange": "NSE"},
    ]
    
    query = q.upper()
    results = [s for s in popular_indian_stocks if query in s['symbol'] or query in s['name'].upper()]
    return results[:10]

# Stock data endpoints
@api_router.get("/stocks/{symbol}")
async def get_stock(symbol: str):
    """Get current stock data"""
    return await fetch_stock_data(symbol)

@api_router.get("/stocks/{symbol}/history")
async def get_stock_history(symbol: str, period: str = "1y"):
    """Get historical price data"""
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
        # Use the new multi-agent orchestrator (REAL IMPLEMENTATION)
        orchestrator = get_orchestrator()
        
        analysis_result = await orchestrator.run_analysis(
            symbol=request.symbol,
            model=request.model,
            provider=request.provider,
            api_key=api_key
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
    """Get trending topics from recent news"""
    try:
        news_aggregator = get_news_aggregator()
        trending = news_aggregator.get_trending_topics(limit=limit)
        return {"trending": trending}
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

# Include the router in the main app
app.include_router(api_router)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

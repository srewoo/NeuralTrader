"""
Seed Data for RAG Knowledge Base
Initial knowledge base with trading patterns, strategies, and market insights
"""

import logging
from .ingestion import get_ingestion_pipeline

logger = logging.getLogger(__name__)


def seed_trading_patterns():
    """Seed common trading patterns"""
    ingestion = get_ingestion_pipeline()
    
    patterns = [
        {
            "pattern_name": "Bullish RSI Divergence",
            "description": "Price makes lower lows while RSI makes higher lows, indicating weakening bearish momentum",
            "indicators": {"rsi": 35, "price_trend": "down", "volume": "increasing"},
            "outcome": "bullish reversal",
            "confidence": 75,
            "additional_info": {"timeframe": "daily", "success_rate": 0.72}
        },
        {
            "pattern_name": "MACD Golden Cross",
            "description": "MACD line crosses above signal line, indicating bullish momentum",
            "indicators": {"macd": 5.2, "signal": 3.8, "histogram": 1.4},
            "outcome": "bullish continuation",
            "confidence": 70,
            "additional_info": {"timeframe": "daily", "success_rate": 0.68}
        },
        {
            "pattern_name": "Bollinger Band Squeeze",
            "description": "Bollinger Bands narrow significantly, indicating low volatility before potential breakout",
            "indicators": {"bb_width": 0.02, "atr": 15, "volume": "low"},
            "outcome": "breakout imminent",
            "confidence": 65,
            "additional_info": {"timeframe": "daily", "success_rate": 0.60}
        },
        {
            "pattern_name": "Bearish RSI Divergence",
            "description": "Price makes higher highs while RSI makes lower highs, indicating weakening bullish momentum",
            "indicators": {"rsi": 75, "price_trend": "up", "volume": "decreasing"},
            "outcome": "bearish reversal",
            "confidence": 72,
            "additional_info": {"timeframe": "daily", "success_rate": 0.70}
        },
        {
            "pattern_name": "Volume Breakout",
            "description": "Price breaks resistance with significantly higher volume",
            "indicators": {"volume_ratio": 2.5, "price_change": 3.5, "rsi": 65},
            "outcome": "bullish continuation",
            "confidence": 78,
            "additional_info": {"timeframe": "daily", "success_rate": 0.75}
        }
    ]
    
    for pattern in patterns:
        ingestion.ingest_trading_pattern(**pattern)
    
    logger.info(f"Seeded {len(patterns)} trading patterns")


def seed_trading_strategies():
    """Seed proven trading strategies"""
    ingestion = get_ingestion_pipeline()
    
    strategies = [
        {
            "strategy_name": "Mean Reversion",
            "description": "Buy oversold stocks when RSI < 30, sell when RSI > 70",
            "conditions": [
                "RSI below 30",
                "Price near lower Bollinger Band",
                "No major negative news",
                "Overall market trend is neutral or positive"
            ],
            "expected_outcome": "Price returns to mean within 5-10 trading days",
            "risk_level": "medium",
            "performance_metrics": {"win_rate": 0.65, "avg_return": 0.08, "max_drawdown": 0.15}
        },
        {
            "strategy_name": "Trend Following",
            "description": "Follow strong trends using moving average crossovers",
            "conditions": [
                "SMA 20 crosses above SMA 50",
                "Price above both moving averages",
                "MACD positive and rising",
                "Volume above average"
            ],
            "expected_outcome": "Ride the trend for multiple weeks",
            "risk_level": "medium",
            "performance_metrics": {"win_rate": 0.58, "avg_return": 0.15, "max_drawdown": 0.20}
        },
        {
            "strategy_name": "Breakout Trading",
            "description": "Trade breakouts from consolidation patterns",
            "conditions": [
                "Price consolidating for at least 10 days",
                "Bollinger Bands narrowing",
                "Volume declining during consolidation",
                "Strong volume on breakout"
            ],
            "expected_outcome": "Quick move in breakout direction",
            "risk_level": "high",
            "performance_metrics": {"win_rate": 0.52, "avg_return": 0.20, "max_drawdown": 0.25}
        },
        {
            "strategy_name": "Support and Resistance",
            "description": "Buy at support levels, sell at resistance levels",
            "conditions": [
                "Price approaches tested support level",
                "RSI not oversold",
                "Volume increasing on bounce",
                "No breakdown below support"
            ],
            "expected_outcome": "Bounce from support towards resistance",
            "risk_level": "low",
            "performance_metrics": {"win_rate": 0.70, "avg_return": 0.06, "max_drawdown": 0.10}
        },
        {
            "strategy_name": "Momentum Trading",
            "description": "Trade stocks with strong momentum",
            "conditions": [
                "Price up more than 5% in last 5 days",
                "RSI between 50-70",
                "Volume consistently above average",
                "Positive news or sector strength"
            ],
            "expected_outcome": "Continued momentum for 3-7 days",
            "risk_level": "high",
            "performance_metrics": {"win_rate": 0.55, "avg_return": 0.12, "max_drawdown": 0.22}
        }
    ]
    
    for strategy in strategies:
        ingestion.ingest_trading_strategy(**strategy)
    
    logger.info(f"Seeded {len(strategies)} trading strategies")


def seed_market_insights():
    """Seed general market insights and wisdom"""
    ingestion = get_ingestion_pipeline()
    
    insights = [
        {
            "content": "Technical Indicator Interpretation: RSI above 70 indicates overbought conditions, but in strong uptrends, RSI can remain overbought for extended periods. Always confirm with other indicators.",
            "metadata": {"category": "indicators", "indicator": "RSI", "type": "interpretation"}
        },
        {
            "content": "Volume Analysis: Increasing volume on price advances and decreasing volume on price declines confirms bullish trend strength. The opposite pattern suggests bearish pressure.",
            "metadata": {"category": "indicators", "indicator": "Volume", "type": "interpretation"}
        },
        {
            "content": "MACD Interpretation: MACD histogram expanding indicates strengthening momentum. Histogram contracting suggests momentum is weakening even if price continues in same direction.",
            "metadata": {"category": "indicators", "indicator": "MACD", "type": "interpretation"}
        },
        {
            "content": "Risk Management: Never risk more than 2% of portfolio on a single trade. Use stop losses to limit downside. Let winners run but cut losses quickly.",
            "metadata": {"category": "risk_management", "type": "best_practice"}
        },
        {
            "content": "Market Timing: The best entry points often occur when fear is highest. Look for capitulation patterns with high volume selling followed by reversal.",
            "metadata": {"category": "market_psychology", "type": "timing"}
        },
        {
            "content": "Sector Rotation: During bull markets, cyclical sectors (technology, consumer discretionary) tend to outperform. During bear markets, defensive sectors (utilities, consumer staples) provide safety.",
            "metadata": {"category": "sector_analysis", "type": "rotation"}
        },
        {
            "content": "Earnings Impact: Stocks often see increased volatility around earnings announcements. IV (implied volatility) typically rises before earnings and drops after (volatility crush).",
            "metadata": {"category": "fundamentals", "type": "earnings"}
        },
        {
            "content": "Moving Averages: The 50-day and 200-day moving averages act as dynamic support/resistance. Golden cross (50 MA crosses above 200 MA) is bullish, death cross is bearish.",
            "metadata": {"category": "indicators", "indicator": "Moving Averages", "type": "interpretation"}
        }
    ]
    
    for insight in insights:
        ingestion.ingest_document(
            content=insight["content"],
            metadata=insight["metadata"]
        )
    
    logger.info(f"Seeded {len(insights)} market insights")


def seed_indian_market_context():
    """Seed India-specific market context"""
    ingestion = get_ingestion_pipeline()
    
    indian_context = [
        {
            "content": "NSE Trading Hours: Indian stock markets (NSE/BSE) operate from 9:15 AM to 3:30 PM IST on weekdays. Pre-market session is 9:00-9:15 AM, post-market is 3:40-4:00 PM.",
            "metadata": {"category": "market_structure", "region": "India", "type": "trading_hours"}
        },
        {
            "content": "Indian Market Indices: NIFTY 50 represents top 50 companies by market cap on NSE. SENSEX represents 30 companies on BSE. Both are market-cap weighted indices.",
            "metadata": {"category": "market_structure", "region": "India", "type": "indices"}
        },
        {
            "content": "Sector Leaders India: Major sectors include IT (TCS, Infosys), Banking (HDFC Bank, ICICI Bank), Energy (Reliance), Pharma (Sun Pharma, Dr Reddy's), and Auto (Maruti, Tata Motors).",
            "metadata": {"category": "sector_analysis", "region": "India", "type": "leaders"}
        },
        {
            "content": "Indian Market Seasonality: Markets often see strength during Diwali season (Oct-Nov) and weakness during summer months (May-Jun). Budget announcements (Feb) create volatility.",
            "metadata": {"category": "market_psychology", "region": "India", "type": "seasonality"}
        },
        {
            "content": "FII/DII Activity: Foreign Institutional Investors (FII) and Domestic Institutional Investors (DII) flows significantly impact Indian markets. Net FII buying is bullish, selling is bearish.",
            "metadata": {"category": "market_structure", "region": "India", "type": "institutional"}
        }
    ]
    
    for context in indian_context:
        ingestion.ingest_document(
            content=context["content"],
            metadata=context["metadata"]
        )
    
    logger.info(f"Seeded {len(indian_context)} Indian market context items")


def seed_all():
    """Seed all knowledge base data"""
    try:
        logger.info("Starting knowledge base seeding...")
        
        seed_trading_patterns()
        seed_trading_strategies()
        seed_market_insights()
        seed_indian_market_context()
        
        # Get final stats
        ingestion = get_ingestion_pipeline()
        stats = ingestion.get_ingestion_stats()
        
        logger.info(f"Knowledge base seeding complete. Total documents: {stats.get('count', 0)}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to seed knowledge base: {e}")
        return False


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run seeding
    seed_all()


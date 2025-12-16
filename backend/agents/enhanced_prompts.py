"""
Enhanced Prompts for LLM Analysis
Provides stronger, more comprehensive prompts with:
- Market context awareness
- Contrarian analysis
- Position sizing guidance
- Scenario analysis
- Better confidence calibration
"""

from typing import Dict, Any, Optional
from datetime import datetime


class EnhancedPromptBuilder:
    """Builds enhanced prompts for LLM stock analysis"""

    @staticmethod
    def build_analysis_prompt(
        symbol: str,
        stock_data: Dict[str, Any],
        indicators: Dict[str, Any],
        signals: Dict[str, Any],
        rag_context: str = "",
        market_context: Optional[Dict[str, Any]] = None,
        patterns: Optional[list] = None,
        news_sentiment: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Build comprehensive analysis prompt with all enhancements.
        """

        # Market context section
        market_section = ""
        if market_context:
            market_section = f"""
=== MARKET CONTEXT ===
Nifty 50: {market_context.get('nifty_level', 'N/A')} ({market_context.get('nifty_change', 'N/A')}%)
Sensex: {market_context.get('sensex_level', 'N/A')} ({market_context.get('sensex_change', 'N/A')}%)
Market Breadth: {market_context.get('advances', 'N/A')} advances / {market_context.get('declines', 'N/A')} declines
India VIX: {market_context.get('vix', 'N/A')} (Fear gauge - above 20 = high fear)
FII Flow (Today): ₹{market_context.get('fii_flow', 'N/A')} Cr
DII Flow (Today): ₹{market_context.get('dii_flow', 'N/A')} Cr
Global Cues: {market_context.get('global_cues', 'N/A')}
Sector Performance: {market_context.get('sector_performance', 'N/A')}
"""

        # Pattern section
        pattern_section = ""
        if patterns and len(patterns) > 0:
            recent_patterns = patterns[:3]  # Last 3 patterns
            pattern_list = "\n".join([
                f"  - {p.get('pattern', '').replace('_', ' ').title()}: {p.get('implication', {}).get('signal', '')} ({p.get('date_formatted', '')})"
                for p in recent_patterns
            ])
            pattern_section = f"""
=== CANDLESTICK PATTERNS DETECTED ===
{pattern_list}
"""

        # News sentiment section
        sentiment_section = ""
        if news_sentiment:
            sentiment_section = f"""
=== NEWS SENTIMENT ===
Overall Sentiment: {news_sentiment.get('aggregate', {}).get('label', 'N/A')}
Sentiment Score: {news_sentiment.get('aggregate', {}).get('average_score', 0):.2f} (-1 to +1 scale)
Recent Headlines Analyzed: {news_sentiment.get('article_count', 0)}
Key Themes: {', '.join(news_sentiment.get('aggregate', {}).get('keywords', ['N/A'])[:5])}
"""

        prompt = f"""You are a SEBI-registered Investment Advisor analyzing Indian equities (NSE/BSE).
Your analysis must be thorough, balanced, and actionable. You must consider BOTH bullish AND bearish scenarios.

{market_section}

{rag_context if rag_context else ""}

=== ANALYSIS FRAMEWORK ===

You MUST follow this structured analysis approach:

**STEP 1: SITUATIONAL AWARENESS**
- Where is the stock in its 52-week range? (Near high = cautious, Near low = opportunity/value trap)
- How is the sector performing relative to Nifty?
- What is the market regime? (Bull/Bear/Sideways)

**STEP 2: TECHNICAL HEALTH CHECK**
Score each factor 1-10:
- Trend Strength (SMA alignment, ADX)
- Momentum (RSI, MACD direction)
- Volatility (ATR, BB width - is it expanding or contracting?)
- Volume (Is volume confirming price moves?)

**STEP 3: RISK ASSESSMENT (CRITICAL)**
- What is the maximum drawdown risk from current levels?
- Where are the nearest support levels?
- What could go wrong? (List 3-5 specific risks)

**STEP 4: CONTRARIAN CHECK**
⚠️ MANDATORY: Before finalizing, ask yourself:
- "What if I'm wrong?"
- "What would make this trade fail?"
- "Is the crowd already positioned this way?"

**STEP 5: POSITION SIZING**
Based on:
- Confidence level
- Volatility (ATR)
- Account risk (assume 2% max risk per trade)

=== FEW-SHOT EXAMPLES ===

Example 1 - High Conviction BUY (Confidence 80+):
{{
  "recommendation": "BUY",
  "conviction": "HIGH",
  "confidence": 82,
  "position_size": "Full position (3-5% of portfolio)",
  "entry_price": 1450,
  "targets": {{
    "target_1": {{"price": 1520, "probability": "70%", "timeframe": "2-3 weeks"}},
    "target_2": {{"price": 1580, "probability": "50%", "timeframe": "4-6 weeks"}},
    "target_3": {{"price": 1650, "probability": "30%", "timeframe": "2-3 months"}}
  }},
  "stop_loss": 1395,
  "trailing_stop": "Move to breakeven after Target 1 hit",
  "risk_reward_ratio": 2.36,
  "scenarios": {{
    "bull_case": "Breaks above 1580 with volume → targets 1700+ (30% probability)",
    "base_case": "Gradual climb to 1550-1580 range (50% probability)",
    "bear_case": "Breaks 1400 support → could test 1320 (20% probability)"
  }},
  "reasoning": "STEP 1: Stock at 52-week midpoint, sector outperforming Nifty by 3%. STEP 2: Technical scores - Trend: 8/10, Momentum: 7/10, Volatility: 6/10, Volume: 7/10. STEP 3: Max drawdown risk 8% to support at 1340. STEP 4: Contrarian check - sentiment not extreme, institutional accumulation visible. STEP 5: Full position justified given high conviction.",
  "contrarian_view": "Bears would argue: RSI approaching overbought, global IT spending concerns persist. Counter: Strong order book and AI tailwinds offset macro concerns.",
  "key_risks": ["Global recession fears", "Rupee volatility", "Client concentration"],
  "catalysts": ["Q3 earnings on Jan 15", "Large deal announcements expected", "AI project wins"],
  "exit_strategy": "Book 50% at Target 1, trail rest with 5% stop"
}}

Example 2 - Moderate Conviction HOLD (Confidence 50-65):
{{
  "recommendation": "HOLD",
  "conviction": "LOW",
  "confidence": 55,
  "position_size": "Maintain existing, don't add",
  "entry_price": null,
  "targets": {{
    "target_1": {{"price": null, "probability": null, "timeframe": null}}
  }},
  "stop_loss": null,
  "scenarios": {{
    "bull_case": "Breakout above 1680 triggers momentum (35% probability)",
    "base_case": "Continues consolidation 1550-1680 (45% probability)",
    "bear_case": "Breakdown below 1550 targets 1450 (20% probability)"
  }},
  "reasoning": "STEP 1: Stock in middle of range, no clear trend. STEP 2: All technical scores around 5/10 - neutral. STEP 3: Risk is sideways chop eating time value. STEP 4: Contrarian check - no extreme positioning. STEP 5: No new position warranted.",
  "contrarian_view": "Both bulls and bears have valid arguments. Wait for clarity.",
  "action_triggers": ["BUY if closes above 1680 with volume", "SELL if closes below 1550"],
  "key_risks": ["Time decay if using options", "Opportunity cost"],
  "catalysts": ["RBI policy decision", "Quarterly results"]
}}

=== CURRENT STOCK DATA ===

Stock: {stock_data.get('name', symbol)} ({symbol})
Sector: {stock_data.get('sector', 'N/A')}
Industry: {stock_data.get('industry', 'N/A')}

Price Data:
- Current Price: ₹{stock_data.get('current_price', 'N/A')}
- Previous Close: ₹{stock_data.get('previous_close', 'N/A')}
- Day Change: {stock_data.get('change_percent', 'N/A')}%
- Volume: {stock_data.get('volume', 'N/A'):,} (vs 20-day avg: {indicators.get('volume_ratio', 'N/A')}x)
- Market Cap: ₹{stock_data.get('market_cap', 'N/A')} Cr
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- 52W High: ₹{stock_data.get('week_52_high', 'N/A')} (current is {_calc_from_high(stock_data)}% below)
- 52W Low: ₹{stock_data.get('week_52_low', 'N/A')} (current is {_calc_from_low(stock_data)}% above)

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A')} [{_rsi_interpretation(indicators.get('rsi'))}]
- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')} [{signals.get('macd', 'N/A')}]
- MACD Histogram: {indicators.get('macd_histogram', 'N/A')} [{_macd_hist_interpretation(indicators.get('macd_histogram'))}]
- SMA 20: ₹{indicators.get('sma_20', 'N/A')} (Price {'above' if _price_above_sma(stock_data, indicators, 'sma_20') else 'below'})
- SMA 50: ₹{indicators.get('sma_50', 'N/A')} (Price {'above' if _price_above_sma(stock_data, indicators, 'sma_50') else 'below'})
- SMA 200: ₹{indicators.get('sma_200', 'N/A')} (Price {'above' if _price_above_sma(stock_data, indicators, 'sma_200') else 'below'})
- Bollinger Bands: Upper ₹{indicators.get('bb_upper', 'N/A')} | Mid ₹{indicators.get('bb_middle', 'N/A')} | Lower ₹{indicators.get('bb_lower', 'N/A')}
- ATR (14): ₹{indicators.get('atr', 'N/A')} ({_atr_pct(stock_data, indicators)}% of price - {_volatility_interpretation(stock_data, indicators)})
- ADX: {indicators.get('adx', 'N/A')} [{_adx_interpretation(indicators.get('adx'))}]
- Stochastic K/D: {indicators.get('stochastic_k', 'N/A')} / {indicators.get('stochastic_d', 'N/A')}

Overall Assessment:
- Trend: {signals.get('trend', 'N/A')}
- Momentum: {signals.get('momentum', 'N/A')}
- Volatility: {signals.get('volatility', 'N/A')}

{pattern_section}

{sentiment_section}

=== OUTPUT FORMAT ===

Respond with ONLY valid JSON (no markdown, no explanation outside JSON):

{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "conviction": "HIGH" | "MEDIUM" | "LOW",
  "confidence": <integer 0-100>,
  "position_size": "<specific guidance based on conviction>",
  "entry_price": <number or null>,
  "targets": {{
    "target_1": {{"price": <number>, "probability": "<X%>", "timeframe": "<duration>"}},
    "target_2": {{"price": <number>, "probability": "<X%>", "timeframe": "<duration>"}},
    "target_3": {{"price": <number or null>, "probability": "<X%>", "timeframe": "<duration>"}}
  }},
  "stop_loss": <number>,
  "trailing_stop": "<trailing stop strategy>",
  "risk_reward_ratio": <number>,
  "scenarios": {{
    "bull_case": "<description with probability>",
    "base_case": "<description with probability>",
    "bear_case": "<description with probability>"
  }},
  "reasoning": "<Detailed STEP 1-5 analysis as shown in examples>",
  "contrarian_view": "<What the opposite side would argue and your counter>",
  "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
  "key_opportunities": ["<opp1>", "<opp2>"],
  "catalysts": ["<upcoming events that could move the stock>"],
  "exit_strategy": "<When and how to exit the position>",
  "confidence_breakdown": {{
    "technical_score": <0-100>,
    "momentum_score": <0-100>,
    "risk_score": <0-100>,
    "sentiment_score": <0-100>
  }},
  "action_triggers": ["<specific price levels or events that would change recommendation>"]
}}

=== CONFIDENCE CALIBRATION GUIDE ===
- 85-95: Rare. Multiple strong signals aligned. Clear trend. High volume confirmation. Low risk.
- 70-84: Good setup. Most indicators aligned. Acceptable risk-reward.
- 55-69: Mixed signals. Some indicators conflicting. Higher uncertainty.
- 40-54: Weak signals. Wait for better setup. HOLD recommended.
- Below 40: Avoid. High uncertainty. No clear edge.

Remember: It's better to miss a trade than to force one. If signals are mixed, recommend HOLD."""

        return prompt


# Helper functions for prompt building
def _calc_from_high(stock_data: Dict) -> str:
    try:
        current = float(stock_data.get('current_price', 0))
        high = float(stock_data.get('week_52_high', current))
        if high > 0:
            return f"{((high - current) / high * 100):.1f}"
    except:
        pass
    return "N/A"


def _calc_from_low(stock_data: Dict) -> str:
    try:
        current = float(stock_data.get('current_price', 0))
        low = float(stock_data.get('week_52_low', current))
        if low > 0:
            return f"{((current - low) / low * 100):.1f}"
    except:
        pass
    return "N/A"


def _rsi_interpretation(rsi) -> str:
    if rsi is None:
        return "N/A"
    try:
        rsi = float(rsi)
        if rsi < 25:
            return "Extremely Oversold - Strong bounce expected"
        elif rsi < 30:
            return "Oversold - Watch for reversal"
        elif rsi < 40:
            return "Approaching oversold"
        elif rsi > 80:
            return "Extremely Overbought - Correction likely"
        elif rsi > 70:
            return "Overbought - Caution advised"
        elif rsi > 60:
            return "Approaching overbought"
        else:
            return "Neutral zone"
    except:
        return "N/A"


def _macd_hist_interpretation(hist) -> str:
    if hist is None:
        return "N/A"
    try:
        hist = float(hist)
        if hist > 0.5:
            return "Strong bullish momentum"
        elif hist > 0:
            return "Bullish momentum building"
        elif hist < -0.5:
            return "Strong bearish momentum"
        elif hist < 0:
            return "Bearish momentum building"
        else:
            return "Neutral"
    except:
        return "N/A"


def _price_above_sma(stock_data: Dict, indicators: Dict, sma_key: str) -> bool:
    try:
        price = float(stock_data.get('current_price', 0))
        sma = float(indicators.get(sma_key, 0))
        return price > sma
    except:
        return False


def _atr_pct(stock_data: Dict, indicators: Dict) -> str:
    try:
        price = float(stock_data.get('current_price', 1))
        atr = float(indicators.get('atr', 0))
        return f"{(atr / price * 100):.1f}"
    except:
        return "N/A"


def _volatility_interpretation(stock_data: Dict, indicators: Dict) -> str:
    try:
        price = float(stock_data.get('current_price', 1))
        atr = float(indicators.get('atr', 0))
        atr_pct = atr / price * 100
        if atr_pct > 4:
            return "HIGH volatility - use wider stops"
        elif atr_pct > 2:
            return "MODERATE volatility"
        else:
            return "LOW volatility - tighter stops possible"
    except:
        return "N/A"


def _adx_interpretation(adx) -> str:
    if adx is None:
        return "N/A"
    try:
        adx = float(adx)
        if adx > 50:
            return "Very strong trend"
        elif adx > 25:
            return "Strong trend - trade with trend"
        elif adx > 20:
            return "Developing trend"
        else:
            return "Weak/No trend - range-bound"
    except:
        return "N/A"


class NaturalLanguageQueryPrompt:
    """Prompt for natural language stock queries"""

    @staticmethod
    def build_query_prompt(
        query: str,
        stock_data: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> str:
        return f"""You are a friendly stock market expert assistant for Indian markets.
Answer the user's question about this stock in simple, clear language.

User Question: {query}

Stock Data:
- Symbol: {stock_data.get('symbol', 'N/A')}
- Name: {stock_data.get('name', 'N/A')}
- Current Price: ₹{stock_data.get('current_price', 'N/A')}
- Change: {stock_data.get('change_percent', 'N/A')}%
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- Trend: {indicators.get('trend', 'N/A')}

Guidelines:
1. Be conversational and helpful
2. Use simple language, avoid jargon unless explaining it
3. Give specific numbers when relevant
4. If asked about buy/sell, remind them this is not financial advice
5. Use ₹ for prices

Respond naturally as if talking to a friend who's learning about stocks."""


class NewsSummaryPrompt:
    """Prompt for summarizing financial news"""

    @staticmethod
    def build_summary_prompt(
        symbol: str,
        articles: list,
        stock_price: float
    ) -> str:
        article_text = "\n".join([
            f"- {a.get('title', '')} ({a.get('source', '')})"
            for a in articles[:10]
        ])

        return f"""Summarize the recent news for {symbol} stock (current price: ₹{stock_price}).

Recent Headlines:
{article_text}

Provide a JSON response:
{{
  "summary": "<2-3 sentence summary of key news>",
  "sentiment": "positive" | "negative" | "neutral" | "mixed",
  "key_themes": ["<theme1>", "<theme2>", "<theme3>"],
  "potential_impact": {{
    "short_term": "<expected impact on stock in 1-5 days>",
    "medium_term": "<expected impact in 1-4 weeks>"
  }},
  "notable_events": ["<any specific events mentioned: earnings, deals, etc>"],
  "risk_alerts": ["<any concerning news that investors should watch>"]
}}"""


class PortfolioAnalysisPrompt:
    """Prompt for portfolio analysis"""

    @staticmethod
    def build_portfolio_prompt(
        holdings: list,
        total_value: float
    ) -> str:
        holdings_text = "\n".join([
            f"- {h.get('symbol')}: {h.get('quantity')} shares @ ₹{h.get('avg_price')} (Current: ₹{h.get('current_price')}, P&L: {h.get('pnl_pct')}%)"
            for h in holdings
        ])

        return f"""Analyze this Indian stock portfolio and provide recommendations.

Portfolio Value: ₹{total_value:,.2f}

Holdings:
{holdings_text}

Provide a JSON response:
{{
  "portfolio_health": "healthy" | "needs_attention" | "high_risk",
  "diversification_score": <0-100>,
  "sector_concentration": {{
    "<sector>": "<percentage>",
    ...
  }},
  "risk_assessment": {{
    "overall_risk": "low" | "medium" | "high",
    "biggest_risk": "<description>",
    "correlation_warning": "<if holdings are too correlated>"
  }},
  "recommendations": [
    {{
      "action": "REDUCE" | "HOLD" | "ADD" | "EXIT",
      "symbol": "<stock>",
      "reason": "<why>",
      "priority": "high" | "medium" | "low"
    }}
  ],
  "rebalancing_suggestions": ["<suggestion1>", "<suggestion2>"],
  "missing_sectors": ["<sectors to consider adding for diversification>"]
}}"""


class MarketCommentaryPrompt:
    """Prompt for daily market commentary"""

    @staticmethod
    def build_commentary_prompt(
        market_data: Dict[str, Any],
        top_gainers: list,
        top_losers: list
    ) -> str:
        gainers = ", ".join([f"{g['symbol']} (+{g['change']}%)" for g in top_gainers[:5]])
        losers = ", ".join([f"{l['symbol']} ({l['change']}%)" for l in top_losers[:5]])

        return f"""Generate a professional daily market commentary for Indian markets.

Market Data:
- Nifty 50: {market_data.get('nifty', 'N/A')} ({market_data.get('nifty_change', 'N/A')}%)
- Sensex: {market_data.get('sensex', 'N/A')} ({market_data.get('sensex_change', 'N/A')}%)
- India VIX: {market_data.get('vix', 'N/A')}
- Advance/Decline: {market_data.get('advances', 'N/A')}/{market_data.get('declines', 'N/A')}
- FII Flow: ₹{market_data.get('fii', 'N/A')} Cr
- DII Flow: ₹{market_data.get('dii', 'N/A')} Cr

Top Gainers: {gainers}
Top Losers: {losers}

Date: {datetime.now().strftime('%d %B %Y')}

Provide a JSON response:
{{
  "headline": "<catchy 5-7 word headline>",
  "summary": "<2-3 paragraph market summary>",
  "key_levels": {{
    "nifty_support": [<level1>, <level2>],
    "nifty_resistance": [<level1>, <level2>],
    "sensex_support": [<level1>, <level2>],
    "sensex_resistance": [<level1>, <level2>]
  }},
  "sector_performance": "<which sectors led/lagged>",
  "fii_dii_analysis": "<interpretation of flows>",
  "global_cues": "<relevant global market impact>",
  "outlook": {{
    "short_term": "<1-3 day outlook>",
    "key_events": ["<upcoming events to watch>"]
  }},
  "trading_idea": {{
    "type": "bullish" | "bearish" | "neutral",
    "suggestion": "<specific actionable idea>",
    "risk": "<associated risk>"
  }}
}}"""

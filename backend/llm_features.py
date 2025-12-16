"""
LLM-Powered Features
Additional AI features for the trading platform:
- Natural language stock queries
- News summarization
- Portfolio analysis
- Market commentary
- Pattern explanation
"""

import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
import json
import yfinance as yf

logger = logging.getLogger(__name__)


class LLMFeatures:
    """LLM-powered features for enhanced user experience"""

    def __init__(self):
        self.supported_providers = ['openai', 'gemini']

    async def ask_about_stock(
        self,
        question: str,
        symbol: Optional[str],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Natural language interface to ask questions about a stock.

        Examples:
        - "Is this a good time to buy?"
        - "What's the risk level?"
        - "Explain the RSI for this stock"
        - "Compare to its 52-week high"
        """
        try:
            stock_context = ""
            current_price = None

            # If symbol provided, fetch stock data
            if symbol:
                ticker = yf.Ticker(f"{symbol}.NS")
                info = ticker.info
                hist = ticker.history(period="3mo")

                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    rsi = self._calculate_rsi(hist['Close'])

                    stock_context = f"""
Stock: {info.get('longName', symbol)} ({symbol})
Current Price: ₹{current_price:.2f}
Day Change: {info.get('regularMarketChangePercent', 0):.2f}%
52W High: ₹{info.get('fiftyTwoWeekHigh', 'N/A')}
52W Low: ₹{info.get('fiftyTwoWeekLow', 'N/A')}
P/E Ratio: {info.get('trailingPE', 'N/A')}
Market Cap: ₹{info.get('marketCap', 0) / 10000000:.0f} Cr
RSI (14): {rsi:.1f}
Sector: {info.get('sector', 'N/A')}
"""

            prompt = f"""You are a friendly stock market assistant for Indian investors.
Answer this question naturally and helpfully.

{stock_context if stock_context else "No specific stock data available. Answer based on general market knowledge."}

User Question: {question}

Guidelines:
1. Be conversational but informative
2. Use simple language, explain jargon
3. Include specific numbers when helpful
4. Add a brief disclaimer if giving buy/sell opinions
5. Use ₹ symbol for Indian Rupee
6. Keep response concise (2-4 sentences for simple questions)

Response:"""

            response = await self._call_llm(prompt, api_key, provider, model)

            result = {
                "question": question,
                "answer": response,
                "timestamp": datetime.now().isoformat()
            }

            if symbol:
                result["stock"] = symbol
                if current_price:
                    result["current_price"] = round(current_price, 2)

            return result

        except Exception as e:
            logger.error(f"Ask about stock failed: {e}")
            return {"error": str(e)}

    async def summarize_news(
        self,
        symbol: str,
        articles: List[Dict],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Summarize news articles and their potential impact on stock.
        """
        try:
            if not articles:
                return {"error": "No articles provided"}

            article_list = "\n".join([
                f"• {a.get('title', 'No title')} - {a.get('source', 'Unknown')}"
                for a in articles[:10]
            ])

            prompt = f"""Analyze these news headlines for {symbol} stock and provide insights.

Headlines:
{article_list}

Provide a JSON response with:
{{
  "summary": "2-3 sentence summary of the news",
  "sentiment": "positive/negative/neutral/mixed",
  "sentiment_score": <-1 to 1>,
  "key_themes": ["theme1", "theme2"],
  "potential_impact": {{
    "direction": "bullish/bearish/neutral",
    "magnitude": "high/medium/low",
    "timeframe": "immediate/short-term/medium-term"
  }},
  "actionable_insight": "One sentence on what investors should do",
  "risk_flags": ["any concerning items"]
}}

Respond with ONLY valid JSON."""

            response = await self._call_llm(prompt, api_key, provider, model, json_mode=True)

            try:
                result = json.loads(response)
            except:
                result = {"summary": response, "sentiment": "unknown"}

            result["symbol"] = symbol
            result["articles_analyzed"] = len(articles)
            result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"News summarization failed: {e}")
            return {"error": str(e)}

    async def analyze_portfolio(
        self,
        holdings: List[Dict],
        investment_goal: Optional[str],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Analyze a portfolio and provide recommendations.

        holdings format:
        [{"symbol": "RELIANCE", "quantity": 10, "avg_price": 2500}, ...]

        investment_goal: "growth", "income", "balanced", "aggressive", etc.
        """
        try:
            # Enrich holdings with current prices
            enriched = []
            total_invested = 0
            total_current = 0

            for h in holdings:
                try:
                    ticker = yf.Ticker(f"{h['symbol']}.NS")
                    info = ticker.info
                    current_price = info.get('regularMarketPrice', h['avg_price'])

                    invested = h['quantity'] * h['avg_price']
                    current = h['quantity'] * current_price
                    pnl_pct = ((current_price / h['avg_price']) - 1) * 100

                    total_invested += invested
                    total_current += current

                    enriched.append({
                        "symbol": h['symbol'],
                        "quantity": h['quantity'],
                        "avg_price": h['avg_price'],
                        "current_price": round(current_price, 2),
                        "invested": round(invested, 2),
                        "current_value": round(current, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "sector": info.get('sector', 'Unknown')
                    })
                except Exception as e:
                    logger.warning(f"Could not fetch {h['symbol']}: {e}")
                    enriched.append({**h, "error": str(e)})

            holdings_text = "\n".join([
                f"• {h['symbol']}: {h['quantity']} shares, Avg ₹{h['avg_price']}, Current ₹{h.get('current_price', 'N/A')}, P&L: {h.get('pnl_pct', 0):.1f}%, Sector: {h.get('sector', 'N/A')}"
                for h in enriched
            ])

            total_pnl = ((total_current / total_invested) - 1) * 100 if total_invested > 0 else 0

            goal = investment_goal or "growth"

            prompt = f"""Analyze this Indian stock portfolio and provide actionable advice.

Investment Goal: {goal.upper()}

Portfolio Summary:
Total Invested: ₹{total_invested:,.0f}
Current Value: ₹{total_current:,.0f}
Overall P&L: {total_pnl:.1f}%

Holdings:
{holdings_text}

Provide a JSON response:
{{
  "portfolio_grade": "A/B/C/D/F",
  "health_score": <0-100>,
  "diversification": {{
    "score": <0-100>,
    "issue": "over-concentrated/well-diversified/under-diversified",
    "sector_breakdown": {{"sector": "percentage"}}
  }},
  "risk_level": "low/medium/high/very-high",
  "top_concerns": ["concern1", "concern2"],
  "recommendations": [
    {{
      "action": "BUY/SELL/HOLD/REDUCE/ADD",
      "symbol": "STOCK",
      "reason": "why",
      "priority": "high/medium/low"
    }}
  ],
  "rebalancing": ["specific rebalancing suggestions"],
  "missing_exposure": ["sectors or themes to consider adding"],
  "overall_advice": "2-3 sentence portfolio advice"
}}

Be specific with Indian market context. Respond with ONLY valid JSON."""

            response = await self._call_llm(prompt, api_key, provider, model, json_mode=True)

            try:
                result = json.loads(response)
            except:
                result = {"overall_advice": response}

            result["holdings"] = enriched
            result["total_invested"] = round(total_invested, 2)
            result["total_current"] = round(total_current, 2)
            result["total_pnl_pct"] = round(total_pnl, 2)
            result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {"error": str(e)}

    async def explain_pattern(
        self,
        symbol: str,
        pattern_name: str,
        pattern_data: Optional[Dict[str, Any]],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Explain a candlestick pattern in simple terms.
        """
        try:
            price = pattern_data.get('price', 0) if pattern_data else 0
            date_str = pattern_data.get('date_formatted', 'recently') if pattern_data else 'recently'
            pattern_type = pattern_data.get('type', 'unknown') if pattern_data else 'unknown'
            strength = pattern_data.get('strength', 'unknown') if pattern_data else 'unknown'

            prompt = f"""Explain the {pattern_name.replace('_', ' ')} candlestick pattern
that was detected on {symbol} at ₹{price:.2f} on {date_str}.
Pattern Type: {pattern_type}
Signal Strength: {strength}

Provide a JSON response:
{{
  "pattern": "{pattern_name}",
  "simple_explanation": "Explain in 1-2 simple sentences what this pattern means, as if talking to a beginner",
  "what_it_shows": "What buyer/seller behavior created this pattern",
  "trading_implication": {{
    "direction": "bullish/bearish/neutral",
    "strength": "strong/moderate/weak",
    "action": "What a trader might consider doing"
  }},
  "success_rate": "Approximate historical success rate of this pattern",
  "confirmation_needed": "What to look for to confirm the pattern",
  "warning": "When this pattern might fail"
}}

Keep explanations simple and jargon-free. Respond with ONLY valid JSON."""

            response = await self._call_llm(prompt, api_key, provider, model, json_mode=True)

            try:
                result = json.loads(response)
            except:
                result = {"simple_explanation": response}

            result["symbol"] = symbol
            result["price"] = price
            result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Pattern explanation failed: {e}")
            return {"error": str(e)}

    async def generate_market_commentary(
        self,
        market_data: Optional[Dict[str, Any]],
        news: Optional[List[Dict]],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Generate daily market commentary.
        """
        try:
            # Use provided market data or fetch it
            if market_data and market_data.get('nifty_current'):
                nifty_price = market_data.get('nifty_current', 0)
                nifty_change = market_data.get('nifty_change_pct', 0)
            else:
                nifty = yf.Ticker("^NSEI")
                nifty_hist = nifty.history(period="2d")
                nifty_price = nifty_hist['Close'].iloc[-1] if not nifty_hist.empty else 0
                nifty_prev = nifty_hist['Close'].iloc[-2] if len(nifty_hist) > 1 else nifty_price
                nifty_change = ((nifty_price / nifty_prev) - 1) * 100 if nifty_prev else 0

            # Format news context
            news_context = ""
            if news:
                news_items = "\n".join([
                    f"• {n.get('title', 'No title')} ({n.get('source', 'Unknown')})"
                    for n in news[:5]
                ])
                news_context = f"\nRecent News:\n{news_items}"

            prompt = f"""Generate a professional market commentary for Indian markets.

Today's Date: {datetime.now().strftime('%d %B %Y, %A')}

Market Levels:
- Nifty 50: {nifty_price:.0f} ({nifty_change:+.2f}%)
{news_context}

Provide a JSON response:
{{
  "headline": "Catchy 5-7 word headline",
  "summary": "2-3 paragraph market summary covering key moves, sectors, and outlook",
  "market_mood": "bullish/bearish/cautious/volatile/range-bound",
  "key_levels": {{
    "nifty_support": [<level1>, <level2>],
    "nifty_resistance": [<level1>, <level2>]
  }},
  "sectors_to_watch": ["sector1", "sector2"],
  "trading_strategy": {{
    "bias": "bullish/bearish/neutral",
    "approach": "Specific trading approach for today",
    "risk_note": "Key risk to watch"
  }},
  "outlook": "1-2 sentence near-term outlook"
}}

Write as a professional market analyst. Respond with ONLY valid JSON."""

            response = await self._call_llm(prompt, api_key, provider, model, json_mode=True)

            try:
                result = json.loads(response)
            except:
                result = {"summary": response}

            result["nifty"] = round(nifty_price, 2)
            result["nifty_change"] = round(nifty_change, 2)
            result["generated_at"] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Market commentary generation failed: {e}")
            return {"error": str(e)}

    async def compare_stocks(
        self,
        symbols: List[str],
        stocks_data: List[Dict[str, Any]],
        criteria: Optional[List[str]],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Compare multiple stocks for investment decision.
        Accepts pre-fetched stock data from server.
        """
        try:
            if len(stocks_data) < 2:
                return {"error": "Need at least 2 stocks to compare"}

            # Build comparison text for each stock
            stocks_text = []
            for i, data in enumerate(stocks_data):
                stock_info = data.get('stock_data', {})
                indicators = data.get('indicators', {})

                if stock_info.get('error'):
                    stocks_text.append(f"Stock {i+1}: {symbols[i]} - Error: {stock_info.get('error')}")
                    continue

                text = f"""Stock {i+1}: {stock_info.get('name', symbols[i])} ({symbols[i]})
- Price: ₹{stock_info.get('current_price', 'N/A')}
- Change: {stock_info.get('change_percent', 'N/A')}%
- P/E: {stock_info.get('pe_ratio', 'N/A')}
- Market Cap: ₹{stock_info.get('market_cap', 'N/A')} Cr
- Sector: {stock_info.get('sector', 'N/A')}
- 52W High: ₹{stock_info.get('week_52_high', 'N/A')}
- 52W Low: ₹{stock_info.get('week_52_low', 'N/A')}
- RSI: {indicators.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}"""
                stocks_text.append(text)

            stocks_comparison = "\n\n".join(stocks_text)

            # Build criteria text
            criteria_list = criteria or ["value", "growth", "momentum", "risk"]
            criteria_text = ", ".join(criteria_list)

            prompt = f"""Compare these Indian stocks for an investment decision.

{stocks_comparison}

Evaluation Criteria: {criteria_text}

Provide a JSON response:
{{
  "summary": "2-3 sentence overall comparison summary",
  "comparison_table": [
    {{
      "criterion": "<criterion>",
      "winner": "<symbol>",
      "reason": "<brief explanation>"
    }}
  ],
  "stock_analysis": [
    {{
      "symbol": "<symbol>",
      "pros": ["<pro1>", "<pro2>"],
      "cons": ["<con1>", "<con2>"],
      "best_for": "<type of investor this suits>"
    }}
  ],
  "verdict": {{
    "top_pick": "<symbol or NONE>",
    "runner_up": "<symbol or NONE>",
    "reasoning": "2-3 sentence explanation of the verdict",
    "confidence": <0-100>
  }},
  "risk_ranking": [
    {{"symbol": "<symbol>", "risk_level": "low/medium/high", "key_risk": "<main risk>"}}
  ],
  "recommendation": {{
    "action": "Which stock(s) to buy and why",
    "allocation": "Suggested allocation if buying multiple"
  }}
}}

Respond with ONLY valid JSON."""

            response = await self._call_llm(prompt, api_key, provider, model, json_mode=True)

            try:
                result = json.loads(response)
            except:
                result = {"summary": response}

            result["symbols"] = symbols
            result["criteria_used"] = criteria_list
            result["timestamp"] = datetime.now().isoformat()

            return result

        except Exception as e:
            logger.error(f"Stock comparison failed: {e}")
            return {"error": str(e)}

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs.iloc[-1]))

    async def _call_llm(
        self,
        prompt: str,
        api_key: str,
        provider: str,
        model: str,
        json_mode: bool = False
    ) -> str:
        """Call LLM API"""
        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7
            }

            if json_mode and "gpt-4" in model.lower():
                kwargs["response_format"] = {"type": "json_object"}

            completion = await client.chat.completions.create(**kwargs)
            return completion.choices[0].message.content

        elif provider == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            config = types.GenerateContentConfig(
                temperature=0.7,
                response_mime_type="application/json" if json_mode else None
            )

            completion = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=types.Content(parts=[types.Part(text=prompt)]),
                config=config
            )

            return completion.text

        else:
            raise ValueError(f"Unsupported provider: {provider}")


# Global instance
_llm_features_instance = None


def get_llm_features() -> LLMFeatures:
    """Get or create global LLM features instance"""
    global _llm_features_instance
    if _llm_features_instance is None:
        _llm_features_instance = LLMFeatures()
    return _llm_features_instance

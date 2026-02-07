"""
Deep Reasoning Agent
Uses chain-of-thought reasoning to generate trading recommendations
"""

from typing import Dict, Any
import json
import asyncio
from .base import BaseAgent


class DeepReasoningAgent(BaseAgent):
    """
    Agent responsible for deep analysis and recommendation generation
    Uses chain-of-thought reasoning with LLM
    """
    
    def __init__(self):
        super().__init__("Deep Reasoning Agent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate trading recommendation using chain-of-thought reasoning
        
        Args:
            state: Current state with all collected data
            
        Returns:
            Updated state with analysis result
        """
        try:
            symbol = state.get("symbol")
            stock_data = state.get("stock_data", {})
            indicators = state.get("technical_indicators", {})
            signals = state.get("technical_signals", {})
            percentile_scores = state.get("percentile_scores", {})
            rag_context = state.get("rag_context", "")
            model = state.get("model", "gpt-4.1")
            provider = state.get("provider", "openai")
            api_key = state.get("api_key")

            # Extract both provider API keys for potential ensemble analysis
            # The orchestrator may pass openai_api_key and/or gemini_api_key
            if provider == "openai" and api_key and not state.get("openai_api_key"):
                state["openai_api_key"] = api_key
            elif provider == "gemini" and api_key and not state.get("gemini_api_key"):
                state["gemini_api_key"] = api_key

            if not api_key:
                raise ValueError("API key not provided")
            
            self.log_execution(f"Analyzing {symbol} with {model}")
            
            # Add running step
            if "agent_steps" not in state:
                state["agent_steps"] = []
            
            state["agent_steps"].append(
                self.create_step_record(
                    status="running",
                    message=f"Performing deep analysis with {model}..."
                )
            )
            
            # Extract discovered patterns and historical events
            discovered_patterns = state.get("discovered_patterns", [])
            historical_events = state.get("historical_events", [])

            # Fetch recent news for context
            news_context = []
            news_sentiment = {}
            try:
                from news.web_search import search_stock_news
                from news.sentiment import get_sentiment_analyzer

                company_name = stock_data.get('name', '')
                news_results = await search_stock_news(
                    symbol=symbol,
                    company_name=company_name,
                    max_results=8,
                    days_back=7
                )
                news_context = news_results

                # Analyze sentiment of news
                if news_results:
                    analyzer = get_sentiment_analyzer()
                    articles = [{"title": n["title"], "content": n.get("snippet", "")} for n in news_results]
                    news_sentiment = analyzer.get_aggregate_sentiment(articles)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"News search failed (non-critical): {e}")

            # Get recent candlestick patterns
            candlestick_patterns = []
            try:
                from patterns.candlestick import get_pattern_detector
                import yfinance as yf

                # Get recent OHLCV data for pattern detection
                ticker = yf.Ticker(f"{symbol}.NS")
                hist = ticker.history(period="1mo")

                if not hist.empty:
                    detector = get_pattern_detector()
                    pattern_result = detector.detect_patterns(hist)
                    candlestick_patterns = pattern_result.get("patterns", [])
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Pattern detection failed (non-critical): {e}")

            # Generate candlestick chart images for visual LLM analysis
            chart_images = []
            try:
                from analysis.chart_generator import CandlestickChartGenerator
                chart_gen = CandlestickChartGenerator()

                # Daily chart from 1-month data (already fetched above as `hist`)
                if 'hist' in dir() and not hist.empty:
                    daily_chart = chart_gen.generate_daily_chart(hist, symbol, indicators)
                    if daily_chart:
                        chart_images.append(daily_chart)

                # Weekly chart from 6-month data (passed through state from analysis_agent)
                ohlcv_6mo = state.get("ohlcv_6mo")
                if ohlcv_6mo is not None and not ohlcv_6mo.empty:
                    weekly_chart = chart_gen.generate_weekly_chart(ohlcv_6mo, symbol, indicators)
                    if weekly_chart:
                        chart_images.append(weekly_chart)

                # 4-hour chart from intraday data
                try:
                    from data_providers.provider_manager import get_provider_manager
                    api_keys = state.get("data_provider_keys", {})
                    provider_mgr = get_provider_manager(api_keys if api_keys else None)
                    intraday_hist = await provider_mgr.get_historical_data(symbol, period="5d", interval="1h")
                    if intraday_hist is not None and len(intraday_hist) >= 10:
                        chart_4h = chart_gen.generate_4h_chart(intraday_hist, symbol, indicators)
                        if chart_4h:
                            chart_images.append(chart_4h)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"4hr chart generation failed (non-critical): {e}")

                if chart_images:
                    self.log_execution(f"Generated {len(chart_images)} chart image(s) for visual analysis")
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Chart generation failed (non-critical): {e}")

            # Build comprehensive prompt with RAG context and percentile scores
            prompt = self._build_analysis_prompt(
                symbol, stock_data, indicators, signals, percentile_scores, rag_context,
                discovered_patterns, historical_events,
                news_context=news_context, news_sentiment=news_sentiment,
                candlestick_patterns=candlestick_patterns,
                has_chart_images=len(chart_images) > 0
            )

            # Call LLM with chain-of-thought prompting (REAL API CALL)
            # Include chart images for multimodal analysis if available
            try:
                analysis_result = await self._call_llm(
                    prompt, model, provider, api_key,
                    images=chart_images if chart_images else None
                )
            except Exception as e:
                # Retry without images if multimodal call fails
                if chart_images:
                    self.log_execution("Multimodal LLM call failed, retrying text-only", "warning")
                    analysis_result = await self._call_llm(
                        prompt, model, provider, api_key, images=None
                    )
                else:
                    raise
            
            # Parse and validate result
            analysis_result = self._parse_llm_response(analysis_result)

            # Check for ensemble opportunity
            secondary_provider = "gemini" if provider == "openai" else "openai"
            secondary_key = state.get(f"{secondary_provider}_api_key")
            secondary_model = "gemini-2.0-flash" if secondary_provider == "gemini" else "gpt-4o-mini"

            ensemble_used = False
            if secondary_key:
                try:
                    secondary_result_raw = await self._call_llm(
                        prompt, secondary_model, secondary_provider, secondary_key,
                        images=chart_images if chart_images else None
                    )
                    # Parse secondary result
                    secondary_result = self._parse_llm_response(secondary_result_raw)

                    if secondary_result:
                        ensemble_used = True
                        analysis_result = self._merge_ensemble_results(analysis_result, secondary_result)
                except Exception as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Secondary model failed (using primary only): {e}")

            # Validate required fields
            required_fields = ["recommendation", "confidence", "reasoning"]
            for field in required_fields:
                if field not in analysis_result:
                    raise ValueError(f"Missing required field: {field}")

            # Add default values for optional enhanced fields if missing
            if "sector_analysis" not in analysis_result:
                analysis_result["sector_analysis"] = {
                    "sector_trend": "neutral",
                    "relative_strength": "inline",
                    "sector_catalysts": "N/A"
                }
            if "percentile_metrics" not in analysis_result:
                analysis_result["percentile_metrics"] = {
                    "rsi_percentile": "N/A",
                    "volatility_percentile": "N/A",
                    "volume_percentile": "N/A"
                }
            if "confidence_breakdown" not in analysis_result:
                # Keep backward compatibility
                analysis_result["confidence_breakdown"] = {
                    "technical_score": analysis_result.get("confidence", 50),
                    "momentum_score": analysis_result.get("confidence", 50),
                    "risk_score": analysis_result.get("confidence", 50),
                    "sector_score": 50
                }
            
            # Add model information
            analysis_result["model_used"] = model
            analysis_result["provider"] = provider
            if ensemble_used:
                analysis_result["ensemble_used"] = True
                analysis_result["models_used"] = [model, secondary_model]
            
            # Update state
            state["analysis_result"] = analysis_result
            state["recommendation"] = analysis_result.get("recommendation")
            state["confidence"] = analysis_result.get("confidence")
            state["reasoning"] = analysis_result.get("reasoning")
            
            # Update step to completed
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=f"Generated {analysis_result['recommendation']} recommendation",
                data={
                    "recommendation": analysis_result["recommendation"],
                    "confidence": analysis_result["confidence"],
                    "model": model
                }
            )
            
            self.log_execution(
                f"Analysis complete: {analysis_result['recommendation']} "
                f"({analysis_result['confidence']}% confidence)"
            )
            
            return state
            
        except Exception as e:
            return await self.handle_error(e, state)
    
    def _build_analysis_prompt(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        indicators: Dict[str, Any],
        signals: Dict[str, Any],
        percentile_scores: Dict[str, Any],
        rag_context: str,
        discovered_patterns: list = None,
        historical_events: list = None,
        news_context: list = None,
        news_sentiment: dict = None,
        candlestick_patterns: list = None,
        has_chart_images: bool = False
    ) -> str:
        """
        Build comprehensive analysis prompt with chain-of-thought structure
        Enhanced with industry benchmarking, role specialization, and percentile scoring

        Args:
            symbol: Stock symbol
            stock_data: Stock data
            indicators: Technical indicators
            signals: Technical signals
            percentile_scores: Percentile-based scores for context
            rag_context: RAG context
            discovered_patterns: Auto-discovered trading patterns
            historical_events: Historical news event impacts
            news_context: Recent news articles for the stock
            news_sentiment: Aggregate sentiment analysis of news
            candlestick_patterns: Detected candlestick patterns

        Returns:
            Formatted prompt
        """
        news_context = news_context or []
        news_sentiment = news_sentiment or {}
        candlestick_patterns = candlestick_patterns or []
        discovered_patterns = discovered_patterns or []
        historical_events = historical_events or []
        # Extract extended features for enhanced analysis
        current_price = indicators.get('current_price', stock_data.get('current_price'))
        lag_1d = indicators.get('lag_1d_price')
        lag_5d = indicators.get('lag_5d_price')
        volatility_10d = indicators.get('volatility_10d')
        volatility_20d = indicators.get('volatility_20d')
        price_vs_sma20 = indicators.get('price_vs_sma20_pct')
        price_vs_sma50 = indicators.get('price_vs_sma50_pct')
        return_1d = indicators.get('return_1d')
        return_5d = indicators.get('return_5d')
        return_20d = indicators.get('return_20d')
        volume_ratio = indicators.get('volume_ratio')

        # Calculate sector-relative metrics
        sector = stock_data.get('sector', 'Unknown')
        industry = stock_data.get('industry', 'Unknown')

        prompt = f"""You are a senior quantitative analyst with 15+ years of experience in Indian equity markets (NSE/BSE).
You specialize in technical analysis, sector rotation strategies, and risk-adjusted returns.

ROLE & EXPERTISE:
- Expert in Indian market microstructure, sectoral trends, and regulatory environment
- Proficient in statistical analysis, pattern recognition, and quantitative modeling
- Focus on data-driven decisions with clear risk-reward frameworks
- Experienced in comparing stocks against industry peers and sector benchmarks

Perform a comprehensive analysis using chain-of-thought reasoning with industry-relative context.

{rag_context if rag_context else ""}

=== FEW-SHOT EXAMPLES ===

Example 1 - Strong BUY Signal:
Stock: INFY at ₹1,450 with RSI=28, MACD bullish crossover, price below SMA20 but above SMA50
Analysis Result:
{{
  "recommendation": "BUY",
  "confidence": 78,
  "entry_price": 1450,
  "target_price": 1580,
  "stop_loss": 1395,
  "risk_reward_ratio": 2.36,
  "time_horizon": "medium_term",
  "reasoning": "1) PRICE ACTION: Stock corrected 8% from highs, now at support zone near SMA50. 2) MOMENTUM: RSI at 28 shows oversold condition, MACD just crossed above signal line indicating momentum shift. 3) TREND: Long-term uptrend intact as price above SMA50 and SMA200. 4) RISK: IT sector facing headwinds, but company fundamentals strong. 5) SYNTHESIS: Oversold bounce setup with favorable risk-reward, recommend accumulating.",
  "key_risks": ["Sector-wide correction in IT", "Rupee appreciation", "Client spending cuts"],
  "key_opportunities": ["Oversold bounce", "Strong order book", "AI/Cloud tailwinds"]
}}

Example 2 - SELL Signal:
Stock: ADANIENT at ₹2,850 with RSI=76, MACD bearish divergence, price 15% above SMA50
Analysis Result:
{{
  "recommendation": "SELL",
  "confidence": 72,
  "entry_price": 2850,
  "target_price": 2550,
  "stop_loss": 2950,
  "risk_reward_ratio": 3.0,
  "time_horizon": "short_term",
  "reasoning": "1) PRICE ACTION: Extended 15% above SMA50, at upper Bollinger Band. 2) MOMENTUM: RSI at 76 overbought, MACD showing bearish divergence with lower highs. 3) TREND: Short-term overextension in otherwise bullish trend. 4) RISK: Short squeeze possible given high short interest. 5) SYNTHESIS: Profit booking opportunity, recommend reducing exposure or shorting with tight stops.",
  "key_risks": ["Short squeeze", "Positive news catalyst", "Momentum continuation"],
  "key_opportunities": ["Mean reversion", "Profit booking by FIIs", "Technical resistance"]
}}

Example 3 - HOLD Signal:
Stock: HDFCBANK at ₹1,620 with RSI=52, MACD flat, price between SMA20 and SMA50
Analysis Result:
{{
  "recommendation": "HOLD",
  "confidence": 55,
  "entry_price": 1620,
  "target_price": 1720,
  "stop_loss": 1550,
  "risk_reward_ratio": 1.43,
  "time_horizon": "medium_term",
  "reasoning": "1) PRICE ACTION: Consolidating in narrow range between SMAs, no clear breakout. 2) MOMENTUM: RSI neutral at 52, MACD histogram flat near zero. 3) TREND: Sideways consolidation after recent rally. 4) RISK: Banking sector concerns persist but priced in. 5) SYNTHESIS: No clear entry signal, wait for breakout above 1,680 or breakdown below 1,560.",
  "key_risks": ["NPA concerns", "Rate cycle uncertainty", "Merger integration"],
  "key_opportunities": ["Breakout potential", "Valuation support", "Credit growth pickup"]
}}

=== CURRENT MARKET DATA ===

Stock: {stock_data.get('name', symbol)} ({symbol})
Sector: {sector}
Industry: {industry}

INDUSTRY CONTEXT:
When analyzing this stock, consider the following sector-specific factors:
- {sector} sector trends and rotation patterns in Indian markets
- Industry-specific regulatory changes or tailwinds/headwinds
- Peer comparison within {industry} (relative strength, valuation multiples)
- Sector-average volatility and typical trading ranges
- Recent FII/DII activity in the sector

Price Data:
- Current Price: ₹{current_price if current_price else stock_data.get('current_price', 'N/A')}
- Previous Close: ₹{stock_data.get('previous_close', 'N/A')}
- 1-Day Change: {stock_data.get('change_percent', 'N/A')}%
- Volume: {stock_data.get('volume', 'N/A'):,}
- Volume Ratio (vs 20D avg): {volume_ratio if volume_ratio else 'N/A'}x
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- 52W High: ₹{stock_data.get('week_52_high', 'N/A')}
- 52W Low: ₹{stock_data.get('week_52_low', 'N/A')}

Recent Price Action (Historical Context):
- Price 1 day ago: ₹{lag_1d if lag_1d else 'N/A'}
- Price 5 days ago: ₹{lag_5d if lag_5d else 'N/A'}
- 1-Day Return: {return_1d if return_1d else 'N/A'}%
- 5-Day Return: {return_5d if return_5d else 'N/A'}%
- 20-Day Return: {return_20d if return_20d else 'N/A'}%

Volatility Metrics (Recent History):
- 10-Day Volatility: {volatility_10d if volatility_10d else 'N/A'}%
- 20-Day Volatility: {volatility_20d if volatility_20d else 'N/A'}%
- ATR (14): {indicators.get('atr', 'N/A')} - Signal: {signals.get('volatility', 'N/A')}

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A')} - Signal: {signals.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')} - Signal: {signals.get('macd', 'N/A')}
- Histogram: {indicators.get('macd_histogram', 'N/A')}
- SMA 20: ₹{indicators.get('sma_20', 'N/A')} (Price {price_vs_sma20 if price_vs_sma20 else 'N/A'}% {'above' if price_vs_sma20 and price_vs_sma20 > 0 else 'below'})
- SMA 50: ₹{indicators.get('sma_50', 'N/A')} (Price {price_vs_sma50 if price_vs_sma50 else 'N/A'}% {'above' if price_vs_sma50 and price_vs_sma50 > 0 else 'below'})
- SMA 200: ₹{indicators.get('sma_200', 'N/A')}
- Bollinger Bands: ₹{indicators.get('bb_upper', 'N/A')} / ₹{indicators.get('bb_middle', 'N/A')} / ₹{indicators.get('bb_lower', 'N/A')}
- Stochastic K/D: {indicators.get('stochastic_k', 'N/A')} / {indicators.get('stochastic_d', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')}
- CCI: {indicators.get('cci', 'N/A')}
- OBV: {indicators.get('obv', 'N/A')}

Trend Analysis:
- Overall Trend: {signals.get('trend', 'N/A')}
- Momentum: {signals.get('momentum', 'N/A')}

PERCENTILE-BASED CONTEXT (6-Month Historical Distribution):
This section provides critical context by comparing current metrics to their 6-month historical range:

- RSI Percentile: {percentile_scores.get('rsi_interpretation', 'N/A')}
- Volatility Percentile: {percentile_scores.get('volatility_interpretation', 'N/A')}
- Volume Percentile: {percentile_scores.get('volume_interpretation', 'N/A')}
- Price Position: {percentile_scores.get('price_position_interpretation', 'N/A')}

COMPOSITE SCORE: {percentile_scores.get('composite_score', 'N/A')}/100
Interpretation: {percentile_scores.get('composite_interpretation', 'N/A')}

USE THESE PERCENTILE INSIGHTS in your analysis - they provide crucial context that raw indicator values alone cannot convey.
For example, RSI=35 might seem neutral, but if it's in the 10th percentile historically, it indicates extreme oversold conditions relative to this stock's normal range.
"""

        # Add discovered patterns section if available
        if discovered_patterns:
            prompt += "\n\n=== DISCOVERED TRADING PATTERNS (AUTO-MINED FROM HISTORICAL DATA) ===\n"
            prompt += "The following patterns were automatically discovered by analyzing 5+ years of historical data:\n\n"
            for i, pattern in enumerate(discovered_patterns[:3], 1):  # Show top 3
                metadata = pattern.get("metadata", {})
                prompt += f"Pattern {i}: {pattern.get('pattern_type', 'Unknown').replace('_', ' ').title()}\n"
                prompt += f"- Success Rate: {metadata.get('success_rate', 0):.1%} ({metadata.get('occurrences', 0)} occurrences)\n"
                prompt += f"- Average Return: {metadata.get('average_return', 0):+.2f}%\n"
                prompt += f"- Conditions: {pattern.get('content', '')[:200]}...\n"
                prompt += f"- Priority Score: {pattern.get('priority', 0)}/100\n\n"
            prompt += "IMPORTANT: Check if current market conditions match any of these validated patterns. If they do, factor this historical success rate into your recommendation.\n"

        # Add historical events section if available
        if historical_events:
            prompt += "\n\n=== HISTORICAL NEWS EVENT IMPACTS ===\n"
            prompt += "Past news events affecting this stock and how the market reacted:\n\n"
            for i, event in enumerate(historical_events[:3], 1):  # Show top 3
                metadata = event.get("metadata", {})
                prompt += f"Event {i}: {event.get('event_type', 'Unknown').replace('_', ' ').title()}\n"
                prompt += f"- Immediate Impact: {metadata.get('immediate_impact', {})}\n"
                prompt += f"- Recovery Timeline: {metadata.get('recovery_timeline', 'N/A')}\n"
                prompt += f"- Details: {event.get('content', '')[:200]}...\n\n"
            prompt += "IMPORTANT: Consider if similar events might occur soon and how the stock historically reacted to such catalysts.\n"

        # Add candlestick patterns section if available
        if candlestick_patterns:
            prompt += "\n\n=== RECENT CANDLESTICK PATTERNS (Last 30 Days) ===\n"
            for i, pattern in enumerate(candlestick_patterns, 1):
                pattern_name = pattern.get("name", "Unknown")
                pattern_date = pattern.get("date", "N/A")
                pattern_price = pattern.get("price", "N/A")
                pattern_signal = pattern.get("signal", "N/A")
                strength_score = pattern.get("strength_score", None)
                implication = pattern.get("implication", "N/A")
                strength_str = f", Strength: {strength_score}/100" if strength_score is not None else ""
                prompt += f"{i}. {pattern_name} — Date: {pattern_date}, Price: ₹{pattern_price}, Signal: {pattern_signal}{strength_str}\n"
                prompt += f"   Implication: {implication}\n"
            prompt += "\nIMPORTANT: Factor these recent candlestick patterns into your technical analysis and recommendation.\n"

        # Add news sentiment section if available
        if news_context:
            prompt += "\n\n=== RECENT NEWS SENTIMENT ===\n"
            if news_sentiment:
                agg_score = news_sentiment.get("score", "N/A")
                positive = news_sentiment.get("positive", 0)
                neutral = news_sentiment.get("neutral", 0)
                negative = news_sentiment.get("negative", 0)
                prompt += f"Aggregate Score: {agg_score}, Distribution: {positive} positive / {neutral} neutral / {negative} negative\n"
            prompt += "\nTop Headlines:\n"
            for i, article in enumerate(news_context[:8], 1):
                title = article.get("title", "N/A")
                source = article.get("source", "Unknown")
                sentiment_score = article.get("sentiment", "N/A")
                prompt += f"{i}. {title} ({source}) — Sentiment: {sentiment_score}\n"
            prompt += "\nIMPORTANT: Factor the news sentiment and recent headlines into your analysis. Positive news flow supports bullish thesis; negative news increases risk.\n"

        # Add visual chart analysis instructions if chart images are provided
        if has_chart_images:
            prompt += """

=== VISUAL CHART ANALYSIS (IMAGES ATTACHED) ===
Candlestick chart images are attached. This is your PRIMARY source for pattern recognition.

DAILY CHART (1-Month) - Image 1:
1. Identify candlestick patterns visible in the last 5-10 trading sessions
2. Assess short-term trend direction from candlestick progression and body sizes
3. Note price position relative to the Bollinger Bands (dashed gray lines) and SMAs (colored lines: blue=SMA20, red=SMA50)
4. Identify any gaps, wicks rejecting levels, or key support/resistance zones
5. Evaluate volume bars at the bottom — is volume confirming or diverging from price movement?

WEEKLY CHART (6-Month) - Image 2 (if present):
1. Identify the broader trend (uptrend channel, downtrend, consolidation range)
2. Spot chart patterns: head & shoulders, triangles, wedges, channels, double tops/bottoms, cup & handle
3. Identify major horizontal support and resistance zones
4. Note any trend line breaks or significant trend changes
5. Assess whether current price is at a historically significant level

4-HOUR CHART (5 Days) - Image 3 (if present):
1. Identify intraday support and resistance levels
2. Spot short-term momentum shifts and micro-patterns
3. Assess entry/exit timing precision for swing trades
4. Note any divergences between 4hr and daily timeframes
5. Evaluate volume patterns during market hours

IMPORTANT: Your visual analysis is the PRIMARY method for chart pattern identification. The rule-based patterns listed above are SUPPLEMENTARY. If you see patterns in the chart that were not detected algorithmically, INCLUDE them. If the chart contradicts the algorithmic detection, TRUST THE CHART.
"""

        prompt += """

=== ANALYSIS INSTRUCTIONS ===

Use the historical knowledge provided above and perform step-by-step chain-of-thought reasoning with INDUSTRY-RELATIVE perspective:

CRITICAL ANALYSIS FRAMEWORK:
1. Sector Context Analysis (NEW):
   - How is {sector} performing in current market conditions?
   - Is the stock outperforming or underperforming its sector peers?
   - Consider sector rotation patterns and institutional flows
   - Evaluate if sector-specific catalysts or headwinds exist

2. Recent Price Momentum (6-Week Window):
   - Analyze 1D, 5D, and 20D returns for trend acceleration/deceleration
   - Compare volume ratio - is accumulation/distribution happening?
   - Check if recent volatility (10D, 20D) is expanding or contracting
   - Assess if price action shows institutional participation

3. Price Action & Position Analysis:
   - Current price vs SMA20/50/200 (use percentage deviation metrics)
   - Evaluate if stock is at extremes (overbought/oversold relative to history)
   - Identify support/resistance zones and Bollinger Band positioning
   - Compare lag prices (1D, 5D ago) to gauge momentum shift

4. Technical Momentum Assessment:
   - RSI, Stochastic for overbought/oversold conditions
   - MACD histogram for momentum direction and divergences
   - ADX for trend strength
   - Synthesize: Is momentum building or fading?

5. Volatility & Risk Profile:
   - ATR and recent volatility (10D, 20D) for risk sizing
   - Compare to sector-average volatility
   - Bollinger Band width for volatility compression/expansion
   - Determine appropriate position sizing and stop-loss width

6. Pattern Recognition & Visual Chart Analysis:
   - Analyze the attached chart images for visual patterns (if provided)
   - Identify chart formations: trendlines, channels, H&S, triangles, wedges, double tops/bottoms
   - Cross-reference visual patterns with algorithmic detections
   - Look for similar setups in the same stock or sector peers
   - Consider seasonal patterns or event-driven catalysts

7. Risk-Reward Framework:
   - List specific risks: technical, sector-specific, macro factors
   - Identify catalysts and opportunities
   - Calculate entry, target, and stop-loss with clear R:R ratio
   - Ensure stop-loss accounts for recent volatility

8. Industry-Relative Scoring:
   - Technical strength vs sector peers
   - Momentum quality (sustained vs choppy)
   - Risk-adjusted return potential
   - Overall conviction level

Provide your analysis in the following JSON format:
{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100 (integer, based on signal strength and conviction),
  "entry_price": suggested entry price (number),
  "target_price": price target (number),
  "stop_loss": stop loss price (number),
  "risk_reward_ratio": calculated ratio (number),
  "time_horizon": "short_term" or "medium_term" or "long_term",
  "reasoning": "Comprehensive step-by-step chain-of-thought analysis. MUST include all 8 framework points: 1) Sector Context, 2) Recent Price Momentum (6-week), 3) Price Action & Position, 4) Technical Momentum, 5) Volatility & Risk, 6) Pattern Recognition, 7) Risk-Reward Framework, 8) Industry-Relative Scoring. Be specific about percentages, price levels, and comparative analysis vs sector peers.",
  "key_risks": ["risk1 (sector-specific)", "risk2 (technical)", "risk3 (macro/regulatory)"],
  "key_opportunities": ["opportunity1 (with catalyst)", "opportunity2 (with timeframe)"],
  "similar_patterns": "Any similar historical patterns from the knowledge base, with recency preference (last 6 weeks)",
  "sector_analysis": {{
    "sector_trend": "bullish/bearish/neutral - brief justification",
    "relative_strength": "outperforming/underperforming/inline - vs sector peers",
    "sector_catalysts": "any sector-specific tailwinds or headwinds"
  }},
  "confidence_breakdown": {{
    "technical_score": 0-100 (indicator alignment and signal quality),
    "momentum_score": 0-100 (trend strength and momentum sustainability),
    "risk_score": 0-100 (volatility, sector risk, macro factors),
    "sector_score": 0-100 (relative performance vs industry peers)
  }},
  "percentile_metrics": {{
    "rsi_percentile": "Current RSI relative to 6-month history (e.g., 'RSI at 28 is in 15th percentile - historically oversold')",
    "volatility_percentile": "Current volatility vs 6-month range (e.g., '10D volatility at 2.3% is in 70th percentile - above average')",
    "volume_percentile": "Volume ratio context (e.g., 'Volume 1.8x average is in 80th percentile - strong participation')"
  }}
}}

CRITICAL REQUIREMENTS:
1. Be SPECIFIC with price levels, percentages, and numerical analysis
2. Use the extended metrics (lag prices, volatility %, position vs SMAs, returns) in your reasoning
3. Provide industry-relative context - don't analyze in isolation
4. Include percentile-based insights for key metrics when possible
5. Prioritize recent patterns (last 6 weeks) over older historical data
6. Ensure reasoning directly references the 8-point analysis framework
7. Make recommendations actionable with clear entry/exit levels accounting for volatility"""
        
        return prompt

    def _parse_llm_response(self, raw_response) -> Dict[str, Any]:
        """
        Parse and extract JSON from an LLM response string.

        Args:
            raw_response: Raw string response from the LLM

        Returns:
            Parsed dictionary from the JSON response
        """
        if isinstance(raw_response, dict):
            return raw_response

        if isinstance(raw_response, str):
            try:
                return json.loads(raw_response)
            except json.JSONDecodeError:
                import re
                # Try to extract JSON from markdown code blocks
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw_response, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group(1))
                else:
                    # Try to find any JSON object
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw_response, re.DOTALL)
                    if json_match:
                        return json.loads(json_match.group(0))
                    else:
                        raise ValueError("Could not parse LLM response as JSON")

        raise ValueError(f"Unexpected response type: {type(raw_response)}")

    def _merge_ensemble_results(
        self, primary: Dict[str, Any], secondary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge results from two LLM providers using weighted averaging.

        Primary model gets 60% weight, secondary gets 40%.
        Direction agreement adds confidence bonus; disagreement applies penalty.

        Args:
            primary: Parsed result from the primary LLM
            secondary: Parsed result from the secondary LLM

        Returns:
            Merged analysis result
        """
        merged = dict(primary)

        # Determine direction agreement
        primary_rec = primary.get("recommendation", "HOLD")
        secondary_rec = secondary.get("recommendation", "HOLD")
        agreement = primary_rec == secondary_rec

        # Confidence: 60/40 weighted average with agreement bonus/penalty
        primary_conf = primary.get("confidence", 50)
        secondary_conf = secondary.get("confidence", 50)
        blended_confidence = int(primary_conf * 0.6 + secondary_conf * 0.4)

        if agreement:
            blended_confidence = min(100, blended_confidence + 5)
        else:
            blended_confidence = max(0, blended_confidence - 10)

        merged["confidence"] = blended_confidence

        # Use primary recommendation (with penalty already applied to confidence if disagreement)
        merged["recommendation"] = primary_rec

        # Merge and deduplicate risks
        primary_risks = primary.get("key_risks", [])
        secondary_risks = secondary.get("key_risks", [])
        seen_risks = set()
        merged_risks = []
        for risk in primary_risks + secondary_risks:
            risk_lower = risk.lower().strip()
            if risk_lower not in seen_risks:
                seen_risks.add(risk_lower)
                merged_risks.append(risk)
        merged["key_risks"] = merged_risks

        # Merge and deduplicate opportunities
        primary_opps = primary.get("key_opportunities", [])
        secondary_opps = secondary.get("key_opportunities", [])
        seen_opps = set()
        merged_opps = []
        for opp in primary_opps + secondary_opps:
            opp_lower = opp.lower().strip()
            if opp_lower not in seen_opps:
                seen_opps.add(opp_lower)
                merged_opps.append(opp)
        merged["key_opportunities"] = merged_opps

        # Add ensemble metadata
        merged["ensemble_used"] = True
        merged["ensemble_agreement"] = agreement

        return merged

    async def _call_llm(
        self,
        prompt: str,
        model: str,
        provider: str,
        api_key: str,
        images: list = None
    ) -> Dict[str, Any]:
        """
        Call LLM API for analysis (REAL API CALL).
        Supports multimodal inputs (text + chart images).

        Args:
            prompt: Analysis prompt
            model: Model name
            provider: Provider (openai/gemini)
            api_key: API key
            images: Optional list of PNG image bytes for visual analysis

        Returns:
            Analysis result
        """
        import base64

        system_message = "You are an expert stock analyst. Always respond with valid JSON matching the requested format exactly."

        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

            model_lower = model.lower()
            is_reasoning_model = any(x in model_lower for x in ['o1', 'o3'])

            # Build content — multimodal if images provided
            if images:
                content_parts = [{"type": "text", "text": prompt}]
                for img_bytes in images:
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{b64}",
                            "detail": "high"
                        }
                    })
                user_content = content_parts
            else:
                user_content = prompt

            if is_reasoning_model:
                # Reasoning models: no temperature, no system message
                if images:
                    # Prepend system message as text part
                    user_content = [{"type": "text", "text": system_message + "\n\n"}] + content_parts
                else:
                    user_content = f"{system_message}\n\n{prompt}"

                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": user_content}
                    ]
                )
            else:
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"} if "gpt-4" in model_lower else None
                )

            return completion.choices[0].message.content

        elif provider == "gemini":
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=api_key)

            # Build content parts — multimodal if images provided
            parts = [types.Part(text=prompt)]
            if images:
                for img_bytes in images:
                    parts.append(
                        types.Part(
                            inline_data=types.Blob(
                                mime_type="image/png",
                                data=img_bytes
                            )
                        )
                    )

            completion = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=types.Content(parts=parts),
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    system_instruction=system_message,
                    response_mime_type="application/json"
                )
            )

            return completion.text

        else:
            raise ValueError(f"Unsupported provider: {provider}")


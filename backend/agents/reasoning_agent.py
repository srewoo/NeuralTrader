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

            # Build comprehensive prompt with RAG context and percentile scores
            prompt = self._build_analysis_prompt(
                symbol, stock_data, indicators, signals, percentile_scores, rag_context,
                discovered_patterns, historical_events
            )
            
            # Call LLM with chain-of-thought prompting (REAL API CALL)
            analysis_result = await self._call_llm(
                prompt, model, provider, api_key
            )
            
            # Parse and validate result
            if isinstance(analysis_result, str):
                try:
                    analysis_result = json.loads(analysis_result)
                except json.JSONDecodeError:
                    # Try to extract JSON from markdown code blocks
                    import re
                    json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_result, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group(1))
                    else:
                        # Try to find any JSON object
                        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', analysis_result, re.DOTALL)
                        if json_match:
                            analysis_result = json.loads(json_match.group(0))
                        else:
                            raise ValueError("Could not parse LLM response as JSON")
            
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
        historical_events: list = None
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

        Returns:
            Formatted prompt
        """
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

6. Pattern Recognition & Historical Context:
   - Identify any chart patterns from RAG knowledge base
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
    
    async def _call_llm(
        self,
        prompt: str,
        model: str,
        provider: str,
        api_key: str
    ) -> Dict[str, Any]:
        """
        Call LLM API for analysis (REAL API CALL)
        
        Args:
            prompt: Analysis prompt
            model: Model name
            provider: Provider (openai/gemini)
            api_key: API key
            
        Returns:
            Analysis result
        """
        system_message = "You are an expert stock analyst. Always respond with valid JSON matching the requested format exactly."
        
        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)

            # OpenAI reasoning models (o1, o3 series) don't support temperature or system messages
            model_lower = model.lower()
            is_reasoning_model = any(x in model_lower for x in ['o1', 'o3'])

            if is_reasoning_model:
                # Reasoning models: no temperature, no system message, no response_format
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"{system_message}\n\n{prompt}"}
                    ]
                )
            else:
                # Standard models: include temperature and optional JSON mode
                completion = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"} if "gpt-4" in model_lower else None
                )

            return completion.choices[0].message.content
            
        elif provider == "gemini":
            from google import genai
            from google.genai import types
            
            client = genai.Client(api_key=api_key)
            
            completion = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=types.Content(
                    parts=[types.Part(text=prompt)]
                ),
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    system_instruction=system_message,
                    response_mime_type="application/json"
                )
            )
            
            return completion.text
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")


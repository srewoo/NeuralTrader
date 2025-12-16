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
            
            # Build comprehensive prompt with RAG context
            prompt = self._build_analysis_prompt(
                symbol, stock_data, indicators, signals, rag_context
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
        rag_context: str
    ) -> str:
        """
        Build comprehensive analysis prompt with chain-of-thought structure
        
        Args:
            symbol: Stock symbol
            stock_data: Stock data
            indicators: Technical indicators
            signals: Technical signals
            rag_context: RAG context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""You are an expert stock market analyst for Indian markets (NSE/BSE).
Perform a comprehensive analysis using chain-of-thought reasoning.

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
Sector: {stock_data.get('sector', 'N/A')}
Industry: {stock_data.get('industry', 'N/A')}

Price Data:
- Current Price: ₹{stock_data.get('current_price', 'N/A')}
- Previous Close: ₹{stock_data.get('previous_close', 'N/A')}
- Change: {stock_data.get('change_percent', 'N/A')}%
- Volume: {stock_data.get('volume', 'N/A'):,}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- 52W High: ₹{stock_data.get('week_52_high', 'N/A')}
- 52W Low: ₹{stock_data.get('week_52_low', 'N/A')}

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A')} - Signal: {signals.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')} - Signal: {signals.get('macd', 'N/A')}
- Histogram: {indicators.get('macd_histogram', 'N/A')}
- SMA 20: ₹{indicators.get('sma_20', 'N/A')}
- SMA 50: ₹{indicators.get('sma_50', 'N/A')}
- SMA 200: ₹{indicators.get('sma_200', 'N/A')}
- Bollinger Bands: ₹{indicators.get('bb_upper', 'N/A')} / ₹{indicators.get('bb_middle', 'N/A')} / ₹{indicators.get('bb_lower', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')} - Volatility: {signals.get('volatility', 'N/A')}
- Stochastic K/D: {indicators.get('stochastic_k', 'N/A')} / {indicators.get('stochastic_d', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')}
- CCI: {indicators.get('cci', 'N/A')}

Trend Analysis:
- Overall Trend: {signals.get('trend', 'N/A')}
- Momentum: {signals.get('momentum', 'N/A')}

=== ANALYSIS INSTRUCTIONS ===

Use the historical knowledge provided above and perform step-by-step chain-of-thought reasoning:

1. Price Action Analysis: Analyze current price relative to moving averages and support/resistance
2. Momentum Assessment: Evaluate RSI, Stochastic, and MACD for momentum direction
3. Trend Confirmation: Verify trend using multiple indicators (SMA, ADX)
4. Volatility Check: Assess risk using ATR and Bollinger Bands
5. Pattern Recognition: Identify any patterns from historical knowledge
6. Risk Assessment: List potential risks and concerns
7. Final Recommendation: Synthesize all factors into actionable recommendation

Provide your analysis in the following JSON format:
{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100 (integer),
  "entry_price": suggested entry price (number),
  "target_price": price target (number),
  "stop_loss": stop loss price (number),
  "risk_reward_ratio": calculated ratio (number),
  "time_horizon": "short_term" or "medium_term" or "long_term",
  "reasoning": "Step-by-step chain-of-thought analysis explaining your decision. Use numbered steps. Include: 1) Price action analysis, 2) Momentum assessment, 3) Trend confirmation, 4) Risk factors, 5) Final synthesis",
  "key_risks": ["risk1", "risk2", "risk3"],
  "key_opportunities": ["opportunity1", "opportunity2"],
  "similar_patterns": "Any similar historical patterns from the knowledge base",
  "confidence_breakdown": {{
    "technical_score": 0-100,
    "momentum_score": 0-100,
    "risk_score": 0-100
  }}
}}

Be specific with price levels and provide actionable, data-driven insights. Use the historical patterns from the knowledge base to support your analysis."""
        
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


"""
Ensemble LLM Analyzer
Uses multiple LLM models and aggregates their recommendations for higher confidence
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import json
import re
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class Recommendation(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class ModelResponse:
    """Response from a single model"""
    model: str
    provider: str
    recommendation: Recommendation
    confidence: int
    reasoning: str
    entry_price: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    key_risks: List[str]
    key_opportunities: List[str]
    success: bool
    error: Optional[str] = None


@dataclass
class EnsembleResult:
    """Aggregated result from ensemble analysis"""
    final_recommendation: Recommendation
    final_confidence: int
    consensus_level: str  # "strong", "moderate", "weak", "split"
    models_used: int
    models_agreed: int
    individual_responses: List[ModelResponse]
    aggregated_reasoning: str
    aggregated_risks: List[str]
    aggregated_opportunities: List[str]
    weighted_entry_price: Optional[float]
    weighted_target_price: Optional[float]
    weighted_stop_loss: Optional[float]
    average_risk_reward: Optional[float]


class EnsembleLLMAnalyzer:
    """
    Ensemble LLM Analyzer

    Calls multiple LLM providers in parallel and aggregates their recommendations
    using weighted voting based on model confidence scores.

    Supported providers:
    - OpenAI (GPT-4, GPT-4-turbo, GPT-3.5-turbo)
    - Google Gemini (gemini-1.5-pro, gemini-1.5-flash)
    - Anthropic Claude (claude-3-opus, claude-3-sonnet)
    """

    # Model weights for voting (based on typical performance)
    MODEL_WEIGHTS = {
        "gpt-4": 1.0,
        "gpt-4-turbo": 1.0,
        "gpt-4o": 1.0,
        "gpt-4.1": 1.0,
        "gpt-4o-mini": 0.8,
        "gpt-3.5-turbo": 0.7,
        "gemini-1.5-pro": 0.95,
        "gemini-1.5-flash": 0.8,
        "gemini-2.0-flash": 0.95,
        "gemini-2.0-flash-exp": 1.0,  # Latest Gemini 2.0
        "claude-3-opus": 1.0,
        "claude-3-sonnet": 0.9,
        "claude-3-haiku": 0.75,
        "claude-sonnet-4-20250514": 1.0,  # Claude Sonnet 4.5
    }

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        timeout_seconds: int = 60
    ):
        """
        Initialize ensemble analyzer with API keys

        Args:
            openai_api_key: OpenAI API key
            gemini_api_key: Google Gemini API key
            anthropic_api_key: Anthropic Claude API key
            timeout_seconds: Timeout for each model call
        """
        self.openai_api_key = openai_api_key
        self.gemini_api_key = gemini_api_key
        self.anthropic_api_key = anthropic_api_key
        self.timeout_seconds = timeout_seconds

    def get_available_models(self) -> List[Tuple[str, str]]:
        """
        Get list of available models based on configured API keys

        Returns:
            List of (model_name, provider) tuples
        """
        available = []

        if self.gemini_api_key:
            available.extend([
                ("gemini-1.5-flash", "gemini"),  # Gemini 1.5 Flash (stable, works with Indian stocks)
            ])

        if self.anthropic_api_key:
            available.extend([
                ("claude-sonnet-4-20250514", "anthropic"),  # Claude Sonnet 4.5
            ])

        if self.openai_api_key:
            available.extend([
                ("gpt-4o", "openai"),  # Latest GPT-4 Omni
            ])

        return available

    async def analyze_with_ensemble(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        technical_signals: Dict[str, Any],
        percentile_scores: Dict[str, Any],
        rag_context: str = "",
        models: Optional[List[Tuple[str, str]]] = None,
        min_models: int = 2
    ) -> EnsembleResult:
        """
        Run ensemble analysis using multiple LLM models

        Args:
            symbol: Stock symbol
            stock_data: Stock market data
            technical_indicators: Technical indicator values
            technical_signals: Technical signal interpretations
            percentile_scores: Percentile-based scores
            rag_context: RAG knowledge context
            models: List of (model, provider) tuples to use. If None, uses all available.
            min_models: Minimum number of models required for ensemble

        Returns:
            EnsembleResult with aggregated recommendation
        """
        # Get models to use
        if models is None:
            models = self.get_available_models()

        if len(models) < min_models:
            raise ValueError(
                f"Ensemble analysis requires at least {min_models} models, "
                f"but only {len(models)} are available. "
                "Please configure additional API keys in Settings."
            )

        logger.info(f"Running ensemble analysis with {len(models)} models: {[m[0] for m in models]}")

        # Build prompt
        prompt = self._build_analysis_prompt(
            symbol, stock_data, technical_indicators,
            technical_signals, percentile_scores, rag_context
        )

        # Call all models in parallel
        tasks = [
            self._call_model(model, provider, prompt)
            for model, provider in models
        ]

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Process responses
        model_responses: List[ModelResponse] = []
        for (model, provider), response in zip(models, responses):
            if isinstance(response, Exception):
                logger.error(f"Model {model} failed: {response}")
                model_responses.append(ModelResponse(
                    model=model,
                    provider=provider,
                    recommendation=Recommendation.HOLD,
                    confidence=0,
                    reasoning=f"Model failed: {str(response)}",
                    entry_price=None,
                    target_price=None,
                    stop_loss=None,
                    risk_reward_ratio=None,
                    key_risks=[],
                    key_opportunities=[],
                    success=False,
                    error=str(response)
                ))
            else:
                model_responses.append(response)

        # Filter successful responses
        successful_responses = [r for r in model_responses if r.success]

        if not successful_responses:
            raise ValueError("All models failed to generate recommendations")

        # Aggregate results
        result = self._aggregate_responses(successful_responses)
        result.individual_responses = model_responses
        result.models_used = len(model_responses)

        logger.info(
            f"Ensemble result: {result.final_recommendation.value} "
            f"({result.final_confidence}% confidence, {result.consensus_level} consensus)"
        )

        return result

    def _build_analysis_prompt(
        self,
        symbol: str,
        stock_data: Dict[str, Any],
        indicators: Dict[str, Any],
        signals: Dict[str, Any],
        percentile_scores: Dict[str, Any],
        rag_context: str
    ) -> str:
        """Build analysis prompt for LLM"""

        current_price = indicators.get('current_price', stock_data.get('current_price'))

        # Calculate key price levels for the prompt
        week_52_high = stock_data.get('week_52_high', current_price)
        week_52_low = stock_data.get('week_52_low', current_price)
        price_range = week_52_high - week_52_low if week_52_high and week_52_low else 0
        price_position = ((current_price - week_52_low) / price_range * 100) if price_range > 0 else 50

        prompt = f"""You are a senior quantitative analyst specializing in Indian equity markets (NSE/BSE) with 15+ years of experience in technical analysis, risk management, and algorithmic trading.

{rag_context if rag_context else ""}

=== MARKET DATA: {stock_data.get('name', symbol)} ({symbol}) ===

**Company Overview:**
- Sector: {stock_data.get('sector', 'Unknown')}
- Industry: {stock_data.get('industry', 'Unknown')}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}

**Price Action:**
- Current Price: ₹{current_price:.2f}
- Previous Close: ₹{stock_data.get('previous_close', 'N/A')}
- Daily Change: {stock_data.get('change_percent', 'N/A')}%
- Volume: {stock_data.get('volume', 'N/A')}
- 52-Week High: ₹{week_52_high}
- 52-Week Low: ₹{week_52_low}
- Position in Range: {price_position:.1f}% (0%=low, 100%=high)

**Technical Indicators:**
- RSI(14): {indicators.get('rsi', 'N/A')} → {signals.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')} → {signals.get('macd', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')} (Trend Strength)
- Stochastic: K={indicators.get('stochastic_k', 'N/A')} D={indicators.get('stochastic_d', 'N/A')}

**Moving Averages:**
- SMA(20): ₹{indicators.get('sma_20', 'N/A')} - Price vs MA20: {((current_price / indicators.get('sma_20', current_price) - 1) * 100) if indicators.get('sma_20') else 0:.2f}%
- SMA(50): ₹{indicators.get('sma_50', 'N/A')} - Price vs MA50: {((current_price / indicators.get('sma_50', current_price) - 1) * 100) if indicators.get('sma_50') else 0:.2f}%
- SMA(200): ₹{indicators.get('sma_200', 'N/A')} - Price vs MA200: {((current_price / indicators.get('sma_200', current_price) - 1) * 100) if indicators.get('sma_200') else 0:.2f}%
- Trend Classification: {signals.get('trend', 'N/A')}

**Volatility & Support/Resistance:**
- ATR(14): {indicators.get('atr', 'N/A')} (Average True Range)
- Bollinger Bands (20,2):
  * Upper: ₹{indicators.get('bb_upper', 'N/A')} (+{((indicators.get('bb_upper', current_price) / current_price - 1) * 100) if indicators.get('bb_upper') else 0:.1f}%)
  * Middle: ₹{indicators.get('bb_middle', 'N/A')}
  * Lower: ₹{indicators.get('bb_lower', 'N/A')} (-{((1 - indicators.get('bb_lower', current_price) / current_price) * 100) if indicators.get('bb_lower') else 0:.1f}%)
  * Width: {((indicators.get('bb_upper', current_price) - indicators.get('bb_lower', current_price)) / indicators.get('bb_middle', current_price) * 100) if indicators.get('bb_middle') else 0:.1f}%

**Market Context:**
- Composite Score: {percentile_scores.get('composite_score', 'N/A')}/100
- RSI Context: {percentile_scores.get('rsi_interpretation', 'N/A')}
- Volume Analysis: {percentile_scores.get('volume_interpretation', 'N/A')}

=== ANALYSIS FRAMEWORK ===

**Your Task:** Provide a professional trading recommendation following institutional-grade analysis standards.

**Analysis Requirements:**

1. **Multi-Timeframe Confirmation:**
   - Analyze alignment across short-term (SMA20), medium-term (SMA50), and long-term (SMA200) trends
   - Identify trend strength using ADX (>25 = strong trend, <20 = weak/ranging)

2. **Support & Resistance Identification:**
   - Use Bollinger Bands, 52W high/low, and moving averages to identify key price levels
   - Current price position relative to these levels determines entry/exit zones

3. **Momentum & Oscillator Analysis:**
   - RSI: <30 oversold, >70 overbought, 40-60 neutral zone
   - Stochastic: Crossovers and divergence signals
   - MACD: Histogram momentum and signal line crosses

4. **Volume Confirmation:**
   - Volume should confirm price moves (strong volume on breakouts = valid signal)
   - Low volume on rallies = weak/suspect moves

5. **Risk Management (CRITICAL):**
   - Stop Loss: Must be based on ATR or technical support/resistance (NOT arbitrary)
   - Risk-Reward Ratio: Minimum 2:1 (target profit ≥ 2x stop loss distance)
   - Entry Price: Specify optimal entry zone (not just current price)
   - Position Sizing: Consider volatility (ATR) for position risk

6. **Confidence Calibration:**
   - 80-100%: Multiple strong confirmations, clear trend, high conviction
   - 60-79%: Good technical setup with minor concerns
   - 40-59%: Mixed signals, borderline setup
   - 20-39%: Weak setup, many conflicting signals
   - 0-19%: Poor setup, avoid trading

**Output Format (STRICT JSON):**

{{
  "recommendation": "BUY" | "SELL" | "HOLD",
  "confidence": <integer 0-100>,
  "entry_price": <number - optimal entry level, not just current price>,
  "target_price": <number - realistic based on resistance/support>,
  "stop_loss": <number - based on ATR or technical level>,
  "risk_reward_ratio": <number - must be ≥1.5, ideally ≥2.0>,
  "reasoning": "<3-5 sentences: Synthesize trend, momentum, support/resistance, and volume. Explain WHY this setup is tradeable. Reference specific indicator values.>",
  "key_risks": [
    "<Specific technical risk - e.g., 'Breaking below SMA50 at ₹1650 invalidates bullish setup'>",
    "<Market/volatility risk with numbers - e.g., 'High ATR of 45 indicates 2.7% daily volatility'>",
    "<Sector/external risk - e.g., 'Sector weakness or adverse news could trigger stop loss'>"
  ],
  "key_opportunities": [
    "<Specific bullish catalyst - e.g., 'Golden cross forming as SMA50 approaches SMA200'>",
    "<Price level opportunity - e.g., 'Breakout above ₹1750 resistance opens path to ₹1850'>",
    "<Technical setup - e.g., 'RSI recovery from oversold with positive divergence'>"
  ]
}}

**CRITICAL RULES:**
✓ Entry price must be realistic (within ±2% of current price or at specific trigger level)
✓ Stop loss must protect capital (3-5% max loss for swing trades, use ATR as guide)
✓ Target must be achievable (based on nearby resistance, not wishful thinking)
✓ Risk-reward ratio ≥ 1.5 (reject setups with poor risk-reward)
✓ Be honest: If setup is unclear → HOLD with lower confidence
✓ Use actual numbers from data (don't invent values)
✓ Calculate risk_reward_ratio = (target_price - entry_price) / (entry_price - stop_loss)

Provide your analysis now in valid JSON format only (no additional text)."""

        return prompt

    async def _call_model(
        self,
        model: str,
        provider: str,
        prompt: str
    ) -> ModelResponse:
        """
        Call a single LLM model

        Args:
            model: Model name
            provider: Provider name (openai, gemini, anthropic)
            prompt: Analysis prompt

        Returns:
            ModelResponse with analysis result
        """
        try:
            system_message = """You are a senior quantitative analyst and portfolio manager with CFA charter and 15+ years of experience in Indian equity markets (NSE/BSE). Your expertise includes:
- Technical analysis (price action, indicators, chart patterns)
- Quantitative risk management (VaR, position sizing, portfolio optimization)
- Algorithmic trading systems and backtesting
- Market microstructure and execution algorithms

You provide institutional-grade analysis with precise entry/exit levels, proper risk-reward ratios (minimum 2:1), and realistic price targets based on technical support/resistance levels. You always respond with valid JSON matching the requested format exactly, using actual data values and calculations."""

            if provider == "openai":
                result = await self._call_openai(model, prompt, system_message)
            elif provider == "gemini":
                result = await self._call_gemini(model, prompt, system_message)
            elif provider == "anthropic":
                result = await self._call_anthropic(model, prompt, system_message)
            else:
                raise ValueError(f"Unsupported provider: {provider}")

            # Parse JSON response
            parsed = self._parse_response(result)

            return ModelResponse(
                model=model,
                provider=provider,
                recommendation=Recommendation(parsed.get('recommendation', 'HOLD')),
                confidence=parsed.get('confidence', 50),
                reasoning=parsed.get('reasoning', ''),
                entry_price=parsed.get('entry_price'),
                target_price=parsed.get('target_price'),
                stop_loss=parsed.get('stop_loss'),
                risk_reward_ratio=parsed.get('risk_reward_ratio'),
                key_risks=parsed.get('key_risks', []),
                key_opportunities=parsed.get('key_opportunities', []),
                success=True
            )

        except Exception as e:
            logger.error(f"Failed to call {model}: {e}")
            return ModelResponse(
                model=model,
                provider=provider,
                recommendation=Recommendation.HOLD,
                confidence=0,
                reasoning=str(e),
                entry_price=None,
                target_price=None,
                stop_loss=None,
                risk_reward_ratio=None,
                key_risks=[],
                key_opportunities=[],
                success=False,
                error=str(e)
            )

    async def _call_openai(self, model: str, prompt: str, system_message: str) -> str:
        """Call OpenAI API"""
        import openai

        client = openai.AsyncOpenAI(api_key=self.openai_api_key)

        completion = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"} if "gpt-4" in model.lower() else None
            ),
            timeout=self.timeout_seconds
        )

        return completion.choices[0].message.content

    async def _call_gemini(self, model: str, prompt: str, system_message: str) -> str:
        """Call Google Gemini API"""
        import google.generativeai as genai

        genai.configure(api_key=self.gemini_api_key)

        # Create model with system instruction
        generation_config = {
            "temperature": 0.7,
            "response_mime_type": "application/json",
        }

        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_message
        )

        # Generate content
        completion = await asyncio.wait_for(
            asyncio.to_thread(
                gemini_model.generate_content,
                prompt
            ),
            timeout=self.timeout_seconds
        )

        return completion.text

    async def _call_anthropic(self, model: str, prompt: str, system_message: str) -> str:
        """Call Anthropic Claude API"""
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=self.anthropic_api_key)

        message = await asyncio.wait_for(
            client.messages.create(
                model=model,
                max_tokens=2000,
                system=system_message,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            ),
            timeout=self.timeout_seconds
        )

        return message.content[0].text

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            # Try to find any JSON object
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))

            raise ValueError("Could not parse LLM response as JSON")

    def _aggregate_responses(self, responses: List[ModelResponse]) -> EnsembleResult:
        """
        Aggregate responses from multiple models using weighted voting

        Args:
            responses: List of successful model responses

        Returns:
            Aggregated EnsembleResult
        """
        # Calculate weighted votes
        vote_scores = {
            Recommendation.BUY: 0.0,
            Recommendation.SELL: 0.0,
            Recommendation.HOLD: 0.0
        }

        total_weight = 0.0

        for response in responses:
            # Get model weight
            weight = self.MODEL_WEIGHTS.get(response.model, 0.8)

            # Adjust weight by confidence
            adjusted_weight = weight * (response.confidence / 100)

            # Add to vote
            vote_scores[response.recommendation] += adjusted_weight
            total_weight += adjusted_weight

        # Normalize scores
        if total_weight > 0:
            for rec in vote_scores:
                vote_scores[rec] /= total_weight

        # Get winning recommendation
        final_recommendation = max(vote_scores, key=vote_scores.get)

        # Count agreements
        models_agreed = sum(1 for r in responses if r.recommendation == final_recommendation)

        # Calculate consensus level
        winning_score = vote_scores[final_recommendation]
        if winning_score >= 0.7:
            consensus_level = "strong"
        elif winning_score >= 0.5:
            consensus_level = "moderate"
        elif winning_score >= 0.4:
            consensus_level = "weak"
        else:
            consensus_level = "split"

        # Calculate weighted average confidence
        total_confidence_weight = 0.0
        weighted_confidence = 0.0

        for response in responses:
            if response.recommendation == final_recommendation:
                weight = self.MODEL_WEIGHTS.get(response.model, 0.8)
                weighted_confidence += response.confidence * weight
                total_confidence_weight += weight

        final_confidence = int(weighted_confidence / total_confidence_weight) if total_confidence_weight > 0 else 50

        # Aggregate price targets (weighted average from agreeing models)
        entry_prices = []
        target_prices = []
        stop_losses = []
        risk_rewards = []

        for response in responses:
            if response.recommendation == final_recommendation:
                weight = self.MODEL_WEIGHTS.get(response.model, 0.8)
                if response.entry_price:
                    entry_prices.append((response.entry_price, weight))
                if response.target_price:
                    target_prices.append((response.target_price, weight))
                if response.stop_loss:
                    stop_losses.append((response.stop_loss, weight))
                if response.risk_reward_ratio:
                    risk_rewards.append((response.risk_reward_ratio, weight))

        def weighted_avg(values_weights: List[Tuple[float, float]]) -> Optional[float]:
            if not values_weights:
                return None
            total_weight = sum(w for _, w in values_weights)
            if total_weight == 0:
                return None
            return round(sum(v * w for v, w in values_weights) / total_weight, 2)

        # Aggregate reasoning
        reasonings = [r.reasoning for r in responses if r.recommendation == final_recommendation and r.reasoning]
        aggregated_reasoning = self._synthesize_reasonings(reasonings, final_recommendation)

        # Aggregate risks and opportunities
        all_risks = []
        all_opportunities = []
        for response in responses:
            all_risks.extend(response.key_risks)
            all_opportunities.extend(response.key_opportunities)

        # Deduplicate and rank by frequency
        def ranked_unique(items: List[str], limit: int = 5) -> List[str]:
            from collections import Counter
            counts = Counter(items)
            return [item for item, _ in counts.most_common(limit)]

        return EnsembleResult(
            final_recommendation=final_recommendation,
            final_confidence=final_confidence,
            consensus_level=consensus_level,
            models_used=len(responses),
            models_agreed=models_agreed,
            individual_responses=[],  # Will be set by caller
            aggregated_reasoning=aggregated_reasoning,
            aggregated_risks=ranked_unique(all_risks),
            aggregated_opportunities=ranked_unique(all_opportunities),
            weighted_entry_price=weighted_avg(entry_prices),
            weighted_target_price=weighted_avg(target_prices),
            weighted_stop_loss=weighted_avg(stop_losses),
            average_risk_reward=weighted_avg(risk_rewards)
        )

    def _synthesize_reasonings(
        self,
        reasonings: List[str],
        recommendation: Recommendation
    ) -> str:
        """Synthesize reasoning from multiple models"""
        if not reasonings:
            return "No detailed reasoning available."

        if len(reasonings) == 1:
            return reasonings[0]

        # Combine key points from each reasoning
        synthesis = f"ENSEMBLE ANALYSIS ({len(reasonings)} models agree on {recommendation.value}):\n\n"

        for i, reasoning in enumerate(reasonings[:3], 1):
            # Truncate long reasonings
            truncated = reasoning[:500] + "..." if len(reasoning) > 500 else reasoning
            synthesis += f"Model {i} perspective:\n{truncated}\n\n"

        return synthesis

    async def get_ai_response(self, prompt: str, model: str = None, system_message: str = None) -> str:
        """
        Get AI response for a custom prompt using available LLM models

        Args:
            prompt: Custom prompt to send to the model
            model: Optional model name (defaults to first available)
            system_message: Optional system message for the model

        Returns:
            str: Model response text
        """
        if system_message is None:
            system_message = "You are a helpful AI assistant specialized in financial analysis."

        # Get available models
        available_models = self.get_available_models()
        if not available_models:
            raise ValueError("No LLM models configured. Please provide API keys.")

        # Use specified model or first available
        if model:
            model_info = next((m for m in available_models if m[0] == model), None)
            if not model_info:
                raise ValueError(f"Model {model} not available")
        else:
            model_info = available_models[0]

        model_name, provider = model_info

        # Call the appropriate model
        try:
            if provider == "openai":
                return await self._call_openai(model_name, prompt, system_message)
            elif provider == "gemini":
                return await self._call_gemini(model_name, prompt, system_message)
            elif provider == "anthropic":
                return await self._call_anthropic(model_name, prompt, system_message)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        except Exception as e:
            logger.error(f"Failed to get AI response from {model_name}: {e}")
            raise

    async def get_market_overview(self) -> Dict[str, Any]:
        """
        Get AI-powered market overview analysis

        Returns:
            Dict with market analysis
        """
        prompt = """Analyze the current state of the Indian stock market (NSE/BSE).

Provide a comprehensive market overview including:
1. Current market trend (Bullish/Bearish/Sideways)
2. Key support and resistance levels for major indices (NIFTY, SENSEX)
3. Sectoral performance and leaders
4. Key market drivers and risks
5. Short-term outlook (1-2 weeks)
6. Investment recommendations (sectors to watch)

Format your response as a structured JSON with the following keys:
{
  "market_trend": "Bullish/Bearish/Sideways",
  "nifty_analysis": "...",
  "sensex_analysis": "...",
  "top_sectors": ["sector1", "sector2", "sector3"],
  "bottom_sectors": ["sector1", "sector2"],
  "key_drivers": ["driver1", "driver2"],
  "key_risks": ["risk1", "risk2"],
  "outlook": "...",
  "recommendations": ["recommendation1", "recommendation2"]
}"""

        response = await self.get_ai_response(prompt)

        try:
            import json
            return json.loads(response)
        except json.JSONDecodeError:
            # If response is not valid JSON, return as text
            return {"analysis": response}


# Factory function
def get_ensemble_analyzer(
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None
) -> EnsembleLLMAnalyzer:
    """
    Create ensemble analyzer with available API keys

    Args:
        openai_api_key: OpenAI API key
        gemini_api_key: Google Gemini API key
        anthropic_api_key: Anthropic Claude API key

    Returns:
        Configured EnsembleLLMAnalyzer
    """
    return EnsembleLLMAnalyzer(
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key
    )

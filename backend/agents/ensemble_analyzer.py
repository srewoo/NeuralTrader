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
        "gemini-2.0-flash": 0.85,
        "claude-3-opus": 1.0,
        "claude-3-sonnet": 0.9,
        "claude-3-haiku": 0.75,
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

        if self.openai_api_key:
            available.extend([
                ("gpt-4o", "openai"),
                ("gpt-4o-mini", "openai"),
            ])

        if self.gemini_api_key:
            available.extend([
                ("gemini-2.0-flash", "gemini"),
                ("gemini-1.5-pro", "gemini"),
            ])

        if self.anthropic_api_key:
            available.extend([
                ("claude-3-sonnet-20240229", "anthropic"),
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

        prompt = f"""You are an expert stock analyst. Analyze the following stock data and provide a trading recommendation.

{rag_context if rag_context else ""}

=== STOCK DATA ===
Stock: {stock_data.get('name', symbol)} ({symbol})
Sector: {stock_data.get('sector', 'Unknown')}
Industry: {stock_data.get('industry', 'Unknown')}

Price Data:
- Current Price: ₹{current_price}
- Previous Close: ₹{stock_data.get('previous_close', 'N/A')}
- Change: {stock_data.get('change_percent', 'N/A')}%
- Volume: {stock_data.get('volume', 'N/A')}
- Market Cap: {stock_data.get('market_cap', 'N/A')}
- P/E Ratio: {stock_data.get('pe_ratio', 'N/A')}
- 52W High: ₹{stock_data.get('week_52_high', 'N/A')}
- 52W Low: ₹{stock_data.get('week_52_low', 'N/A')}

Technical Indicators:
- RSI (14): {indicators.get('rsi', 'N/A')} - Signal: {signals.get('rsi', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')}
- SMA 20: ₹{indicators.get('sma_20', 'N/A')}
- SMA 50: ₹{indicators.get('sma_50', 'N/A')}
- SMA 200: ₹{indicators.get('sma_200', 'N/A')}
- Bollinger Bands: ₹{indicators.get('bb_upper', 'N/A')} / ₹{indicators.get('bb_middle', 'N/A')} / ₹{indicators.get('bb_lower', 'N/A')}
- ATR: {indicators.get('atr', 'N/A')}
- Stochastic K/D: {indicators.get('stochastic_k', 'N/A')} / {indicators.get('stochastic_d', 'N/A')}
- ADX: {indicators.get('adx', 'N/A')}

Trend Analysis:
- Overall Trend: {signals.get('trend', 'N/A')}
- Momentum: {signals.get('momentum', 'N/A')}

Percentile Context:
- Composite Score: {percentile_scores.get('composite_score', 'N/A')}/100
- RSI Interpretation: {percentile_scores.get('rsi_interpretation', 'N/A')}
- Volume Interpretation: {percentile_scores.get('volume_interpretation', 'N/A')}

=== ANALYSIS INSTRUCTIONS ===
Perform a comprehensive technical analysis and provide your recommendation in the following JSON format:

{{
  "recommendation": "BUY" or "SELL" or "HOLD",
  "confidence": 0-100 (integer),
  "entry_price": suggested entry price (number),
  "target_price": price target (number),
  "stop_loss": stop loss price (number),
  "risk_reward_ratio": calculated ratio (number),
  "reasoning": "Your detailed analysis and reasoning",
  "key_risks": ["risk1", "risk2", "risk3"],
  "key_opportunities": ["opportunity1", "opportunity2", "opportunity3"]
}}

Be specific with price levels and percentages. Provide actionable recommendations."""

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
            system_message = "You are an expert stock analyst. Always respond with valid JSON matching the requested format exactly."

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
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.gemini_api_key)

        completion = await asyncio.wait_for(
            asyncio.to_thread(
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

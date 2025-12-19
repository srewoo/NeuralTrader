"""
Qualitative Analysis Module
Uses LLM to analyze "soft" factors like Economic Moat, Management Quality, and Brand Strength.
"""

import logging
import json
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class QualitativeAnalyzer:
    """
    Analyzes qualitative aspects of a company using LLM.
    """

    async def analyze_moat_and_management(
        self,
        symbol: str,
        company_summary: str,
        news_headlines: list[str],
        api_key: str,
        provider: str = "openai",
        model: str = "gpt-4.1"
    ) -> Dict[str, Any]:
        """
        Analyze competitive advantage (Moat) and Management Quality.
        """
        try:
            if not company_summary:
                return {"error": "No company summary provided"}

            news_text = "\n".join([f"- {h}" for h in news_headlines[:10]])

            prompt = f"""Analyze the competitive advantage (Economic Moat) and Management Quality for {symbol}.
            
            Company Description:
            {company_summary}
            
            Recent News Headlines:
            {news_text}
            
            Provide a JSON response with the following analysis:
            {{
                "economic_moat": {{
                    "rating": "Wide/Narrow/None",
                    "source": "Network Effect/Switching Costs/Cost Advantage/Intangible Assets/Efficient Scale/None",
                    "explanation": "Why this moat exists or doesn't"
                }},
                "management_quality": {{
                    "rating": "Exemplary/Standard/Poor",
                    "integrity_check": "Any news indicating fraud or controversy?",
                    "capital_allocation": "Good/Poor/Unknown based on news/description"
                }},
                "brand_strength": {{
                    "rating": "Strong/Moderate/Weak",
                    "explanation": "Assessment of brand power"
                }},
                "overall_qualitative_score": <0-100 score based on above factors>
            }}
            
            Respond with ONLY valid JSON.
            """
            
            response_text = await self._call_llm(prompt, api_key, provider, model, json_mode=True)
            
            try:
                result = json.loads(response_text)
            except:
                result = {"error": "Failed to parse LLM response", "raw_response": response_text}
                
            return result

        except Exception as e:
            logger.error(f"Qualitative analysis failed: {e}")
            return {"error": str(e)}

    async def _call_llm(
        self,
        prompt: str,
        api_key: str,
        provider: str,
        model: str,
        json_mode: bool = False
    ) -> str:
        """Helper to call LLM (duplicated from LLMFeatures to avoid circular dep for now/keep independent)"""
        # ... logic similar to LLMFeatures ...
        if provider == "openai":
            import openai
            client = openai.AsyncOpenAI(api_key=api_key)
            
            kwargs = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3 # Lower temp for analysis
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
                temperature=0.3,
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

_qualitative_analyzer = None

def get_qualitative_analyzer() -> QualitativeAnalyzer:
    global _qualitative_analyzer
    if _qualitative_analyzer is None:
        _qualitative_analyzer = QualitativeAnalyzer()
    return _qualitative_analyzer

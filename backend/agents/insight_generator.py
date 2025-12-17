"""
Natural Language Insight Generator
Converts technical analysis into clear, actionable insights for users
"""

from typing import Dict, Any, List


class InsightGenerator:
    """
    Generate natural language insights from technical analysis
    Based on best practices from repository analysis
    """

    def generate_insights(
        self,
        analysis_result: Dict[str, Any],
        technical_indicators: Dict[str, Any],
        percentile_scores: Dict[str, Any],
        stock_data: Dict[str, Any]
    ) -> List[str]:
        """
        Generate natural language insights from analysis

        Args:
            analysis_result: AI analysis result
            technical_indicators: Technical indicators
            percentile_scores: Percentile scores
            stock_data: Stock data

        Returns:
            List of insight strings
        """
        insights = []

        # 1. Primary Recommendation Insight
        recommendation = analysis_result.get("recommendation", "HOLD")
        confidence = analysis_result.get("confidence", 50)
        entry_price = analysis_result.get("entry_price")
        target_price = analysis_result.get("target_price")
        stop_loss = analysis_result.get("stop_loss")

        if recommendation == "BUY":
            action_insight = f"🟢 **{recommendation}** signal with {confidence}% confidence"
            if entry_price and target_price:
                upside = ((target_price - entry_price) / entry_price) * 100
                action_insight += f" - Potential upside of {upside:.1f}% to target ₹{target_price:.2f}"
        elif recommendation == "SELL":
            action_insight = f"🔴 **{recommendation}** signal with {confidence}% confidence"
            if entry_price and target_price:
                downside = ((entry_price - target_price) / entry_price) * 100
                action_insight += f" - Potential downside of {downside:.1f}% to target ₹{target_price:.2f}"
        else:
            action_insight = f"🟡 **{recommendation}** recommendation with {confidence}% confidence - Wait for clearer signals"

        insights.append(action_insight)

        # 2. Risk-Reward Insight
        risk_reward = analysis_result.get("risk_reward_ratio")
        if risk_reward and stop_loss:
            if risk_reward >= 2.0:
                insights.append(
                    f"✅ Favorable risk-reward ratio of {risk_reward:.2f}:1 with stop-loss at ₹{stop_loss:.2f}"
                )
            elif risk_reward >= 1.5:
                insights.append(
                    f"⚖️ Acceptable risk-reward ratio of {risk_reward:.2f}:1 with stop-loss at ₹{stop_loss:.2f}"
                )
            else:
                insights.append(
                    f"⚠️ Limited risk-reward ratio of {risk_reward:.2f}:1 - Consider waiting for better setup"
                )

        # 3. Percentile Context Insights
        composite_score = percentile_scores.get("composite_score")
        if composite_score is not None:
            if composite_score >= 70:
                insights.append(
                    f"📊 Composite score of {composite_score}/100 indicates strong bullish alignment across indicators"
                )
            elif composite_score <= 30:
                insights.append(
                    f"📊 Composite score of {composite_score}/100 indicates strong bearish alignment across indicators"
                )
            else:
                insights.append(
                    f"📊 Composite score of {composite_score}/100 shows mixed signals - exercise caution"
                )

        # 4. RSI Context Insight
        rsi_interpretation = percentile_scores.get("rsi_interpretation")
        if rsi_interpretation and rsi_interpretation != "Insufficient data":
            insights.append(f"📈 {rsi_interpretation}")

        # 5. Volume Insight
        volume_interpretation = percentile_scores.get("volume_interpretation")
        if volume_interpretation and volume_interpretation != "Insufficient data":
            insights.append(f"📊 {volume_interpretation}")

        # 6. Volatility Insight
        volatility_interpretation = percentile_scores.get("volatility_interpretation")
        if volatility_interpretation and volatility_interpretation != "Insufficient data":
            if "extreme" in volatility_interpretation.lower() or "elevated" in volatility_interpretation.lower():
                insights.append(f"⚠️ {volatility_interpretation} - Use wider stops")
            else:
                insights.append(f"📉 {volatility_interpretation}")

        # 7. Sector Analysis Insight
        sector_analysis = analysis_result.get("sector_analysis", {})
        if sector_analysis:
            sector_trend = sector_analysis.get("sector_trend", "")
            relative_strength = sector_analysis.get("relative_strength", "")

            if sector_trend and relative_strength:
                if "bullish" in sector_trend.lower() and "outperforming" in relative_strength.lower():
                    insights.append(
                        f"🚀 Stock is outperforming in a bullish sector - strong positive tailwinds"
                    )
                elif "bearish" in sector_trend.lower() and "underperforming" in relative_strength.lower():
                    insights.append(
                        f"⚠️ Stock is underperforming in a bearish sector - significant headwinds"
                    )
                elif "outperforming" in relative_strength.lower():
                    insights.append(
                        f"💪 Stock showing relative strength vs sector peers"
                    )
                elif "underperforming" in relative_strength.lower():
                    insights.append(
                        f"⚠️ Stock showing relative weakness vs sector peers"
                    )

        # 8. Momentum Insight
        return_5d = technical_indicators.get("return_5d")
        return_20d = technical_indicators.get("return_20d")
        if return_5d is not None and return_20d is not None:
            if return_5d > 0 and return_20d > 0:
                if return_5d > return_20d:
                    insights.append(
                        f"⚡ Momentum accelerating - 5D return ({return_5d:.1f}%) exceeds 20D return ({return_20d:.1f}%)"
                    )
                else:
                    insights.append(
                        f"📈 Positive momentum maintained - 5D: {return_5d:.1f}%, 20D: {return_20d:.1f}%"
                    )
            elif return_5d < 0 and return_20d < 0:
                if return_5d < return_20d:
                    insights.append(
                        f"⚠️ Momentum deteriorating - 5D return ({return_5d:.1f}%) worse than 20D ({return_20d:.1f}%)"
                    )
                else:
                    insights.append(
                        f"📉 Negative momentum - 5D: {return_5d:.1f}%, 20D: {return_20d:.1f}%"
                    )

        # 9. Price Position Insight
        price_position_interpretation = percentile_scores.get("price_position_interpretation")
        if price_position_interpretation and price_position_interpretation != "Insufficient data":
            if "near lows" in price_position_interpretation.lower():
                insights.append(f"💡 {price_position_interpretation} - potential bounce zone")
            elif "near highs" in price_position_interpretation.lower():
                insights.append(f"💡 {price_position_interpretation} - watch for breakout or rejection")
            else:
                insights.append(f"💡 {price_position_interpretation}")

        # 10. Key Opportunities
        opportunities = analysis_result.get("key_opportunities", [])
        if opportunities and len(opportunities) > 0:
            top_opportunity = opportunities[0]
            insights.append(f"💎 Key catalyst: {top_opportunity}")

        # 11. Top Risk
        risks = analysis_result.get("key_risks", [])
        if risks and len(risks) > 0:
            top_risk = risks[0]
            insights.append(f"⚠️ Primary risk: {top_risk}")

        # 12. Time Horizon
        time_horizon = analysis_result.get("time_horizon")
        if time_horizon:
            horizon_map = {
                "short_term": "Short-term trade (days to weeks)",
                "medium_term": "Medium-term position (weeks to months)",
                "long_term": "Long-term investment (months+)"
            }
            horizon_text = horizon_map.get(time_horizon, time_horizon)
            insights.append(f"⏱️ Recommended timeframe: {horizon_text}")

        return insights

    def generate_summary_insight(
        self,
        analysis_result: Dict[str, Any],
        percentile_scores: Dict[str, Any]
    ) -> str:
        """
        Generate a single-sentence executive summary

        Args:
            analysis_result: AI analysis result
            percentile_scores: Percentile scores

        Returns:
            Summary string
        """
        recommendation = analysis_result.get("recommendation", "HOLD")
        confidence = analysis_result.get("confidence", 50)
        composite_score = percentile_scores.get("composite_score", 50)
        confidence_breakdown = analysis_result.get("confidence_breakdown", {})
        sector_score = confidence_breakdown.get("sector_score", 50)

        # Determine overall sentiment
        if recommendation == "BUY" and confidence >= 70 and composite_score >= 60:
            sentiment = "Strong buying opportunity"
        elif recommendation == "BUY" and confidence >= 50:
            sentiment = "Moderate buying opportunity"
        elif recommendation == "SELL" and confidence >= 70 and composite_score <= 40:
            sentiment = "Strong selling pressure"
        elif recommendation == "SELL" and confidence >= 50:
            sentiment = "Consider reducing exposure"
        elif composite_score >= 60:
            sentiment = "Bullish setup, but wait for confirmation"
        elif composite_score <= 40:
            sentiment = "Bearish setup, avoid entry"
        else:
            sentiment = "Neutral setup with mixed signals"

        # Add sector context
        sector_analysis = analysis_result.get("sector_analysis", {})
        relative_strength = sector_analysis.get("relative_strength", "")

        if relative_strength:
            if "outperforming" in relative_strength.lower():
                sector_context = "with relative sector strength"
            elif "underperforming" in relative_strength.lower():
                sector_context = "despite sector weakness"
            else:
                sector_context = "in line with sector"
        else:
            sector_context = ""

        # Combine into summary
        summary = f"{sentiment}"
        if sector_context:
            summary += f" {sector_context}"

        summary += f" (Confidence: {confidence}%, Composite: {composite_score}/100)"

        return summary

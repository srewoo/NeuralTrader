"""
Validator Agent
Validates and critiques the analysis for quality assurance
"""

from typing import Dict, Any
import json
from .base import BaseAgent


class ValidatorAgent(BaseAgent):
    """
    Agent responsible for validating analysis and performing self-critique
    """
    
    def __init__(self):
        super().__init__("Validator Agent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate and critique the analysis
        
        Args:
            state: Current state with analysis_result
            
        Returns:
            Updated state with validation results
        """
        try:
            analysis_result = state.get("analysis_result")
            if not analysis_result:
                raise ValueError("No analysis result to validate")
            
            symbol = state.get("symbol")
            stock_data = state.get("stock_data", {})
            indicators = state.get("technical_indicators", {})
            
            self.log_execution(f"Validating analysis for {symbol}")
            
            # Add running step
            if "agent_steps" not in state:
                state["agent_steps"] = []
            
            state["agent_steps"].append(
                self.create_step_record(
                    status="running",
                    message="Validating analysis quality and consistency..."
                )
            )
            
            # Perform validation checks
            validation_results = {
                "is_valid": True,
                "warnings": [],
                "suggestions": [],
                "quality_score": 100
            }
            
            # 1. Validate recommendation consistency
            recommendation = analysis_result.get("recommendation", "").upper()
            if recommendation not in ["BUY", "SELL", "HOLD"]:
                validation_results["is_valid"] = False
                validation_results["warnings"].append(
                    f"Invalid recommendation: {recommendation}"
                )
                validation_results["quality_score"] -= 50
            
            # 2. Validate confidence level
            confidence = analysis_result.get("confidence", 0)
            if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 100):
                validation_results["warnings"].append(
                    f"Invalid confidence level: {confidence}"
                )
                validation_results["quality_score"] -= 10
            
            # 3. Validate price targets
            entry_price = analysis_result.get("entry_price")
            target_price = analysis_result.get("target_price")
            stop_loss = analysis_result.get("stop_loss")
            current_price = stock_data.get("current_price")
            
            if entry_price and target_price and stop_loss and current_price:
                # Check if prices are reasonable
                if recommendation == "BUY":
                    if target_price <= entry_price:
                        validation_results["warnings"].append(
                            "BUY signal but target price is not higher than entry"
                        )
                        validation_results["quality_score"] -= 15
                    
                    if stop_loss >= entry_price:
                        validation_results["warnings"].append(
                            "BUY signal but stop loss is not lower than entry"
                        )
                        validation_results["quality_score"] -= 15
                
                elif recommendation == "SELL":
                    if target_price >= entry_price:
                        validation_results["warnings"].append(
                            "SELL signal but target price is not lower than entry"
                        )
                        validation_results["quality_score"] -= 15
                    
                    if stop_loss <= entry_price:
                        validation_results["warnings"].append(
                            "SELL signal but stop loss is not higher than entry"
                        )
                        validation_results["quality_score"] -= 15
                
                # Check if prices are within reasonable range of current price
                price_deviation = abs(entry_price - current_price) / current_price
                if price_deviation > 0.1:  # More than 10% deviation
                    validation_results["suggestions"].append(
                        f"Entry price deviates {price_deviation*100:.1f}% from current price"
                    )
                    validation_results["quality_score"] -= 5
            
            # 4. Validate technical alignment
            rsi = indicators.get("rsi")
            if rsi and recommendation == "BUY" and rsi > 70:
                validation_results["warnings"].append(
                    "BUY recommendation but RSI indicates overbought (>70)"
                )
                validation_results["quality_score"] -= 10
            
            if rsi and recommendation == "SELL" and rsi < 30:
                validation_results["warnings"].append(
                    "SELL recommendation but RSI indicates oversold (<30)"
                )
                validation_results["quality_score"] -= 10

            # 4b. MACD alignment check
            macd = indicators.get("macd")
            macd_signal_val = indicators.get("macd_signal")
            if macd is not None and macd_signal_val is not None:
                if recommendation == "BUY" and macd < macd_signal_val:
                    validation_results["warnings"].append(
                        f"BUY recommendation but MACD ({macd:.2f}) is below signal line ({macd_signal_val:.2f}) — bearish momentum"
                    )
                    validation_results["quality_score"] -= 8
                elif recommendation == "SELL" and macd > macd_signal_val:
                    validation_results["warnings"].append(
                        f"SELL recommendation but MACD ({macd:.2f}) is above signal line ({macd_signal_val:.2f}) — bullish momentum"
                    )
                    validation_results["quality_score"] -= 8

            # 4c. Trend alignment check (SMA20 and SMA50)
            sma_20 = indicators.get("sma_20")
            sma_50 = indicators.get("sma_50")
            current_price_val = indicators.get("current_price", stock_data.get("current_price"))
            if sma_20 and sma_50 and current_price_val:
                if recommendation == "BUY" and current_price_val < sma_20 and current_price_val < sma_50:
                    validation_results["warnings"].append(
                        f"BUY recommendation but price (₹{current_price_val:.2f}) is below both SMA20 (₹{sma_20:.2f}) and SMA50 (₹{sma_50:.2f}) — bearish trend"
                    )
                    validation_results["quality_score"] -= 10
                elif recommendation == "SELL" and current_price_val > sma_20 and current_price_val > sma_50:
                    validation_results["warnings"].append(
                        f"SELL recommendation but price (₹{current_price_val:.2f}) is above both SMA20 (₹{sma_20:.2f}) and SMA50 (₹{sma_50:.2f}) — bullish trend"
                    )
                    validation_results["quality_score"] -= 10

            # 4d. Volume confirmation check
            volume_ratio = indicators.get("volume_ratio")
            if volume_ratio is not None and volume_ratio < 0.5:
                validation_results["suggestions"].append(
                    f"Low volume ratio ({volume_ratio:.2f}x avg) — recommendation may lack conviction"
                )
                validation_results["quality_score"] -= 5

            # 4e. Stochastic alignment check
            stochastic_k = indicators.get("stochastic_k")
            if stochastic_k is not None:
                if recommendation == "BUY" and stochastic_k > 80:
                    validation_results["warnings"].append(
                        f"BUY recommendation but Stochastic K ({stochastic_k:.1f}) indicates overbought (>80)"
                    )
                    validation_results["quality_score"] -= 8
                elif recommendation == "SELL" and stochastic_k < 20:
                    validation_results["warnings"].append(
                        f"SELL recommendation but Stochastic K ({stochastic_k:.1f}) indicates oversold (<20)"
                    )
                    validation_results["quality_score"] -= 8

            # 4f. Comprehensive signal alignment scoring
            alignment_count = 0
            total_checks = 0

            # RSI alignment
            if rsi is not None:
                total_checks += 1
                if recommendation == "BUY" and rsi < 70:
                    alignment_count += 1
                elif recommendation == "SELL" and rsi > 30:
                    alignment_count += 1
                elif recommendation == "HOLD":
                    alignment_count += 1

            # MACD alignment
            if macd is not None and macd_signal_val is not None:
                total_checks += 1
                if recommendation == "BUY" and macd > macd_signal_val:
                    alignment_count += 1
                elif recommendation == "SELL" and macd < macd_signal_val:
                    alignment_count += 1
                elif recommendation == "HOLD":
                    alignment_count += 1

            # Trend alignment
            if sma_20 and sma_50 and current_price_val:
                total_checks += 1
                if recommendation == "BUY" and (current_price_val > sma_20 or current_price_val > sma_50):
                    alignment_count += 1
                elif recommendation == "SELL" and (current_price_val < sma_20 or current_price_val < sma_50):
                    alignment_count += 1
                elif recommendation == "HOLD":
                    alignment_count += 1

            # Stochastic alignment
            if stochastic_k is not None:
                total_checks += 1
                if recommendation == "BUY" and stochastic_k < 80:
                    alignment_count += 1
                elif recommendation == "SELL" and stochastic_k > 20:
                    alignment_count += 1
                elif recommendation == "HOLD":
                    alignment_count += 1

            # Volume alignment
            if volume_ratio is not None:
                total_checks += 1
                if volume_ratio >= 0.7:
                    alignment_count += 1

            # Calculate alignment ratio
            if total_checks >= 3:
                alignment_ratio = alignment_count / total_checks
                if alignment_ratio < 0.6:
                    validation_results["warnings"].append(
                        f"Low signal alignment: only {alignment_count}/{total_checks} indicators support {recommendation} — high contradiction risk"
                    )
                    validation_results["quality_score"] -= 10

                # Store for confidence recalibration
                validation_results["signal_alignment"] = {
                    "aligned": alignment_count,
                    "total": total_checks,
                    "ratio": round(alignment_ratio, 2)
                }

            # 5. Validate reasoning quality
            reasoning = analysis_result.get("reasoning", "")
            if len(reasoning) < 100:
                validation_results["warnings"].append(
                    "Reasoning is too brief (< 100 characters)"
                )
                validation_results["quality_score"] -= 15
            
            if not any(keyword in reasoning.lower() for keyword in ["rsi", "macd", "trend", "support", "resistance"]):
                validation_results["suggestions"].append(
                    "Reasoning could include more technical analysis details"
                )
                validation_results["quality_score"] -= 5
            
            # 6. Validate risk assessment
            key_risks = analysis_result.get("key_risks", [])
            if len(key_risks) < 2:
                validation_results["suggestions"].append(
                    "Consider identifying more risk factors (at least 2-3)"
                )
                validation_results["quality_score"] -= 5
            
            # 7. Check risk-reward ratio
            risk_reward = analysis_result.get("risk_reward_ratio")
            if risk_reward:
                if risk_reward < 1.5:
                    validation_results["suggestions"].append(
                        f"Risk-reward ratio ({risk_reward:.2f}) is below recommended minimum (1.5)"
                    )
                    validation_results["quality_score"] -= 5
            
            # Floor quality score at 0
            validation_results["quality_score"] = max(0, validation_results["quality_score"])

            # Final quality assessment
            if validation_results["quality_score"] >= 90:
                quality_rating = "excellent"
            elif validation_results["quality_score"] >= 75:
                quality_rating = "good"
            elif validation_results["quality_score"] >= 60:
                quality_rating = "acceptable"
            else:
                quality_rating = "needs_improvement"
            
            validation_results["quality_rating"] = quality_rating
            
            # Update state
            state["validation"] = validation_results
            state["quality_score"] = validation_results["quality_score"]

            # Confidence recalibration based on signal alignment
            alignment_info = validation_results.get("signal_alignment")
            if alignment_info:
                current_confidence = analysis_result.get("confidence", 50)
                alignment_ratio = alignment_info["ratio"]

                if alignment_ratio < 0.5:
                    # Very low alignment — cap confidence
                    if current_confidence > 60:
                        analysis_result["confidence"] = 60
                        analysis_result["confidence_capped"] = True
                        analysis_result["confidence_cap_reason"] = f"Low signal alignment ({alignment_info['aligned']}/{alignment_info['total']})"
                        state["confidence"] = 60
                elif alignment_ratio >= 0.8 and current_confidence < 80:
                    # Strong alignment — modest boost
                    boosted = min(current_confidence + 5, 85)
                    analysis_result["confidence"] = boosted
                    analysis_result["confidence_boosted"] = True
                    state["confidence"] = boosted

            # Add any critical warnings to analysis result
            if validation_results["warnings"]:
                if "validation_warnings" not in analysis_result:
                    analysis_result["validation_warnings"] = []
                analysis_result["validation_warnings"].extend(validation_results["warnings"])
            
            # Update step to completed
            step_message = f"Validation complete: {quality_rating} quality"
            if validation_results["warnings"]:
                step_message += f" ({len(validation_results['warnings'])} warnings)"
            
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=step_message,
                data={
                    "quality_score": validation_results["quality_score"],
                    "quality_rating": quality_rating,
                    "warnings": len(validation_results["warnings"]),
                    "suggestions": len(validation_results["suggestions"])
                }
            )
            
            self.log_execution(
                f"Validation complete: {quality_rating} "
                f"(score: {validation_results['quality_score']}/100)"
            )
            
            return state
            
        except Exception as e:
            return await self.handle_error(e, state)


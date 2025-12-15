"""
Unit Tests for NeuralTrader Agents
Tests for BaseAgent, DataCollectionAgent, TechnicalAnalysisAgent,
RAGKnowledgeAgent, DeepReasoningAgent, and ValidatorAgent
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ============================================================================
# BaseAgent Tests
# ============================================================================

class TestBaseAgent:
    """Tests for BaseAgent class"""

    def test_agent_initialization(self):
        """Test that agents initialize correctly with name"""
        from agents.base import BaseAgent

        # Create a concrete implementation for testing
        class ConcreteAgent(BaseAgent):
            async def execute(self, state):
                return state

        agent = ConcreteAgent("Test Agent")
        assert agent.name == "Test Agent"

    def test_create_step_record(self):
        """Test step record creation"""
        from agents.base import BaseAgent

        class ConcreteAgent(BaseAgent):
            async def execute(self, state):
                return state

        agent = ConcreteAgent("Test Agent")
        step = agent.create_step_record(
            status="running",
            message="Processing data",
            data={"key": "value"}
        )

        assert step["agent_name"] == "Test Agent"
        assert step["status"] == "running"
        assert step["message"] == "Processing data"
        assert step["data"] == {"key": "value"}
        assert "timestamp" in step

    def test_create_step_record_without_data(self):
        """Test step record creation without optional data"""
        from agents.base import BaseAgent

        class ConcreteAgent(BaseAgent):
            async def execute(self, state):
                return state

        agent = ConcreteAgent("Test Agent")
        step = agent.create_step_record(status="completed", message="Done")

        assert step["data"] == {}

    @pytest.mark.asyncio
    async def test_handle_error(self):
        """Test error handling in agents"""
        from agents.base import BaseAgent

        class ConcreteAgent(BaseAgent):
            async def execute(self, state):
                return state

        agent = ConcreteAgent("Test Agent")
        state = {"symbol": "TEST"}
        error = ValueError("Test error")

        updated_state = await agent.handle_error(error, state)

        assert updated_state["has_errors"] is True
        assert "Test error" in updated_state["last_error"]
        assert len(updated_state["agent_steps"]) == 1
        assert updated_state["agent_steps"][0]["status"] == "failed"


# ============================================================================
# DataCollectionAgent Tests
# ============================================================================

class TestDataCollectionAgent:
    """Tests for DataCollectionAgent"""

    def test_agent_name(self):
        """Test agent has correct name"""
        from agents.data_agent import DataCollectionAgent

        agent = DataCollectionAgent()
        assert agent.name == "Data Collection Agent"

    @pytest.mark.asyncio
    async def test_execute_missing_symbol(self):
        """Test error when symbol is missing"""
        from agents.data_agent import DataCollectionAgent

        agent = DataCollectionAgent()
        state = {}

        result = await agent.execute(state)

        assert result["has_errors"] is True
        assert "Symbol not provided" in result["last_error"]

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_yfinance):
        """Test successful data collection"""
        from agents.data_agent import DataCollectionAgent

        agent = DataCollectionAgent()
        state = {"symbol": "TEST.NS", "agent_steps": []}

        result = await agent.execute(state)

        assert "stock_data" in result
        assert result["stock_data"]["symbol"] == "TEST.NS"
        assert "current_price" in result["stock_data"]
        assert "volume" in result["stock_data"]

    @pytest.mark.asyncio
    async def test_execute_no_data_available(self):
        """Test handling when no data is available"""
        from agents.data_agent import DataCollectionAgent

        with patch('yfinance.Ticker') as mock:
            ticker = MagicMock()
            ticker.info = {}
            ticker.history.return_value = pd.DataFrame()  # Empty dataframe
            mock.return_value = ticker

            agent = DataCollectionAgent()
            state = {"symbol": "INVALID", "agent_steps": []}

            result = await agent.execute(state)

            assert result["has_errors"] is True
            assert "No data available" in result["last_error"]


# ============================================================================
# TechnicalAnalysisAgent Tests
# ============================================================================

class TestTechnicalAnalysisAgent:
    """Tests for TechnicalAnalysisAgent"""

    def test_agent_name(self):
        """Test agent has correct name"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        assert agent.name == "Technical Analysis Agent"

    @pytest.mark.asyncio
    async def test_execute_missing_symbol(self):
        """Test error when symbol is missing"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        state = {}

        result = await agent.execute(state)

        assert result["has_errors"] is True

    @pytest.mark.asyncio
    async def test_execute_success(self, mock_yfinance):
        """Test successful technical analysis"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        state = {
            "symbol": "TEST.NS",
            "agent_steps": [],
            "stock_data": {"current_price": 100.0}
        }

        result = await agent.execute(state)

        assert "technical_indicators" in result
        indicators = result["technical_indicators"]

        # Check all indicators are present
        assert "rsi" in indicators
        assert "macd" in indicators
        assert "sma_20" in indicators
        assert "sma_50" in indicators
        assert "bb_upper" in indicators
        assert "atr" in indicators

    @pytest.mark.asyncio
    async def test_insufficient_data(self):
        """Test handling of insufficient historical data"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        with patch('yfinance.Ticker') as mock:
            ticker = MagicMock()
            # Only 10 days of data (less than 50 required)
            dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
            hist_data = pd.DataFrame({
                'Open': [100] * 10,
                'High': [105] * 10,
                'Low': [95] * 10,
                'Close': [102] * 10,
                'Volume': [1000000] * 10
            }, index=dates)
            ticker.history.return_value = hist_data
            mock.return_value = ticker

            agent = TechnicalAnalysisAgent()
            state = {"symbol": "TEST", "agent_steps": []}

            result = await agent.execute(state)

            assert result["has_errors"] is True
            assert "Insufficient data" in result["last_error"]

    def test_analyze_signals_oversold(self):
        """Test RSI oversold signal detection"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        indicators = {"rsi": 25.0, "stochastic_k": 15.0}
        stock_data = {"current_price": 100.0}

        signals = agent._analyze_signals(indicators, stock_data)

        assert signals["rsi"] == "oversold"
        assert signals["momentum"] == "oversold"

    def test_analyze_signals_overbought(self):
        """Test RSI overbought signal detection"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        indicators = {"rsi": 75.0, "stochastic_k": 85.0}
        stock_data = {"current_price": 100.0}

        signals = agent._analyze_signals(indicators, stock_data)

        assert signals["rsi"] == "overbought"
        assert signals["momentum"] == "overbought"

    def test_analyze_signals_trend(self):
        """Test trend signal detection"""
        from agents.analysis_agent import TechnicalAnalysisAgent

        agent = TechnicalAnalysisAgent()
        indicators = {
            "sma_20": 95.0,
            "sma_50": 90.0,
            "macd": 5.0,
            "macd_signal": 3.0
        }
        stock_data = {"current_price": 100.0}

        signals = agent._analyze_signals(indicators, stock_data)

        assert signals["trend"] == "strong_uptrend"
        assert signals["macd"] == "bullish"


# ============================================================================
# RAGKnowledgeAgent Tests
# ============================================================================

class TestRAGKnowledgeAgent:
    """Tests for RAGKnowledgeAgent"""

    def test_agent_name(self):
        """Test agent has correct name"""
        from agents.knowledge_agent import RAGKnowledgeAgent

        agent = RAGKnowledgeAgent()
        assert agent.name == "RAG Knowledge Agent"

    @pytest.mark.asyncio
    async def test_execute_graceful_fallback(self):
        """Test graceful fallback when RAG fails"""
        from agents.knowledge_agent import RAGKnowledgeAgent

        with patch('rag.retrieval.get_retriever') as mock:
            mock.side_effect = Exception("RAG system unavailable")

            agent = RAGKnowledgeAgent()
            state = {
                "symbol": "TEST",
                "stock_data": {},
                "technical_indicators": {},
                "technical_signals": {},
                "agent_steps": []
            }

            result = await agent.execute(state)

            # Should not fail, just use fallback
            assert result["rag_context"] == ""
            assert result["rag_results"] == []
            assert "has_errors" not in result or result.get("has_errors") is False


# ============================================================================
# DeepReasoningAgent Tests
# ============================================================================

class TestDeepReasoningAgent:
    """Tests for DeepReasoningAgent"""

    def test_agent_name(self):
        """Test agent has correct name"""
        from agents.reasoning_agent import DeepReasoningAgent

        agent = DeepReasoningAgent()
        assert agent.name == "Deep Reasoning Agent"

    @pytest.mark.asyncio
    async def test_execute_missing_api_key(self):
        """Test error when API key is missing"""
        from agents.reasoning_agent import DeepReasoningAgent

        agent = DeepReasoningAgent()
        state = {
            "symbol": "TEST",
            "stock_data": {},
            "technical_indicators": {},
            "technical_signals": {},
            "agent_steps": []
        }

        result = await agent.execute(state)

        assert result["has_errors"] is True
        assert "API key not provided" in result["last_error"]

    def test_build_analysis_prompt(self, sample_stock_data, sample_technical_indicators, sample_technical_signals):
        """Test analysis prompt building"""
        from agents.reasoning_agent import DeepReasoningAgent

        agent = DeepReasoningAgent()
        prompt = agent._build_analysis_prompt(
            symbol="RELIANCE.NS",
            stock_data=sample_stock_data,
            indicators=sample_technical_indicators,
            signals=sample_technical_signals,
            rag_context="Historical pattern: bullish divergence"
        )

        assert "RELIANCE.NS" in prompt
        assert "RSI" in prompt
        assert "MACD" in prompt
        assert "chain-of-thought" in prompt.lower()
        assert "JSON" in prompt

    @pytest.mark.asyncio
    async def test_execute_openai_success(self, sample_stock_data, sample_technical_indicators, sample_technical_signals):
        """Test successful execution with OpenAI"""
        from agents.reasoning_agent import DeepReasoningAgent

        with patch('openai.AsyncOpenAI') as mock_client:
            mock_instance = AsyncMock()
            completion = MagicMock()
            completion.choices = [MagicMock()]
            completion.choices[0].message.content = '{"recommendation": "BUY", "confidence": 75, "reasoning": "Strong technical setup"}'
            mock_instance.chat.completions.create = AsyncMock(return_value=completion)
            mock_client.return_value = mock_instance

            agent = DeepReasoningAgent()
            state = {
                "symbol": "TEST",
                "model": "gpt-4.1",
                "provider": "openai",
                "api_key": "test-key",
                "stock_data": sample_stock_data,
                "technical_indicators": sample_technical_indicators,
                "technical_signals": sample_technical_signals,
                "rag_context": "",
                "agent_steps": []
            }

            result = await agent.execute(state)

            assert "analysis_result" in result
            assert result["recommendation"] == "BUY"
            assert result["confidence"] == 75

    @pytest.mark.asyncio
    async def test_execute_unsupported_provider(self, sample_stock_data, sample_technical_indicators):
        """Test error with unsupported provider"""
        from agents.reasoning_agent import DeepReasoningAgent

        agent = DeepReasoningAgent()
        state = {
            "symbol": "TEST",
            "model": "unknown-model",
            "provider": "unsupported",
            "api_key": "test-key",
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "technical_signals": {},
            "rag_context": "",
            "agent_steps": []
        }

        result = await agent.execute(state)

        assert result["has_errors"] is True
        assert "Unsupported provider" in result["last_error"]


# ============================================================================
# ValidatorAgent Tests
# ============================================================================

class TestValidatorAgent:
    """Tests for ValidatorAgent"""

    def test_agent_name(self):
        """Test agent has correct name"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        assert agent.name == "Validator Agent"

    @pytest.mark.asyncio
    async def test_execute_missing_analysis_result(self):
        """Test error when analysis result is missing"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        state = {"symbol": "TEST", "agent_steps": []}

        result = await agent.execute(state)

        assert result["has_errors"] is True
        assert "No analysis result" in result["last_error"]

    @pytest.mark.asyncio
    async def test_validate_valid_buy_recommendation(
        self, sample_analysis_result, sample_stock_data, sample_technical_indicators
    ):
        """Test validation of valid BUY recommendation"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        state = {
            "symbol": "TEST",
            "analysis_result": sample_analysis_result,
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        assert "validation" in result
        assert result["validation"]["is_valid"] is True
        assert result["quality_score"] >= 60

    @pytest.mark.asyncio
    async def test_validate_invalid_recommendation(self, sample_stock_data, sample_technical_indicators):
        """Test validation of invalid recommendation"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        invalid_result = {
            "recommendation": "INVALID",  # Not BUY/SELL/HOLD
            "confidence": 75,
            "reasoning": "Test"
        }
        state = {
            "symbol": "TEST",
            "analysis_result": invalid_result,
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        assert result["validation"]["is_valid"] is False
        assert len(result["validation"]["warnings"]) > 0

    @pytest.mark.asyncio
    async def test_validate_buy_with_overbought_rsi(self, sample_stock_data):
        """Test validation warns on BUY with overbought RSI"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        analysis_result = {
            "recommendation": "BUY",
            "confidence": 75,
            "entry_price": 100,
            "target_price": 110,
            "stop_loss": 95,
            "reasoning": "Long reasoning with RSI, MACD, trend analysis and support resistance levels."
        }
        indicators = {"rsi": 75.0}  # Overbought

        state = {
            "symbol": "TEST",
            "analysis_result": analysis_result,
            "stock_data": sample_stock_data,
            "technical_indicators": indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        # Should have warning about RSI
        warnings = result["validation"]["warnings"]
        assert any("RSI" in w and "overbought" in w for w in warnings)

    @pytest.mark.asyncio
    async def test_validate_invalid_price_targets(self, sample_stock_data, sample_technical_indicators):
        """Test validation warns on invalid price targets for BUY"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        analysis_result = {
            "recommendation": "BUY",
            "confidence": 75,
            "entry_price": 100,
            "target_price": 90,  # Target lower than entry for BUY
            "stop_loss": 105,    # Stop loss higher than entry for BUY
            "reasoning": "Long reasoning with RSI, MACD, trend analysis and support resistance levels."
        }

        state = {
            "symbol": "TEST",
            "analysis_result": analysis_result,
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        warnings = result["validation"]["warnings"]
        assert len(warnings) >= 2  # Should have warnings about both target and stop loss

    @pytest.mark.asyncio
    async def test_validate_brief_reasoning(self, sample_stock_data, sample_technical_indicators):
        """Test validation warns on brief reasoning"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()
        analysis_result = {
            "recommendation": "HOLD",
            "confidence": 50,
            "reasoning": "Short"  # Less than 100 chars
        }

        state = {
            "symbol": "TEST",
            "analysis_result": analysis_result,
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        warnings = result["validation"]["warnings"]
        assert any("brief" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_quality_ratings(self, sample_stock_data, sample_technical_indicators):
        """Test quality rating calculation"""
        from agents.validator_agent import ValidatorAgent

        agent = ValidatorAgent()

        # Excellent quality
        excellent_result = {
            "recommendation": "BUY",
            "confidence": 80,
            "entry_price": 2500,
            "target_price": 2700,
            "stop_loss": 2400,
            "risk_reward_ratio": 2.0,
            "reasoning": "Detailed analysis: 1) Price action shows RSI support. 2) MACD confirms trend. 3) Support at SMA levels.",
            "key_risks": ["Risk 1", "Risk 2", "Risk 3"]
        }

        state = {
            "symbol": "TEST",
            "analysis_result": excellent_result,
            "stock_data": sample_stock_data,
            "technical_indicators": sample_technical_indicators,
            "agent_steps": []
        }

        result = await agent.execute(state)

        assert result["validation"]["quality_rating"] in ["excellent", "good", "acceptable"]

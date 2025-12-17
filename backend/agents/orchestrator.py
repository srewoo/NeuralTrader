"""
LangGraph Orchestrator
Manages the multi-agent workflow using LangGraph state machine
"""

from typing import Dict, Any, TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, END
import operator
import logging
from .data_agent import DataCollectionAgent
from .analysis_agent import TechnicalAnalysisAgent
from .knowledge_agent import RAGKnowledgeAgent
from .reasoning_agent import DeepReasoningAgent
from .validator_agent import ValidatorAgent
from .insight_generator import InsightGenerator

logger = logging.getLogger(__name__)


class AnalysisState(TypedDict):
    """
    State schema for the analysis workflow
    """
    # Input parameters
    symbol: str
    model: str
    provider: str
    api_key: str
    data_provider_keys: Dict[str, Any]
    
    # Data collected by agents
    stock_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    technical_signals: Dict[str, Any]
    percentile_scores: Dict[str, Any]
    rag_context: str
    rag_results: list
    similar_patterns: list
    strategy_recommendations: list
    analysis_result: Dict[str, Any]
    validation: Dict[str, Any]
    insights: list
    summary_insight: str
    
    # Workflow management
    agent_steps: Annotated[list, operator.add]
    has_errors: bool
    last_error: str
    
    # Output
    recommendation: str
    confidence: int
    reasoning: str
    quality_score: int


class AnalysisOrchestrator:
    """
    Orchestrates the multi-agent analysis workflow using LangGraph
    """
    
    def __init__(self):
        """Initialize orchestrator with agents"""
        self.data_agent = DataCollectionAgent()
        self.analysis_agent = TechnicalAnalysisAgent()
        self.knowledge_agent = RAGKnowledgeAgent()
        self.reasoning_agent = DeepReasoningAgent()
        self.validator_agent = ValidatorAgent()
        self.insight_generator = InsightGenerator()

        # Build the workflow graph
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile()
    
    def _build_workflow(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            Compiled workflow graph
        """
        # Create workflow graph
        workflow = StateGraph(AnalysisState)
        
        # Add agent nodes
        workflow.add_node("collect_data", self._data_node)
        workflow.add_node("analyze_technical", self._analysis_node)
        workflow.add_node("retrieve_knowledge", self._knowledge_node)
        workflow.add_node("reason_deeply", self._reasoning_node)
        workflow.add_node("validate", self._validator_node)
        
        # Define workflow edges (sequential execution)
        workflow.set_entry_point("collect_data")
        workflow.add_edge("collect_data", "analyze_technical")
        workflow.add_edge("analyze_technical", "retrieve_knowledge")
        workflow.add_edge("retrieve_knowledge", "reason_deeply")
        workflow.add_edge("reason_deeply", "validate")
        workflow.add_edge("validate", END)
        
        return workflow
    
    async def _data_node(self, state: AnalysisState) -> AnalysisState:
        """Data collection node"""
        return await self.data_agent.execute(state)
    
    async def _analysis_node(self, state: AnalysisState) -> AnalysisState:
        """Technical analysis node"""
        return await self.analysis_agent.execute(state)
    
    async def _knowledge_node(self, state: AnalysisState) -> AnalysisState:
        """Knowledge retrieval node"""
        return await self.knowledge_agent.execute(state)
    
    async def _reasoning_node(self, state: AnalysisState) -> AnalysisState:
        """Deep reasoning node"""
        return await self.reasoning_agent.execute(state)
    
    async def _validator_node(self, state: AnalysisState) -> AnalysisState:
        """Validation node"""
        return await self.validator_agent.execute(state)
    
    async def run_analysis(
        self,
        symbol: str,
        model: str,
        provider: str,
        api_key: str,
        data_provider_keys: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete analysis workflow

        Args:
            symbol: Stock symbol
            model: LLM model to use
            provider: Provider (openai/gemini)
            api_key: API key for LLM
            data_provider_keys: Optional API keys for data providers (Finnhub, Alpaca, FMP)

        Returns:
            Complete analysis result with all agent outputs
        """
        try:
            logger.info(f"Starting analysis workflow for {symbol} with {model}")

            # Initialize state
            initial_state = {
                "symbol": symbol,
                "model": model,
                "provider": provider,
                "api_key": api_key,
                "data_provider_keys": data_provider_keys or {},
                "agent_steps": [],
                "has_errors": False,
                "last_error": ""
            }
            
            # Run the workflow
            final_state = await self.app.ainvoke(initial_state)

            # Generate natural language insights
            analysis_result = final_state.get("analysis_result", {})
            technical_indicators = final_state.get("technical_indicators", {})
            percentile_scores = final_state.get("percentile_scores", {})
            stock_data = final_state.get("stock_data", {})

            insights = self.insight_generator.generate_insights(
                analysis_result, technical_indicators, percentile_scores, stock_data
            )
            summary_insight = self.insight_generator.generate_summary_insight(
                analysis_result, percentile_scores
            )

            # Extract result
            result = {
                "symbol": symbol,
                "model_used": model,
                "provider": provider,
                "recommendation": final_state.get("recommendation", "HOLD"),
                "confidence": final_state.get("confidence", 50),
                "reasoning": final_state.get("reasoning", "Analysis incomplete"),
                "entry_price": final_state.get("analysis_result", {}).get("entry_price"),
                "target_price": final_state.get("analysis_result", {}).get("target_price"),
                "stop_loss": final_state.get("analysis_result", {}).get("stop_loss"),
                "risk_reward_ratio": final_state.get("analysis_result", {}).get("risk_reward_ratio"),
                "time_horizon": final_state.get("analysis_result", {}).get("time_horizon", "medium_term"),
                "key_risks": final_state.get("analysis_result", {}).get("key_risks", []),
                "key_opportunities": final_state.get("analysis_result", {}).get("key_opportunities", []),
                "agent_steps": final_state.get("agent_steps", []),
                "quality_score": final_state.get("quality_score", 0),
                "validation_warnings": final_state.get("analysis_result", {}).get("validation_warnings", []),
                "technical_indicators": final_state.get("technical_indicators", {}),
                "technical_signals": final_state.get("technical_signals", {}),
                "percentile_scores": percentile_scores,
                "insights": insights,
                "summary_insight": summary_insight,
                "rag_patterns_found": len(final_state.get("rag_results", [])),
                "similar_patterns_count": len(final_state.get("similar_patterns", [])),
                "has_errors": final_state.get("has_errors", False)
            }
            
            logger.info(
                f"Analysis complete: {result['recommendation']} "
                f"({result['confidence']}% confidence, "
                f"quality: {result['quality_score']}/100)"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise


# Global orchestrator instance
_orchestrator_instance = None


def get_orchestrator() -> AnalysisOrchestrator:
    """
    Get or create global orchestrator instance (Singleton)
    
    Returns:
        AnalysisOrchestrator instance
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = AnalysisOrchestrator()
    return _orchestrator_instance


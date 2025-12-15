"""
RAG Knowledge Agent
Retrieves relevant historical knowledge for context
"""

from typing import Dict, Any
from .base import BaseAgent


class RAGKnowledgeAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant knowledge using RAG
    """
    
    def __init__(self):
        super().__init__("RAG Knowledge Agent")
    
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge from RAG system
        
        Args:
            state: Current state with stock_data and technical_indicators
            
        Returns:
            Updated state with rag_context
        """
        try:
            symbol = state.get("symbol")
            stock_data = state.get("stock_data", {})
            indicators = state.get("technical_indicators", {})
            signals = state.get("technical_signals", {})
            
            self.log_execution(f"Retrieving historical knowledge for {symbol}")
            
            # Add running step
            if "agent_steps" not in state:
                state["agent_steps"] = []
            
            state["agent_steps"].append(
                self.create_step_record(
                    status="running",
                    message="Searching knowledge base for relevant patterns..."
                )
            )
            
            # Import RAG system (REAL implementation)
            from rag.retrieval import get_retriever
            retriever = get_retriever()
            
            # Build query from current market conditions
            query_parts = []
            query_parts.append(f"Stock {symbol}")
            
            if indicators.get("rsi"):
                query_parts.append(f"RSI {indicators['rsi']}")
            
            if indicators.get("macd"):
                query_parts.append(f"MACD {indicators['macd']}")
            
            if signals.get("trend"):
                query_parts.append(f"{signals['trend']}")
            
            if signals.get("rsi"):
                query_parts.append(f"{signals['rsi']}")
            
            rag_query = " ".join(query_parts) + " trading pattern analysis"
            
            # Retrieve relevant documents (REAL RAG query)
            rag_results = retriever.retrieve(
                query=rag_query,
                n_results=5,
                min_similarity=0.5
            )
            
            # Build context for LLM
            rag_context = retriever.build_context(
                query=rag_query,
                n_results=5,
                max_tokens=2000
            )
            
            # Get similar patterns
            if indicators:
                similar_patterns = retriever.get_similar_patterns(
                    technical_indicators=indicators,
                    n_results=3
                )
            else:
                similar_patterns = []
            
            # Get strategy recommendations
            if signals.get("trend"):
                strategy_recs = retriever.get_strategy_recommendations(
                    market_condition=signals["trend"],
                    n_results=2
                )
            else:
                strategy_recs = []
            
            # Update state
            state["rag_context"] = rag_context
            state["rag_results"] = rag_results
            state["similar_patterns"] = similar_patterns
            state["strategy_recommendations"] = strategy_recs
            
            # Calculate statistics
            avg_similarity = (
                sum(r["similarity"] for r in rag_results) / len(rag_results)
                if rag_results else 0
            )
            
            categories = list(set(
                r.get("metadata", {}).get("category", "unknown")
                for r in rag_results
            ))
            
            # Update step to completed
            state["agent_steps"][-1] = self.create_step_record(
                status="completed",
                message=f"Retrieved {len(rag_results)} relevant historical patterns",
                data={
                    "patterns_found": len(rag_results),
                    "avg_similarity": round(avg_similarity, 2),
                    "categories": categories,
                    "similar_patterns": len(similar_patterns),
                    "strategies": len(strategy_recs)
                }
            )
            
            self.log_execution(
                f"Retrieved {len(rag_results)} documents with "
                f"avg similarity {avg_similarity:.2f}"
            )
            
            return state
            
        except Exception as e:
            # Graceful fallback if RAG fails
            self.log_execution(f"RAG retrieval failed, continuing without context: {e}", "warning")
            
            state["rag_context"] = ""
            state["rag_results"] = []
            
            if "agent_steps" not in state:
                state["agent_steps"] = []
            
            state["agent_steps"].append(
                self.create_step_record(
                    status="completed",
                    message="RAG system unavailable, proceeding with direct analysis",
                    data={"patterns_found": 0, "note": "Using fallback mode"}
                )
            )
            
            return state


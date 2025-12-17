"""
RAG Knowledge Agent
Retrieves relevant historical knowledge for context
"""

from typing import Dict, Any
from .base import BaseAgent
from motor.motor_asyncio import AsyncIOMotorDatabase


class RAGKnowledgeAgent(BaseAgent):
    """
    Agent responsible for retrieving relevant knowledge using RAG
    """

    def __init__(self, db: AsyncIOMotorDatabase = None):
        super().__init__("RAG Knowledge Agent")
        self.db = db
    
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

            # **NEW: Query discovered patterns and news events from knowledge system**
            discovered_patterns = []
            historical_events = []

            if self.db is not None:
                from knowledge.rag_ingestion import get_rag_ingestion
                rag_ingestion = get_rag_ingestion(self.db)

                # Query for discovered technical patterns
                discovered_patterns = await rag_ingestion.query_relevant_knowledge(
                    symbol=symbol,
                    query_type="pattern",
                    limit=3
                )

                # Query for historical news event impacts
                historical_events = await rag_ingestion.query_relevant_knowledge(
                    symbol=symbol,
                    query_type="event",
                    limit=3
                )

                # Format discovered patterns for context
                pattern_context = self._format_discovered_patterns(discovered_patterns)
                event_context = self._format_historical_events(historical_events)

                # Append to RAG context
                if pattern_context:
                    rag_context += f"\n\n--- DISCOVERED PATTERNS ---\n{pattern_context}"
                if event_context:
                    rag_context += f"\n\n--- HISTORICAL EVENT IMPACTS ---\n{event_context}"

            # Update state
            state["rag_context"] = rag_context
            state["rag_results"] = rag_results
            state["similar_patterns"] = similar_patterns
            state["strategy_recommendations"] = strategy_recs
            state["discovered_patterns"] = discovered_patterns
            state["historical_events"] = historical_events

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
                    "strategies": len(strategy_recs),
                    "discovered_patterns": len(discovered_patterns),
                    "historical_events": len(historical_events)
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
            state["discovered_patterns"] = []
            state["historical_events"] = []

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

    def _format_discovered_patterns(self, patterns: list) -> str:
        """Format discovered trading patterns for RAG context"""
        if not patterns:
            return ""

        formatted = []
        for pattern in patterns:
            metadata = pattern.get("metadata", {})
            formatted.append(
                f"Pattern: {pattern.get('pattern_type', 'Unknown').replace('_', ' ').title()}\n"
                f"Success Rate: {metadata.get('success_rate', 0):.1%}\n"
                f"Occurrences: {metadata.get('occurrences', 0)}\n"
                f"Avg Return: {metadata.get('average_return', 0):+.2f}%\n"
                f"Priority: {pattern.get('priority', 0)}/100\n"
                f"Details: {pattern.get('content', '')[:300]}..."
            )

        return "\n\n".join(formatted)

    def _format_historical_events(self, events: list) -> str:
        """Format historical news events for RAG context"""
        if not events:
            return ""

        formatted = []
        for event in events:
            metadata = event.get("metadata", {})
            formatted.append(
                f"Event Type: {event.get('event_type', 'Unknown').replace('_', ' ').title()}\n"
                f"Symbols Affected: {', '.join(event.get('symbols_affected', []))}\n"
                f"Immediate Impact: {metadata.get('immediate_impact', {})}\n"
                f"Recovery Timeline: {metadata.get('recovery_timeline', 'N/A')}\n"
                f"Priority: {event.get('priority', 0)}/100\n"
                f"Details: {event.get('content', '')[:300]}..."
            )

        return "\n\n".join(formatted)


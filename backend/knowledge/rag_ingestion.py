"""
RAG Knowledge Ingestion Pipeline
Automatically feeds discovered patterns and news events into RAG vector database
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)


class RAGKnowledgeIngestion:
    """
    Ingests discovered patterns and news events into RAG vector database
    Enables AI agents to leverage historical knowledge in recommendations
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.rag_collection = db.rag_knowledge  # Vector embeddings
        self.metadata_collection = db.rag_metadata  # Metadata for tracking

    async def ingest_discovered_pattern(
        self,
        pattern: Dict[str, Any],
        generate_embedding: bool = True
    ) -> str:
        """
        Ingest a discovered trading pattern into RAG

        Args:
            pattern: Pattern dictionary from pattern_mining
            generate_embedding: Whether to generate vector embedding

        Returns:
            RAG entry ID
        """
        try:
            # Create rich text description for embedding
            pattern_text = self._format_pattern_for_rag(pattern)

            # Create RAG entry
            rag_entry = {
                "type": "discovered_pattern",
                "source": "pattern_mining",
                "symbol": pattern["symbol"],
                "pattern_type": pattern["pattern_type"],
                "content": pattern_text,
                "metadata": {
                    "success_rate": pattern["statistics"]["success_rate"],
                    "occurrences": pattern["statistics"]["total_occurrences"],
                    "average_return": pattern["statistics"]["average_return"],
                    "discovered_at": pattern["discovered_at"],
                    "conditions": pattern["conditions"]
                },
                "tags": [
                    pattern["symbol"],
                    pattern["pattern_type"],
                    f"success_rate_{int(pattern['statistics']['success_rate'] * 100)}",
                    "technical_pattern"
                ],
                "ingested_at": datetime.now(timezone.utc),
                "priority": self._calculate_priority(pattern)
            }

            # TODO: Generate vector embedding using OpenAI/Gemini
            # For now, store without embedding (can be added later)
            if generate_embedding:
                # rag_entry["embedding"] = await self._generate_embedding(pattern_text)
                pass

            result = await self.rag_collection.insert_one(rag_entry)
            entry_id = str(result.inserted_id)

            # Track in metadata
            await self.metadata_collection.insert_one({
                "rag_entry_id": entry_id,
                "pattern_id": pattern.get("id", str(pattern.get("_id"))),
                "ingested_at": datetime.now(timezone.utc),
                "type": "pattern"
            })

            logger.info(
                f"✅ Ingested pattern into RAG: {pattern['symbol']} - "
                f"{pattern['pattern_type']} (success: {pattern['statistics']['success_rate']:.0%})"
            )

            return entry_id

        except Exception as e:
            logger.error(f"Failed to ingest pattern into RAG: {e}")
            raise

    async def ingest_news_event(
        self,
        event: Dict[str, Any],
        generate_embedding: bool = True
    ) -> str:
        """
        Ingest a news event impact into RAG

        Args:
            event: Event dictionary from news_events
            generate_embedding: Whether to generate vector embedding

        Returns:
            RAG entry ID
        """
        try:
            # Create rich text description
            event_text = self._format_event_for_rag(event)

            # Create RAG entry
            rag_entry = {
                "type": "news_event_impact",
                "source": "news_knowledge",
                "event_type": event["event_type"],
                "sector": event.get("sector"),
                "symbols_affected": event["symbols_affected"],
                "content": event_text,
                "metadata": {
                    "event_date": event["event_date"],
                    "immediate_impact": event.get("immediate_impact", {}),
                    "recovery_timeline": event.get("recovery_timeline"),
                    "sentiment": event.get("sentiment")
                },
                "tags": [
                    event["event_type"],
                    event.get("sector", "general"),
                    *event["symbols_affected"],
                    "news_impact"
                ],
                "ingested_at": datetime.now(timezone.utc),
                "priority": self._calculate_event_priority(event)
            }

            if generate_embedding:
                # rag_entry["embedding"] = await self._generate_embedding(event_text)
                pass

            result = await self.rag_collection.insert_one(rag_entry)
            entry_id = str(result.inserted_id)

            # Track in metadata
            await self.metadata_collection.insert_one({
                "rag_entry_id": entry_id,
                "event_id": event.get("id", str(event.get("_id"))),
                "ingested_at": datetime.now(timezone.utc),
                "type": "event"
            })

            logger.info(
                f"✅ Ingested event into RAG: {event['event_type']} - "
                f"{len(event['symbols_affected'])} symbols affected"
            )

            return entry_id

        except Exception as e:
            logger.error(f"Failed to ingest event into RAG: {e}")
            raise

    def _format_pattern_for_rag(self, pattern: Dict[str, Any]) -> str:
        """Format pattern as rich text for RAG embedding"""
        stats = pattern["statistics"]
        conditions = pattern["conditions"]

        text = f"""
DISCOVERED TRADING PATTERN: {pattern['symbol']}

Pattern Type: {pattern['pattern_type'].replace('_', ' ').title()}

Description: {pattern['description']}

Historical Performance:
- Total Occurrences: {stats['total_occurrences']}
- Success Rate: {stats['success_rate']:.1%}
- Average Return (Profitable): {stats['average_profitable_return']:+.2f}%
- Average Return (All): {stats['average_return']:+.2f}%
- Best Return: {stats['best_return']:+.2f}%
- Worst Return: {stats['worst_return']:+.2f}%

Signal Conditions:
"""
        for key, value in conditions.items():
            text += f"- {key.replace('_', ' ').title()}: {value}\n"

        text += f"""
Discovery Date: {pattern['discovered_at']}
Data Period: {pattern['data_period']['start']} to {pattern['data_period']['end']}

Trading Strategy:
When these conditions are met for {pattern['symbol']}, historical data shows a {stats['success_rate']:.0%} probability of profit with an average gain of {stats['average_profitable_return']:.2f}% over the holding period.
"""
        return text.strip()

    def _format_event_for_rag(self, event: Dict[str, Any]) -> str:
        """Format news event as rich text for RAG embedding"""
        text = f"""
HISTORICAL NEWS EVENT IMPACT

Event: {event['event_title']}
Type: {event['event_type'].replace('_', ' ').title()}
Date: {event['event_date']}
"""
        if event.get('sector'):
            text += f"Sector: {event['sector']}\n"

        text += f"\nSymbols Affected: {', '.join(event['symbols_affected'])}\n"

        if event.get('immediate_impact'):
            text += "\nImmediate Price Impact:\n"
            for symbol, impact in event['immediate_impact'].items():
                text += f"- {symbol}: {impact:+.2f}%\n"

        if event.get('recovery_timeline'):
            text += f"\nRecovery Timeline: {event['recovery_timeline']}\n"

        if event.get('pattern_observed'):
            text += f"\nPattern Observed: {event['pattern_observed']}\n"

        if event.get('sentiment'):
            text += f"Sentiment: {event['sentiment'].title()}\n"

        text += f"""
Historical Context:
This event type has occurred {event.get('occurrence_count', 1)} time(s) in our database. Use this information to predict similar market reactions when comparable events occur in the future.
"""
        return text.strip()

    def _calculate_priority(self, pattern: Dict[str, Any]) -> int:
        """
        Calculate priority score for pattern (1-100)
        Higher priority patterns are retrieved first during RAG queries
        """
        success_rate = pattern["statistics"]["success_rate"]
        occurrences = pattern["statistics"]["total_occurrences"]
        avg_return = abs(pattern["statistics"]["average_profitable_return"])

        # Priority factors
        success_score = success_rate * 40  # Max 40 points
        occurrence_score = min(occurrences / 50 * 30, 30)  # Max 30 points (capped at 50 occurrences)
        return_score = min(avg_return / 10 * 30, 30)  # Max 30 points (capped at 10% return)

        priority = int(success_score + occurrence_score + return_score)
        return min(priority, 100)

    def _calculate_event_priority(self, event: Dict[str, Any]) -> int:
        """Calculate priority score for news event (1-100)"""
        # Recent events are more relevant
        event_date = event.get('event_date')
        if isinstance(event_date, datetime):
            days_ago = (datetime.now(timezone.utc) - event_date).days
            recency_score = max(50 - (days_ago / 30), 0)  # Decay over months
        else:
            recency_score = 25

        # High-impact events get higher priority
        impacts = event.get('immediate_impact', {}).values()
        if impacts:
            avg_impact = sum(abs(i) for i in impacts) / len(impacts)
            impact_score = min(avg_impact * 5, 50)  # Max 50 points
        else:
            impact_score = 25

        priority = int(recency_score + impact_score)
        return min(priority, 100)

    async def bulk_ingest_patterns(
        self,
        min_success_rate: float = 0.60
    ) -> Dict[str, Any]:
        """
        Bulk ingest all discovered patterns meeting criteria

        Args:
            min_success_rate: Minimum success rate to include

        Returns:
            Ingestion summary
        """
        from .pattern_mining import get_pattern_miner

        pattern_miner = get_pattern_miner(self.db)
        patterns = await pattern_miner.get_all_valid_patterns(min_success_rate)

        ingested = 0
        failed = 0

        for pattern in patterns:
            try:
                await self.ingest_discovered_pattern(pattern, generate_embedding=False)
                ingested += 1
            except Exception as e:
                logger.error(f"Failed to ingest pattern {pattern.get('id')}: {e}")
                failed += 1

        summary = {
            "total_patterns": len(patterns),
            "ingested": ingested,
            "failed": failed,
            "timestamp": datetime.now(timezone.utc)
        }

        logger.info(
            f"Bulk ingestion complete: {ingested}/{len(patterns)} patterns ingested into RAG"
        )

        return summary

    async def bulk_ingest_events(
        self,
        days_back: int = 365
    ) -> Dict[str, Any]:
        """
        Bulk ingest recent news events

        Args:
            days_back: How many days back to ingest

        Returns:
            Ingestion summary
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)

        cursor = self.db.news_event_knowledge.find({
            "event_date": {"$gte": cutoff}
        })

        events = await cursor.to_list(length=1000)

        ingested = 0
        failed = 0

        for event in events:
            try:
                await self.ingest_news_event(event, generate_embedding=False)
                ingested += 1
            except Exception as e:
                logger.error(f"Failed to ingest event {event.get('_id')}: {e}")
                failed += 1

        summary = {
            "total_events": len(events),
            "ingested": ingested,
            "failed": failed,
            "timestamp": datetime.now(timezone.utc)
        }

        logger.info(
            f"Bulk ingestion complete: {ingested}/{len(events)} events ingested into RAG"
        )

        return summary

    async def query_relevant_knowledge(
        self,
        symbol: str,
        query_type: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query RAG for relevant knowledge about a symbol

        Args:
            symbol: Stock symbol
            query_type: Optional filter (pattern/event)
            limit: Max results

        Returns:
            List of relevant knowledge entries
        """
        query = {"$or": [
            {"symbol": symbol},
            {"symbols_affected": symbol}
        ]}

        if query_type:
            if query_type == "pattern":
                query["type"] = "discovered_pattern"
            elif query_type == "event":
                query["type"] = "news_event_impact"

        cursor = self.rag_collection.find(query).sort("priority", -1).limit(limit)
        results = await cursor.to_list(length=limit)

        for result in results:
            result['id'] = str(result.pop('_id'))

        return results

    async def create_indexes(self):
        """Create database indexes for efficient querying"""
        await self.rag_collection.create_index([("type", 1)])
        await self.rag_collection.create_index([("symbol", 1)])
        await self.rag_collection.create_index([("symbols_affected", 1)])
        await self.rag_collection.create_index([("priority", -1)])
        await self.rag_collection.create_index([("tags", 1)])
        await self.metadata_collection.create_index([("rag_entry_id", 1)])
        logger.info("Created indexes for RAG knowledge collections")


# Singleton instance
_rag_ingestion: Optional[RAGKnowledgeIngestion] = None


def get_rag_ingestion(db: AsyncIOMotorDatabase) -> RAGKnowledgeIngestion:
    """Get or create singleton instance"""
    global _rag_ingestion
    if _rag_ingestion is None:
        _rag_ingestion = RAGKnowledgeIngestion(db)
    return _rag_ingestion

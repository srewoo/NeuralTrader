"""
News & Events Knowledge Building System
Stores historical context of how news/events affected specific stocks
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, timezone
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

logger = logging.getLogger(__name__)


class NewsEventKnowledge:
    """
    Manages historical knowledge about news impact on stocks
    Enables pattern recognition and learning from past events
    """

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.news_event_knowledge

    async def store_event_impact(
        self,
        event_title: str,
        event_date: datetime,
        event_type: str,
        symbols_affected: List[str],
        sector: Optional[str] = None,
        immediate_impact: Dict[str, float] = None,
        recovery_timeline: Optional[str] = None,
        pattern_observed: Optional[str] = None,
        source: Optional[str] = None,
        sentiment: Optional[str] = None
    ) -> str:
        """
        Store a news/event and its observed market impact

        Args:
            event_title: Title/description of the event
            event_date: When the event occurred
            event_type: Category (rbi_policy, earnings, merger, sector_news, etc.)
            symbols_affected: List of stock symbols impacted
            sector: Sector affected (if sector-wide impact)
            immediate_impact: Dict of {symbol: percent_change} for 1-2 days after
            recovery_timeline: How long it took to recover (e.g., "1 week", "3 days")
            pattern_observed: Description of the observed pattern
            source: News source
            sentiment: Event sentiment (positive/negative/neutral)

        Returns:
            Event ID
        """
        event_doc = {
            "event_title": event_title,
            "event_date": event_date,
            "event_type": event_type,
            "symbols_affected": symbols_affected,
            "sector": sector,
            "immediate_impact": immediate_impact or {},
            "recovery_timeline": recovery_timeline,
            "pattern_observed": pattern_observed,
            "source": source,
            "sentiment": sentiment,
            "created_at": datetime.now(timezone.utc),
            "occurrence_count": 1  # Track if similar events happen multiple times
        }

        result = await self.collection.insert_one(event_doc)
        event_id = str(result.inserted_id)

        logger.info(f"Stored event impact: {event_title} affecting {len(symbols_affected)} symbols")
        return event_id

    async def find_similar_events(
        self,
        event_type: str,
        sector: Optional[str] = None,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical events to learn from past patterns

        Args:
            event_type: Type of event to search for
            sector: Optional sector filter
            limit: Maximum number of results

        Returns:
            List of similar historical events
        """
        query = {"event_type": event_type}
        if sector:
            query["sector"] = sector

        cursor = self.collection.find(query).sort("event_date", -1).limit(limit)
        events = await cursor.to_list(length=limit)

        # Remove MongoDB _id from results
        for event in events:
            event['id'] = str(event.pop('_id'))

        return events

    async def get_symbol_event_history(
        self,
        symbol: str,
        lookback_days: int = 365
    ) -> List[Dict[str, Any]]:
        """
        Get all events that affected a specific symbol in the past

        Args:
            symbol: Stock symbol
            lookback_days: How far back to look

        Returns:
            List of events affecting this symbol
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        cursor = self.collection.find({
            "symbols_affected": symbol,
            "event_date": {"$gte": cutoff_date}
        }).sort("event_date", -1)

        events = await cursor.to_list(length=100)

        for event in events:
            event['id'] = str(event.pop('_id'))

        return events

    async def build_pattern_summary(self, event_type: str, sector: Optional[str] = None) -> str:
        """
        Generate a pattern summary from historical events

        Args:
            event_type: Type of event
            sector: Optional sector filter

        Returns:
            Natural language pattern summary
        """
        similar_events = await self.find_similar_events(event_type, sector, limit=20)

        if not similar_events:
            return f"No historical data for {event_type} events"

        # Analyze patterns
        total_events = len(similar_events)
        negative_impacts = sum(1 for e in similar_events if e.get('sentiment') == 'negative')
        positive_impacts = sum(1 for e in similar_events if e.get('sentiment') == 'positive')

        # Calculate average impact
        impacts = []
        for event in similar_events:
            for symbol, change in event.get('immediate_impact', {}).items():
                impacts.append(change)

        avg_impact = sum(impacts) / len(impacts) if impacts else 0

        # Build summary
        summary = f"Historical Pattern: {event_type.replace('_', ' ').title()}\n"
        summary += f"Based on {total_events} similar events:\n"
        summary += f"- Average immediate impact: {avg_impact:+.2f}%\n"
        summary += f"- Negative outcomes: {negative_impacts}/{total_events} ({negative_impacts/total_events*100:.0f}%)\n"
        summary += f"- Positive outcomes: {positive_impacts}/{total_events} ({positive_impacts/total_events*100:.0f}%)\n"

        # Recovery patterns
        recovery_times = [e.get('recovery_timeline') for e in similar_events if e.get('recovery_timeline')]
        if recovery_times:
            from collections import Counter
            most_common_recovery = Counter(recovery_times).most_common(1)[0][0]
            summary += f"- Most common recovery time: {most_common_recovery}\n"

        return summary

    async def auto_detect_event_impact(
        self,
        symbol: str,
        news_item: Dict[str, Any],
        price_data_before: float,
        price_data_after: float,
        days_elapsed: int = 2
    ) -> Optional[str]:
        """
        Automatically detect if a news event had significant market impact

        Args:
            symbol: Stock symbol
            news_item: News item with title, date, sentiment, etc.
            price_data_before: Price before news
            price_data_after: Price after news (1-2 days later)
            days_elapsed: Days between before/after prices

        Returns:
            Event ID if stored, None if impact too small
        """
        impact_percent = ((price_data_after - price_data_before) / price_data_before) * 100

        # Only store if impact is significant (> 2%)
        if abs(impact_percent) < 2.0:
            return None

        # Determine event type from news
        event_type = self._classify_event_type(news_item.get('title', ''))

        # Store the event impact
        event_id = await self.store_event_impact(
            event_title=news_item.get('title', 'Unknown Event'),
            event_date=news_item.get('published_date', datetime.now(timezone.utc)),
            event_type=event_type,
            symbols_affected=[symbol],
            sector=news_item.get('sector'),
            immediate_impact={symbol: round(impact_percent, 2)},
            recovery_timeline=f"{days_elapsed} days observed",
            pattern_observed=f"Price moved {impact_percent:+.2f}% following {event_type} news",
            source=news_item.get('source'),
            sentiment=news_item.get('sentiment')
        )

        logger.info(
            f"Auto-detected event impact: {news_item.get('title')} → "
            f"{symbol} {impact_percent:+.2f}%"
        )

        return event_id

    def _classify_event_type(self, title: str) -> str:
        """Classify event type based on title keywords"""
        title_lower = title.lower()

        if any(word in title_lower for word in ['rbi', 'rate', 'policy', 'repo', 'inflation']):
            return 'rbi_policy'
        elif any(word in title_lower for word in ['earning', 'profit', 'revenue', 'result', 'q1', 'q2', 'q3', 'q4']):
            return 'earnings'
        elif any(word in title_lower for word in ['merger', 'acquisition', 'takeover', 'deal']):
            return 'merger_acquisition'
        elif any(word in title_lower for word in ['regulation', 'sebi', 'compliance', 'ban']):
            return 'regulatory'
        elif any(word in title_lower for word in ['product', 'launch', 'innovation']):
            return 'product_launch'
        elif any(word in title_lower for word in ['fraud', 'scam', 'investigation', 'probe']):
            return 'negative_event'
        else:
            return 'general_news'

    async def create_indexes(self):
        """Create database indexes for efficient querying"""
        await self.collection.create_index([("event_type", 1)])
        await self.collection.create_index([("sector", 1)])
        await self.collection.create_index([("symbols_affected", 1)])
        await self.collection.create_index([("event_date", -1)])
        await self.collection.create_index([("event_type", 1), ("sector", 1)])
        logger.info("Created indexes for news_event_knowledge collection")


# Singleton instance
_news_event_knowledge: Optional[NewsEventKnowledge] = None


def get_news_event_knowledge(db: AsyncIOMotorDatabase) -> NewsEventKnowledge:
    """Get or create singleton instance"""
    global _news_event_knowledge
    if _news_event_knowledge is None:
        _news_event_knowledge = NewsEventKnowledge(db)
    return _news_event_knowledge

"""
API Cost Tracking System
Tracks LLM API usage and costs for budgeting and monitoring
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase

logger = logging.getLogger(__name__)

# Pricing as of Dec 2024 (per 1M tokens)
PRICING = {
    "openai": {
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.150, "output": 0.600},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
        "o1-preview": {"input": 15.00, "output": 60.00},
        "o1-mini": {"input": 3.00, "output": 12.00},
    },
    "gemini": {
        "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
        "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-1.0-pro": {"input": 0.50, "output": 1.50},
    }
}


class CostTracker:
    """Tracks API costs for LLM usage"""

    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self.collection = db.api_costs

    async def track_call(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        request_type: str = "analysis",
        symbol: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track an API call and calculate cost

        Args:
            provider: "openai" or "gemini"
            model: Model name (e.g., "gpt-4", "gemini-1.5-pro")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            request_type: Type of request (analysis, qa, portfolio, etc.)
            symbol: Stock symbol if applicable
            user_id: User identifier if multi-user system

        Returns:
            Dict with cost information
        """
        # Get pricing for this model
        pricing = self._get_pricing(provider, model)

        # Calculate costs (convert from per 1M tokens to actual)
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        total_cost = input_cost + output_cost

        # Create tracking record
        record = {
            "timestamp": datetime.now(timezone.utc),
            "provider": provider,
            "model": model,
            "request_type": request_type,
            "symbol": symbol,
            "user_id": user_id or "default",
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens
            },
            "cost": {
                "input": round(input_cost, 6),
                "output": round(output_cost, 6),
                "total": round(total_cost, 6),
                "currency": "USD"
            },
            "pricing_snapshot": pricing
        }

        # Save to database
        try:
            await self.collection.insert_one(record)
            logger.info(
                f"Tracked API call: {provider}/{model} - "
                f"{input_tokens + output_tokens} tokens - ${total_cost:.6f}"
            )
        except Exception as e:
            logger.error(f"Failed to save cost tracking record: {e}")

        return {
            "tokens": record["tokens"],
            "cost": record["cost"],
            "timestamp": record["timestamp"]
        }

    def _get_pricing(self, provider: str, model: str) -> Dict[str, float]:
        """Get pricing for a specific model"""
        provider_pricing = PRICING.get(provider.lower(), {})

        # Normalize model name (remove version suffixes, etc.)
        normalized_model = self._normalize_model_name(model)

        # Try exact match first
        if normalized_model in provider_pricing:
            return provider_pricing[normalized_model]

        # Try partial match (e.g., "gpt-4-0125-preview" -> "gpt-4")
        for key in provider_pricing.keys():
            if normalized_model.startswith(key):
                return provider_pricing[key]

        # Default to most expensive model as fallback
        if provider.lower() == "openai":
            logger.warning(f"Unknown OpenAI model: {model}, using gpt-4 pricing")
            return PRICING["openai"]["gpt-4"]
        else:
            logger.warning(f"Unknown Gemini model: {model}, using gemini-1.5-pro pricing")
            return PRICING["gemini"]["gemini-1.5-pro"]

    def _normalize_model_name(self, model: str) -> str:
        """Normalize model name for pricing lookup"""
        model = model.lower()

        # Handle common variations
        if "gpt-4o-mini" in model:
            return "gpt-4o-mini"
        elif "gpt-4o" in model:
            return "gpt-4o"
        elif "gpt-4-turbo" in model or "gpt-4-1106" in model or "gpt-4-0125" in model:
            return "gpt-4-turbo"
        elif "gpt-4" in model:
            return "gpt-4"
        elif "gpt-3.5" in model or "gpt-35" in model:
            return "gpt-3.5-turbo"
        elif "o1-preview" in model:
            return "o1-preview"
        elif "o1-mini" in model:
            return "o1-mini"
        elif "gemini-1.5-pro" in model or "gemini-pro-1.5" in model:
            return "gemini-1.5-pro"
        elif "gemini-1.5-flash" in model or "gemini-flash" in model:
            return "gemini-1.5-flash"
        elif "gemini-1.0-pro" in model or "gemini-pro" in model:
            return "gemini-1.0-pro"

        return model

    async def get_usage_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        user_id: Optional[str] = None,
        group_by: str = "day"  # day, week, month, provider, model
    ) -> Dict[str, Any]:
        """
        Get usage summary for a time period

        Args:
            start_date: Start of period (default: 30 days ago)
            end_date: End of period (default: now)
            user_id: Filter by user
            group_by: How to group results

        Returns:
            Summary statistics
        """
        from datetime import timedelta

        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)

        # Build query
        query = {
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }
        if user_id:
            query["user_id"] = user_id

        # Aggregation pipeline
        pipeline = [
            {"$match": query},
            {
                "$group": {
                    "_id": f"${group_by}" if group_by in ["provider", "model", "request_type"] else {
                        "$dateToString": {
                            "format": "%Y-%m-%d" if group_by == "day" else "%Y-%W" if group_by == "week" else "%Y-%m",
                            "date": "$timestamp"
                        }
                    },
                    "total_calls": {"$sum": 1},
                    "total_tokens": {"$sum": "$tokens.total"},
                    "total_cost": {"$sum": "$cost.total"},
                    "input_tokens": {"$sum": "$tokens.input"},
                    "output_tokens": {"$sum": "$tokens.output"}
                }
            },
            {"$sort": {"_id": 1}}
        ]

        try:
            results = await self.collection.aggregate(pipeline).to_list(length=None)

            # Calculate totals
            total_calls = sum(r["total_calls"] for r in results)
            total_tokens = sum(r["total_tokens"] for r in results)
            total_cost = sum(r["total_cost"] for r in results)

            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "totals": {
                    "calls": total_calls,
                    "tokens": total_tokens,
                    "cost": round(total_cost, 2),
                    "currency": "USD"
                },
                "breakdown": results,
                "group_by": group_by
            }
        except Exception as e:
            logger.error(f"Failed to generate usage summary: {e}")
            return {"error": str(e)}

    async def get_current_month_cost(self, user_id: Optional[str] = None) -> float:
        """Get total cost for current month"""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        start_of_month = datetime(now.year, now.month, 1, tzinfo=timezone.utc)

        query = {
            "timestamp": {"$gte": start_of_month}
        }
        if user_id:
            query["user_id"] = user_id

        try:
            pipeline = [
                {"$match": query},
                {
                    "$group": {
                        "_id": None,
                        "total": {"$sum": "$cost.total"}
                    }
                }
            ]
            results = await self.collection.aggregate(pipeline).to_list(length=1)

            if results:
                return round(results[0]["total"], 2)
            return 0.0
        except Exception as e:
            logger.error(f"Failed to get current month cost: {e}")
            return 0.0


# Singleton instance
_tracker_instance: Optional[CostTracker] = None


def get_cost_tracker(db: AsyncIOMotorDatabase) -> CostTracker:
    """Get or create cost tracker instance"""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = CostTracker(db)
    return _tracker_instance

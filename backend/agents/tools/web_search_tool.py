"""
Web Search Tool for LangGraph Agents

Provides web search capabilities to agents for real-time news and information.
"""

from typing import Dict, Any, Optional, List
import logging
from langchain_core.tools import tool

from news.web_search import get_web_search_service

logger = logging.getLogger(__name__)


@tool
async def search_stock_news_tool(
    symbol: str,
    company_name: Optional[str] = None,
    max_results: int = 5,
    days_back: int = 7
) -> str:
    """
    Search for recent news about a stock.

    Use this tool when you need current news, events, or developments
    about a specific stock or company.

    Args:
        symbol: Stock ticker symbol (e.g., RELIANCE, TCS, INFY)
        company_name: Optional company name for better results
        max_results: Maximum number of articles (default 5)
        days_back: How many days back to search (default 7)

    Returns:
        Formatted string with news articles including titles, sources, and snippets
    """
    try:
        service = get_web_search_service()
        results = await service.search_stock_news(
            symbol=symbol,
            company_name=company_name,
            max_results=max_results,
            days_back=days_back
        )

        if not results:
            return f"No recent news found for {symbol} in the last {days_back} days."

        # Deduplicate
        results = service.deduplicate_results(results)

        # Format results
        output = [f"Recent news for {symbol} ({len(results)} articles):\n"]

        for i, result in enumerate(results, 1):
            date_str = ""
            if result.published_date:
                date_str = f" ({result.published_date.strftime('%Y-%m-%d')})"

            output.append(
                f"{i}. {result.title}{date_str}\n"
                f"   Source: {result.source}\n"
                f"   {result.snippet}\n"
                f"   URL: {result.url}\n"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Stock news search failed: {e}")
        return f"Error searching for news: {str(e)}"


@tool
async def search_market_news_tool(
    topic: str = "Indian stock market",
    max_results: int = 5,
    days_back: int = 3
) -> str:
    """
    Search for general market news and trends.

    Use this tool when you need information about:
    - Overall market conditions
    - Economic indicators
    - Policy changes
    - Market sentiment

    Args:
        topic: Market topic to search (default "Indian stock market")
        max_results: Maximum number of articles (default 5)
        days_back: How many days back to search (default 3)

    Returns:
        Formatted string with market news articles
    """
    try:
        service = get_web_search_service()
        results = await service.search_market_news(
            topic=topic,
            max_results=max_results,
            days_back=days_back
        )

        if not results:
            return f"No recent market news found for '{topic}'."

        # Deduplicate
        results = service.deduplicate_results(results)

        # Format results
        output = [f"Market news: {topic} ({len(results)} articles):\n"]

        for i, result in enumerate(results, 1):
            date_str = ""
            if result.published_date:
                date_str = f" ({result.published_date.strftime('%Y-%m-%d')})"

            output.append(
                f"{i}. {result.title}{date_str}\n"
                f"   Source: {result.source}\n"
                f"   {result.snippet}\n"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Market news search failed: {e}")
        return f"Error searching for market news: {str(e)}"


@tool
async def search_sector_news_tool(
    sector: str,
    max_results: int = 5,
    days_back: int = 7
) -> str:
    """
    Search for sector-specific news and developments.

    Use this tool when analyzing sector trends or comparing stocks
    within a sector.

    Args:
        sector: Sector name (e.g., "banking", "IT", "pharma", "auto")
        max_results: Maximum number of articles (default 5)
        days_back: How many days back to search (default 7)

    Returns:
        Formatted string with sector news
    """
    try:
        service = get_web_search_service()
        results = await service.search_sector_news(
            sector=sector,
            max_results=max_results,
            days_back=days_back
        )

        if not results:
            return f"No recent news found for {sector} sector."

        # Deduplicate
        results = service.deduplicate_results(results)

        # Format results
        output = [f"{sector.title()} sector news ({len(results)} articles):\n"]

        for i, result in enumerate(results, 1):
            date_str = ""
            if result.published_date:
                date_str = f" ({result.published_date.strftime('%Y-%m-%d')})"

            output.append(
                f"{i}. {result.title}{date_str}\n"
                f"   Source: {result.source}\n"
                f"   {result.snippet}\n"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Sector news search failed: {e}")
        return f"Error searching for sector news: {str(e)}"


@tool
async def general_web_search_tool(
    query: str,
    max_results: int = 5,
    days_back: int = 7
) -> str:
    """
    Perform a general web search for any query.

    Use this tool when you need information that isn't covered by
    other specialized search tools.

    Args:
        query: Search query
        max_results: Maximum number of results (default 5)
        days_back: How many days back to search (default 7)

    Returns:
        Formatted string with search results
    """
    try:
        service = get_web_search_service()
        results = await service.search_news(
            query=query,
            max_results=max_results,
            days_back=days_back
        )

        if not results:
            return f"No results found for '{query}'."

        # Deduplicate
        results = service.deduplicate_results(results)

        # Format results
        output = [f"Search results for '{query}' ({len(results)} results):\n"]

        for i, result in enumerate(results, 1):
            date_str = ""
            if result.published_date:
                date_str = f" ({result.published_date.strftime('%Y-%m-%d')})"

            output.append(
                f"{i}. {result.title}{date_str}\n"
                f"   Source: {result.source}\n"
                f"   {result.snippet}\n"
                f"   URL: {result.url}\n"
            )

        return "\n".join(output)

    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error performing search: {str(e)}"


# Export all tools
WEB_SEARCH_TOOLS = [
    search_stock_news_tool,
    search_market_news_tool,
    search_sector_news_tool,
    general_web_search_tool
]


def get_web_search_tools() -> List:
    """Get list of all web search tools for agent use"""
    return WEB_SEARCH_TOOLS

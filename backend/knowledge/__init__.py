"""
Knowledge Management Package
Handles news events and pattern mining for continuous learning
"""

from .news_events import NewsEventKnowledge, get_news_event_knowledge
from .pattern_mining import PatternMiner, get_pattern_miner, PatternMatch
from .rag_ingestion import RAGKnowledgeIngestion, get_rag_ingestion
from .auto_discovery import AutoDiscoveryPipeline, get_discovery_pipeline

__all__ = [
    'NewsEventKnowledge',
    'get_news_event_knowledge',
    'PatternMiner',
    'get_pattern_miner',
    'PatternMatch',
    'RAGKnowledgeIngestion',
    'get_rag_ingestion',
    'AutoDiscoveryPipeline',
    'get_discovery_pipeline'
]

"""
Knowledge Retrieval System
Performs semantic search and context retrieval for RAG
"""

from typing import List, Dict, Optional, Any
import logging
from .vector_store import get_vector_store
from .embeddings import get_embedding_generator

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """
    Retrieves relevant knowledge from vector store for RAG
    """
    
    def __init__(self):
        """Initialize retriever with vector store and embeddings"""
        self.vector_store = get_vector_store()
        self.embedding_generator = get_embedding_generator()
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        min_similarity: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            n_results: Maximum number of results to return
            filters: Metadata filters (e.g., {"category": "technical_analysis"})
            min_similarity: Minimum similarity threshold (0-1)
            
        Returns:
            List of relevant documents with metadata and scores
        """
        try:
            # Query vector store
            results = self.vector_store.query(
                query_texts=[query],
                n_results=n_results,
                where=filters
            )
            
            # Format results
            retrieved_docs = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for i in range(len(results["ids"][0])):
                    # FAISS returns distances (lower is better)
                    # Convert to similarity score (higher is better)
                    distance = results["distances"][0][i]
                    similarity = 1 / (1 + distance)  # Convert distance to similarity
                    
                    # Filter by minimum similarity
                    if similarity >= min_similarity:
                        retrieved_docs.append({
                            "id": results["ids"][0][i],
                            "content": results["documents"][0][i],
                            "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                            "similarity": similarity,
                            "distance": distance
                        })
            
            logger.info(f"Retrieved {len(retrieved_docs)} relevant documents for query")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            return []
    
    def retrieve_by_category(
        self,
        query: str,
        category: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents filtered by category
        
        Args:
            query: Search query
            category: Category to filter by (e.g., "patterns", "strategies", "indicators")
            n_results: Number of results
            
        Returns:
            List of relevant documents
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            filters={"category": category}
        )
    
    def retrieve_for_stock(
        self,
        query: str,
        symbol: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents related to a specific stock
        
        Args:
            query: Search query
            symbol: Stock symbol
            n_results: Number of results
            
        Returns:
            List of relevant documents
        """
        return self.retrieve(
            query=query,
            n_results=n_results,
            filters={"symbol": symbol}
        )
    
    def build_context(
        self,
        query: str,
        n_results: int = 5,
        max_tokens: int = 2000
    ) -> str:
        """
        Build context string from retrieved documents for LLM prompt
        
        Args:
            query: Search query
            n_results: Number of documents to retrieve
            max_tokens: Approximate maximum tokens for context (rough estimate)
            
        Returns:
            Formatted context string
        """
        try:
            # Retrieve relevant documents
            docs = self.retrieve(query=query, n_results=n_results)
            
            if not docs:
                return "No relevant historical knowledge found."
            
            # Build context string
            context_parts = ["=== RELEVANT HISTORICAL KNOWLEDGE ===\n"]
            
            total_chars = 0
            max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
            
            for i, doc in enumerate(docs, 1):
                doc_text = f"\n[Source {i}] (Relevance: {doc['similarity']:.2f})\n"
                doc_text += f"{doc['content']}\n"
                
                # Add metadata if available
                if doc.get("metadata"):
                    metadata = doc["metadata"]
                    if metadata.get("category"):
                        doc_text += f"Category: {metadata['category']}\n"
                    if metadata.get("date"):
                        doc_text += f"Date: {metadata['date']}\n"
                
                # Check if adding this doc would exceed max tokens
                if total_chars + len(doc_text) > max_chars:
                    break
                
                context_parts.append(doc_text)
                total_chars += len(doc_text)
            
            context = "".join(context_parts)
            context += "\n=== END OF HISTORICAL KNOWLEDGE ===\n"
            
            logger.info(f"Built context with {len(docs)} documents (~{total_chars} chars)")
            return context
            
        except Exception as e:
            logger.error(f"Failed to build context: {e}")
            return "Error retrieving historical knowledge."
    
    def get_similar_patterns(
        self,
        technical_indicators: Dict[str, float],
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar historical patterns based on technical indicators
        
        Args:
            technical_indicators: Dict of indicator values
            n_results: Number of similar patterns to find
            
        Returns:
            List of similar historical patterns
        """
        try:
            # Create query from indicators
            query_parts = []
            for indicator, value in technical_indicators.items():
                query_parts.append(f"{indicator}: {value:.2f}")
            
            query = "Technical pattern: " + ", ".join(query_parts)
            
            # Retrieve similar patterns
            return self.retrieve(
                query=query,
                n_results=n_results,
                filters={"category": "patterns"}
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve similar patterns: {e}")
            return []
    
    def get_strategy_recommendations(
        self,
        market_condition: str,
        n_results: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get strategy recommendations for current market conditions
        
        Args:
            market_condition: Description of market condition
            n_results: Number of strategies to retrieve
            
        Returns:
            List of recommended strategies
        """
        query = f"Trading strategy for {market_condition}"
        return self.retrieve(
            query=query,
            n_results=n_results,
            filters={"category": "strategies"}
        )


# Global instance
_retriever_instance = None


def get_retriever() -> KnowledgeRetriever:
    """
    Get or create global KnowledgeRetriever instance (Singleton pattern)
    """
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = KnowledgeRetriever()
    return _retriever_instance


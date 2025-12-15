"""
Document Ingestion Pipeline
Processes and ingests documents into the vector store
"""

from typing import List, Dict, Any, Optional
import logging
import hashlib
import json
from datetime import datetime
from .vector_store import get_vector_store
from .embeddings import get_embedding_generator

logger = logging.getLogger(__name__)


class DocumentIngestion:
    """
    Handles document processing and ingestion into vector store
    """
    
    def __init__(self):
        """Initialize ingestion pipeline"""
        self.vector_store = get_vector_store()
        self.embedding_generator = get_embedding_generator()
    
    def _generate_document_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Generate unique document ID based on content and metadata
        
        Args:
            content: Document content
            metadata: Document metadata
            
        Returns:
            Unique document ID
        """
        # Create hash from content and key metadata
        hash_input = content + json.dumps(metadata, sort_keys=True)
        doc_id = hashlib.md5(hash_input.encode()).hexdigest()
        return doc_id
    
    def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        doc_id: Optional[str] = None
    ) -> bool:
        """
        Ingest a single document
        
        Args:
            content: Document text content
            metadata: Optional metadata (category, source, date, etc.)
            doc_id: Optional custom document ID
            
        Returns:
            bool: Success status
        """
        try:
            # Prepare metadata
            if metadata is None:
                metadata = {}
            
            # Add timestamp if not present
            if "ingested_at" not in metadata:
                metadata["ingested_at"] = datetime.now().isoformat()
            
            # Generate ID if not provided
            if doc_id is None:
                doc_id = self._generate_document_id(content, metadata)
            
            # Generate embedding
            embedding = self.embedding_generator.generate_single_embedding(content)
            
            # Add to vector store
            success = self.vector_store.add_documents(
                documents=[content],
                metadatas=[metadata],
                ids=[doc_id],
                embeddings=[embedding]
            )
            
            if success:
                logger.info(f"Successfully ingested document: {doc_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to ingest document: {e}")
            return False
    
    def ingest_batch(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 32
    ) -> Dict[str, int]:
        """
        Ingest multiple documents in batches
        
        Args:
            documents: List of dicts with 'content' and optional 'metadata', 'id'
            batch_size: Number of documents to process at once
            
        Returns:
            Dict with success and failure counts
        """
        try:
            total = len(documents)
            success_count = 0
            failure_count = 0
            
            # Process in batches
            for i in range(0, total, batch_size):
                batch = documents[i:i + batch_size]
                
                contents = []
                metadatas = []
                ids = []
                
                for doc in batch:
                    content = doc.get("content", "")
                    metadata = doc.get("metadata", {})
                    doc_id = doc.get("id")
                    
                    # Add timestamp
                    if "ingested_at" not in metadata:
                        metadata["ingested_at"] = datetime.now().isoformat()
                    
                    # Generate ID if not provided
                    if doc_id is None:
                        doc_id = self._generate_document_id(content, metadata)
                    
                    contents.append(content)
                    metadatas.append(metadata)
                    ids.append(doc_id)
                
                # Generate embeddings for batch
                embeddings = self.embedding_generator.generate_embeddings(
                    contents,
                    batch_size=batch_size
                )
                
                # Convert numpy array to list of lists
                embeddings_list = [emb.tolist() for emb in embeddings]
                
                # Add batch to vector store
                success = self.vector_store.add_documents(
                    documents=contents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings_list
                )
                
                if success:
                    success_count += len(batch)
                else:
                    failure_count += len(batch)
                
                logger.info(f"Processed batch {i // batch_size + 1}: {len(batch)} documents")
            
            logger.info(f"Batch ingestion complete. Success: {success_count}, Failed: {failure_count}")
            
            return {
                "total": total,
                "success": success_count,
                "failed": failure_count
            }
            
        except Exception as e:
            logger.error(f"Batch ingestion failed: {e}")
            return {"total": len(documents), "success": 0, "failed": len(documents)}
    
    def ingest_trading_pattern(
        self,
        pattern_name: str,
        description: str,
        indicators: Dict[str, float],
        outcome: str,
        confidence: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Ingest a trading pattern with structured data
        
        Args:
            pattern_name: Name of the pattern
            description: Pattern description
            indicators: Technical indicators at the time
            outcome: What happened (e.g., "bullish reversal", "bearish continuation")
            confidence: Confidence score (0-100)
            additional_info: Additional metadata
            
        Returns:
            bool: Success status
        """
        # Build content string
        content = f"Pattern: {pattern_name}\n"
        content += f"Description: {description}\n"
        content += f"Technical Indicators: {json.dumps(indicators)}\n"
        content += f"Outcome: {outcome}\n"
        content += f"Confidence: {confidence}%\n"
        
        # Build metadata
        metadata = {
            "category": "patterns",
            "pattern_name": pattern_name,
            "outcome": outcome,
            "confidence": confidence,
            "date": datetime.now().isoformat()
        }
        
        if additional_info:
            metadata.update(additional_info)
        
        return self.ingest_document(content, metadata)
    
    def ingest_trading_strategy(
        self,
        strategy_name: str,
        description: str,
        conditions: List[str],
        expected_outcome: str,
        risk_level: str,
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> bool:
        """
        Ingest a trading strategy
        
        Args:
            strategy_name: Name of the strategy
            description: Strategy description
            conditions: List of conditions when to apply
            expected_outcome: Expected result
            risk_level: Risk level (low, medium, high)
            performance_metrics: Historical performance data
            
        Returns:
            bool: Success status
        """
        # Build content string
        content = f"Strategy: {strategy_name}\n"
        content += f"Description: {description}\n"
        content += f"Conditions:\n"
        for condition in conditions:
            content += f"  - {condition}\n"
        content += f"Expected Outcome: {expected_outcome}\n"
        content += f"Risk Level: {risk_level}\n"
        
        if performance_metrics:
            content += f"Performance Metrics: {json.dumps(performance_metrics)}\n"
        
        # Build metadata
        metadata = {
            "category": "strategies",
            "strategy_name": strategy_name,
            "risk_level": risk_level,
            "date": datetime.now().isoformat()
        }
        
        return self.ingest_document(content, metadata)
    
    def ingest_market_commentary(
        self,
        symbol: str,
        commentary: str,
        date: str,
        source: str,
        sentiment: Optional[str] = None
    ) -> bool:
        """
        Ingest market commentary or analysis
        
        Args:
            symbol: Stock symbol
            commentary: Commentary text
            date: Date of commentary
            source: Source of commentary
            sentiment: Optional sentiment (positive, negative, neutral)
            
        Returns:
            bool: Success status
        """
        # Build content
        content = f"Stock: {symbol}\n"
        content += f"Date: {date}\n"
        content += f"Commentary: {commentary}\n"
        
        # Build metadata
        metadata = {
            "category": "commentary",
            "symbol": symbol,
            "date": date,
            "source": source
        }
        
        if sentiment:
            metadata["sentiment"] = sentiment
        
        return self.ingest_document(content, metadata)
    
    def get_ingestion_stats(self) -> Dict[str, Any]:
        """
        Get statistics about ingested documents
        
        Returns:
            Dict with collection statistics
        """
        return self.vector_store.get_collection_info()


# Global instance
_ingestion_instance = None


def get_ingestion_pipeline() -> DocumentIngestion:
    """
    Get or create global DocumentIngestion instance (Singleton pattern)
    """
    global _ingestion_instance
    if _ingestion_instance is None:
        _ingestion_instance = DocumentIngestion()
    return _ingestion_instance


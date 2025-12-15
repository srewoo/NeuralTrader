"""
Vector Store Management using ChromaDB
Handles initialization, storage, and retrieval of embeddings
"""

import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional, Any
import logging
import os

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages ChromaDB vector store for RAG system
    """
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        """
        Initialize ChromaDB client and collection
        
        Args:
            persist_directory: Directory to persist ChromaDB data
        """
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client with persistence"""
        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)
            
            # Initialize ChromaDB client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="stock_knowledge",
                metadata={
                    "description": "Stock market knowledge base for RAG",
                    "hnsw:space": "cosine"  # Use cosine similarity
                }
            )
            
            logger.info(f"ChromaDB initialized successfully. Collection size: {self.collection.count()}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        embeddings: Optional[List[List[float]]] = None
    ) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            embeddings: Optional pre-computed embeddings
            
        Returns:
            bool: Success status
        """
        try:
            if embeddings:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
            else:
                # ChromaDB will auto-generate embeddings
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            return False
    
    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store for similar documents
        
        Args:
            query_texts: List of query strings
            n_results: Number of results to return
            where: Metadata filter
            where_document: Document content filter
            
        Returns:
            Dict containing ids, documents, metadatas, and distances
        """
        try:
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            
            logger.info(f"Query returned {len(results['ids'][0])} results")
            return results
            
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve documents by their IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Dict containing documents and metadatas
        """
        try:
            results = self.collection.get(ids=ids)
            return results
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return {"ids": [], "documents": [], "metadatas": []}
    
    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: Success status
        """
        try:
            self.collection.delete(ids=ids)
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False
    
    def count(self) -> int:
        """Get total number of documents in collection"""
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    def reset(self) -> bool:
        """
        Reset the collection (delete all documents)
        WARNING: This is destructive!
        """
        try:
            self.client.delete_collection(name="stock_knowledge")
            self.collection = self.client.create_collection(
                name="stock_knowledge",
                metadata={
                    "description": "Stock market knowledge base for RAG",
                    "hnsw:space": "cosine"
                }
            )
            logger.warning("Vector store has been reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        try:
            return {
                "name": self.collection.name,
                "count": self.collection.count(),
                "metadata": self.collection.metadata
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return {}


# Global instance
_vector_store_instance = None


def get_vector_store() -> VectorStore:
    """
    Get or create global VectorStore instance (Singleton pattern)
    """
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
    return _vector_store_instance


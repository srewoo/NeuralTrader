"""
Vector Store Management using FAISS
Handles initialization, storage, and retrieval of embeddings
"""

import logging
import os
import json
import pickle
from typing import List, Dict, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    logger.warning("FAISS not available - vector store will be disabled")
    FAISS_AVAILABLE = False


class VectorStore:
    """
    Manages FAISS vector store for RAG system
    """

    def __init__(self, persist_directory: str = "./faiss_db"):
        """
        Initialize FAISS index and document store

        Args:
            persist_directory: Directory to persist FAISS data
        """
        self.persist_directory = persist_directory
        self.index = None
        self.documents: Dict[str, Dict[str, Any]] = {}  # id -> {document, metadata}
        self.id_to_idx: Dict[str, int] = {}  # id -> faiss index position
        self.idx_to_id: Dict[int, str] = {}  # faiss index position -> id
        self.dimension = 384  # Default dimension for all-MiniLM-L6-v2
        self._embedding_generator = None
        self._initialize_store()

    def _get_embedding_generator(self):
        """Lazy load embedding generator to avoid circular imports"""
        if self._embedding_generator is None:
            from .embeddings import get_embedding_generator
            self._embedding_generator = get_embedding_generator()
            self.dimension = self._embedding_generator.get_embedding_dimension()
        return self._embedding_generator

    def _initialize_store(self):
        """Initialize FAISS index with persistence"""
        if not FAISS_AVAILABLE:
            logger.info("FAISS is disabled - vector store will not function")
            return

        try:
            # Create persist directory if it doesn't exist
            os.makedirs(self.persist_directory, exist_ok=True)

            index_path = os.path.join(self.persist_directory, "faiss.index")
            data_path = os.path.join(self.persist_directory, "documents.pkl")

            # Try to load existing index
            if os.path.exists(index_path) and os.path.exists(data_path):
                self.index = faiss.read_index(index_path)
                with open(data_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.id_to_idx = data.get('id_to_idx', {})
                    self.idx_to_id = data.get('idx_to_id', {})
                    self.dimension = data.get('dimension', 384)
                logger.info(f"FAISS index loaded. Collection size: {len(self.documents)}")
            else:
                # Create new index (IndexFlatIP for inner product / cosine similarity)
                # We'll normalize vectors before adding, so inner product = cosine similarity
                self.index = faiss.IndexFlatIP(self.dimension)
                logger.info(f"FAISS index created with dimension {self.dimension}")

        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            self.index = None

    def _save(self):
        """Persist FAISS index and documents to disk"""
        if self.index is None:
            return

        try:
            index_path = os.path.join(self.persist_directory, "faiss.index")
            data_path = os.path.join(self.persist_directory, "documents.pkl")

            faiss.write_index(self.index, index_path)
            with open(data_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'id_to_idx': self.id_to_idx,
                    'idx_to_id': self.idx_to_id,
                    'dimension': self.dimension
                }, f)
            logger.debug("FAISS index saved to disk")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")

    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors for cosine similarity"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms

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
        if self.index is None:
            logger.warning("Vector store not initialized - cannot add documents")
            return False

        try:
            # Generate embeddings if not provided
            if embeddings is None:
                embedding_gen = self._get_embedding_generator()
                embeddings = embedding_gen.generate_embeddings(documents).tolist()

            # Convert to numpy array
            vectors = np.array(embeddings, dtype=np.float32)

            # Normalize for cosine similarity
            vectors = self._normalize_vectors(vectors)

            # Add to FAISS index
            start_idx = self.index.ntotal
            self.index.add(vectors)

            # Store documents and mappings
            for i, (doc_id, doc, meta) in enumerate(zip(ids, documents, metadatas)):
                idx = start_idx + i
                self.documents[doc_id] = {
                    'document': doc,
                    'metadata': meta or {}
                }
                self.id_to_idx[doc_id] = idx
                self.idx_to_id[idx] = doc_id

            # Persist to disk
            self._save()

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
            where: Metadata filter (simple key-value matching)
            where_document: Document content filter (not implemented for FAISS)

        Returns:
            Dict containing ids, documents, metadatas, and distances
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store not initialized or empty - cannot query")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            # Generate query embedding
            embedding_gen = self._get_embedding_generator()
            query_vectors = embedding_gen.generate_embeddings(query_texts)
            query_vectors = self._normalize_vectors(query_vectors).astype(np.float32)

            # Search with more results if filtering (we'll filter after)
            search_k = n_results * 3 if where else n_results
            search_k = min(search_k, self.index.ntotal)

            # FAISS search returns (distances, indices)
            # For IndexFlatIP with normalized vectors, higher scores = more similar
            distances, indices = self.index.search(query_vectors, search_k)

            # Format results (one list per query)
            result_ids = []
            result_docs = []
            result_metas = []
            result_dists = []

            for q_idx in range(len(query_texts)):
                q_ids = []
                q_docs = []
                q_metas = []
                q_dists = []

                for i in range(len(indices[q_idx])):
                    idx = int(indices[q_idx][i])
                    if idx < 0:  # FAISS returns -1 for empty slots
                        continue

                    doc_id = self.idx_to_id.get(idx)
                    if doc_id is None:
                        continue

                    doc_data = self.documents.get(doc_id)
                    if doc_data is None:
                        continue

                    # Apply metadata filter
                    if where:
                        match = True
                        for key, value in where.items():
                            if doc_data['metadata'].get(key) != value:
                                match = False
                                break
                        if not match:
                            continue

                    # Convert similarity score to distance (higher similarity = lower distance)
                    # FAISS inner product gives similarity, convert to distance-like format
                    similarity = float(distances[q_idx][i])
                    distance = 1.0 - similarity  # Convert to distance (0 = identical)

                    q_ids.append(doc_id)
                    q_docs.append(doc_data['document'])
                    q_metas.append(doc_data['metadata'])
                    q_dists.append(distance)

                    if len(q_ids) >= n_results:
                        break

                result_ids.append(q_ids)
                result_docs.append(q_docs)
                result_metas.append(q_metas)
                result_dists.append(q_dists)

            logger.info(f"Query returned {len(result_ids[0])} results")
            return {
                "ids": result_ids,
                "documents": result_docs,
                "metadatas": result_metas,
                "distances": result_dists
            }

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
        if self.index is None:
            return {"ids": [], "documents": [], "metadatas": []}

        try:
            result_ids = []
            result_docs = []
            result_metas = []

            for doc_id in ids:
                if doc_id in self.documents:
                    doc_data = self.documents[doc_id]
                    result_ids.append(doc_id)
                    result_docs.append(doc_data['document'])
                    result_metas.append(doc_data['metadata'])

            return {
                "ids": result_ids,
                "documents": result_docs,
                "metadatas": result_metas
            }
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return {"ids": [], "documents": [], "metadatas": []}

    def delete_by_ids(self, ids: List[str]) -> bool:
        """
        Delete documents by their IDs
        Note: FAISS doesn't support true deletion, so we rebuild the index

        Args:
            ids: List of document IDs to delete

        Returns:
            bool: Success status
        """
        if self.index is None:
            return False

        try:
            # Remove from documents dict
            ids_set = set(ids)
            remaining_docs = []
            remaining_metas = []
            remaining_ids = []

            for doc_id, doc_data in self.documents.items():
                if doc_id not in ids_set:
                    remaining_ids.append(doc_id)
                    remaining_docs.append(doc_data['document'])
                    remaining_metas.append(doc_data['metadata'])

            # Clear and rebuild
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
            self.index = faiss.IndexFlatIP(self.dimension)

            if remaining_docs:
                self.add_documents(remaining_docs, remaining_metas, remaining_ids)
            else:
                self._save()

            logger.info(f"Deleted {len(ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    def count(self) -> int:
        """Get total number of documents in collection"""
        if self.index is None:
            return 0
        return len(self.documents)

    def reset(self) -> bool:
        """
        Reset the collection (delete all documents)
        WARNING: This is destructive!
        """
        if not FAISS_AVAILABLE:
            return False

        try:
            self.documents = {}
            self.id_to_idx = {}
            self.idx_to_id = {}
            self.index = faiss.IndexFlatIP(self.dimension)
            self._save()
            logger.warning("Vector store has been reset")
            return True
        except Exception as e:
            logger.error(f"Failed to reset vector store: {e}")
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection"""
        if self.index is None:
            return {"name": None, "count": 0, "metadata": {}}

        try:
            return {
                "name": "stock_knowledge",
                "count": len(self.documents),
                "metadata": {
                    "backend": "FAISS",
                    "dimension": self.dimension,
                    "index_type": "IndexFlatIP"
                }
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

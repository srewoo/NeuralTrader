"""
Embedding Generation using Sentence Transformers
Converts text to vector embeddings for semantic search
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers models
    """
    
    # Recommended models for financial/semantic search
    MODELS = {
        "default": "all-MiniLM-L6-v2",  # Fast, good for general purpose (384 dim)
        "large": "all-mpnet-base-v2",   # Better quality (768 dim)
        "multilingual": "paraphrase-multilingual-MiniLM-L12-v2"  # Supports multiple languages
    }
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the model to use (default, large, multilingual)
        """
        self.model_name = self.MODELS.get(model_name, self.MODELS["default"])
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded successfully. Embedding dimension: {self.get_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def generate_embeddings(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for text(s)
        
        Args:
            texts: Single text string or list of texts
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings
        """
        try:
            # Convert single string to list
            if isinstance(texts, str):
                texts = [texts]
            
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            List of floats representing the embedding
        """
        embedding = self.generate_embeddings(text)
        return embedding[0].tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model"""
        return self.model.get_sentence_embedding_dimension()
    
    def compute_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """
        Compute cosine similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between -1 and 1
        """
        try:
            embeddings = self.generate_embeddings([text1, text2])
            
            # Compute cosine similarity
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    def batch_compute_similarity(
        self,
        query: str,
        documents: List[str]
    ) -> List[float]:
        """
        Compute similarity between a query and multiple documents
        
        Args:
            query: Query text
            documents: List of document texts
            
        Returns:
            List of similarity scores
        """
        try:
            # Generate embeddings
            query_embedding = self.generate_embeddings(query)[0]
            doc_embeddings = self.generate_embeddings(documents)
            
            # Compute similarities
            similarities = []
            for doc_embedding in doc_embeddings:
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                similarities.append(float(similarity))
            
            return similarities
            
        except Exception as e:
            logger.error(f"Failed to compute batch similarities: {e}")
            return [0.0] * len(documents)


# Global instance
_embedding_generator_instance = None


def get_embedding_generator(model_name: str = "default") -> EmbeddingGenerator:
    """
    Get or create global EmbeddingGenerator instance (Singleton pattern)
    
    Args:
        model_name: Name of the model to use
    """
    global _embedding_generator_instance
    if _embedding_generator_instance is None:
        _embedding_generator_instance = EmbeddingGenerator(model_name)
    return _embedding_generator_instance


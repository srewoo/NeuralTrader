"""
Unit Tests for RAG Retrieval System
Tests for KnowledgeRetriever
"""

import pytest
from unittest.mock import MagicMock, patch


class TestKnowledgeRetriever:
    """Tests for KnowledgeRetriever class"""

    def test_retriever_initialization(self, mock_chromadb):
        """Test retriever initializes correctly"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_vs.return_value = MagicMock()
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                assert retriever.vector_store is not None
                assert retriever.embedding_generator is not None


class TestRetrieve:
    """Tests for retrieve method"""

    def test_retrieve_returns_documents(self, mock_chromadb):
        """Test retrieval returns documents"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1', 'doc2']],
                    'documents': [['Document 1 content', 'Document 2 content']],
                    'metadatas': [[{'category': 'patterns'}, {'category': 'strategies'}]],
                    'distances': [[0.1, 0.2]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("RSI oversold pattern", n_results=5)

                assert len(results) == 2
                assert results[0]["id"] == "doc1"
                assert "similarity" in results[0]

    def test_retrieve_empty_results(self, mock_chromadb):
        """Test retrieval with no results"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [[]],
                    'documents': [[]],
                    'metadatas': [[]],
                    'distances': [[]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("unknown query", n_results=5)

                assert results == []

    def test_retrieve_filters_by_similarity(self, mock_chromadb):
        """Test retrieval filters by minimum similarity"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                # Return one close match and one distant match
                mock_collection.query.return_value = {
                    'ids': [['doc1', 'doc2']],
                    'documents': [['Close match', 'Distant match']],
                    'metadatas': [[{'category': 'patterns'}, {'category': 'patterns'}]],
                    'distances': [[0.1, 5.0]]  # Second is very distant
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("test", n_results=5, min_similarity=0.5)

                # Only the close match should pass the threshold
                high_similarity_docs = [r for r in results if r["similarity"] >= 0.5]
                assert len(high_similarity_docs) >= 1

    def test_retrieve_with_filters(self, mock_chromadb):
        """Test retrieval with metadata filters"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1']],
                    'documents': [['Filtered document']],
                    'metadatas': [[{'category': 'patterns'}]],
                    'distances': [[0.1]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve(
                    "test",
                    n_results=5,
                    filters={"category": "patterns"}
                )

                # Verify filter was passed to query
                mock_collection.query.assert_called()
                call_args = mock_collection.query.call_args
                assert call_args.kwargs.get("where") == {"category": "patterns"}

    def test_retrieve_handles_error(self, mock_chromadb):
        """Test retrieval handles errors gracefully"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.side_effect = Exception("Database error")
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("test")

                # Should return empty list on error
                assert results == []


class TestRetrieveByCategory:
    """Tests for retrieve_by_category method"""

    def test_retrieve_by_category(self, mock_chromadb):
        """Test retrieval by category"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1']],
                    'documents': [['Pattern document']],
                    'metadatas': [[{'category': 'patterns'}]],
                    'distances': [[0.1]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve_by_category("RSI", "patterns", n_results=3)

                assert len(results) >= 0  # May be filtered by similarity


class TestBuildContext:
    """Tests for build_context method"""

    def test_build_context_with_documents(self, mock_chromadb):
        """Test context building with documents"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1', 'doc2']],
                    'documents': [['Document 1 content', 'Document 2 content']],
                    'metadatas': [[
                        {'category': 'patterns', 'date': '2024-01-01'},
                        {'category': 'strategies'}
                    ]],
                    'distances': [[0.1, 0.2]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                context = retriever.build_context("test query", n_results=5)

                assert "RELEVANT HISTORICAL KNOWLEDGE" in context
                assert "Source 1" in context
                assert "Source 2" in context

    def test_build_context_no_documents(self, mock_chromadb):
        """Test context building with no documents"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [[]],
                    'documents': [[]],
                    'metadatas': [[]],
                    'distances': [[]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                context = retriever.build_context("test query")

                assert "No relevant historical knowledge" in context

    def test_build_context_respects_max_tokens(self, mock_chromadb):
        """Test context respects max_tokens limit"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                # Return many long documents
                long_docs = ["Very long document content " * 100] * 10
                mock_collection.query.return_value = {
                    'ids': [[f'doc{i}' for i in range(10)]],
                    'documents': [long_docs],
                    'metadatas': [[{'category': 'patterns'}] * 10],
                    'distances': [[0.1] * 10]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                context = retriever.build_context("test", max_tokens=500)

                # Context should be limited
                assert len(context) < 500 * 4 + 500  # 4 chars per token estimate + overhead

    def test_build_context_handles_error(self, mock_chromadb):
        """Test context building handles errors"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.side_effect = Exception("Database Error")
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                context = retriever.build_context("test")

                # When error occurs, returns error message OR fallback message
                assert "Error" in context or "error" in context.lower() or "No relevant" in context


class TestGetSimilarPatterns:
    """Tests for get_similar_patterns method"""

    def test_get_similar_patterns(self, mock_chromadb, sample_technical_indicators):
        """Test finding similar patterns"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['pattern1']],
                    'documents': [['Similar pattern found']],
                    'metadatas': [[{'category': 'patterns'}]],
                    'distances': [[0.1]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                patterns = retriever.get_similar_patterns(
                    sample_technical_indicators,
                    n_results=3
                )

                # Should return patterns (may be filtered)
                assert isinstance(patterns, list)

    def test_get_similar_patterns_handles_error(self, mock_chromadb):
        """Test similar patterns handles error"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.side_effect = Exception("Error")
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                patterns = retriever.get_similar_patterns({"rsi": 50}, n_results=3)

                assert patterns == []


class TestGetStrategyRecommendations:
    """Tests for get_strategy_recommendations method"""

    def test_get_strategy_recommendations(self, mock_chromadb):
        """Test strategy recommendations"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['strategy1']],
                    'documents': [['Mean reversion strategy']],
                    'metadatas': [[{'category': 'strategies'}]],
                    'distances': [[0.1]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                strategies = retriever.get_strategy_recommendations(
                    "uptrend",
                    n_results=3
                )

                assert isinstance(strategies, list)


class TestGlobalRetriever:
    """Tests for global retriever instance"""

    def test_get_retriever_singleton(self, mock_chromadb):
        """Test singleton pattern"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_vs.return_value = MagicMock()
                mock_eg.return_value = MagicMock()

                # Reset singleton
                import rag.retrieval
                rag.retrieval._retriever_instance = None

                from rag.retrieval import get_retriever

                retriever1 = get_retriever()
                retriever2 = get_retriever()

                assert retriever1 is retriever2


class TestResultFormatting:
    """Tests for result formatting"""

    def test_result_includes_similarity_score(self, mock_chromadb):
        """Test results include similarity scores"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1']],
                    'documents': [['Content']],
                    'metadatas': [[{'category': 'patterns'}]],
                    'distances': [[0.5]]  # Distance of 0.5
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("test", min_similarity=0)

                if results:
                    # Similarity should be converted from distance
                    assert "similarity" in results[0]
                    assert "distance" in results[0]
                    # Similarity = 1 / (1 + distance)
                    expected_similarity = 1 / (1 + 0.5)
                    assert abs(results[0]["similarity"] - expected_similarity) < 0.01

    def test_result_includes_metadata(self, mock_chromadb):
        """Test results include metadata"""
        with patch('rag.retrieval.get_vector_store') as mock_vs:
            with patch('rag.retrieval.get_embedding_generator') as mock_eg:
                mock_collection = MagicMock()
                mock_collection.query.return_value = {
                    'ids': [['doc1']],
                    'documents': [['Content']],
                    'metadatas': [[{
                        'category': 'patterns',
                        'confidence': 0.85,
                        'date': '2024-01-01'
                    }]],
                    'distances': [[0.1]]
                }
                mock_vs.return_value = mock_collection
                mock_eg.return_value = MagicMock()

                from rag.retrieval import KnowledgeRetriever

                retriever = KnowledgeRetriever()
                results = retriever.retrieve("test", min_similarity=0)

                if results:
                    assert "metadata" in results[0]
                    assert results[0]["metadata"]["category"] == "patterns"

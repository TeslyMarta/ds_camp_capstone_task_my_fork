"""Tests for search index functionality."""

# Standard library imports
# Standart library imports
import os
import tempfile
from unittest.mock import MagicMock, patch

# Third-party imports
import numpy as np

# Local imports
from ds_capstone.search_index import FaissSearchHandler


class TestFaissSearchHandler:
    """Test cases for the FaissSearchHandler class."""

    def test_init(self):
        """Test initialization of FaissSearchHandler.

        Verifies that:
        - Handler is initialized with correct dimension
        - FAISS index is created with correct parameters
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        assert handler.dimension == dimension
        assert hasattr(handler, 'index')
        # Check that index is of correct type and dimension
        assert handler.index.d == dimension

    def test_build(self):
        """Test building the search index with embeddings.

        Verifies that:
        - Embeddings are added to the index
        - Index count is updated correctly
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Create sample embeddings
        n_vectors = 5
        embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)

        # Build the index
        handler.build(embeddings)

        # Verify that vectors were added
        assert handler.index.ntotal == n_vectors

    def test_search(self):
        """Test searching for similar vectors.

        Verifies that:
        - Search returns correct number of results
        - Distances and indices have correct shapes
        - Results are meaningful
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Create and add sample embeddings
        n_vectors = 10
        embeddings = np.random.randn(n_vectors, dimension).astype(np.float32)
        handler.build(embeddings)

        # Create query embedding
        query_embedding = np.random.randn(1, dimension).astype(np.float32)

        # Search for top 3 similar vectors
        k = 3
        distances, indices = handler.search(query_embedding, k)

        # Verify output shapes
        assert distances.shape == (1, k)
        assert indices.shape == (1, k)

        # Verify that indices are within valid range
        assert all(0 <= idx < n_vectors for idx in indices[0])

    def test_search_exact_match(self):
        """Test search with exact match scenario.

        Verifies that:
        - Exact match returns highest similarity
        - Index of exact match is returned
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Create embeddings with one specific vector
        embeddings = np.random.randn(5, dimension).astype(np.float32)
        target_vector = embeddings[2]  # Third vector
        handler.build(embeddings)

        # Search with the exact same vector
        query_embedding = target_vector.reshape(1, -1)
        distances, indices = handler.search(query_embedding, k=1)

        # Should return the exact match (index 2) with highest similarity
        assert indices[0][0] == 2
        # For inner product, exact match should have high similarity
        assert distances[0][0] > 0

    def test_save_and_load(self):
        """Test saving and loading index functionality.

        Verifies that:
        - Index can be saved to file
        - Saved index can be loaded
        - Loaded index maintains same vectors
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Create and build index
        embeddings = np.random.randn(3, dimension).astype(np.float32)
        handler.build(embeddings)
        original_ntotal = handler.index.ntotal

        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.faiss') as temp_file:
            temp_path = temp_file.name

        try:
            # Save the index
            handler.save(temp_path)

            # Verify file was created
            assert os.path.exists(temp_path)

            # Create new handler and load
            new_handler = FaissSearchHandler(dimension)
            load_success = new_handler.load(temp_path)

            # Verify load was successful
            assert load_success is True
            assert new_handler.index.ntotal == original_ntotal

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file.

        Verifies that:
        - Load returns False for nonexistent files
        - Original index is unchanged
        """
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Try to load from nonexistent file
        nonexistent_path = "/path/that/does/not/exist.faiss"
        load_success = handler.load(nonexistent_path)

        # Should return False and maintain empty index
        assert load_success is False
        assert handler.index.ntotal == 0

    @patch('ds_capstone.search_index.faiss')
    def test_save_calls_faiss_write(self, mock_faiss):
        """Test that save method calls FAISS write_index function."""
        dimension = 384
        handler = FaissSearchHandler(dimension)

        test_path = "/test/path.faiss"
        handler.save(test_path)

        mock_faiss.write_index.assert_called_once_with(handler.index, test_path)

    @patch('ds_capstone.search_index.faiss')
    @patch('ds_capstone.search_index.os.path.exists')
    def test_load_calls_faiss_read(self, mock_exists, mock_faiss):
        """Test that load method calls FAISS read_index function when file exists."""
        mock_exists.return_value = True
        mock_index = MagicMock()
        mock_faiss.read_index.return_value = mock_index

        dimension = 384
        handler = FaissSearchHandler(dimension)
        test_path = "/test/path.faiss"

        result = handler.load(test_path)

        assert result is True
        mock_faiss.read_index.assert_called_once_with(test_path)
        assert handler.index == mock_index

    def test_search_with_more_k_than_vectors(self):
        """Test search when k is larger than number of vectors in index."""
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Add only 2 vectors but search for 5
        embeddings = np.random.randn(2, dimension).astype(np.float32)
        handler.build(embeddings)

        query_embedding = np.random.randn(1, dimension).astype(np.float32)
        distances, indices = handler.search(query_embedding, k=5)

        # Should still work and return results for available vectors
        assert distances.shape == (1, 5)
        assert indices.shape == (1, 5)
        # Only first 2 indices should be valid (0, 1), rest should be -1
        valid_indices = indices[0][indices[0] >= 0]
        assert len(valid_indices) == 2

    def test_multiple_query_search(self):
        """Test search with multiple query vectors simultaneously."""
        dimension = 384
        handler = FaissSearchHandler(dimension)

        # Build index with sample data
        embeddings = np.random.randn(5, dimension).astype(np.float32)
        handler.build(embeddings)

        # Search with multiple queries
        n_queries = 3
        query_embeddings = np.random.randn(n_queries, dimension).astype(np.float32)
        distances, indices = handler.search(query_embeddings, k=2)

        # Verify output shapes for multiple queries
        assert distances.shape == (n_queries, 2)
        assert indices.shape == (n_queries, 2)

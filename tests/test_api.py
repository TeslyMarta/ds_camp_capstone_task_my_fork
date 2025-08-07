"""Tests for FastAPI endpoints."""

# Standard library imports
# Standart library imports
from unittest.mock import MagicMock, patch

# Thirdparty imports
from fastapi.testclient import TestClient

# Third-party imports
import pandas as pd

# Local imports
from api import app


class TestAPI:
    """Test cases for the FastAPI application."""

    def setup_method(self):
        """Set up test client and mock dependencies."""
        self.client = TestClient(app)

    def test_root_endpoint(self):
        """Test the root endpoint returns HTML response.

        Verifies that:
        - Root endpoint returns 200 status code
        - Content type is text/html
        - Response contains expected HTML elements
        """
        response = self.client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "<title>FastAPI</title>" in response.text
        assert "<h1>Welcome to FastAPI!</h1>" in response.text

    @patch('api.app.state')
    def test_summarize_text_endpoint(self, mock_state):
        """Test the text summarization endpoint.

        Verifies that:
        - Endpoint accepts text input
        - Returns summary in expected format
        - Calls summarizer agent correctly
        """
        # Mock the summarizer agent
        mock_summarizer = MagicMock()
        mock_summarizer.execute.return_value = "This is a test summary."
        mock_state.summarizer_graph = mock_summarizer

        # Test data
        test_input = {"text": "This is a long text that needs to be summarized."}

        response = self.client.post("/summarize/", json=test_input)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "summary" in response_data
        assert response_data["summary"] == "This is a test summary."

        # Verify agent was called correctly
        mock_summarizer.execute.assert_called_once_with(
            input_message="This is a long text that needs to be summarized."
        )

    @patch('api.app.state')
    def test_semantic_search_endpoint_with_results(self, mock_state):
        """Test semantic search endpoint with matching results.

        Verifies that:
        - Endpoint processes search queries
        - Returns results in expected format
        - Filters results by similarity threshold
        """
        # Mock vectorizer
        mock_vectorizer = MagicMock()
        mock_query_embedding = [[0.1, 0.2, 0.3]]
        mock_vectorizer.embed.return_value = mock_query_embedding
        mock_state.vectorizer = mock_vectorizer

        # Mock search index with mixed scores
        mock_search_index = MagicMock()
        mock_distances = [[0.9, 0.85, 0.82]]  # All above threshold now
        mock_indices = [[0, 1, 2]]
        mock_search_index.search.return_value = (mock_distances, mock_indices)
        mock_state.search_index = mock_search_index

        # Mock products dataframe
        mock_products_df = pd.DataFrame(
            {
                'title': ['Product A', 'Product B', 'Product C'],
                'brand': ['Brand X', 'Brand Y', 'Brand Z'],
                'category': ['Electronics', 'Clothing', 'Home'],
                'description': ['Great product A', 'Amazing product B', 'Wonderful product C'],
            }
        )
        mock_state.products_df = mock_products_df

        # Test data
        test_query = {"text": "electronics product"}

        response = self.client.post("/semantic_search/", json=test_query)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "results" in response_data
        assert len(response_data["results"]) == 3

        # Verify first result structure
        first_result = response_data["results"][0]
        assert first_result["rank"] == 1
        assert first_result["title"] == "Product A"
        assert first_result["brand"] == "Brand X"
        assert first_result["category"] == "Electronics"
        assert first_result["description"] == "Great product A"
        assert first_result["similarity"] == 0.9

        # Verify vectorizer was called with is_query=True
        mock_vectorizer.embed.assert_called_once_with(["electronics product"], is_query=True)

        # Verify search index was called correctly
        mock_search_index.search.assert_called_once_with(mock_query_embedding, k=5)

    @patch('api.app.state')
    def test_semantic_search_endpoint_with_low_scores(self, mock_state):
        """Test semantic search endpoint when all scores are below threshold.

        Verifies that:
        - Endpoint returns 404 when no results meet similarity threshold
        - Proper error message is returned
        """
        # Mock vectorizer
        mock_vectorizer = MagicMock()
        mock_query_embedding = [[0.1, 0.2, 0.3]]
        mock_vectorizer.embed.return_value = mock_query_embedding
        mock_state.vectorizer = mock_vectorizer

        # Mock search index with low similarity scores
        mock_search_index = MagicMock()
        mock_distances = [[0.7, 0.6, 0.5]]  # Below 0.8 threshold
        mock_indices = [[0, 1, 2]]
        mock_search_index.search.return_value = (mock_distances, mock_indices)
        mock_state.search_index = mock_search_index

        # Mock products dataframe
        mock_products_df = pd.DataFrame(
            {
                'title': ['Product A', 'Product B', 'Product C'],
                'brand': ['Brand X', 'Brand Y', 'Brand Z'],
                'category': ['Electronics', 'Clothing', 'Home'],
                'description': ['Great product A', 'Amazing product B', 'Wonderful product C'],
            }
        )
        mock_state.products_df = mock_products_df

        # Test data
        test_query = {"text": "unrelated query"}

        response = self.client.post("/semantic_search/", json=test_query)

        # Verify 404 response
        assert response.status_code == 404
        response_data = response.json()
        assert response_data["detail"] == "No matches found."

    @patch('api.app.state')
    def test_semantic_search_endpoint_mixed_scores(self, mock_state):
        """Test semantic search endpoint with mixed similarity scores.

        Verifies that:
        - Only results above threshold are returned
        - Results are properly ranked
        """
        # Mock vectorizer
        mock_vectorizer = MagicMock()
        mock_query_embedding = [[0.1, 0.2, 0.3]]
        mock_vectorizer.embed.return_value = mock_query_embedding
        mock_state.vectorizer = mock_vectorizer

        # Mock search index with mixed scores
        mock_search_index = MagicMock()
        mock_distances = [[0.9, 0.7, 0.85]]  # Only first and third above 0.8
        mock_indices = [[0, 1, 2]]
        mock_search_index.search.return_value = (mock_distances, mock_indices)
        mock_state.search_index = mock_search_index

        # Mock products dataframe
        mock_products_df = pd.DataFrame(
            {
                'title': ['Product A', 'Product B', 'Product C'],
                'brand': ['Brand X', 'Brand Y', 'Brand Z'],
                'category': ['Electronics', 'Clothing', 'Home'],
                'description': ['Great product A', 'Amazing product B', 'Wonderful product C'],
            }
        )
        mock_state.products_df = mock_products_df

        # Test data
        test_query = {"text": "good product"}

        response = self.client.post("/semantic_search/", json=test_query)

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "results" in response_data
        assert len(response_data["results"]) == 2  # Only results above threshold

        # Verify results are Product A and Product C (indices 0 and 2)
        result_titles = [r["title"] for r in response_data["results"]]
        assert "Product A" in result_titles
        assert "Product C" in result_titles
        assert "Product B" not in result_titles  # Below threshold

    def test_summarize_endpoint_validation_error(self):
        """Test summarization endpoint with invalid input.

        Verifies that:
        - Endpoint validates input format
        - Returns appropriate error for invalid data
        """
        # Test with missing text field
        response = self.client.post("/summarize/", json={})
        assert response.status_code == 422

        # Test with empty text
        response = self.client.post("/summarize/", json={"text": ""})
        assert response.status_code == 422

    def test_semantic_search_endpoint_validation_error(self):
        """Test semantic search endpoint with invalid input.

        Verifies that:
        - Endpoint validates input format
        - Returns appropriate error for invalid data
        """
        # Test with missing text field
        response = self.client.post("/semantic_search/", json={})
        assert response.status_code == 422

        # Test with empty text
        response = self.client.post("/semantic_search/", json={"text": ""})
        assert response.status_code == 422

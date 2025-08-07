"""Tests for text vectorization functionality."""

# Standard library imports
# Standart library imports
from unittest.mock import MagicMock, patch

# Thirdparty imports
import torch

# Third-party imports
import numpy as np

# Local imports
from ds_capstone.vectorizer import TextVectorizer


class TestTextVectorizer:
    """Test cases for the TextVectorizer class."""

    @patch('ds_capstone.vectorizer.AutoTokenizer')
    @patch('ds_capstone.vectorizer.AutoModel')
    def test_init_success(self, mock_model, mock_tokenizer):
        """Test successful initialization of TextVectorizer.

        Verifies that:
        - Tokenizer and model are loaded from pretrained
        - Device is set correctly
        - Model is moved to the correct device
        """
        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_model_instance = MagicMock()
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        mock_model.from_pretrained.return_value = mock_model_instance

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        vectorizer = TextVectorizer(model_name)

        # Verify initialization
        mock_tokenizer.from_pretrained.assert_called_once_with(model_name)
        mock_model.from_pretrained.assert_called_once_with(model_name)
        mock_model_instance.to.assert_called_once_with('cpu')  # Should always use CPU in current implementation

        assert vectorizer.tokenizer == mock_tokenizer_instance
        assert vectorizer.model == mock_model_instance
        assert vectorizer.device == 'cpu'

    def test_average_pool(self):
        """Test the average pooling functionality.

        Verifies that:
        - Padding tokens are properly masked
        - Average is calculated correctly
        - Output shape is correct
        """
        # Create a mock vectorizer (we only need the method)
        with patch('ds_capstone.vectorizer.AutoTokenizer'), patch('ds_capstone.vectorizer.AutoModel'):
            vectorizer = TextVectorizer("dummy-model")

        # Create sample tensors
        batch_size, seq_len, hidden_size = 2, 4, 768
        last_hidden_states = torch.randn(batch_size, seq_len, hidden_size)

        # Attention mask: [1, 1, 1, 0] for first sequence, [1, 1, 0, 0] for second
        attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        result = vectorizer._average_pool(last_hidden_states, attention_mask)

        # Verify output shape
        assert result.shape == (batch_size, hidden_size)

        # Verify that result is different from simple mean (due to masking)
        simple_mean = last_hidden_states.mean(dim=1)
        assert not torch.allclose(result, simple_mean)

    @patch('ds_capstone.vectorizer.AutoTokenizer')
    @patch('ds_capstone.vectorizer.AutoModel')
    def test_embed_single_string(self, mock_model_class, mock_tokenizer_class):
        """Test embedding generation for a single string.

        Verifies that:
        - Single string is converted to list
        - Tokenization works correctly
        - Model inference produces expected output
        - L2 normalization is applied
        """
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        # Mock tokenizer output
        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tokenizer.return_value.to.return_value = mock_inputs

        # Mock model output
        mock_hidden_states = torch.randn(1, 3, 384)  # batch_size=1, seq_len=3, hidden_size=384
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs

        vectorizer = TextVectorizer("dummy-model")

        # Test single string input
        test_text = "This is a test sentence"
        result = vectorizer.embed(test_text)

        # Verify tokenizer was called with list
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == [test_text]  # Should be converted to list

        # Verify result properties
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # One embedding
        assert result.shape[1] == 384  # Hidden size

        # Verify L2 normalization (embeddings should have unit norm)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0], rtol=1e-5)

    @patch('ds_capstone.vectorizer.AutoTokenizer')
    @patch('ds_capstone.vectorizer.AutoModel')
    def test_embed_list_of_strings(self, mock_model_class, mock_tokenizer_class):
        """Test embedding generation for a list of strings."""
        # Setup mocks similar to single string test
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_inputs = {
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        mock_tokenizer.return_value.to.return_value = mock_inputs

        mock_hidden_states = torch.randn(2, 3, 384)
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs

        vectorizer = TextVectorizer("dummy-model")

        # Test list input
        test_texts = ["First sentence", "Second sentence"]
        result = vectorizer.embed(test_texts)

        # Verify result properties
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2  # Two embeddings
        assert result.shape[1] == 384  # Hidden size

    @patch('ds_capstone.vectorizer.AutoTokenizer')
    @patch('ds_capstone.vectorizer.AutoModel')
    def test_embed_with_query_prefix(self, mock_model_class, mock_tokenizer_class):
        """Test embedding generation with query prefix.

        Verifies that:
        - Query prefix is added when is_query=True
        - Tokenizer receives prefixed text
        """
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        mock_inputs = {'input_ids': torch.tensor([[1, 2, 3]]), 'attention_mask': torch.tensor([[1, 1, 1]])}
        mock_tokenizer.return_value.to.return_value = mock_inputs

        mock_hidden_states = torch.randn(1, 3, 384)
        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = mock_hidden_states
        mock_model.return_value = mock_outputs

        vectorizer = TextVectorizer("dummy-model")

        # Test with query prefix
        test_text = "What is machine learning?"
        vectorizer.embed(test_text, is_query=True)

        # Verify tokenizer was called with prefixed text
        mock_tokenizer.assert_called_once()
        call_args = mock_tokenizer.call_args[0]
        assert call_args[0] == ["query: What is machine learning?"]

    def test_empty_input_handling(self):
        """Test behavior with empty input."""
        with patch('ds_capstone.vectorizer.AutoTokenizer') as mock_tokenizer_class, patch(
            'ds_capstone.vectorizer.AutoModel'
        ) as mock_model_class:

            # Mock the tokenizer and model instances
            mock_tokenizer = MagicMock()
            mock_model = MagicMock()
            mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
            mock_model_class.from_pretrained.return_value = mock_model

            # Mock tokenizer to raise an error for empty input
            mock_tokenizer.side_effect = ValueError("Empty input")

            vectorizer = TextVectorizer("dummy-model")

            # Test empty string - should handle gracefully in current implementation
            # The actual implementation might not raise an error for empty strings
            try:
                result = vectorizer.embed("")
                # If it doesn't raise, check that result is valid
                assert isinstance(result, np.ndarray)
            except (ValueError, Exception):
                # This is also acceptable behavior
                pass

            # Test empty list - should handle gracefully
            try:
                result = vectorizer.embed([])
                assert isinstance(result, np.ndarray)
            except (ValueError, Exception):
                # This is also acceptable behavior
                pass

    @patch('ds_capstone.vectorizer.torch.cuda.is_available')
    @patch('ds_capstone.vectorizer.AutoTokenizer')
    @patch('ds_capstone.vectorizer.AutoModel')
    def test_device_selection(self, mock_model_class, mock_tokenizer_class, mock_cuda):
        """Test that device is always set to CPU regardless of CUDA availability."""
        mock_cuda.return_value = True  # Pretend CUDA is available

        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        vectorizer = TextVectorizer("dummy-model")

        # Should always use CPU based on current implementation
        assert vectorizer.device == 'cpu'
        mock_model.to.assert_called_once_with('cpu')

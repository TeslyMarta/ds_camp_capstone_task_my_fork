"""Text vectorization module using transformer models for embedding generation.

This module provides functionality to convert text into numerical embeddings using
pre-trained transformer models. It supports both document and query embeddings
with appropriate pooling and normalization strategies.

Classes
-------
TextVectorizer : class
    A text vectorization class that converts text into normalized embeddings.
"""

# Standart library imports
from typing import List, Union

# Thirdparty imports
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

import numpy as np


class TextVectorizer:
    """A text vectorization class that converts text into normalized embeddings.

    This class uses pre-trained transformer models to generate text embeddings
    with average pooling and L2 normalization. It supports GPU acceleration
    when available and can handle both document and query embeddings.

    Parameters
    ----------
    model_name : str
        Name of the pre-trained transformer model to use from HuggingFace.

    Attributes
    ----------
    tokenizer : AutoTokenizer
        Tokenizer instance for the specified model.
    model : AutoModel
        Pre-trained transformer model instance.
    device : str
        Computing device ('cuda' or 'cpu') based on availability.

    Examples
    --------
    >>> vectorizer = TextVectorizer("sentence-transformers/all-MiniLM-L6-v2")
    >>> texts = ["Hello world", "This is a test"]
    >>> embeddings = vectorizer.embed(texts)
    >>> print(embeddings.shape)
    (2, 384)
    """

    def __init__(self, model_name: str) -> None:
        """Initialize the TextVectorizer with a pre-trained model.

        Parameters
        ----------
        model_name : str
            Name of the pre-trained transformer model from HuggingFace.

        Raises
        ------
        OSError
            If the specified model cannot be loaded.
        """
        # Load tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name) #TODO: load tokenizer
        self.model = AutoModel.from_pretrained(model_name) #TODO: load model

        # Set device based on CUDA availability
        self.device: str = "cpu" #TODO: set device, you can use "cuda" or "cpu" directly
        # I recommend using cpu to avoid any issues with CUDA availability

        # Move model to the appropriate device
        self.model.to(self.device)  #TODO: move model to device

    def _average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Perform average pooling on the last hidden states using attention mask.

        This method computes the average of token embeddings while ignoring
        padded tokens using the attention mask. This is a common pooling
        strategy for sentence-level representations.

        Parameters
        ----------
        last_hidden_states : Tensor
            Hidden states from the last layer of the transformer model.
            Shape: (batch_size, sequence_length, hidden_size)
        attention_mask : Tensor
            Binary mask indicating which tokens are actual tokens (1)
            vs padding tokens (0). Shape: (batch_size, sequence_length)

        Returns
        -------
        Tensor
            Pooled embeddings of shape (batch_size, hidden_size).

        Notes
        -----
        The pooling is performed by:
        1. Masking out padding tokens by setting them to 0
        2. Summing along the sequence dimension
        3. Dividing by the number of actual tokens (not padding)
        """
        # Mask out padding tokens by setting their hidden states to 0
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        last_hidden = last_hidden_states * input_mask_expanded #TODO: mask out padding tokens

        # Calculate average by summing and dividing by number of actual tokens
        sum_hidden = torch.sum(last_hidden, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_hidden / sum_mask #TODO: calculate average pooling

    def embed(self, texts: Union[List[str], str], is_query: bool = False) -> np.ndarray:
        """Convert text(s) into normalized embedding vectors.

        This method tokenizes the input texts, processes them through the
        transformer model, applies average pooling, and returns L2-normalized
        embeddings suitable for semantic similarity tasks.

        Parameters
        ----------
        texts : Union[List[str], str]
            Input text(s) to be converted to embeddings. Can be a single
            string or a list of strings.
        is_query : bool, optional
            Whether the input texts are queries. If True, prepends "query: "
            to each text for better query-document similarity (default: False).

        Returns
        -------
        np.ndarray
            Normalized embedding vectors of shape (n_texts, embedding_dim).
            Each row represents the embedding for one input text.

        Raises
        ------
        ValueError
            If texts is empty or contains only empty strings.

        Examples
        --------
        >>> vectorizer = TextVectorizer("sentence-transformers/all-MiniLM-L6-v2")
        >>> docs = ["Machine learning is fascinating", "AI will change the world"]
        >>> doc_embeddings = vectorizer.embed(docs)
        >>> query_embedding = vectorizer.embed("What is AI?", is_query=True)
        """
        # Convert single string to list for uniform processing
        if isinstance(texts, str):
            texts = [texts]

        # Prepend query prefix if this is a query embedding
        if is_query:
            texts = [f"query: {text}" for text in texts] #TODO: prepend "query: " to each text

        # Tokenize texts with padding and truncation
        inputs = self.tokenizer(
            texts,  #TODO: add inputs and parameters
            padding=True,  # Pad shorter sequences to match longest
            truncation=True,  # Truncate longer sequences
            max_length=512,  # Maximum sequence length
            return_tensors="pt",  # Return PyTorch tensors
        ).to(self.device)

        # Generate embeddings without gradient computation (inference mode)
        with torch.no_grad():  #TODO: disable gradient computation
            # Get model outputs (last hidden states and attention)
            outputs = self.model(**inputs)

            # Apply average pooling to get sentence-level embeddings
            embeddings = self._average_pool(outputs.last_hidden_state, inputs['attention_mask'])  #TODO: apply average pooling

            # L2 normalize embeddings for cosine similarity compatibility
            normalized_embeddings = F.normalize(embeddings, p=2, dim=1) #TODO: normalize embeddings

        # Return embeddings as numpy array on CPU
        return normalized_embeddings.cpu().numpy() #TODO: convert to numpy array and return

"""
Data models for the DS Camp Capstone project.

This module defines Pydantic data models used for API request/response serialization
and data validation throughout the application. These models ensure type safety and
provide clear interfaces for text processing, summarization, and semantic search functionality.

Classes
-------
TextInput : BaseModel
    Model for text input data validation.
SummaryOutput : BaseModel
    Model for text summarization output.
SearchResultItem : BaseModel
    Model for individual search result items.
SearchOutput : BaseModel
    Model for search results collection.
"""

# Standart library imports
from typing import List

# Thirdparty imports
from pydantic import BaseModel, Field


class TextInput(BaseModel):
    """
    Pydantic model for validating text input data.

    This model is used to validate and serialize text data that will be processed
    by summarization or other text analysis functions.

    Attributes
    ----------
    text : str
        The input text to be processed. Must be a non-empty string.

    Examples
    --------
    >>> text_input = TextInput(text="This is sample text to process.")
    >>> print(text_input.text)
    'This is sample text to process.'
    """

    text: str = Field(..., min_length=1, description="The input text to be processed")


class SummaryOutput(BaseModel):
    """
    Pydantic model for text summarization output.

    This model encapsulates the result of text summarization operations,
    providing a structured way to return summarized content.

    Attributes
    ----------
    summary : str
        The generated summary text.

    Examples
    --------
    >>> summary_output = SummaryOutput(summary="This is a summary of the text.")
    >>> print(summary_output.summary)
    'This is a summary of the text.'
    """

    summary: str = Field(..., description="The generated summary text")


class SearchResultItem(BaseModel):
    """
    Pydantic model representing a single search result item.

    This model structures individual search results with product information
    and relevance scoring for semantic search functionality.

    Attributes
    ----------
    rank : int
        The ranking position of this result (1-based indexing).
    title : str
        The product title or name.
    brand : str
        The brand name of the product.
    category : str
        The product category classification.
    description : str
        The detailed product description.
    similarity : float
        The cosine similarity score indicating relevance (0.0 to 1.0).

    Examples
    --------
    >>> item = SearchResultItem(
    ...     rank=1,
    ...     title="Wireless Headphones",
    ...     brand="TechBrand",
    ...     category="Electronics",
    ...     description="High-quality wireless headphones with noise cancellation.",
    ...     similarity=0.95
    ... )
    >>> print(f"Rank: {item.rank}, Similarity: {item.similarity}")
    Rank: 1, Similarity: 0.95
    """

    rank: int = Field(..., ge=1, description="The ranking position of this result")
    title: str = Field(..., description="The product title or name")
    brand: str = Field(..., description="The brand name of the product")
    category: str = Field(..., description="The product category classification")
    description: str = Field(..., description="The detailed product description")
    similarity: float = Field(..., ge=0.0, le=1.0, description="The cosine similarity score indicating relevance")


class SearchOutput(BaseModel):
    """
    Pydantic model for search results collection.

    This model contains a list of search result items, providing a structured
    way to return multiple search results from semantic search operations.

    Attributes
    ----------
    results : List[SearchResultItem]
        A list of search result items, typically ordered by relevance.

    Examples
    --------
    >>> item1 = SearchResultItem(
    ...     rank=1, title="Product 1", brand="Brand A",
    ...     category="Category 1", description="Description 1", similarity=0.95
    ... )
    >>> item2 = SearchResultItem(
    ...     rank=2, title="Product 2", brand="Brand B",
    ...     category="Category 2", description="Description 2", similarity=0.88
    ... )
    >>> search_output = SearchOutput(results=[item1, item2])
    >>> print(f"Found {len(search_output.results)} results")
    Found 2 results
    """

    results: List[SearchResultItem] = Field(
        default_factory=list, description="A list of search result items, ordered by relevance"
    )

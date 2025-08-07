"""Data loading and preprocessing utilities.

This module provides functions for loading product data from CSV files
and preprocessing it for downstream tasks like text vectorization and
semantic search.
"""

import pandas as pd


def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """Load product data from CSV file and preprocess it.

    Loads product data from a CSV file, cleans column names by removing spaces,
    sets the 'id' column as index, and creates a combined 'text' field from
    multiple product attributes for text analysis tasks.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing product data.

    Returns
    -------
    pd.DataFrame
        Preprocessed DataFrame with 'id' as index and a combined 'text' column.
        Returns empty DataFrame if file is not found.

    Notes
    -----
    The 'text' column combines title, description, category, brand, and keywords
    fields separated by periods for comprehensive text representation of each product.
    """
    try:
        products = pd.read_csv(file_path)

        products.columns = products.columns.str.replace(' ', '')
        products.set_index('id', inplace=True)

        products["text"] = products.apply(
            lambda row: f"{row['title']}. {row['description']}. {row['category']}. {row['brand']}. {row['keywords']}",
            axis=1,
        )
        return products

    except FileNotFoundError:
        return pd.DataFrame()

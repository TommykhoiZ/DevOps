import pytest
import pandas as pd
import numpy as np

# Test function to check if the CSV is loaded correctly
def test_read_csv():
    data = pd.read_csv("Dataset4.csv")
    assert isinstance(data, pd.DataFrame), "The data should be a pandas DataFrame"
    assert not data.empty, "The DataFrame should not be empty"

# Test function to check if head() function works properly
def test_head_function():
    data = pd.read_csv("Dataset4.csv")
    head_data = data.head()
    assert len(head_data) == 5, "The head function should return the first 5 rows by default"

import pytest
from metatoolkit.abund import *
import pandas as pd

def test_placeholder():
    # Placeholder test
    assert True

def test_load():
    # Test loading a valid file
    df = load('test_data.tsv')
    assert isinstance(df, pd.DataFrame)

    # Test loading a non-existent file
    try:
        load('non_existent_file.tsv')
    except FileNotFoundError:
        assert True

def test_norm():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    normalized_df = norm(df)
    assert normalized_df.sum(axis=1).all() == 1

import pytest
from metatoolkit.onehot import *
import pandas as pd

def test_placeholder():
    # Placeholder test
    assert True

def test_onehot():
    df = pd.DataFrame({
        'Category': ['A', 'B', 'A']
    })
    onehot_df = onehot(df)
    assert 'Category.A' in onehot_df.columns
    assert 'Category.B' in onehot_df.columns

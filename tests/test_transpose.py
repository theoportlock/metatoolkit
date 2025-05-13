import pytest
from metatoolkit.transpose import *
import pandas as pd

def test_placeholder():
    # Placeholder test
    assert True

def test_transpose():
    df = pd.DataFrame({
        'A': [1, 2],
        'B': [3, 4]
    })
    transposed_df = df.transpose()
    assert transposed_df.shape == (2, 2)

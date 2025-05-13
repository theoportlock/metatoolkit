import pytest
import pandas as pd
from metatoolkit.heatmap import *

def test_placeholder():
    # Placeholder test
    assert True

def test_heatmap():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    # Assuming f.heatmap() generates a heatmap without errors
    f.heatmap(df)
    assert True

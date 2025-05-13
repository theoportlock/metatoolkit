import pandas as pd
from metatoolkit.bar import load

def test_load():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    loaded_df = load('test_subject')
    assert isinstance(loaded_df, pd.DataFrame)
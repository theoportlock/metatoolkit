import pandas as pd

def test_fisher():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    result = fisher(df)
    assert isinstance(result, pd.DataFrame)
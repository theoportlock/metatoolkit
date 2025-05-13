import pandas as pd

def test_explained_variance():
    df1 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    df2 = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    result = explained_variance([df1, df2], df1)
    assert isinstance(result, pd.DataFrame)
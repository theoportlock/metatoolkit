import pandas as pd

def test_upset():
    df1 = pd.DataFrame({'A': [1, 2, 3]})
    df2 = pd.DataFrame({'A': [2, 3, 4]})
    indices = [df1.index, df2.index]
    assert len(indices) == 2
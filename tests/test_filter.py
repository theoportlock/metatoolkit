import pandas as pd

def test_filter():
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    })
    filtered_df = filter(df, A=2)
    assert len(filtered_df) == 1
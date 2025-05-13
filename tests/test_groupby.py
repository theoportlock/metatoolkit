import pytest
from metatoolkit.groupby import *
import pandas as pd

def test_placeholder():
    # Placeholder test
    assert True

def test_group():
    df = pd.DataFrame({
        'A': ['foo', 'bar', 'foo', 'bar'],
        'B': ['one', 'one', 'two', 'two'],
        'C': [1, 2, 3, 4]
    })
    grouped = group(df, group_by=['A'], func='sum')
    assert grouped.loc['foo', 'C'] == 4
    assert grouped.loc['bar', 'C'] == 6

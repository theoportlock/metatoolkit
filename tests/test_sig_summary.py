import pytest
import pandas as pd
from metatoolkit.sig_summary import *

def test_placeholder():
    # Placeholder test
    assert True

def test_change_summary():
    df = pd.DataFrame({
        'coef': [1.5, -2.0, 0.5],
        'qval': [0.01, 0.05, 0.2]
    })
    summary = change_summary(df, change='coef', sig='qval', pval=0.05)
    assert len(summary) == 3

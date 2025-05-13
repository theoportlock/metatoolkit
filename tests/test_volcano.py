import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from metatoolkit.volcano import volcano

def test_volcano():
    df = pd.DataFrame({
        'Log2FC': [1.5, -2.0, 0.5],
        'MWW_pval': [0.01, 0.05, 0.2]
    })
    ax = volcano(df, change='Log2FC', sig='MWW_pval', fc=1, pval=0.05, annot=False)
    assert isinstance(ax, plt.Axes)
import pytest
from metatoolkit.shap_residuals import *
import numpy as np

def test_placeholder():
    # Placeholder test
    assert True

def test_hypercube():
    cube = Hypercube(3)
    assert cube.n_vertices == 3
    assert isinstance(cube.V, list)

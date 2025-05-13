import pytest
from metatoolkit.shortest_path import *
import networkx as nx

def test_placeholder():
    # Placeholder test
    assert True

def test_shortest_path():
    G = nx.Graph()
    G.add_edge('A', 'B', weight=1)
    G.add_edge('B', 'C', weight=2)
    path = nx.shortest_path(G, source='A', target='C', weight='weight')
    assert path == ['A', 'B', 'C']

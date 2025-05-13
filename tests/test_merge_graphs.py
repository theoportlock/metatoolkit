import networkx as nx

def test_merge_graphs():
    G1 = nx.Graph()
    G1.add_node(1, x=0)
    G2 = nx.Graph()
    G2.add_node(2, x=10)
    merged_graph = nx.compose(G1, G2)
    assert len(merged_graph.nodes) == 2
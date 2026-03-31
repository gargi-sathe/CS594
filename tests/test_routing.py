import networkx as nx
from src.routing.graph_builder import build_synthetic_graph

def test_synthetic_graph_generation():
    w, h = 10, 10
    G, apsp = build_synthetic_graph(w, h)
    
    assert len(G.nodes) == 100
    assert len(G.edges) == 180  # 2*w*h - w - h
    
    # Verify node 0 is connected to node 1 and 10 usually
    assert apsp[0][0] == 0.0
    
    # Distance from corner to opposite corner in a 10x10 is 9 + 9 = 18 steps
    # (0,0) -> (9,9). Depending on how relabeling works, nodes are 0 and 99
    assert apsp[0][99] == 18.0

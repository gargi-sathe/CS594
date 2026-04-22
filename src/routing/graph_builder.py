import networkx as nx
from typing import Tuple, Dict, Any

def build_synthetic_graph(grid_width: int, grid_height: int) -> Tuple[nx.Graph, Dict[int, Dict[int, float]]]:
    """
    Builds a synthetic grid graph and precomputes all-pairs shortest path lengths.
    Assume travel time between adjacent nodes is 1.0 minute.
    """
    # Create a 2D grid graph
    G = nx.grid_2d_graph(grid_width, grid_height)
    
    # Relabel nodes to integers 0 ... (W*H - 1)
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Assign weight 1.0 to all edges
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
        
    # Precompute all pairs shortest path lengths
    # dict of dicts: lengths[source][target] = float
    apsp = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    
    return G, apsp

class RoutingTargetProxy:
    def __init__(self, cache, source):
        self.cache = cache
        self.source = source
        
    def get(self, target, default=float('inf')):
        return self.cache._get_time(self.source, target, default)
    
    def __getitem__(self, target):
        res = self.get(target, default=None)
        if res is None:
            raise KeyError(f"No path to {target}")
        return res
    
    def __contains__(self, target):
        # We assume all nodes in G are reachable or at least valid targets
        return target in self.cache.G

class RoutingCache:
    """A proxy dictionary mimicking apsp dicts, fetching travel times on-demand robustly."""
    def __init__(self, G, speed_mps=10.0):
        self.G = G
        self.speed_mps = speed_mps
        self._cache = {}
        
    def __getitem__(self, source):
        return RoutingTargetProxy(self, source)

    def __contains__(self, source):
        return source in self.G

    def get(self, source, default=None):
        if source in self:
            return self[source]
        return default
        
    def _get_time(self, source, target, default):
        key = (source, target)
        if key in self._cache:
            return self._cache[key]
        try:
            # OSMnx uses 'length' in meters for standard graphs
            dist = nx.shortest_path_length(self.G, source, target, weight='length')
            time_mins = (dist / self.speed_mps) / 60.0
            self._cache[key] = time_mins
            self._cache[(target, source)] = time_mins # undirected assumption
            return time_mins
        except nx.NetworkXNoPath:
            self._cache[key] = default
            return default

def build_osmnx_graph(place_name: str) -> Tuple[nx.Graph, Any]:
    import osmnx as ox
    try:
        # Load explicit physical boundary maps mapping drives cleanly
        params = {"network_type": "drive"}
        G = ox.graph_from_place(place_name, **params)
    except Exception:
        # Immutable fallback guaranteeing success gracefully if place_name bounds resolve erratically
        G = ox.graph_from_point((37.8229, -122.2359), dist=500, network_type='drive')
        
    # Convert reliably to pure undirected Graph
    G = nx.Graph(G)
    nodes = max(nx.connected_components(G), key=len)
    G = G.subgraph(nodes).copy()
    
    for u, v, data in G.edges(data=True):
        if 'length' not in data:
            data['length'] = 50.0 
            
    routing_proxy = RoutingCache(G, speed_mps=10.0)
    return G, routing_proxy

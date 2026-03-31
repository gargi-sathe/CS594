import numpy as np
from sklearn.cluster import KMeans
from typing import List, Any
import math
from src.config.schema import RunConfig

def demand_clustering_baseline(candidates: List[Any], k: int, G: Any, apsp: Any, config: RunConfig) -> List[Any]:
    """
    Demand-weighted clustering baseline selecting K sites using KMeans and snapping to available candidate space via minimum Euclidean limits.
    """
    if k >= len(candidates):
        sorted_candidates = sorted(candidates, key=lambda c: str(c))
        return sorted_candidates[:k]
        
    points = []
    weights = []
    
    for n in G.nodes():
        data = G.nodes[n]
        if config.map.type == "grid":
            w_grid = config.map.grid_size[0]
            if isinstance(n, int): 
                x, y = n % w_grid, n // w_grid
            elif isinstance(n, tuple): 
                x, y = n[0], n[1]
            else: 
                x, y = 0, 0
        else:
            x, y = data.get("x", 0.0), data.get("y", 0.0)
            
        points.append((x, y))
        weights.append(data.get("demand_weight", 1.0))
        
    X = np.array(points)
    W = np.array(weights)
    
    kmeans = KMeans(n_clusters=k, random_state=config.seed, n_init=10)
    kmeans.fit(X, sample_weight=W)
    
    centroids = kmeans.cluster_centers_
    
    selected = set()
    result = []
    
    cand_coords = {}
    for c in candidates:
        data = G.nodes.get(c, {})
        if config.map.type == "grid":
            w_grid = config.map.grid_size[0]
            if isinstance(c, int): x, y = c % w_grid, c // w_grid
            elif isinstance(c, tuple): x, y = c[0], c[1]
            else: x, y = 0, 0
        else:
            x, y = data.get("x", 0.0), data.get("y", 0.0)
        cand_coords[c] = (x, y)
        
    for cx, cy in centroids:
        best_cand = None
        best_dist = float('inf')
        
        # Tie breaker via shortest ID
        sorted_candidates = sorted(candidates, key=lambda c: str(c))
        
        for c in sorted_candidates:
            if c in selected:
                continue
            x, y = cand_coords[c]
            dist = math.hypot(x - cx, y - cy)
            
            if dist < best_dist:
                best_dist = dist
                best_cand = c
                
        if best_cand is not None:
            selected.add(best_cand)
            result.append(best_cand)
            
    return result

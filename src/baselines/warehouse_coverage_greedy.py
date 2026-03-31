from typing import List, Any
from src.config.schema import RunConfig

def coverage_greedy_baseline(candidates: List[Any], k: int, G: Any, apsp: Any, config: RunConfig) -> List[Any]:
    """
    Coverage-based baseline choosing incremental maximum candidate covering demand under target budget thresholds.
    """
    if k >= len(candidates):
        sorted_candidates = sorted(candidates, key=lambda c: str(c))
        return sorted_candidates[:k]
        
    travel_budget = max(config.parameters.delivery_target_mins - config.parameters.pick_pack_time_mins, 0.0)
    
    demand_field = {}
    for node in G.nodes():
        demand_field[node] = G.nodes[node].get("demand_weight", 1.0)
        
    selected = []
    covered_nodes = set()
    
    for _ in range(k):
        best_cand = None
        best_gain = -1.0
        
        sorted_candidates = sorted(candidates, key=lambda c: str(c))
        
        for c in sorted_candidates:
            if c in selected:
                continue
                
            incremental_gain = 0.0
            
            for target_node, weight in demand_field.items():
                if target_node in covered_nodes:
                    continue  # Only score strictly NEW covered demand
                
                dist = apsp.get(target_node, {}).get(c, float('inf'))
                if dist == float('inf'):
                    dist = apsp.get(c, {}).get(target_node, float('inf'))
                
                if dist <= travel_budget:
                    incremental_gain += weight
                    
            if incremental_gain > best_gain:
                best_gain = incremental_gain
                best_cand = c
                
        if best_cand is not None:
            selected.append(best_cand)
            # Commit the newly covered nodes to the shared state
            for target_node in demand_field.keys():
                if target_node not in covered_nodes:
                    dist = apsp.get(target_node, {}).get(best_cand, float('inf'))
                    if dist == float('inf'):
                        dist = apsp.get(best_cand, {}).get(target_node, float('inf'))
                    if dist <= travel_budget:
                        covered_nodes.add(target_node)
            
    return selected

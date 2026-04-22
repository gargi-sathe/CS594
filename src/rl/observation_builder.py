import numpy as np
from typing import Dict, Any, List

# Documented normalization references
K_REF = 50.0
D_REF = 100.0
T_REF_MINS = 60.0
HORIZON_REF_MINS = 1440.0
PICK_PACK_REF_MINS = 30.0
LAMBDA_REF_PER_HOUR = 600.0  # Assumes order_lambda is mapped to hourly volume locally if needed

def build_global_features(config: Any, step_index: int, k_total: int) -> np.ndarray:
    """Returns exactly global_features: float32[13]"""
    feats = np.zeros(13, dtype=np.float32)
    feats[0] = k_total / K_REF
    feats[1] = step_index / max(1, k_total)
    feats[2] = (k_total - step_index) / max(1, k_total)
    feats[3] = config.entities.num_drivers / D_REF
    feats[4] = config.parameters.delivery_target_mins / T_REF_MINS
    feats[5] = config.run_horizon_mins / HORIZON_REF_MINS
    feats[6] = config.parameters.pick_pack_time_mins / PICK_PACK_REF_MINS
    
    # order_lambda treated as arrivals per hour
    # if it's stored as per minute in the raw config, we rely on the simulator bounds, 
    # but the user said "Our config schema treats order_lambda as arrivals per hour", 
    # so we divide config.parameters.order_lambda by LAMBDA_REF_PER_HOUR exactly.
    feats[7] = config.parameters.order_lambda / LAMBDA_REF_PER_HOUR
    
    feats[8] = 0.0 if config.map.type == "grid" else 1.0
    
    if getattr(config, "stress_tests", None) is not None:
        if getattr(config.stress_tests, "demand_spike", None): feats[9] = 1.0
        if getattr(config.stress_tests, "driver_shortage", None): feats[10] = 1.0
        if getattr(config.stress_tests, "warehouse_closure", None): feats[11] = 1.0
        if getattr(config.stress_tests, "time_shift", None): feats[12] = 1.0
        
    return np.clip(feats, 0.0, 1.0)

def build_candidate_features(
    candidates: List[Any], 
    G: Any, 
    apsp: Dict[Any, Dict[Any, float]], 
    selected_indices: List[int],
    config: Any
) -> np.ndarray:
    """Returns candidate_features: float32[N_candidates, 8]"""
    N = len(candidates)
    feats = np.zeros((N, 8), dtype=np.float32)
    
    # Grid sizes tracking
    if config.map.type == "grid":
        max_width = max(config.map.grid_size[0] - 1, 1)
        max_height = max(config.map.grid_size[1] - 1, 1)
    else:
        # Generic map fallback
        xs = [d.get("x", 0.0) for n, d in G.nodes(data=True)]
        ys = [d.get("y", 0.0) for n, d in G.nodes(data=True)]
        min_x, max_x_val = min(xs) if xs else 0.0, max(xs) if xs else 1.0
        min_y, max_y_val = min(ys) if ys else 0.0, max(ys) if ys else 1.0
        max_width = max(max_x_val - min_x, 1)
        max_height = max(max_y_val - min_y, 1)

    # Graph maximum shortest-path tracking and max degree
    max_dist = 1.0
    if isinstance(apsp, dict) and apsp:
        max_dist = max((dist for row in apsp.values() for dist in row.values()), default=1.0)
    else:
        # For lazy caches (real maps), we use the delivery target as a normalization reference
        # to ensure features remain roughly scaled even if we don't know the diameter of the graph.
        max_dist = max(config.parameters.delivery_target_mins * 2.0, 1.0)
    max_dist = max(max_dist, 1.0)
    
    max_degree = max((d for n, d in G.degree()), default=1)
    max_degree = max(max_degree, 1)

    # Demand Field Initialization
    demand_field = {}
    total_demand = 0.0
    for node in G.nodes():
        w = G.nodes[node].get("demand_weight", 1.0)
        demand_field[node] = w
        total_demand += w
    max_demand = max(demand_field.values()) if demand_field else 1.0

    travel_budget = max(config.parameters.delivery_target_mins - config.parameters.pick_pack_time_mins, 0.0)
    
    for i, node in enumerate(candidates):
        # 0 & 1: x_or_lon_norm, y_or_lat_norm
        node_data = G.nodes.get(node, {})
        if config.map.type == "grid":
            w_grid = config.map.grid_size[0]
            if isinstance(node, int):
                x = node % w_grid
                y = node // w_grid
            elif isinstance(node, tuple):
                x, y = node[0], node[1]
            else:
                x, y = 0, 0
            feats[i, 0] = x / max_width
            feats[i, 1] = y / max_height
        else:
            x = node_data.get("x", 0.0)
            y = node_data.get("y", 0.0)
            feats[i, 0] = (x - min_x) / max_width
            feats[i, 1] = (y - min_y) / max_height
            
        # 2: local_demand_mass_norm
        feats[i, 2] = demand_field.get(node, 1.0) / max(1e-6, max_demand)
        
        # 3 & 4: avg_travel_time_to_demand_norm & reachable_demand_fraction_within_T
        weighted_sum = 0.0
        sum_of_weights = 0.0
        reachable_demand = 0.0
        if apsp and node in apsp:
            # If lazy (real maps), we only estimate based on candidates to stay ultra-fast.
            # If dense (grid), we use the full dictionary.
            if isinstance(apsp, dict):
                for target_node, dist in apsp[node].items():
                    w = demand_field.get(target_node, 1.0)
                    weighted_sum += dist * w
                    sum_of_weights += w
                    if dist <= travel_budget:
                        reachable_demand += w
            else:
                # Lazy cache path: estimate using candidates as a proxy for the demand field
                for target_node in candidates:
                    dist = apsp[node][target_node]
                    w = demand_field.get(target_node, 1.0)
                    weighted_sum += dist * w
                    sum_of_weights += w
                    if dist <= travel_budget:
                        reachable_demand += w
                    
        avg_travel = (weighted_sum / sum_of_weights) if sum_of_weights > 0 else 0.0
        feats[i, 3] = avg_travel / max_dist
        feats[i, 4] = reachable_demand / max(1e-6, total_demand)
        
        # 5: distance_to_nearest_selected_norm
        if not selected_indices:
            feats[i, 5] = 1.0 # Sentinel defaults when no sites selected
        else:
            min_dist = float('inf')
            for sel_idx in selected_indices:
                sel_node = candidates[sel_idx]
                dist = apsp.get(node, {}).get(sel_node, float('inf'))
                min_dist = min(min_dist, dist)
            feats[i, 5] = min_dist / max_dist if min_dist != float('inf') else 1.0
            
        # 6: node_degree_or_connectivity_norm
        feats[i, 6] = G.degree(node) / max_degree if G.has_node(node) else 0.0
        
        # 7: selected_flag
        feats[i, 7] = 1.0 if i in selected_indices else 0.0
        
    return np.clip(feats, 0.0, 1.0)

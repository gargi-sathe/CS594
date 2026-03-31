import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple, List
import copy

from src.config.schema import parse_config, RunConfig
from src.core.engine import run_simulation
from src.metrics.aggregator import calculate_metrics
from src.rl.observation_builder import build_global_features, build_candidate_features

class WarehousePlacementEnv(gym.Env):
    """
    Gymnasium environment for Warehouse Placement.
    Step: Agent selects one candidate site.
    Terminal Step: K selected, simulator rolls out.
    """
    def __init__(self, scenario_sampler, max_candidates: int = 100):
        super().__init__()
        self.scenario_sampler = scenario_sampler
        self.max_candidates = max_candidates
        
        self.action_space = gym.spaces.Discrete(self.max_candidates)
        
        self.observation_space = gym.spaces.Dict({
            "global_features": gym.spaces.Box(low=0.0, high=1.0, shape=(13,), dtype=np.float32),
            "candidate_features": gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_candidates, 8), dtype=np.float32),
            "selected_mask": gym.spaces.MultiBinary(self.max_candidates),
            "action_mask": gym.spaces.MultiBinary(self.max_candidates),
            "step_index": gym.spaces.Box(low=0, high=1000, shape=(1,), dtype=np.int32)
        })
        
        self.current_config_raw = None
        self.current_config_parsed = None
        self.candidates = []
        self.G = None
        self.apsp = None
        self.K = 0
        
        self.selected_indices = []
        self.obs = None

    def reset(self, seed=None, options=None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        super().reset(seed=seed)
        
        self.current_config_raw = self.scenario_sampler.sample()
        if seed is not None:
            self.current_config_raw.setdefault("run_config", {})["seed"] = seed
            
        self.current_config_parsed = parse_config(self.current_config_raw)
        
        # Build map to get candidates
        config = self.current_config_parsed
        if config.map.type == "grid":
            from src.routing.graph_builder import build_synthetic_graph
            self.G, self.apsp = build_synthetic_graph(config.map.grid_size[0], config.map.grid_size[1])
        elif config.map.type == "osmnx":
            from src.routing.graph_builder import build_osmnx_graph
            self.G, self.apsp = build_osmnx_graph(config.map.osmnx_place)
        
        # All nodes are candidates for now
        self.candidates = list(self.G.nodes())
        
        # Truncate or pad candidates to max_candidates
        if len(self.candidates) > self.max_candidates:
            self.candidates = self.candidates[:self.max_candidates]
            
        self.K = config.entities.num_warehouses
        self.selected_indices = []
        self.step_index = 0
        
        # Build obs
        global_feats = build_global_features(config, self.step_index, self.K)
        cand_feats = build_candidate_features(
            candidates=self.candidates,
            G=self.G,
            apsp=self.apsp,
            selected_indices=self.selected_indices,
            config=config
        )
        
        # Pad cand_feats if we have fewer candidates than max_candidates
        actual_n = len(self.candidates)
        padded_cand_feats = np.zeros((self.max_candidates, 8), dtype=np.float32)
        padded_cand_feats[:actual_n] = cand_feats
        
        sel_mask = np.zeros(self.max_candidates, dtype=np.int8)
        act_mask = np.zeros(self.max_candidates, dtype=np.int8)
        act_mask[:actual_n] = 1 # Valid actions are available candidates
        
        self.obs = {
            "global_features": global_feats,
            "candidate_features": padded_cand_feats,
            "selected_mask": sel_mask,
            "action_mask": act_mask,
            "step_index": np.array([self.step_index], dtype=np.int32)
        }
        
        info = {
            "candidate_node_ids": self.candidates,
            "scenario_id": "default",
            "scenario_type": "default",
            "demand_summary": {},
            "config_snapshot": self.current_config_raw
        }
        return self.obs, info

    def action_masks(self):
        """For MaskablePPO sb3-contrib compatibility."""
        return self.obs["action_mask"]

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        # Validate action
        if self.obs["action_mask"][action] == 0:
            raise ValueError(f"Invalid action {action}. Action mask is 0.")
            
        self.selected_indices.append(action)
        self.step_index += 1
        
        # Re-build features with updated state
        global_feats = build_global_features(self.current_config_parsed, self.step_index, self.K)
        cand_feats = build_candidate_features(
            candidates=self.candidates,
            G=self.G,
            apsp=self.apsp,
            selected_indices=self.selected_indices,
            config=self.current_config_parsed
        )
        
        actual_n = len(self.candidates)
        padded_cand_feats = np.zeros((self.max_candidates, 8), dtype=np.float32)
        padded_cand_feats[:actual_n] = cand_feats
        
        self.obs["global_features"] = global_feats
        self.obs["candidate_features"] = padded_cand_feats
        self.obs["selected_mask"][action] = 1
        self.obs["action_mask"][action] = 0
        self.obs["step_index"][0] = self.step_index
        
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        if len(self.selected_indices) == self.K:
            terminated = True
            selected_nodes = [self.candidates[idx] for idx in self.selected_indices]
            
            # Inject warehouse locations
            config_rollout = copy.deepcopy(self.current_config_parsed)
            if config_rollout.entities.warehouse_locations is None:
                config_rollout.entities.warehouse_locations = []
            config_rollout.entities.warehouse_locations = selected_nodes
            
            # Run simulator
            final_state = run_simulation(config_rollout)
            metrics = calculate_metrics(final_state, config_rollout)
            
            raw_total_orders_generated = metrics.get("total_orders_generated", 0)
            denom_total_orders = max(raw_total_orders_generated, 1)
            
            missed_unserved = metrics.get("orders_missed_or_unserved", 0)
            missed_rate = missed_unserved / denom_total_orders
            
            simple_cost = metrics.get("simple_cost_estimate", 0.0)
            cost_per_order = simple_cost / denom_total_orders
            
            rw_config = config_rollout.reward_weights
            cost_norm_ref = getattr(rw_config, "cost_norm_ref", 100.0)
            normalized_cost_per_order = min(cost_per_order / cost_norm_ref, 5.0)
            
            T = max(config_rollout.parameters.delivery_target_mins, 1e-6)
            p95 = metrics.get("p95_delivery_time", 0.0)
            tail_penalty = max(0.0, (p95 - T) / T)
            
            on_time_delivery_rate = metrics.get("on_time_delivery_rate", 0.0)
            delivered_success_rate = metrics.get("delivered_success_rate", 0.0)
            
            w_on_time = getattr(rw_config, "weight_on_time", 2.0)
            w_success = getattr(rw_config, "weight_delivered_success", 0.5)
            w_missed = getattr(rw_config, "weight_missed_rate", 1.0)
            w_tail = getattr(rw_config, "weight_tail_penalty", 0.5)
            w_cost = getattr(rw_config, "weight_cost", 0.1)
            
            reward = (
                (w_on_time * on_time_delivery_rate) +
                (w_success * delivered_success_rate) -
                (w_missed * missed_rate) -
                (w_tail * tail_penalty) -
                (w_cost * normalized_cost_per_order)
            )
            
            info = {
                "selected_warehouse_node_ids": selected_nodes,
                "total_orders_generated": raw_total_orders_generated,
                "delivered_count": metrics.get("delivered_count", 0),
                "delivered_success_rate": delivered_success_rate,
                "on_time_delivery_rate": on_time_delivery_rate,
                "p50_delivery_time": metrics.get("p50_delivery_time", 0.0),
                "p90_delivery_time": metrics.get("p90_delivery_time", 0.0),
                "p95_delivery_time": p95,
                "average_delivery_time": metrics.get("average_delivery_time", 0.0),
                "average_queue_time": metrics.get("average_queue_time", 0.0),
                "average_pick_pack_time": metrics.get("average_pick_pack_time", 0.0),
                "average_travel_to_customer_time": metrics.get("average_travel_to_customer_time", 0.0),
                "driver_utilization": metrics.get("driver_utilization", 0.0),
                "orders_missed_or_unserved": missed_unserved,
                "simple_cost_estimate": simple_cost,
                "cost_per_order": cost_per_order,
                "missed_rate": missed_rate,
                "scenario_type": "default",
                "scenario_config": self.current_config_raw
            }
            
        return self.obs, reward, terminated, truncated, info

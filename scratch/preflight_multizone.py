import yaml
import numpy as np
from src.rl.warehouse_env import WarehousePlacementEnv
from src.rl.multi_zone_sampler import MultiZoneScenarioSampler
from src.config.schema import parse_config

# Load Multi-Zone Config
with open("configs/chicago_multizone_train.yaml", "r") as f:
    raw_config = yaml.safe_load(f)

mz_list = raw_config["run_config"]["training"]["multi_zone_list"]
sampler = MultiZoneScenarioSampler(raw_config, mz_list)
env = WarehousePlacementEnv(sampler, max_candidates=25)

print("\n--- TEST 1: Multi-zone reset preflight ---")
for i in range(10):
    obs, info = env.reset()
    zone = info["config_snapshot"]["run_config"]["map"]["osmnx_place"]
    K = info["config_snapshot"]["run_config"]["entities"]["num_warehouses"]
    D = info["config_snapshot"]["run_config"]["entities"]["num_drivers"]
    
    print(f"Episode {i}: Zone={zone.split(',')[0]}, K={K}, D={D}")
    print(f"  candidate_features shape: {obs['candidate_features'].shape}")
    print(f"  action_mask length: {len(obs['action_mask'])}")
    print(f"  global_features (K={obs['global_features'][0]:.2f}, D={obs['global_features'][3]:.2f})")
    
    # Assertions
    assert obs["candidate_features"].shape == (25, 8)
    assert len(obs["action_mask"]) == 25
    if "Near West Side" in zone:
        assert K == 5
    else:
        assert K == 3

print("\n--- TEST 2: Variable-K step preflight ---")
for target_k in [3, 5]:
    print(f"\nTesting K={target_k}...")
    # Reset until we get the target K
    found = False
    for _ in range(20):
        obs, info = env.reset()
        if info["config_snapshot"]["run_config"]["entities"]["num_warehouses"] == target_k:
            found = True
            break
    if not found:
        print(f"Could not find zone with K={target_k}")
        continue
        
    terminated = False
    steps = 0
    selected_nodes = []
    while not terminated:
        mask = env.action_masks()
        valid_actions = np.where(mask == 1)[0]
        action = valid_actions[0]
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        print(f"  Step {steps}: Action={action}, Terminated={terminated}")
    
    print(f"Finished. Total steps: {steps} (Expected {target_k})")
    assert steps == target_k
    assert len(info["selected_warehouse_node_ids"]) == target_k
    assert len(set(info["selected_warehouse_node_ids"])) == target_k

print("\n--- All Multi-Zone Preflight Checks Passed! ---")

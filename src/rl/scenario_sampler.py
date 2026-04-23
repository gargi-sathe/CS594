import copy
import random
from typing import Dict, Any

class ScenarioSampler:
    def __init__(self, base_config_raw: Dict[str, Any]):
        self.base_config_raw = base_config_raw
        self.training_cfg = base_config_raw.get("run_config", {}).get("training", {})
        self.robust_cfg = self.training_cfg.get("robust_training", {})

    def sample(self) -> Dict[str, Any]:
        """Returns a modified dict config with optional domain randomization."""
        cfg = copy.deepcopy(self.base_config_raw)
        
        if not self.robust_cfg:
            return cfg
            
        # Dice roll for stress
        prob_stress = self.robust_cfg.get("prob_stress", 0.5)
        if random.random() > prob_stress:
            return cfg # Normal episode
            
        # Select stress type
        stress_probs = self.robust_cfg.get("stress_types", {
            "demand_spike": 0.25,
            "driver_shortage": 0.25,
            "warehouse_closure": 0.25,
            "time_shift": 0.25
        })
        
        stype = random.choices(list(stress_probs.keys()), weights=list(stress_probs.values()))[0]
        
        cfg["run_config"].setdefault("stress_tests", {})
        
        if stype == "demand_spike":
            cfg["run_config"]["stress_tests"]["demand_spike"] = {
                "multiplier": random.uniform(2.0, 4.0),
                "quadrant": random.randint(0, 3)
            }
        elif stype == "driver_shortage":
            cfg["run_config"]["stress_tests"]["driver_shortage"] = {
                "reduction_fraction": random.uniform(0.2, 0.6)
            }
        elif stype == "warehouse_closure":
            cfg["run_config"]["stress_tests"]["warehouse_closure"] = {
                "close_at_min": random.uniform(20.0, 60.0),
                "target_warehouse_id": random.choice(["W1", "W2", "W3"])
            }
        elif stype == "time_shift":
            # Profile: morning peak or lunch peak
            if random.random() > 0.5:
                # Lunch peak
                profile = [[0, 40, 15.0], [40, 80, 75.0], [80, 120, 20.0]]
            else:
                # Evening taper
                profile = [[0, 60, 60.0], [60, 120, 15.0]]
            cfg["run_config"]["stress_tests"]["time_shift"] = {"lambda_profile": profile}
            
        return cfg

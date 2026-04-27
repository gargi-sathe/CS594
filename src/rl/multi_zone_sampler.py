import copy
import random
from typing import List, Dict, Any

class MultiZoneScenarioSampler:
    """
    Samples from a set of Chicago zones with zone-specific capacities (K, D).
    """
    def __init__(self, base_config: Dict[str, Any], zones_config: List[Dict[str, Any]]):
        self.base_config = copy.deepcopy(base_config)
        self.zones_config = zones_config
        # We'll use a local random instance to avoid interfering with global seed if needed
        self.rng = random.Random()

    def sample(self) -> Dict[str, Any]:
        # Pick a random zone config
        zone_info = self.rng.choice(self.zones_config)
        
        config = copy.deepcopy(self.base_config)
        
        # Override zone-specific settings
        config["run_config"]["map"]["osmnx_place"] = zone_info["osmnx_place"]
        config["run_config"]["entities"]["num_warehouses"] = zone_info["num_warehouses"]
        config["run_config"]["entities"]["num_drivers"] = zone_info["num_drivers"]
        
        # If the zone has specific stress settings, they could be added here,
        # but for Phase 1B Tier 3, we mostly focus on topology + capacity.
        
        return config

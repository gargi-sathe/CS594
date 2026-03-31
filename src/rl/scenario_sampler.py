import copy
from typing import Dict, Any

class ScenarioSampler:
    def __init__(self, base_config_raw: Dict[str, Any]):
        self.base_config_raw = base_config_raw

    def sample(self) -> Dict[str, Any]:
        """Returns a modified dict config."""
        return copy.deepcopy(self.base_config_raw)

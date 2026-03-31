import random
from typing import List, Any
from src.config.schema import RunConfig

def random_baseline(candidates: List[Any], k: int, G: Any, apsp: Any, config: RunConfig) -> List[Any]:
    """
    Randomly selects K candidate nodes using a local RNG seeded by config.seed.
    """
    rng = random.Random(config.seed)
    # Ensure candidates are sorted so the random sampling is cleanly deterministic
    # regardless of iteration order differences in sets/dicts.
    sorted_candidates = sorted(candidates, key=lambda c: str(c))
    return rng.sample(sorted_candidates, k)

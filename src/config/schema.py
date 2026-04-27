from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class MapConfig:
    type: str
    grid_size: Optional[List[int]] = None
    osmnx_place: Optional[str] = None
    candidate_generation_mode: Optional[str] = None
    candidate_subsample_size: Optional[int] = None
    demand_distribution: Optional[str] = None

@dataclass
class EntitiesConfig:
    num_warehouses: int
    num_drivers: int
    warehouse_locations: Optional[List[Any]] = None

@dataclass
class ParametersConfig:
    delivery_target_mins: float
    pick_pack_time_mins: float
    order_lambda: float

@dataclass
class PoliciesConfig:
    staging: str
    dispatch: str

@dataclass
class CostsConfig:
    warehouse_base: float
    driver_hourly: float
    order_op_cost: float

@dataclass
class OutputsConfig:
    output_dir: str

@dataclass
class RewardWeightsConfig:
    weight_on_time: float = 2.0
    weight_delivered_success: float = 0.5
    weight_missed_rate: float = 1.0
    weight_tail_penalty: float = 0.5
    weight_cost: float = 0.1
    cost_norm_ref: float = 100.0

@dataclass
class EvalConfig:
    scenarios_per_bucket: int = 3
    buckets: Dict[str, float] = field(default_factory=lambda: {"Low": 15.0, "Medium": 30.0, "High": 60.0})
    output_dir: str = "results/eval"
    model_path: str = "logs/checkpoints/final_model.zip"

@dataclass
class TrainingConfig:
    total_timesteps: int = 5000
    checkpoint_freq: int = 1000
    log_dir: str = "logs/tb/"
    checkpoint_dir: str = "logs/checkpoints/"
    learning_rate: float = 0.0003
    n_steps: int = 256
    batch_size: int = 64
    policy_name: str = "MultiInputPolicy"
    device: str = "auto"
    tensorboard_log_name: str = "MaskablePPO"
    seed: int = 42
    multi_zone_list: Optional[List[Dict[str, Any]]] = None
    robust_training: Optional[Dict[str, Any]] = None

@dataclass
class StressTestConfig:
    warehouse_closure: Optional[Dict[str, Any]] = None
    driver_shortage: Optional[Dict[str, Any]] = None
    demand_spike: Optional[Dict[str, Any]] = None
    time_shift: Optional[Dict[str, Any]] = None

@dataclass
class RunConfig:
    seed: int
    mode: str
    run_horizon_mins: float
    map: MapConfig
    entities: EntitiesConfig
    parameters: ParametersConfig
    policies: PoliciesConfig
    costs: CostsConfig
    outputs: OutputsConfig
    reward_weights: RewardWeightsConfig = field(default_factory=RewardWeightsConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    stress_tests: Optional[StressTestConfig] = None

def parse_config(raw: dict) -> RunConfig:
    cfg = raw.get("run_config", {})
    return RunConfig(
        seed=cfg["seed"],
        mode=cfg["mode"],
        run_horizon_mins=cfg["run_horizon_mins"],
        map=MapConfig(**cfg.get("map", {})),
        entities=EntitiesConfig(**cfg.get("entities", {})),
        parameters=ParametersConfig(**cfg.get("parameters", {})),
        policies=PoliciesConfig(**cfg.get("policies", {})),
        costs=CostsConfig(**cfg.get("costs", {})),
        outputs=OutputsConfig(**cfg.get("outputs", {})),
        reward_weights=RewardWeightsConfig(**cfg.get("reward_weights", {})),
        training=TrainingConfig(**cfg.get("training", {})),
        eval=EvalConfig(**cfg.get("eval", {})),
        stress_tests=StressTestConfig(**cfg.get("stress_tests")) if cfg.get("stress_tests") else None
    )

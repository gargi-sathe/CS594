# UrbanScale

**Robust Sectoring and Staging for Q-Commerce via Reinforcement Learning**

UrbanScale is a modular city-scale delivery simulator and RL-based warehouse placement system. Phase 0 provides a discrete-event simulator baseline for ultra-fast (10–15 minute) delivery operations using SimPy and NetworkX. Phase 1 layers a Gymnasium-compatible RL environment on top for strategic warehouse placement using MaskablePPO, with full baseline comparison and evaluation pipelines.

---

## Project Structure

```
project/
├── configs/                    # YAML configuration files
│   ├── synthetic_smoke.yaml    # Small RL training config
│   ├── eval_synthetic.yaml     # Synthetic evaluation config
│   └── ...
├── src/
│   ├── core/                   # SimPy engine, state, order generation, dispatch
│   ├── config/                 # Dataclass-based config schema
│   ├── routing/                # Graph builder (synthetic grids + OSMnx)
│   ├── entities/               # Order, Driver, Warehouse models
│   ├── metrics/                # Metric aggregator (SLA, cost, utilization)
│   ├── policies/               # Heuristic assignment and dispatch policies
│   ├── rl/                     # RL environment, observation builder, training
│   ├── baselines/              # Random, clustering, greedy baselines + harness
│   ├── eval/                   # Evaluation runner with frozen scenarios
│   └── visualization/          # Map plotting utilities
├── tests/                      # Pytest suites for Phase 0 and Phase 1
├── experiments/                # Phase 0 standalone experiment runners
└── requirements.txt
```

## Setup

```bash
# Python 3.10+ required
pip install -r requirements.txt

# Set PYTHONPATH (Windows PowerShell)
$env:PYTHONPATH = "."

# Set PYTHONPATH (Linux/macOS)
export PYTHONPATH="."
```

## Phase 0: Discrete-Event Simulator

Phase 0 is a functional baseline simulator modeling drivers, warehouses, orders, and sectors on synthetic grid graphs or real-world OSMnx topologies.

### Key Features
- **SimPy-based event loop**: Time-accurate driver movements over graph edges
- **Dual topology**: Synthetic grids or real-world OSMnx road networks
- **Heuristic policies**: Nearest-warehouse assignment, nearest-idle-driver dispatch
- **Full metrics**: On-time rate, P50/P90/P95 delivery times, cost estimates, driver utilization

### Running Phase 0

```bash
# Synthetic scenario
python experiments/run_synthetic.py --config synthetic_base.yaml --run_id synthetic_demo

# Real-map scenario (requires OSMnx)
python experiments/run_synthetic.py --config real_base.yaml --run_id real_demo
```

### Output Artifacts
- `run_summary.json` — Top-level metrics
- `metrics.json` — Detailed SLA percentiles and delivery statistics
- `event_log.csv` — Raw chronological event trace
- `config_snapshot.yaml` — Reproducibility snapshot
- `map_plot.png` — Static topology visualization

---

## Phase 1: RL Warehouse Placement

Phase 1 builds a Gymnasium-compatible RL environment for strategic warehouse placement. An agent sequentially selects K warehouse sites from candidate nodes, then the Phase 0 simulator evaluates the placement via a full rollout.

### Architecture
- **Environment**: `WarehousePlacementEnv` — Dict observation space with global features (13), candidate features (N×8), action/selected masks, and step index
- **Algorithm**: MaskablePPO from sb3-contrib with `MultiInputPolicy`
- **Reward**: Terminal-only composite reward based on on-time rate, delivery success, missed orders, P95 tail penalty, and normalized cost
- **Baselines**: Random, demand-only K-Means clustering, coverage-based greedy placement

### Observation Space
| Component | Shape | Description |
|-----------|-------|-------------|
| `global_features` | (13,) | Normalized scenario parameters (K, D, T, lambda, horizon, etc.) |
| `candidate_features` | (N, 8) | Per-candidate normalized features (position, demand, travel time, coverage) |
| `selected_mask` | (N,) | Binary mask of already-selected candidates |
| `action_mask` | (N,) | Valid actions (unselected candidates within bounds) |
| `step_index` | (1,) | Current selection step |

### Reward Formula
```
reward = 2.0 * on_time_delivery_rate
       + 0.5 * delivered_success_rate
       - 1.0 * missed_rate
       - 0.5 * tail_penalty
       - 0.1 * normalized_cost_per_order
```
Where:
- `missed_rate = orders_missed / max(total_orders, 1)`
- `tail_penalty = max(0, (p95 - T) / T)`
- `normalized_cost_per_order = min(cost_per_order / cost_norm_ref, 5.0)`

### Training

```bash
# Smoke training run (small synthetic grid)
python -m src.rl.train --config configs/synthetic_smoke.yaml

# Outputs: logs/checkpoints/final_model.zip, logs/tb/ (TensorBoard)
```

### Evaluation

```bash
# Evaluate RL policy vs baselines on synthetic scenarios
python -m src.eval.run_evaluation --config configs/eval_synthetic.yaml

# Outputs: results/eval/eval_raw_results.csv, eval_summary_table.csv/.md
```

The evaluation harness:
1. Generates frozen scenarios per load bucket (Low/Medium/High via `order_lambda`)
2. Evaluates all methods on the **same** frozen scenario (same graph, candidates, demand)
3. Reports per-method metrics in standardized CSV/Markdown tables

### Baselines
| Baseline | Strategy |
|----------|----------|
| **Random** | Selects K candidates uniformly at random (seeded) |
| **Demand Clustering** | K-Means on demand-weighted node coordinates, snapped to nearest candidate |
| **Coverage Greedy** | Iteratively selects the candidate maximizing incremental covered demand within delivery budget |

### Tests

```bash
# Run all Phase 1 tests
python -m pytest tests/test_phase1_env.py tests/test_phase1_baselines.py tests/test_phase1_reward.py tests/test_phase1_training.py tests/test_phase1_eval.py -v
```

---

## Known Limitations

### Phase 0
- Routing uses mathematical edge weights without stochastic traffic delays
- Complex OSMnx MultiDiGraphs are normalized to undirected graphs
- Single-order dispatch only (no batching)

### Phase 1
- Policy trained on synthetic grids only (real-map generalization is M7/future)
- Demand is synthetic/uniform in v1
- Simulator core is used as-is from Phase 0 (no throughput optimization for large-scale training)
- Terminal-only reward (no intermediate shaping)

---

## Configuration

All behavior is config-driven via YAML files parsed into typed dataclasses. Key config sections:

| Section | Purpose |
|---------|---------|
| `map` | Topology type (`grid` or `osmnx`), grid size, place name |
| `entities` | Number of warehouses (K) and drivers (D) |
| `parameters` | `delivery_target_mins`, `pick_pack_time_mins`, `order_lambda` |
| `costs` | `warehouse_base`, `driver_hourly`, `order_op_cost` |
| `reward_weights` | Terminal reward component weights and `cost_norm_ref` |
| `training` | `total_timesteps`, `n_steps`, `batch_size`, `learning_rate`, etc. |
| `eval` | `scenarios_per_bucket`, load buckets, model path, output directory |

---

## Dependencies

Key dependencies (see `requirements.txt` for full list):
- `simpy` — Discrete-event simulation engine
- `networkx` — Graph data structures and algorithms
- `gymnasium` — RL environment interface
- `stable-baselines3` + `sb3-contrib` — MaskablePPO training
- `scikit-learn` — K-Means clustering baseline
- `numpy`, `pandas` — Numerical computing and data aggregation
- `osmnx` — Real-world road network graphs (optional, for real-map mode)

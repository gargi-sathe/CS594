# UrbanScale Phase 0

UrbanScale is a modular city-scale discrete-event simulator written in Python. Phase 0 provides a functional baseline for ultra-fast (10-15 minute) delivery operations, emphasizing strict event-flow physics designed for eventual integration with Reinforcement Learning (RL) strategies.

## Overview
The simulator models drivers, warehouses, orders, and sectors spanning either purely synthetic grid-graphs or genuine geographical layouts powered natively by OSMnx.

### Key Capabilities
*   **Time-Accurate Event Loops**: Driver movements are accurately extrapolated natively against explicit node topological limits leveraging `SimPy`.
*   **Dual-Topology Backend**: Instantly supports arbitrary synthetic boundary bounds or real-world city structures completely interchangeably.
*   **Passive Tracing**: Emits 100% stable `json`, `csv`, `yaml`, and visual `png` formats decoupled natively from engine computation loops.

## Setup Instructions
```bash
# 1. Ensure Python 3.10+
# 2. Install dependencies mapping explicit execution bounds:
pip install -r requirements.txt

# 3. Secure environmental paths natively:
export PYTHONPATH="."  # (Or $env:PYTHONPATH="." on Windows PS)
```

## Running Simulations

### Synthetic Demo
Evaluates 10 drivers across 2 localized warehouses running safely on isolated 10x10 synthetic mathematical arrays:
```bash
python experiments/run_synthetic.py --config synthetic_base.yaml --run_id synthetic_demo
```

### Real-Map Demo
Bootstraps explicitly from realistic mapping coordinates, pulling geographical road distances actively via OSMnx:
```bash
python experiments/run_synthetic.py --config real_base.yaml --run_id real_demo
```

### Testing Suite
We maintain comprehensive execution verification explicitly securing timing bounds natively:
```bash
python -m pytest tests/ -v
```

## Output Artifacts
Simulation endpoints inherently package holistic isolated directory traces natively storing:
*   `run_summary.json`: Top-level validation metrics (SLA bounds, global overhead costing, generated boundaries).
*   `metrics.json`: High-precision SLA logic (Percentiles, delivery delay times arrays natively processed).
*   `event_log.csv`: Raw timeline rows (Chronological outputs intrinsically housing JSON dictionary mappings safely nested natively inside CSV columns).
*   `config_snapshot.yaml`: Strict environment execution reproduction dependencies.
*   `map_plot.png`: Simple static topological boundaries generating deterministic visual maps securely onto the rendered topologies directly.

## Known Limitations (Phase 0)
*   **Routing Limits**: Simulation physics strictly maps mathematical lengths over edges without implementing genuine stochastic real-world traffic delay variations.
*   **Topology Boundaries**: High-complexity OSMnx environments (MultiDiGraphs explicit edge orientations) are deliberately normalized back into simple undirected maps for deterministic physics verification mapping.
*   **Batched Order Boundaries**: Dynamic batch deliveries are out of scope. Every single dispatch fundamentally consumes one Driver exactly.

## RL Extension Points
The codebase natively guarantees architectural stability mapping specific abstractions explicitly capable of intercepting reinforcement overrides actively:
- `assign_warehouse(...)` inside `policies/assignment.py`
- `assign_driver(...)` inside `policies/dispatch.py`
- Stage allocations mapping directly via `driver_process.py` transition logic explicit returns.

# Configuration Schema

Runs are defined by a YAML file containing:

```yaml
run_config:
  seed: int                    # fixed random seed
  mode: str                    # "synthetic" | "real_map"
  map:
    type: str                  # "grid" | "osmnx"
    grid_size: [int, int]      # required if type == grid
    osmnx_place: str           # required if type == osmnx
  entities:
    num_warehouses: int        # K 
    num_drivers: int           # D
  parameters:
    delivery_target_mins: float  # T limit (e.g., 15)
    pick_pack_time_mins: float   # internal delay 
    order_lambda: float          # base arrival rate
  policies:
    staging: str               # "sector_centroid" | "warehouse"
  stress_tests:                # toggle scenarios
    warehouse_closure: 
      time_mins: float
      warehouse_id: int
    driver_shortage: 
      fraction: float
    demand_spike: 
      time_range_mins: [float, float]
      node_id: int
      lambda_mult: float
  costs:                       # metric cost parameters
    warehouse_base: float
    driver_hourly: float
    order_op_cost: float
  run_horizon_mins: float      # strict shutdown limit
```

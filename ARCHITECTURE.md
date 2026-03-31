# Architecture

## Directory Layout
```
src/
  config/      # config parsers and schema definitions
  core/        # discrete-event engine (SimPy)
  entities/    # State machines and dataclasses
  routing/     # NetworkX graphs, A* caching
  policies/    # Base interfaces and Baseline implementations
  metrics/     # Observers for event_log and aggregations
  visuals/     # Plotting logic
```

## Module Boundaries
- **Simulator Core (`core/`)**: Holds the `simpy.Environment`. Progresses time. 
- **Entity State Managers**: E.g., `Driver` object holds its own state (`IDLE`, `EN_ROUTE_TO_DELIVERY`, etc.).
- **Policies**: Pure functions implementing interfaces (`StagingPolicy`, `DispatchPolicy`). The Core invokes these. The Core does NOT make routing decisions itself.
- **Event Bus**: The core yields events to `metrics/` which passively writes to `event_log.csv`.

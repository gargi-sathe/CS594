import simpy
from src.core.state import SimulatorState
from src.config.schema import RunConfig
from src.entities.models import Warehouse, Driver, DriverState, WarehouseState
from src.routing.graph_builder import build_synthetic_graph
from src.core.order_generator import order_generator
from src.policies.assignment import assign_warehouse
from src.policies.dispatch import assign_driver
from src.core.driver_process import driver_lifecycle

def bootstrap_simulation(config: RunConfig) -> SimulatorState:
    env = simpy.Environment()
    
    # Surgical Hook: Driver Shortage
    num_drivers = config.entities.num_drivers
    if config.stress_tests and config.stress_tests.driver_shortage:
        reduction = config.stress_tests.driver_shortage.get('reduction_fraction', 0.5)
        num_drivers = max(1, int(num_drivers * (1 - reduction)))
    
    # 1. Graph Generation dynamically bounds mapping
    if config.map.type == "grid":
        from src.routing.graph_builder import build_synthetic_graph
        G, apsp = build_synthetic_graph(config.map.grid_size[0], config.map.grid_size[1])
    elif config.map.type == "osmnx":
        from src.routing.graph_builder import build_osmnx_graph
        G, apsp = build_osmnx_graph(config.map.osmnx_place)
    else:
        raise ValueError(f"Invalid map type parameter boundaries: {config.map.type}")
        
    state = SimulatorState(env, G, apsp)
    state.pick_pack_mins = config.parameters.pick_pack_time_mins
    
    # 2. Warehouses (placed dynamically targeting correct topological structures)
    valid_nodes = list(G.nodes())
    import random
    random.seed(config.seed)
    
    if getattr(config.entities, "warehouse_locations", None) is not None:
        wh_nodes = config.entities.warehouse_locations
    elif config.map.type == "grid":
        wh_nodes = [i * 2 for i in range(config.entities.num_warehouses)]
    else:
        # Uniform discrete sampling guaranteeing explicitly stable nodes strictly mapped
        wh_nodes = random.sample(valid_nodes, min(len(valid_nodes), config.entities.num_warehouses))
        
    for i in range(config.entities.num_warehouses):
        w_id = f"W{i+1}"
        loc = wh_nodes[i % len(wh_nodes)] 
        state.warehouses[w_id] = Warehouse(id=w_id, location=loc, state=WarehouseState.OPEN)
        
    # 3. Drivers
    for i in range(num_drivers):
        d_id = f"D{i+1}"
        w_idx = i % config.entities.num_warehouses
        assigned_w = list(state.warehouses.values())[w_idx]
        # Drivers mapped to baseline facility staging for Step 9 metrics
        state.drivers[d_id] = Driver(
            id=d_id, 
            current_location=assigned_w.location, 
            assigned_sector_centroid=assigned_w.location, 
            state=DriverState.IDLE
        )
        
    return state

def closure_process(state: SimulatorState, config: RunConfig):
    """Surgical Hook: Fails a warehouse mid-run."""
    settings = config.stress_tests.warehouse_closure
    yield state.env.timeout(settings.get('close_at_min', 30.0))
    target = settings.get('target_warehouse_id', 'W1')
    if target in state.warehouses:
        state.warehouses[target].state = WarehouseState.CLOSED
        state.log_event("Warehouse", target, "CLOSED_FOR_ROBUSTNESS", f"time={state.env.now}")

def dispatch_loop(state: SimulatorState, config: RunConfig):
    """
    Active chronological ticker periodically interrogating queue lengths and triggering 
    evaluations against passive discrete objects. (Polled cleanly to emulate asynchronous bounds).
    """
    while state.env.now < config.run_horizon_mins:
        # Assign newly arrived UNASSIGNED orders seamlessly directly to the grid mapping
        for o_id, order in list(state.orders.items()):
            if order.state.name == "UNASSIGNED":
                assign_warehouse(state, order)
                
        # Continuously exhaust driver dispatch allocations per-warehouse logic 
        for w_id in state.warehouses:
            while True: 
                match = assign_driver(state, w_id)
                if match:
                    driver, order = match
                    state.env.process(driver_lifecycle(state, driver, order))
                else:
                    break # Break naturally once queues empty or drivers exhausted
                    
        yield state.env.timeout(0.1) # Precision check interval every 0.1 simulated minutes

def run_simulation(config: RunConfig) -> SimulatorState:
    """Holistic wrapper binding strictly initialized config instances to physics pipelines."""
    state = bootstrap_simulation(config)
    # Instantiate concurrent SimPy loops
    state.env.process(order_generator(state, config.parameters.order_lambda, config.run_horizon_mins, config))
    state.env.process(dispatch_loop(state, config))
    
    # Surgical Hook: Warehouse Closure
    if config.stress_tests and config.stress_tests.warehouse_closure:
        state.env.process(closure_process(state, config))
        
    # Begin bounds limit evaluation
    state.env.run(until=config.run_horizon_mins)
    return state

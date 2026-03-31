import simpy
import networkx as nx
from src.core.state import SimulatorState
from src.metrics.aggregator import calculate_metrics, get_percentile
from src.config.schema import RunConfig, MapConfig, EntitiesConfig, ParametersConfig, PoliciesConfig, CostsConfig, OutputsConfig
from src.entities.models import Order, OrderState, Driver, DriverState

def test_percentile_logic():
    data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    assert get_percentile(data, 50) == 50.0  
    assert get_percentile(data, 90) == 90.0  
    assert get_percentile(data, 100) == 100.0 

def test_metric_aggregation():
    env = simpy.Environment()
    state = SimulatorState(env, nx.Graph(), {})
    
    cfg = RunConfig(
        seed=1, mode="synthetic", run_horizon_mins=60.0,
        map=MapConfig("grid", [5,5]),
        entities=EntitiesConfig(1, 2), # 1 W, 2 Drivers
        parameters=ParametersConfig(15.0, 2.0, 10.0),
        policies=PoliciesConfig("centroid", "nearest"),
        costs=CostsConfig(100.0, 20.0, 2.0),
        outputs=OutputsConfig("out")
    )
    
    # Validation data: 2 on-time, 1 late, 2 unserved
    o1 = Order("O1", 1, 0.0, OrderState.DELIVERED, "W1", "D1", 2.0, 10.0) # On time (del=10 <= 15 target)
    o2 = Order("O2", 1, 0.0, OrderState.DELIVERED, "W1", "D2", 2.0, 20.0) # Late (del=20 > 15)
    o3 = Order("O3", 1, 0.0, OrderState.QUEUED, "W1") # unserved
    o4 = Order("O4", 1, 0.0, OrderState.DELIVERED, "W1", "D1", 2.0, 12.0) # On time
    o5 = Order("O5", 1, 0.0, OrderState.UNASSIGNED) # unserved
    
    state.orders = {"O1": o1, "O2": o2, "O3": o3, "O4": o4, "O5": o5}
    state.drivers = {"D1": Driver("D1", 0), "D2": Driver("D2", 0)}
    
    # Mock event log purely for Queue tracking and Utilization passive formulas
    state.event_log = [
        (1.0, "Dispatcher", "W1", "DISPATCHED", "order=O1, driver=D1, dist=2"),
        (2.0, "Dispatcher", "W1", "DISPATCHED", "order=O2, driver=D2, dist=2"),
        (15.0, "Driver", "D1", "RETURNED_TO_STAGING", "loc=0")
        # D2 explicitly simulated to run until horizon
    ]
    
    metrics = calculate_metrics(state, cfg)
    
    # 1. Delivery times: O1=10, O2=20, O4=12 (avg = 42/3 = 14)
    assert metrics["average_delivery_time"] == 14.0
    
    # 2. On-time delivery classification
    assert metrics["on_time_delivery_rate"] == (2 / 5)
    assert metrics["delivered_success_rate"] == (3 / 5)
    
    # 3. Unserved orders bounds mapping
    assert metrics["orders_missed_or_unserved"] == 2
    
    # 4. Queue times: O1 dispatched at 1.0 (arr 0) -> 1.0. O2 dispatched at 2.0 -> 2.0. Avg = 1.5
    assert metrics["average_queue_time"] == 1.5
    
    # 5. Driver utilization physics
    # D1 active from 1.0 to 15.0 = 14.0
    # D2 active from 2.0 to horizon 60.0 = 58.0
    # Total active = 72.0
    # Total availability = 2 drivers * 60 = 120.0
    assert metrics["driver_utilization"] == (72.0 / 120.0)
    
    assert metrics["average_pick_pack_time"] == 2.0
    
    # Cost = 1W * 100 + 2D * 20 * (60/60) + 3 ord * 2.0 = 100 + 40 + 6 = 146
    assert metrics["simple_cost_estimate"] == 146.0

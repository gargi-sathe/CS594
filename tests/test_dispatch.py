import simpy
from src.core.state import SimulatorState
from src.entities.models import Order, OrderState, Driver, DriverState, Warehouse, WarehouseState
from src.routing.graph_builder import build_synthetic_graph
from src.policies.dispatch import assign_driver

def test_dispatch_logic():
    env = simpy.Environment()
    G, apsp = build_synthetic_graph(5, 5) # nodes 0 to 24
    state = SimulatorState(env, G, apsp)
    
    w1 = Warehouse(id="W1", location=12) # Center
    state.warehouses["W1"] = w1
    
    o1 = Order(id="O1", location=0, arrival_time=0.0, state=OrderState.QUEUED, assigned_warehouse_id="W1")
    o2 = Order(id="O2", location=1, arrival_time=1.0, state=OrderState.QUEUED, assigned_warehouse_id="W1")
    
    state.orders = {"O1": o1, "O2": o2}
    state.warehouse_queues["W1"] = ["O1", "O2"]
    
    d1 = Driver(id="D1", current_location=0, state=DriverState.IDLE)
    d2 = Driver(id="D2", current_location=13, state=DriverState.IDLE)
    d3 = Driver(id="D3", current_location=11, state=DriverState.IDLE)
    
    state.drivers = {"D1": d1, "D2": d2, "D3": d3}
    
    result = assign_driver(state, "W1")
    assert result is not None
    driver, order = result
    
    assert driver.id == "D2"
    assert order.id == "O1"
    assert driver.state == DriverState.EN_ROUTE_TO_PICKUP
    assert order.state == OrderState.DISPATCHED
    
    assert state.warehouse_queues["W1"] == ["O2"]
    
    result2 = assign_driver(state, "W1")
    assert result2 is not None
    driver2, order2 = result2
    
    assert driver2.id == "D3"
    assert order2.id == "O2"
    assert driver2.state == DriverState.EN_ROUTE_TO_PICKUP
    assert order2.state == OrderState.DISPATCHED
    
    assert len(state.warehouse_queues["W1"]) == 0
    assert assign_driver(state, "W1") is None

def test_dispatch_numeric_tie_breaker():
    env = simpy.Environment()
    G, apsp = build_synthetic_graph(5, 5) # 0 to 24
    state = SimulatorState(env, G, apsp)
    
    w1 = Warehouse(id="W1", location=12)
    state.warehouses["W1"] = w1
    
    o1 = Order(id="O1", location=0, arrival_time=0.0, state=OrderState.QUEUED, assigned_warehouse_id="W1")
    state.orders = {"O1": o1}
    state.warehouse_queues["W1"] = ["O1"]
    
    # Distance to 12 from 11 is 1
    # distance to 12 from 13 is 1
    # We want to test D2 vs D10. Lexicographically, D10 < D2. 
    # But numerically, D2 < D10.
    
    d10 = Driver(id="D10", current_location=11, state=DriverState.IDLE)
    d2 = Driver(id="D2", current_location=13, state=DriverState.IDLE)
    
    state.drivers = {"D10": d10, "D2": d2}
    
    result = assign_driver(state, "W1")
    assert result is not None
    driver, order = result
    
    # Numerically D2 is preferred
    assert driver.id == "D2"

import simpy
from src.core.state import SimulatorState
from src.entities.models import Order, Driver, OrderState, DriverState, Warehouse
from src.routing.graph_builder import build_synthetic_graph
from src.core.driver_process import driver_lifecycle

def setup_baseline():
    env = simpy.Environment()
    G, apsp = build_synthetic_graph(5, 5)  # nodes 0 to 24
    state = SimulatorState(env, G, apsp)
    state.pick_pack_mins = 5.0
    
    w1 = Warehouse(id="W1", location=12) # Center distance to 0 is 4, distance to 24 is 4
    state.warehouses["W1"] = w1
    
    o1 = Order(id="O1", location=24, arrival_time=0.0, state=OrderState.DISPATCHED, assigned_warehouse_id="W1")
    state.orders["O1"] = o1
    
    d1 = Driver(id="D1", current_location=0, assigned_sector_centroid=0, state=DriverState.EN_ROUTE_TO_PICKUP)
    state.drivers["D1"] = d1
    
    return env, state, d1, o1

def test_driver_arrives_before_pick_pack():
    env, state, d1, o1 = setup_baseline()
    state.pick_pack_mins = 6.0  # Driver travel is 4.0 mins
    
    env.process(driver_lifecycle(state, d1, o1))
    
    # Run to minute 3.0 (Driver still en route, 4 mins needed)
    env.run(until=3.1)
    assert d1.state == DriverState.EN_ROUTE_TO_PICKUP
    assert o1.state == OrderState.DISPATCHED
    
    # Run to minute 5.0 (Driver arrived at 4.0, enters wait because pick is 6.0)
    env.run(until=5.1)
    assert d1.current_location == 12
    assert d1.state == DriverState.WAITING_FOR_ORDER
    # Order hasn't been picked up yet
    assert o1.state == OrderState.DISPATCHED
    
    # Run to exactly post-pickpack (6.1)
    env.run(until=6.1)
    assert o1.state == OrderState.OUT_FOR_DELIVERY
    assert d1.state == DriverState.EN_ROUTE_TO_DELIVERY
    assert o1.picked_time == 6.0
    
    # Travel from 12 to 24 is 4 mins. Arrives at 10.0
    env.run(until=10.1)
    assert o1.state == OrderState.DELIVERED
    assert o1.delivered_time == 10.0
    assert d1.state == DriverState.RETURNING
    
    # Return to staging point 0 from 24 is 8 mins. Arrives at 18.0
    env.run(until=18.1)
    assert d1.state == DriverState.IDLE
    assert d1.current_location == 0


def test_pick_pack_done_before_driver_arrives():
    env, state, d1, o1 = setup_baseline()
    state.pick_pack_mins = 2.0  # Driver travel is 4.0 mins
    
    env.process(driver_lifecycle(state, d1, o1))
    
    env.run(until=4.1)
    # At 4.0 driver arrives, sees pick_pack is done (since 4.0 > 2.0)
    # Immediately transitions to EN_ROUTE_TO_DELIVERY
    assert d1.current_location == 12
    assert d1.state == DriverState.EN_ROUTE_TO_DELIVERY
    assert o1.state == OrderState.OUT_FOR_DELIVERY
    assert o1.picked_time == 4.0
    
    # Travel to customer 4 mins -> 8.0
    env.run(until=8.1)
    assert o1.state == OrderState.DELIVERED
    assert d1.state == DriverState.RETURNING
    
    # Travel to stage 8 mins -> 16.0
    env.run(until=16.1)
    assert d1.state == DriverState.IDLE
    assert d1.current_location == 0

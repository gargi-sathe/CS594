import simpy
from src.core.state import SimulatorState
from src.entities.models import Order, OrderState, Warehouse, WarehouseState
from src.routing.graph_builder import build_synthetic_graph
from src.policies.assignment import assign_warehouse

def test_warehouse_assignment():
    env = simpy.Environment()
    G, apsp = build_synthetic_graph(5, 5) # 0 to 24 nodes
    state = SimulatorState(env, G, apsp)
    
    w1 = Warehouse(id="W1", location=0, state=WarehouseState.OPEN)
    w2 = Warehouse(id="W2", location=24, state=WarehouseState.OPEN)
    
    state.warehouses = {"W1": w1, "W2": w2}
    
    o1 = Order(id="O1", location=1, arrival_time=0.0) # near node 0
    o2 = Order(id="O2", location=23, arrival_time=0.0) # near node 24
    
    assert assign_warehouse(state, o1) == True
    assert assign_warehouse(state, o2) == True
    
    assert o1.state == OrderState.QUEUED
    assert o1.assigned_warehouse_id == "W1"
    
    assert o2.state == OrderState.QUEUED
    assert o2.assigned_warehouse_id == "W2"
    
    assert state.warehouse_queues["W1"] == ["O1"]
    assert state.warehouse_queues["W2"] == ["O2"]
    
def test_no_open_warehouse():
    env = simpy.Environment()
    G, apsp = build_synthetic_graph(2, 2)
    state = SimulatorState(env, G, apsp)
    
    w1 = Warehouse(id="W1", location=0, state=WarehouseState.CLOSED)
    state.warehouses = {"W1": w1}
    
    o1 = Order(id="O1", location=3, arrival_time=0.0)
    
    assert assign_warehouse(state, o1) == False
    assert o1.state == OrderState.UNASSIGNED

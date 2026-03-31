import simpy
import networkx as nx
import random
from src.core.state import SimulatorState
from src.core.order_generator import order_generator
from src.entities.models import OrderState

def test_order_generation():
    random.seed(42)
    env = simpy.Environment()
    G = nx.Graph()
    G.add_node(1)
    
    state = SimulatorState(env, G, apsp={})
    
    # Generate ~60 orders per hour (1 per minute) for 100 minutes.
    env.process(order_generator(state, 60.0, 100.0))
    env.run(until=100.0)
    
    assert 50 < len(state.orders) < 150
    assert len(state.event_log) == len(state.orders)
    
    # Verify exact state logic is matched
    first_order = list(state.orders.values())[0]
    assert first_order.state == OrderState.UNASSIGNED
    assert first_order.arrival_time > 0.0

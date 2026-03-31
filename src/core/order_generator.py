import simpy
import random
from src.core.state import SimulatorState
from src.entities.models import Order, OrderState

def order_generator(state: SimulatorState, order_lambda_per_hour: float, horizon_mins: float):
    """
    SimPy process that generates orders across time using a Poisson process.
    """
    order_id_counter = 0
    # Convert hourly rate to minute-based lambda
    lambda_per_min = order_lambda_per_hour / 60.0
    
    if lambda_per_min <= 0.0:
        yield state.env.timeout(horizon_mins)
        return
        
    nodes = list(state.graph.nodes)
    
    while state.env.now < horizon_mins:
        # Inter-arrival time from exponential distribution
        inter_arrival = random.expovariate(lambda_per_min)
        yield state.env.timeout(inter_arrival)
        
        # If we jump past horizon_mins, break
        if state.env.now >= horizon_mins:
            break
            
        order_id_counter += 1
        oid = f"O-{order_id_counter}"
        loc = random.choice(nodes)
        
        o = Order(id=oid, location=loc, arrival_time=state.env.now, state=OrderState.UNASSIGNED)
        state.orders[oid] = o
        
        state.log_event("Order", oid, "CREATED", f"location={loc}")

import simpy
import random
from typing import Any
from src.core.state import SimulatorState
from src.entities.models import Order, OrderState

def order_generator(state: SimulatorState, order_lambda_per_hour: float, horizon_mins: float, config: Any = None):
    """
    SimPy process that generates orders across time using a Poisson process.
    """
    order_id_counter = 0
    nodes = list(state.graph.nodes)
    stress = getattr(config, 'stress_tests', None) if config else None

    while state.env.now < horizon_mins:
        curr_lmbda = order_lambda_per_hour
        if stress and stress.time_shift:
            for s, e, l in stress.time_shift.get('lambda_profile', []):
                if s <= state.env.now < e:
                    curr_lmbda = l
                    break
        l_per_m = curr_lmbda / 60.0
        if l_per_m <= 0:
            yield state.env.timeout(1.0)
            continue
            
        yield state.env.timeout(random.expovariate(l_per_m))
        if state.env.now >= horizon_mins: break
        
        loc = random.choice(nodes)
        if stress and stress.demand_spike:
            sp = stress.demand_spike
            if isinstance(loc, int):
                gw = 5
                x, y = loc // gw, loc % gw
                mid = gw / 2
                in_q = False
                q = sp.get('quadrant', 0)
                if q == 0 and x < mid and y < mid: in_q = True
                elif q == 1 and x >= mid and y < mid: in_q = True
                elif q == 2 and x < mid and y >= mid: in_q = True
                elif q == 3 and x >= mid and y >= mid: in_q = True
                
                if in_q:
                    for _ in range(int(sp.get('multiplier', 3))):
                        order_id_counter += 1
                        oid = f"O-{order_id_counter}"
                        state.orders[oid] = Order(id=oid, location=loc, arrival_time=state.env.now, state=OrderState.UNASSIGNED)
                    continue

        order_id_counter += 1
        oid = f"O-{order_id_counter}"
        state.orders[oid] = Order(id=oid, location=loc, arrival_time=state.env.now, state=OrderState.UNASSIGNED)
        state.log_event("Order", oid, "CREATED", f"location={loc}")

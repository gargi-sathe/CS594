import re
from typing import List, Tuple, Optional
from src.core.state import SimulatorState
from src.entities.models import Order, Driver, DriverState, OrderState

def _driver_sort_key(driver_id: str) -> tuple:
    """
    Extracts numeric suffix for properly ordering IDs like 'D2' vs 'D10'.
    'D10' becomes ('D', 10), which is greater than ('D', 2).
    Arbitrary strings fallback to (driver_id, 0).
    """
    match = re.search(r'^([a-zA-Z\-_]*)(\d+)$', driver_id)
    if match:
        prefix, num = match.groups()
        return (prefix, int(num))
    return (driver_id, 0)

def assign_driver(state: SimulatorState, warehouse_id: str) -> Optional[Tuple[Driver, Order]]:
    """
    Attempts to dispatch an IDLE driver to the oldest QUEUED order at a specific warehouse.
    
    Tie-breaking rules for drivers:
    1. Shortest travel time to the warehouse.
    2. Lowest alphanumeric parsed driver ID (e.g., D2 preferred over D10).
    """
    queue = state.warehouse_queues.get(warehouse_id, [])
    if not queue:
        return None
        
    order_id = queue[0]
    order = state.orders[order_id]
    warehouse_loc = state.warehouses[warehouse_id].location
    
    best_driver = None
    best_dist = float('inf')
    
    for d_id, d in state.drivers.items():
        if d.state == DriverState.IDLE:
            dist = state.apsp[d.current_location].get(warehouse_loc, float('inf'))
            
            # Tie breaking: distance first, then numeric-aware driver_id parsing
            if dist < best_dist:
                best_dist = dist
                best_driver = d
            elif dist == best_dist and best_driver is not None:
                if _driver_sort_key(d_id) < _driver_sort_key(best_driver.id):
                    best_driver = d
                    
    if best_driver is None:
        return None
        
    # Valid dispatch found
    queue.pop(0)  # drain from pending queue
    
    # State transitions mapped
    order.state = OrderState.DISPATCHED
    order.dispatched_driver_id = best_driver.id
    
    best_driver.state = DriverState.EN_ROUTE_TO_PICKUP
    
    state.log_event("Dispatcher", warehouse_id, "DISPATCHED", f"order={order_id}, driver={best_driver.id}, dist={best_dist}")
    
    return best_driver, order

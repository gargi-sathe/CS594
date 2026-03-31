from src.core.state import SimulatorState
from src.entities.models import Order, OrderState, WarehouseState

def assign_warehouse(state: SimulatorState, order: Order) -> bool:
    """
    Finds the nearest OPEN warehouse and assigns the order to it.
    Returns True if successfully assigned, False if no open warehouse available.
    """
    best_w_id = None
    best_dist = float('inf')
    
    for w_id, w in state.warehouses.items():
        if w.state == WarehouseState.OPEN:
            # Synthetic mode caches full all-pairs map
            # Fallback to float('inf') if isolated logically
            dist = state.apsp[w.location].get(order.location, float('inf'))
            if dist < best_dist:
                best_dist = dist
                best_w_id = w_id
                
    if best_w_id is None:
        state.log_event("Order", order.id, "REJECTED_NO_OPEN_WAREHOUSE", "No open warehouse found")
        return False
        
    order.assigned_warehouse_id = best_w_id
    order.state = OrderState.QUEUED
    
    if best_w_id not in state.warehouse_queues:
        state.warehouse_queues[best_w_id] = []
    
    state.warehouse_queues[best_w_id].append(order.id)
    state.log_event("Order", order.id, "ASSIGNED_WAREHOUSE", f"warehouse_id={best_w_id}, dist={best_dist}")
    
    return True

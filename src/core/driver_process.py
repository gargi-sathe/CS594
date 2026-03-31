import simpy
from src.core.state import SimulatorState
from src.entities.models import Order, Driver, DriverState, OrderState

def driver_lifecycle(state: SimulatorState, driver: Driver, order: Order):
    """
    SimPy process modeling the physical lifecycle of a dispatched driver.
    Handles travel, wait queues for prep, pick, and explicit staging returns.
    """
    warehouse = state.warehouses[order.assigned_warehouse_id]
    
    # 1. Travel to warehouse
    travel_to_wh = state.apsp[driver.current_location].get(warehouse.location, float('inf'))
    if travel_to_wh == float('inf'):
        travel_to_wh = 0.0 # Synthetic isolated graph fallback
        
    yield state.env.timeout(travel_to_wh)
    driver.current_location = warehouse.location
    driver.state = DriverState.WAITING_FOR_ORDER
    state.log_event("Driver", driver.id, "ARRIVED_AT_WAREHOUSE", f"warehouse={warehouse.id}")
    
    # 2. Wait for pick/pack completion
    time_since_arrival = state.env.now - order.arrival_time
    pick_pack_time = state.pick_pack_mins
    
    if time_since_arrival < pick_pack_time:
        wait_time = pick_pack_time - time_since_arrival
        yield state.env.timeout(wait_time)
        
    # Pickup completion
    order.picked_time = state.env.now
    order.state = OrderState.OUT_FOR_DELIVERY
    driver.state = DriverState.EN_ROUTE_TO_DELIVERY
    state.log_event("Order", order.id, "PICKED_UP", f"driver={driver.id}")
    
    # 3. Travel to customer
    travel_to_cust = state.apsp[driver.current_location].get(order.location, float('inf'))
    if travel_to_cust == float('inf'):
        travel_to_cust = 0.0
        
    yield state.env.timeout(travel_to_cust)
    driver.current_location = order.location
    order.delivered_time = state.env.now
    order.state = OrderState.DELIVERED
    state.log_event("Order", order.id, "DELIVERED", f"driver={driver.id}")
    
    # 4. Transition into RETURNING
    driver.state = DriverState.RETURNING
    staging_target = driver.assigned_sector_centroid if driver.assigned_sector_centroid is not None else warehouse.location
    travel_to_stage = state.apsp[driver.current_location].get(staging_target, float('inf'))
    if travel_to_stage == float('inf'):
        travel_to_stage = 0.0
        
    yield state.env.timeout(travel_to_stage)
    driver.current_location = staging_target
    driver.state = DriverState.IDLE
    state.log_event("Driver", driver.id, "RETURNED_TO_STAGING", f"location={staging_target}")

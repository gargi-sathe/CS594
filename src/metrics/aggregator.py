import statistics
import re
from typing import Dict, Any, List
from src.core.state import SimulatorState
from src.config.schema import RunConfig
from src.entities.models import OrderState

def get_percentile(data: List[float], p: float) -> float:
    """Helper method bypassing external deps like NumPy for precise bounds."""
    if not data:
        return 0.0
    s_data = sorted(data)
    idx = int((p / 100.0) * (len(s_data) - 1))
    return s_data[idx]

def calculate_metrics(state: SimulatorState, config: RunConfig) -> Dict[str, Any]:
    delivered_times = []
    travel_to_cust_times = []
    
    unserved_count = 0
    delivered_count = 0
    
    for o in state.orders.values():
        if o.state == OrderState.DELIVERED and o.delivered_time is not None and o.picked_time is not None:
            del_time = o.delivered_time - o.arrival_time
            delivered_times.append(del_time)
            travel_to_cust_times.append(o.delivered_time - o.picked_time)
            delivered_count += 1
        else:
            unserved_count += 1
            
    # Parse event log for Queue time and Driver utilization
    dispatch_times = {} # order_id -> dispatch timestamp
    driver_busy_intervals = {d_id: [] for d_id in state.drivers.keys()}
    driver_last_dispatched = {d_id: None for d_id in state.drivers.keys()}
    
    for ts, e_type, e_id, e_event, details in state.event_log:
        if e_type == "Dispatcher" and e_event == "DISPATCHED":
            # details format: "order=O-1, driver=D-1, dist=xx"
            match_o = re.search(r'order=([^,]+)', details)
            match_d = re.search(r'driver=([^,]+)', details)
            if match_o and match_d:
                oid = match_o.group(1)
                did = match_d.group(1)
                dispatch_times[oid] = ts
                driver_last_dispatched[did] = ts
                
        elif e_type == "Driver" and e_event == "RETURNED_TO_STAGING":
            did = e_id
            last_disp = driver_last_dispatched.get(did)
            if last_disp is not None:
                driver_busy_intervals[did].append(ts - last_disp)
                driver_last_dispatched[did] = None
                
    horizon = config.run_horizon_mins
    # close out busy intervals actively pending when simulation terminates
    for did, last_disp in driver_last_dispatched.items():
        if last_disp is not None:
            driver_busy_intervals[did].append(horizon - last_disp)
            
    queue_times = []
    for oid, d_ts in dispatch_times.items():
        if oid in state.orders:
            queue_times.append(d_ts - state.orders[oid].arrival_time)
            
    total_driver_time = config.entities.num_drivers * horizon
    total_busy_time = sum(sum(intervals) for intervals in driver_busy_intervals.values())
    driver_util = (total_busy_time / total_driver_time) if total_driver_time > 0 else 0.0
    
    on_time = 0
    target = config.parameters.delivery_target_mins
    for dt in delivered_times:
        if dt <= target:
            on_time += 1
            
    avg_del = statistics.mean(delivered_times) if delivered_times else 0.0
    p50 = get_percentile(delivered_times, 50)
    p90 = get_percentile(delivered_times, 90)
    p95 = get_percentile(delivered_times, 95)
        
    total_orders = len(state.orders)
    on_time_rate = (on_time / total_orders) if total_orders > 0 else 0.0
    delivered_success_rate = (delivered_count / total_orders) if total_orders > 0 else 0.0
    
    avg_q = statistics.mean(queue_times) if queue_times else 0.0
    avg_ttc = statistics.mean(travel_to_cust_times) if travel_to_cust_times else 0.0
    
    # Simple cost
    w_cost = config.costs.warehouse_base * config.entities.num_warehouses
    d_cost = config.costs.driver_hourly * config.entities.num_drivers * (horizon / 60.0)
    op_cost = config.costs.order_op_cost * delivered_count
    total_cost = w_cost + d_cost + op_cost
    
    return {
        "on_time_delivery_rate": on_time_rate,
        "delivered_success_rate": delivered_success_rate,
        "average_delivery_time": avg_del,
        "p50_delivery_time": p50,
        "p90_delivery_time": p90,
        "p95_delivery_time": p95,
        "average_queue_time": avg_q,
        "average_pick_pack_time": getattr(state, 'pick_pack_mins', config.parameters.pick_pack_time_mins),
        "average_travel_to_customer_time": avg_ttc,
        "driver_utilization": driver_util,
        "orders_missed_or_unserved": unserved_count,
        "simple_cost_estimate": total_cost,
        "delivered_count": delivered_count,
        "total_orders_generated": total_orders
    }

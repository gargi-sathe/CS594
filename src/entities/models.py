from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class OrderState(Enum):
    UNASSIGNED = "UNASSIGNED"
    QUEUED = "QUEUED"
    DISPATCHED = "DISPATCHED"
    PICKING_UP = "PICKING_UP"
    OUT_FOR_DELIVERY = "OUT_FOR_DELIVERY"
    DELIVERED = "DELIVERED"

class DriverState(Enum):
    IDLE = "IDLE"
    EN_ROUTE_TO_PICKUP = "EN_ROUTE_TO_PICKUP"
    WAITING_FOR_ORDER = "WAITING_FOR_ORDER"
    EN_ROUTE_TO_DELIVERY = "EN_ROUTE_TO_DELIVERY"
    RETURNING = "RETURNING"

class WarehouseState(Enum):
    OPEN = "OPEN"
    CLOSED = "CLOSED"

@dataclass
class Warehouse:
    id: str
    location: int
    state: WarehouseState = WarehouseState.OPEN

@dataclass
class Sector:
    id: str
    assigned_warehouse_id: str
    centroid_node: int
    member_nodes: List[int]

@dataclass
class Driver:
    id: str
    current_location: int
    assigned_sector_centroid: Optional[int] = None
    state: DriverState = DriverState.IDLE

@dataclass
class Order:
    id: str
    location: int
    arrival_time: float
    state: OrderState = OrderState.UNASSIGNED
    assigned_warehouse_id: Optional[str] = None
    dispatched_driver_id: Optional[str] = None
    picked_time: Optional[float] = None
    delivered_time: Optional[float] = None

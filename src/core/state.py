import simpy
import networkx as nx
from typing import Dict, List, Tuple
from src.entities.models import Order, Driver, Warehouse, Sector

class SimulatorState:
    def __init__(self, env: simpy.Environment, G: nx.Graph, apsp: Dict[int, Dict[int, float]]):
        self.env = env
        self.graph = G
        self.apsp = apsp
        
        self.orders: Dict[str, Order] = {}
        self.drivers: Dict[str, Driver] = {}
        self.warehouses: Dict[str, Warehouse] = {}
        self.sectors: Dict[str, Sector] = {}
        
        # Pending queue of order IDs at each warehouse ID
        self.warehouse_queues: Dict[str, List[str]] = {}
        
        # Event log: timestamp, entity_type, entity_id, event_type, details
        self.event_log: List[Tuple[float, str, str, str, str]] = []
        
        self.pick_pack_mins: float = 2.0  # Phase 0 default
        
    def log_event(self, entity_type: str, entity_id: str, event_type: str, details: str):
        self.event_log.append((self.env.now, entity_type, entity_id, event_type, details))

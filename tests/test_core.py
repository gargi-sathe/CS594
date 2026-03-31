import pytest
from src.entities.models import Order, Driver, Warehouse, Sector, OrderState, DriverState, WarehouseState
from src.config.schema import parse_config

def test_entity_initialization():
    w = Warehouse(id="W1", location=1)
    assert w.state == WarehouseState.OPEN
    
    s = Sector(id="S1", assigned_warehouse_id="W1", centroid_node=5, member_nodes=[5,6])
    assert s.centroid_node == 5
    
    d = Driver(id="D1", current_location=1, assigned_sector_centroid=5)
    assert d.state == DriverState.IDLE
    
    o = Order(id="O1", location=10, arrival_time=2.0)
    assert o.state == OrderState.UNASSIGNED

def test_config_parsing():
    raw_cfg = {
        "run_config": {
            "seed": 42,
            "mode": "synthetic",
            "run_horizon_mins": 120.0,
            "map": {"type": "grid", "grid_size": [10, 10]},
            "entities": {"num_warehouses": 2, "num_drivers": 10},
            "parameters": {"delivery_target_mins": 15.0, "pick_pack_time_mins": 2.0, "order_lambda": 5.0},
            "policies": {"staging": "sector_centroid", "dispatch": "nearest_idle"},
            "costs": {"warehouse_base": 1000, "driver_hourly": 20, "order_op_cost": 2},
            "outputs": {"output_dir": "results/"}
        }
    }
    config = parse_config(raw_cfg)
    assert config.seed == 42
    assert config.map.grid_size == [10, 10]
    assert config.parameters.pick_pack_time_mins == 2.0

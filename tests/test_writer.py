import os
import tempfile
import csv
import json
import yaml
import simpy
import networkx as nx
from src.core.state import SimulatorState
from src.metrics.writer import write_outputs
from src.config.schema import RunConfig, MapConfig, EntitiesConfig, ParametersConfig, PoliciesConfig, CostsConfig, OutputsConfig
from src.entities.models import Order, OrderState

def test_write_outputs():
    with tempfile.TemporaryDirectory() as tmpdir:
        env = simpy.Environment()
        from src.routing.graph_builder import build_synthetic_graph
        G, apsp = build_synthetic_graph(5, 5)
        state = SimulatorState(env, G, apsp)
        
        cfg = RunConfig(
            seed=1, mode="synthetic", run_horizon_mins=60.0,
            map=MapConfig("grid", [5,5]),
            entities=EntitiesConfig(1, 2),
            parameters=ParametersConfig(15.0, 2.0, 10.0),
            policies=PoliciesConfig("centroid", "nearest"),
            costs=CostsConfig(100.0, 20.0, 2.0),
            outputs=OutputsConfig(tmpdir)
        )
        
        state.orders["O1"] = Order("O1", 1, 0.0, OrderState.DELIVERED, "W1", "D1", 2.0, 10.0)
        from src.entities.models import Driver
        state.drivers["D1"] = Driver(id="D1", current_location=0)
        state.event_log = [
            (1.0, "Dispatcher", "W1", "DISPATCHED", "order=O1, driver=D1, dist=2")
        ]
        
        write_outputs(state, cfg, run_id="test_run")
        
        run_dir = os.path.join(tmpdir, "test_run")
        assert os.path.exists(run_dir)
        
        # 1. Check summary JSON
        with open(os.path.join(run_dir, "run_summary.json"), "r") as f:
            summary = json.load(f)
            assert summary["run_id"] == "test_run"
            assert summary["total_orders_generated"] == 1
            
        # 2. Check metrics JSON
        with open(os.path.join(run_dir, "metrics.json"), "r") as f:
            metrics = json.load(f)
            assert "on_time_delivery_rate" in metrics
            
        # 3. Check yaml config
        with open(os.path.join(run_dir, "config_snapshot.yaml"), "r") as f:
            snap = yaml.safe_load(f)
            assert snap["run_config"]["seed"] == 1
            
        # 4. Map assertions cleanly mapped
        assert os.path.exists(os.path.join(run_dir, "map_plot.png"))
            
        # 5. Check event log csv and stable JSON bounds
        with open(os.path.join(run_dir, "event_log.csv"), "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["timestamp", "entity_type", "entity_id", "event_type", "details"]
            row1 = next(reader)
            assert row1[1] == "Dispatcher"
            details = json.loads(row1[4]) 
            assert details["order"] == "O1"
            assert details["dist"] == "2"

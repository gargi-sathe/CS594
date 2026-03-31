import os
import tempfile
import json
import csv
import pytest
from src.config.schema import RunConfig, MapConfig, EntitiesConfig, ParametersConfig, PoliciesConfig, CostsConfig, OutputsConfig
from src.core.engine import run_simulation
from src.metrics.writer import write_outputs
from src.entities.models import OrderState, DriverState

def test_end_to_end_synthetic():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RunConfig(
            seed=42, mode="synthetic", run_horizon_mins=60.0,
            map=MapConfig("grid", [10, 10]),
            entities=EntitiesConfig(2, 5), # 2 WH, 5 Drivers
            parameters=ParametersConfig(15.0, 2.0, 2.0), # 2 orders/hr ensures drivers sit IDLE eventually
            policies=PoliciesConfig("centroid", "nearest"),
            costs=CostsConfig(100.0, 20.0, 2.0),
            outputs=OutputsConfig(tmpdir)
        )
        
        # 1. Run holistic simulation
        state = run_simulation(cfg)
        
        # 2. Extract bounding passive mappings
        write_outputs(state, cfg, run_id="e2e_run")
        run_dir = os.path.join(tmpdir, "e2e_run")
        
        # 3. Output artifact generation explicit assertions
        assert os.path.exists(os.path.join(run_dir, "run_summary.json"))
        assert os.path.exists(os.path.join(run_dir, "event_log.csv"))
        assert os.path.exists(os.path.join(run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(run_dir, "config_snapshot.yaml"))
        assert os.path.exists(os.path.join(run_dir, "map_plot.png"))
        
        # 4. JSON-Parsing assertions against CSV row columns manually
        with open(os.path.join(run_dir, "event_log.csv"), "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == ["timestamp", "entity_type", "entity_id", "event_type", "details"]
            for row in reader:
                # Every details cell must map correctly directly onto valid dictionary boundaries
                details = row[4]
                try:
                    parsed = json.loads(details)
                    assert isinstance(parsed, dict)
                except json.JSONDecodeError:
                    pytest.fail(f"Invalid JSON string encoded securely inside event details schema mapping logs. Found: {details}")
                    
        # 5. Core state-flow logic sanity validations directly
        delivered = [o for o in state.orders.values() if o.state == OrderState.DELIVERED]
        assert len(delivered) > 0, "Failed: Minimum one order MUST reach DELIVERED state inside normal bounds."
        
        returned = [d for d in state.drivers.values() if d.state == DriverState.IDLE]
        assert len(returned) > 0, "Failed: Baseline limits require at least one driver to naturally conclude loops returning natively to IDLE statuses."
        
        # 6. JSON output logic bounding 
        with open(os.path.join(run_dir, "metrics.json"), "r", encoding="utf-8") as f:
            metrics = json.load(f)
            assert metrics["delivered_count"] >= 0     
            assert 0.0 <= metrics["on_time_delivery_rate"] <= 1.0
            assert 0.0 <= metrics["delivered_success_rate"] <= 1.0     
            assert metrics["orders_missed_or_unserved"] >= 0

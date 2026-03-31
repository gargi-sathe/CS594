import os
import tempfile
import pytest
from src.config.schema import RunConfig, MapConfig, EntitiesConfig, ParametersConfig, PoliciesConfig, CostsConfig, OutputsConfig
from src.core.engine import run_simulation
from src.metrics.writer import write_outputs

def test_end_to_end_real_map():
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = RunConfig(
            seed=42, mode="real_map", run_horizon_mins=30.0,
            map=MapConfig("osmnx", [0, 0], "Piedmont, California, USA"),
            entities=EntitiesConfig(1, 2), # 1 WH, 2 Drivers
            parameters=ParametersConfig(15.0, 2.0, 5.0), # small volume testing stable layouts
            policies=PoliciesConfig("centroid", "nearest"),
            costs=CostsConfig(100.0, 20.0, 2.0),
            outputs=OutputsConfig(tmpdir)
        )
        
        state = run_simulation(cfg)
        write_outputs(state, cfg, run_id="real_test")
        run_dir = os.path.join(tmpdir, "real_test")
        
        # Verify basic files output successfully exactly like synthetic mode
        assert os.path.exists(os.path.join(run_dir, "run_summary.json"))
        assert os.path.exists(os.path.join(run_dir, "event_log.csv"))
        assert os.path.exists(os.path.join(run_dir, "metrics.json"))
        assert os.path.exists(os.path.join(run_dir, "config_snapshot.yaml"))
        assert os.path.exists(os.path.join(run_dir, "map_plot.png"))

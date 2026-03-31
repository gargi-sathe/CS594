import matplotlib.pyplot as plt
import networkx as nx
import os
from src.core.state import SimulatorState
from src.config.schema import RunConfig

def plot_synthetic_run(state: SimulatorState, config: RunConfig, run_id: str, output_dir: str):
    """Generates an explicit static standalone visualization snapshot cleanly bounding state values."""
    G = state.graph
    
    if config.map.type == "grid":
        w, h = config.map.grid_size
        # Synthesize topological constraints onto simple bounded grid dimensions naturally
        pos = {n: (n % w, n // w) for n in G.nodes()}
    elif config.map.type == "osmnx":
        pos = {n: (data.get('x', 0), data.get('y', 0)) for n, data in G.nodes(data=True)}
    else:
        return
    
    plt.figure(figsize=(10, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.3)
    nx.draw_networkx_nodes(G, pos, node_size=50, node_color='lightgray', alpha=0.5)
    
    # 1. Orders (Red circles)
    order_nodes = [o.location for o in state.orders.values()]
    if order_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=order_nodes, node_size=80, node_color='red', node_shape='o', label='Orders')
        
    # 2. Drivers (Green triangles) natively at final boundary states (Return-to-stage IDs)
    driver_nodes = [d.current_location for d in state.drivers.values()]
    if driver_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=driver_nodes, node_size=120, node_color='green', node_shape='^', label='Drivers')
        
    # 3. Warehouses (Blue squares) dynamically on highest explicit z-order
    warehouse_nodes = [w.location for w in state.warehouses.values()]
    if warehouse_nodes:
        nx.draw_networkx_nodes(G, pos, nodelist=warehouse_nodes, node_size=200, node_color='blue', node_shape='s', label='Warehouses')
        
    plt.title(f"UrbanScale Simulation\nRun: {run_id} | Mode: {config.mode}")
    
    # Prune legend deduplication arrays efficiently
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        plt.legend(by_label.values(), by_label.keys(), loc='upper right')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "map_plot.png"), dpi=150)
    plt.close()

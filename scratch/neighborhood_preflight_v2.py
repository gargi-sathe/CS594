import osmnx as ox
import pandas as pd

zones = [
    "Downtown, Chicago, Illinois, USA",
    "Lincoln Park, Chicago, Illinois, USA",
    "Near North Side, Chicago, Illinois, USA",
    "Near West Side, Chicago, Illinois, USA",
    "Hyde Park, Chicago, Illinois, USA"
]

results = []
for zone in zones:
    print(f"Checking {zone}...")
    try:
        G = ox.graph_from_place(zone, network_type='drive')
        results.append({
            "zone": zone,
            "nodes": len(G.nodes),
            "edges": len(G.edges),
            "status": "SUCCESS"
        })
    except Exception as e:
        results.append({
            "zone": zone,
            "status": f"FAILED: {str(e)}"
        })

df = pd.DataFrame(results)
print("\n# Chicago Neighborhood Preflight (Refined)")
print(df.to_markdown())

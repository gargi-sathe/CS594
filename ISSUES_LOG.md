# Issues Log

*This log records actively tracked limitations explicitly bounding the Phase 0 simulator engine architecture.*

## Current Known Limitations
- **OSMnx Graph Simplicity**: Currently `MultiDiGraphs` natively imported over authentic street mappings are actively flattened cleanly into simple `nx.Graph()` components safely retaining distance bounds `length`. Highly complex multi-directional routing structures remain fully abstracted identically handling symmetrical distances.
- **Batched Deliveries**: The framework limits absolute single-dispatch architectures precisely mapping one payload per active transport agent bounds seamlessly. Delivery payloads wrapping multi-stage driver allocation logic mapping natively is out-of-scope for Phase 0 execution wrappers explicitly.
- **Event Logger Structuring Limits**: Simulator metrics strictly bundle nested attributes mapping rigidly back sequentially internally. Native explicit dictionary boundaries inside `log_event()` remain intentionally deferred, utilizing post-simulation string parser limits exactly on execution completion traces.

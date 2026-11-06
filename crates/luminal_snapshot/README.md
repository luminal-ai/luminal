# Luminal compiler snapshots

Core-owned compact e-graph storage and export, shared by Luminal backends.
`luminal::snapshot` re-exports this API; backend code normally imports
`CompactSnapshotExt` and `egraph_serialize` from `luminal::prelude`.

`serialize_compact_filtered` exports numeric node/class identities, shared
operand storage and lazily cached class/operator indexes. The filter selects
function tables. Diagnostic names remain available through `node_label` and
`class_label`; numeric IDs are local to a snapshot, not persistent egglog IDs.

`refresh_compact_subsumption` updates only subsumption flags after
validating the originating graph and table versions. A structural change,
union, fork or changed projection requires a fresh export. The legacy egglog
serializer remains available for diagnostic JSON/HTML and differential tests.

The serializer uses egglog's generic read-only `SnapshotReader`; no Luminal
storage or compiler policy lives in egglog. This small crate is separate from
the main library so its public types are identical across backend dependencies
and the core library's cyclic development dependencies.

## Backend migration

Replace `egglog.serialize(SerializeConfig::default()).egraph` with
`egglog.serialize_compact_filtered(|_| true)`. Existing extraction functions,
`ExtractionSite`, and `EGraphView::new` consume these snapshots. `EGraphView` is
`Clone` rather than `Copy`; its clones share caches scoped to that immutable
snapshot. Build a fresh view after editing a graph.

The prelude's `egraph_serialize` alias now names the compact types. Use
`legacy_egraph_serialize` for existing diagnostic JSON; `compact::EGraph::from`
converts a legacy graph when needed. IDs are newtypes, operators are shared
strings, and `Children` accepts a `Vec<NodeId>` through `.into()`. Graph indexes
are lazy: structural edits to public maps must call `invalidate_indexes()`
before further reads, or construct a fresh graph as the memory-pruning passes do.

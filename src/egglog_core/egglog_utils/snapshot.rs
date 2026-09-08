//! A LUMINAL-OWNED SNAPSHOT of a live e-graph, read through egglog's
//! public read API (`functions_iter` for the schema, `constructor_enodes`
//! / `function_entries` for the rows) instead of `EGraph::serialize`.
//!
//! The shape is the database's, not a term DAG's: TABLES hold ROWS, and
//! every column of a row — an e-class, a literal, a container — interns to
//! a dense [`ClassId`]. Nothing is rendered to a string and nothing is
//! re-parsed, so a reader asks the snapshot what egglog stored rather than
//! what a serializer chose to print, and a literal comes back as the Rust
//! value it is.
//!
//! WHAT IS NOT DECODED: a container value (our one `(sort SigmaMap (Map
//! IntExpr IntExpr))`) is kept as the raw `Value`, the identity egglog
//! itself uses, and so is a base sort this module has no decoder for. The
//! serializer expands a container into inner nodes; nothing here needs
//! that, and inventing structure we do not read is how a reader starts
//! disagreeing with the database.
//!
//! Values stored in a table are canonical after a run returns, so a
//! column's `Value` is interned as-is; the debug tripwire checks that
//! against egglog's own `value_to_class_id` on a sample of each table.

use crate::prelude::FxHashMap;
use anyhow::Result;
use egglog::sort::{F, S, Z};
use egglog::{ArcSort, Core, EGraph, Read, ReadState, Value};
use std::ops::Range;

// =============================================================================
// Ids
// =============================================================================

/// A dense class id, interned per (sort, `Value`) in build order over
/// EVERY class: e-classes, literals and containers alike.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClassId(u32);

/// A row's index into [`Snapshot::nodes`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NodeId(u32);

/// A table's index into [`Snapshot::tables`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TableId(u32);

macro_rules! dense_id {
    ($name:ident) => {
        impl $name {
            pub fn index(self) -> usize {
                self.0 as usize
            }
            fn from_len(len: usize) -> Self {
                Self(u32::try_from(len).expect("snapshot fits in u32 ids"))
            }
        }
    };
}
dense_id!(ClassId);
dense_id!(NodeId);
dense_id!(TableId);

// =============================================================================
// The snapshot
// =============================================================================

/// How a table is declared. Egglog has exactly two subtypes: a
/// `(relation ...)` desugars to a constructor over a fresh non-unionable
/// sort, so its rows arrive on the constructor path and its output column
/// is an ordinary class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TableKind {
    Constructor,
    Function,
}

#[derive(Debug, Clone)]
pub struct Table {
    pub name: String,
    pub kind: TableKind,
    /// Declared sort name per input column.
    pub inputs: Vec<String>,
    /// Declared sort name of the output column.
    pub output: String,
    /// This table's rows, as a contiguous range of [`NodeId`]s.
    pub rows: Range<u32>,
}

/// A decoded base value. The Rust types are egglog's own, so a literal
/// prints exactly as egglog prints it.
#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    I64(i64),
    F64(f64),
    Bool(bool),
    BigInt(Z),
    Str(String),
}

/// What a class holds. A literal and a container are classes with no rows,
/// so every column of every row is one id.
#[derive(Debug, Clone)]
pub enum ClassBody {
    /// An e-class: the rows whose output column is this class, in row order.
    Eq { nodes: Vec<NodeId> },
    /// One base value, interned per (sort, value) like any other class.
    Lit(Literal),
    /// A value this reader does not decode: a container, or a base sort
    /// with no decoder here.
    Container(Value),
}

#[derive(Debug, Clone)]
pub struct Class {
    pub sort: String,
    pub body: ClassBody,
    /// Global `(let ...)` names bound to this class, in declaration order.
    pub let_names: Vec<String>,
}

impl Class {
    pub fn literal(&self) -> Option<&Literal> {
        match &self.body {
            ClassBody::Lit(literal) => Some(literal),
            _ => None,
        }
    }
}

/// One row: a table, its input columns, its output column.
#[derive(Debug, Clone)]
pub struct Node {
    pub table: TableId,
    pub children: Vec<ClassId>,
    pub output: ClassId,
    pub subsumed: bool,
}

#[derive(Debug, Clone)]
pub struct Snapshot {
    pub tables: Vec<Table>,
    pub nodes: Vec<Node>,
    pub classes: Vec<Class>,
}

impl Snapshot {
    /// Read the e-graph's current contents. Let-binding tables contribute
    /// their names to the bound class and no rows; every other table —
    /// hidden ones included, as egglog's own serializer does — contributes
    /// its rows.
    pub fn build(egraph: &EGraph) -> Result<Self> {
        Ok(Self::read(egraph)?.1.snapshot)
    }

    fn read(egraph: &EGraph) -> Result<(Schema, Built)> {
        let schema = Schema::collect(egraph);
        let built = egraph.read(|state| Builder::run(&state, &schema))?;
        #[cfg(debug_assertions)]
        built.tripwire.check(egraph, &schema, &built.snapshot)?;
        Ok((schema, built))
    }

    /// TEST ONLY: the snapshot, plus egglog's own serialized class id per
    /// class. Interning is keyed by exactly the `(sort, value)` pair that
    /// `value_to_class_id` names, so this is the correspondence to the
    /// serializer's output — no ordering assumed anywhere.
    #[cfg(test)]
    fn build_named(egraph: &EGraph) -> Result<(Self, Vec<egraph_serialize::ClassId>)> {
        let (schema, built) = Self::read(egraph)?;
        let names = built
            .class_values
            .iter()
            .map(|(sort, value)| {
                egraph.value_to_class_id(&schema.arc_sorts[*sort as usize], *value)
            })
            .collect();
        Ok((built.snapshot, names))
    }

    pub fn table(&self, name: &str) -> Option<TableId> {
        self.tables
            .iter()
            .position(|table| table.name == name)
            .map(TableId::from_len)
    }

    pub fn table_at(&self, table: TableId) -> &Table {
        &self.tables[table.index()]
    }

    pub fn rows_of(&self, table: TableId) -> &[Node] {
        let rows = &self.tables[table.index()].rows;
        &self.nodes[rows.start as usize..rows.end as usize]
    }

    pub fn node(&self, node: NodeId) -> &Node {
        &self.nodes[node.index()]
    }

    pub fn class(&self, class: ClassId) -> &Class {
        &self.classes[class.index()]
    }

    /// The rows of an e-class; empty for a literal or container class,
    /// which hold none.
    pub fn nodes_of(&self, class: ClassId) -> &[NodeId] {
        match &self.classes[class.index()].body {
            ClassBody::Eq { nodes } => nodes,
            _ => &[],
        }
    }

    pub fn literal(&self, class: ClassId) -> Option<&Literal> {
        self.classes[class.index()].literal()
    }

    /// The name of the table `node`'s row came from — a row's operator.
    pub fn op(&self, node: NodeId) -> &str {
        &self.tables[self.nodes[node.index()].table.index()].name
    }
}

// =============================================================================
// Schema: every column resolved once, so the row scan does no sort work
// =============================================================================

/// Index into [`Schema::sorts`].
type SortId = u32;

/// How one column's `Value` is read.
#[derive(Clone, Copy)]
struct Column {
    sort: SortId,
    kind: ColumnKind,
}

#[derive(Clone, Copy)]
enum ColumnKind {
    Eq,
    Base(BaseKind),
    /// A container sort, or a base sort with no decoder here.
    Opaque,
}

#[derive(Clone, Copy)]
enum BaseKind {
    I64,
    F64,
    Bool,
    BigInt,
    Str,
}

struct TableSchema {
    name: String,
    kind: TableKind,
    inputs: Vec<Column>,
    output: Column,
    input_sorts: Vec<String>,
    output_sort: String,
    let_binding: bool,
}

struct Schema {
    tables: Vec<TableSchema>,
    sorts: Vec<String>,
    /// Parallel to `sorts`, for `value_to_class_id`.
    arc_sorts: Vec<ArcSort>,
}

impl Schema {
    fn collect(egraph: &EGraph) -> Self {
        let mut schema = Schema {
            tables: Vec::new(),
            sorts: Vec::new(),
            arc_sorts: Vec::new(),
        };
        let mut ids: FxHashMap<String, SortId> = FxHashMap::default();
        for (_, function) in egraph.functions_iter() {
            let func_type = function.func_type();
            let mut column = |sort: &ArcSort| -> Column {
                let next = SortId::try_from(schema.sorts.len()).expect("sorts fit in u32");
                let id = *ids.entry(sort.name().to_string()).or_insert_with(|| {
                    schema.sorts.push(sort.name().to_string());
                    schema.arc_sorts.push(sort.clone());
                    next
                });
                let kind = if sort.is_eq_sort() {
                    ColumnKind::Eq
                } else if sort.is_container_sort() {
                    ColumnKind::Opaque
                } else {
                    match sort.name() {
                        "i64" => ColumnKind::Base(BaseKind::I64),
                        "f64" => ColumnKind::Base(BaseKind::F64),
                        "bool" => ColumnKind::Base(BaseKind::Bool),
                        "BigInt" => ColumnKind::Base(BaseKind::BigInt),
                        "String" => ColumnKind::Base(BaseKind::Str),
                        _ => ColumnKind::Opaque,
                    }
                };
                Column { sort: id, kind }
            };
            let inputs: Vec<Column> = func_type.input.iter().map(&mut column).collect();
            let output = column(&func_type.output);
            schema.tables.push(TableSchema {
                name: function.name().to_string(),
                kind: match func_type.subtype {
                    egglog::ast::FunctionSubtype::Constructor => TableKind::Constructor,
                    egglog::ast::FunctionSubtype::Custom => TableKind::Function,
                },
                inputs,
                output,
                input_sorts: func_type
                    .input
                    .iter()
                    .map(|s| s.name().to_string())
                    .collect(),
                output_sort: func_type.output.name().to_string(),
                let_binding: function.is_let_binding(),
            });
        }
        schema
    }
}

// =============================================================================
// The row scan
// =============================================================================

/// (sort, value) samples taken while scanning, checked against egglog's
/// own class ids once the read is over.
#[derive(Default)]
struct Tripwire {
    #[cfg(debug_assertions)]
    samples: Vec<(ClassId, SortId, Value)>,
}

struct Built {
    snapshot: Snapshot,
    #[cfg_attr(not(debug_assertions), allow(dead_code))]
    tripwire: Tripwire,
    /// The `(sort, value)` pair behind each class, in `ClassId` order.
    #[cfg(test)]
    class_values: Vec<(SortId, Value)>,
}

struct Builder<'s> {
    schema: &'s Schema,
    snapshot: Snapshot,
    classes: FxHashMap<(SortId, Value), ClassId>,
    tripwire: Tripwire,
    #[cfg(test)]
    class_values: Vec<(SortId, Value)>,
    /// Rows sampled from the table being scanned.
    sampled: usize,
}

/// Rows sampled per table for the debug tripwire.
#[cfg(debug_assertions)]
const SAMPLE_ROWS: usize = 64;

impl<'s> Builder<'s> {
    fn run(state: &ReadState<'_, '_>, schema: &'s Schema) -> Result<Built> {
        let mut builder = Builder {
            schema,
            snapshot: Snapshot {
                tables: Vec::with_capacity(schema.tables.len()),
                nodes: Vec::new(),
                classes: Vec::new(),
            },
            classes: FxHashMap::default(),
            tripwire: Tripwire::default(),
            #[cfg(test)]
            class_values: Vec::new(),
            sampled: 0,
        };
        for (index, table) in schema.tables.iter().enumerate() {
            let id = TableId::from_len(index);
            builder.sampled = 0;
            let start = u32::try_from(builder.snapshot.nodes.len()).expect("rows fit in u32");
            if table.let_binding {
                builder.scan(state, table, |b, _, output, _| {
                    b.bind_let(state, table, output)
                })?;
            } else {
                builder.scan(state, table, |b, children, output, subsumed| {
                    b.push_row(state, id, table, children, output, subsumed)
                })?;
            }
            let end = u32::try_from(builder.snapshot.nodes.len()).expect("rows fit in u32");
            builder.snapshot.tables.push(Table {
                name: table.name.clone(),
                kind: table.kind,
                inputs: table.input_sorts.clone(),
                output: table.output_sort.clone(),
                rows: start..end,
            });
        }
        Ok(Built {
            snapshot: builder.snapshot,
            tripwire: builder.tripwire,
            #[cfg(test)]
            class_values: builder.class_values,
        })
    }

    /// Walk one table's rows, handing each `(inputs, output, subsumed)` to
    /// `row`. The subtype decides which read method the rows come from.
    fn scan(
        &mut self,
        state: &ReadState<'_, '_>,
        table: &TableSchema,
        mut row: impl FnMut(&mut Self, &[Value], Value, bool),
    ) -> Result<()> {
        let result = match table.kind {
            TableKind::Constructor => state.constructor_enodes_while(&table.name, |enode| {
                row(self, enode.children, enode.eclass, enode.subsumed);
                true
            }),
            TableKind::Function => state.function_entries_while(&table.name, |entry| {
                row(self, entry.inputs, entry.output, entry.subsumed);
                true
            }),
        };
        result.map_err(|err| anyhow::anyhow!("reading table `{}`: {err}", table.name))
    }

    fn push_row(
        &mut self,
        state: &ReadState<'_, '_>,
        id: TableId,
        table: &TableSchema,
        children: &[Value],
        output: Value,
        subsumed: bool,
    ) {
        let node = NodeId::from_len(self.snapshot.nodes.len());
        let children: Vec<ClassId> = table
            .inputs
            .iter()
            .zip(children)
            .map(|(column, value)| self.intern(state, *column, *value))
            .collect();
        let output = self.intern(state, table.output, output);
        if let ClassBody::Eq { nodes } = &mut self.snapshot.classes[output.index()].body {
            nodes.push(node);
        }
        self.snapshot.nodes.push(Node {
            table: id,
            children,
            output,
            subsumed,
        });
        self.sampled += 1;
    }

    /// A let-binding table is one nullary row whose output is the bound
    /// value; the name lands on that class and the row itself is not a node.
    fn bind_let(&mut self, state: &ReadState<'_, '_>, table: &TableSchema, output: Value) {
        let class = self.intern(state, table.output, output);
        self.snapshot.classes[class.index()]
            .let_names
            .push(table.name.clone());
    }

    fn intern(&mut self, state: &ReadState<'_, '_>, column: Column, value: Value) -> ClassId {
        let classes = &mut self.snapshot.classes;
        let sorts = &self.schema.sorts;
        #[cfg(test)]
        let values = &mut self.class_values;
        let id = *self.classes.entry((column.sort, value)).or_insert_with(|| {
            let id = ClassId::from_len(classes.len());
            classes.push(Class {
                sort: sorts[column.sort as usize].clone(),
                body: match column.kind {
                    ColumnKind::Eq => ClassBody::Eq { nodes: Vec::new() },
                    ColumnKind::Base(kind) => ClassBody::Lit(decode_base(state, kind, value)),
                    ColumnKind::Opaque => ClassBody::Container(value),
                },
                let_names: Vec::new(),
            });
            #[cfg(test)]
            values.push((column.sort, value));
            id
        });
        #[cfg(debug_assertions)]
        if self.sampled < SAMPLE_ROWS {
            self.tripwire.samples.push((id, column.sort, value));
        }
        id
    }
}

fn decode_base(state: &ReadState<'_, '_>, kind: BaseKind, value: Value) -> Literal {
    match kind {
        BaseKind::I64 => Literal::I64(state.value_to_base::<i64>(value)),
        BaseKind::F64 => Literal::F64(state.value_to_base::<F>(value).into_inner().into_inner()),
        BaseKind::Bool => Literal::Bool(state.value_to_base::<bool>(value)),
        BaseKind::BigInt => Literal::BigInt(state.value_to_base::<Z>(value)),
        BaseKind::Str => Literal::Str(state.value_to_base::<S>(value).into_inner()),
    }
}

// =============================================================================
// The debug tripwire
// =============================================================================

impl Tripwire {
    /// Sampled columns agree with egglog's own class ids — one class id
    /// string per interned class, and no two of our classes sharing one —
    /// every e-class holds at least one row, and an e-class whose rows are
    /// all subsumed is a `LayoutTensorOp` (the preamble subsumes an op's
    /// `Lit` without unioning it, and that row is the only carrier of the
    /// op's operand lists; any other sort with no live row means a rule
    /// subsumed something it never replaced).
    #[cfg(debug_assertions)]
    fn check(&self, egraph: &EGraph, schema: &Schema, snapshot: &Snapshot) -> Result<()> {
        use anyhow::bail;

        let mut ours: FxHashMap<ClassId, String> = FxHashMap::default();
        let mut theirs: FxHashMap<String, ClassId> = FxHashMap::default();
        for (class, sort, value) in &self.samples {
            let id = egraph
                .value_to_class_id(&schema.arc_sorts[*sort as usize], *value)
                .to_string();
            if let Some(seen) = ours.insert(*class, id.clone())
                && seen != id
            {
                bail!("class {class:?} interned two egglog class ids: {seen} and {id}");
            }
            if let Some(seen) = theirs.insert(id.clone(), *class)
                && seen != *class
            {
                bail!("egglog class {id} interned as both {seen:?} and {class:?}");
            }
        }
        for (index, class) in snapshot.classes.iter().enumerate() {
            let ClassBody::Eq { nodes } = &class.body else {
                continue;
            };
            if nodes.is_empty() {
                bail!(
                    "class {index} of sort {} holds no row (let names: {:?})",
                    class.sort,
                    class.let_names
                );
            }
            if class.sort != "LayoutTensorOp"
                && nodes.iter().all(|node| snapshot.node(*node).subsumed)
            {
                let ops: Vec<&str> = nodes.iter().map(|node| snapshot.op(*node)).collect();
                bail!(
                    "class {index} of sort {} has no live row: {ops:?}",
                    class.sort
                );
            }
        }
        Ok(())
    }
}

// =============================================================================
// The differential: this snapshot vs egglog's own serializer
// =============================================================================

#[cfg(test)]
mod differential {
    use super::*;
    use egraph_serialize as ser;
    use std::collections::{BTreeMap, HashMap};

    /// Run a `test_scripts` fixture under the reference program.
    fn saturate(script: &str) -> EGraph {
        let path = format!(
            "{}/src/egglog_core/test_scripts/{script}",
            env!("CARGO_MANIFEST_DIR")
        );
        let source = std::fs::read_to_string(&path).unwrap_or_else(|_| panic!("{path} readable"));
        let program = format!("{}\n\n{source}", luminal_reference::assembled_program());
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(Some(script.to_string()), &program)
            .unwrap_or_else(|err| panic!("{script} saturates: {err}"));
        egraph
    }

    /// A recorded graph under the reference runtime's own schedule — the
    /// strata a fixture script's schedule leaves out, `cleanup` among
    /// them, where an input producer's `LayoutTensorOpLit` is subsumed
    /// without being unioned away.
    fn saturate_model() -> EGraph {
        let mut graph = luminal::prelude::Graph::new();
        let x = graph.tensor((2, 3), luminal::prelude::DType::F32);
        let y = graph.tensor((2, 3), luminal::prelude::DType::F32);
        let _ = (x * y + x).sum(1).output();
        let (pre_schedule, _inputs, _outputs, post_checks, _labeled) = graph
            .logical
            .bound_parts(&luminal_reference::bindings::ReferenceBindings)
            .expect("the recorded graph binds");
        let program = format!(
            "{}\n\n{pre_schedule}{}{post_checks}",
            luminal_reference::assembled_program(),
            luminal_reference::bindings::ReferenceBindings::SCHEDULE
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &program)
            .expect("the model saturates");
        egraph
    }

    /// How egglog prints a base value: the base type's own `Debug`.
    fn spelling(literal: &Literal) -> String {
        match literal {
            Literal::I64(value) => format!("{value:?}"),
            Literal::F64(value) => format!("{value:?}"),
            Literal::Bool(value) => format!("{value:?}"),
            Literal::BigInt(value) => format!("{value:?}"),
            Literal::Str(value) => format!("{value:?}"),
        }
    }

    /// The op of the node egglog wrote for a class's own value: what it
    /// printed for a literal, the sort's serialized name for a container.
    fn printed(
        serialized: &ser::EGraph,
        egraph: &EGraph,
        class: &ser::ClassId,
        script: &str,
    ) -> String {
        let mut values = serialized.classes()[class].nodes.iter().filter(|node| {
            matches!(
                egraph.from_node_id(node),
                egglog::SerializedNode::Primitive(_)
            )
        });
        let node = values
            .next()
            .unwrap_or_else(|| panic!("{script}: class {class} has no value node"));
        assert!(
            values.next().is_none(),
            "{script}: class {class} has more than one value node"
        );
        serialized.nodes[node].op.clone()
    }

    /// One row, spelled entirely in the serializer's vocabulary:
    /// operator, columns, output, subsumption.
    type Row = (String, Vec<ser::ClassId>, ser::ClassId, bool);

    /// What one comparison found, for the failure message.
    struct Census {
        rows: usize,
        classes: usize,
        serialized_nodes: usize,
        serialized_classes: usize,
        dummies: usize,
    }

    fn compare(script: &str, egraph: &EGraph) -> Census {
        let (snapshot, names) = Snapshot::build_named(egraph).expect("snapshot builds");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;

        // (a) The renaming is a bijection over EVERY class on both sides.
        let mut backward: HashMap<&ser::ClassId, ClassId> = HashMap::new();
        for (index, name) in names.iter().enumerate() {
            let ours = ClassId::from_len(index);
            assert!(
                backward.insert(name, ours).is_none(),
                "{script}: two classes share the serialized id {name}"
            );
        }
        for class in serialized.class_data.keys() {
            assert!(
                backward.contains_key(class),
                "{script}: serialized class {class} has no counterpart"
            );
        }

        // (b) The same rows in the same class, on both sides. Ours are
        // grouped by the e-class that lists them, and the rows whose
        // output is a literal or a container — which no class lists — by
        // the class their output names.
        let row = |node: &Node| -> Row {
            (
                snapshot.table_at(node.table).name.clone(),
                node.children
                    .iter()
                    .map(|child| names[child.index()].clone())
                    .collect(),
                names[node.output.index()].clone(),
                node.subsumed,
            )
        };
        let mut ours: BTreeMap<ser::ClassId, Vec<Row>> = BTreeMap::new();
        for (index, class) in snapshot.classes.iter().enumerate() {
            if let ClassBody::Eq { nodes } = &class.body {
                let rows: Vec<Row> = nodes.iter().map(|n| row(snapshot.node(*n))).collect();
                ours.insert(names[index].clone(), rows);
            }
        }
        for node in &snapshot.nodes {
            if !matches!(snapshot.class(node.output).body, ClassBody::Eq { .. }) {
                ours.entry(names[node.output.index()].clone())
                    .or_default()
                    .push(row(node));
            }
        }
        let mut theirs: BTreeMap<ser::ClassId, Vec<Row>> = BTreeMap::new();
        let mut dummies = 0;
        for (class, nodes) in serialized.classes() {
            for id in &nodes.nodes {
                let node = &serialized.nodes[id];
                match egraph.from_node_id(id) {
                    egglog::SerializedNode::Function { .. } => {
                        theirs.entry(class.clone()).or_default().push((
                            node.op.clone(),
                            node.children
                                .iter()
                                .map(|child| serialized.nodes[child].eclass.clone())
                                .collect(),
                            node.eclass.clone(),
                            node.subsumed,
                        ))
                    }
                    egglog::SerializedNode::Primitive(_) => {}
                    _ => dummies += 1,
                }
            }
        }
        let missing: Vec<&ser::ClassId> = theirs
            .keys()
            .filter(|class| !ours.contains_key(*class))
            .take(4)
            .collect();
        let extra: Vec<&ser::ClassId> = ours
            .keys()
            .filter(|class| !theirs.contains_key(*class))
            .take(4)
            .collect();
        assert!(
            missing.is_empty() && extra.is_empty(),
            "{script}: classes only in the serializer {missing:?}, only here {extra:?}"
        );
        for (class, rows) in &mut ours {
            rows.sort();
            let theirs = theirs.get_mut(class).expect("class on both sides");
            theirs.sort();
            assert_eq!(rows, theirs, "{script}: rows of class {class}");
        }

        for (index, class) in snapshot.classes.iter().enumerate() {
            let name = &names[index];
            let data = &serialized.class_data[name];
            // (c) the same let names, in the same order.
            let ours_let = class.let_names.join(", ");
            assert_eq!(
                data.extra.get("let").map(String::as_str),
                (!ours_let.is_empty()).then_some(ours_let.as_str()),
                "{script}: let names of class {name}"
            );
            // (d) the same sort, a body that sort agrees with, and a
            // literal holding the value egglog printed there.
            assert_eq!(
                data.typ.as_deref(),
                Some(class.sort.as_str()),
                "{script}: sort of class {name}"
            );
            let sort = egraph
                .get_sort_by_name(&class.sort)
                .unwrap_or_else(|| panic!("{script}: sort {} is declared", class.sort));
            match &class.body {
                ClassBody::Eq { .. } => assert!(
                    sort.is_eq_sort(),
                    "{script}: {} is not an eq sort",
                    class.sort
                ),
                ClassBody::Lit(literal) => {
                    assert!(
                        !sort.is_eq_sort() && !sort.is_container_sort(),
                        "{script}: {} is not a base sort",
                        class.sort
                    );
                    assert_eq!(
                        spelling(literal),
                        printed(&serialized, egraph, name, script),
                        "{script}: value of class {name}"
                    );
                }
                ClassBody::Container(_) => assert!(
                    sort.is_container_sort(),
                    "{script}: {} is not a container sort",
                    class.sort
                ),
            }
        }

        Census {
            rows: snapshot.nodes.len(),
            classes: snapshot.classes.len(),
            serialized_nodes: serialized.nodes.len(),
            serialized_classes: serialized.class_data.len(),
            dummies,
        }
    }

    /// The snapshot read out of egglog holds exactly what egglog's own
    /// serializer holds — same rows per class, same operators, same
    /// columns, same subsumption, same literal values, same let names,
    /// same sorts — under the renaming egglog's own `value_to_class_id`
    /// gives each class.
    #[test]
    fn snapshot_matches_the_serializer_up_to_renaming() {
        let check = |script: &str, egraph: EGraph| {
            let census = compare(script, &egraph);
            assert_eq!(
                census.dummies,
                0,
                "{script}: the serializer emitted {} placeholder nodes ({} rows, {} classes; \
                 serialized {} nodes, {} classes)",
                census.dummies,
                census.rows,
                census.classes,
                census.serialized_nodes,
                census.serialized_classes
            );
        };
        for script in [
            "boundary_scalar.egg",
            "boundary_scatter.egg",
            "boundary_gather.egg",
            "transformer.egg",
        ] {
            check(script, saturate(script));
        }
        check("recorded model", saturate_model());
    }
}

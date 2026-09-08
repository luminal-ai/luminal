//! A LUMINAL-OWNED SNAPSHOT of a live e-graph, read through egglog's
//! public read API (`functions_iter` for the schema, `constructor_enodes`
//! / `function_entries` for the rows) instead of `EGraph::serialize`.
//!
//! The shape is the database's, not a term DAG's: TABLES hold ROWS, a
//! row's columns are decoded by their DECLARED sort, and the eq-sort
//! columns intern to dense [`ClassId`]s. Nothing is rendered to a string
//! and nothing is re-parsed, so a reader asks the snapshot what egglog
//! stored rather than what a serializer chose to print.
//!
//! WHAT IS NOT DECODED: container values (our one `(sort SigmaMap (Map
//! IntExpr IntExpr))`) and base sorts this module has no decoder for stay
//! [`Child::Opaque`] — the raw `Value`, which is the identity egglog
//! itself uses. The serializer expands a container into inner nodes;
//! nothing here needs that, and inventing structure we do not read is how
//! a reader starts disagreeing with the database.
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

/// A dense e-class id, interned per (eq-sort, `Value`) in build order.
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

/// One column of a row.
#[derive(Debug, Clone, PartialEq)]
pub enum Child {
    Class(ClassId),
    Lit(Literal),
    /// A value this reader does not decode: a container, or a base sort
    /// with no decoder here.
    Opaque(Value),
}

impl Child {
    pub fn class(&self) -> Option<ClassId> {
        match self {
            Child::Class(id) => Some(*id),
            _ => None,
        }
    }
}

/// One row: a table, its input columns, its output column.
#[derive(Debug, Clone)]
pub struct Node {
    pub table: TableId,
    pub children: Vec<Child>,
    /// `Class` for constructors and for eq-sort-valued functions, `Lit` or
    /// `Opaque` for a function whose output sort is not an eq sort.
    pub output: Child,
    pub subsumed: bool,
}

/// One e-class: every row whose output column is this class, in row order.
#[derive(Debug, Clone)]
pub struct Class {
    pub sort: String,
    pub nodes: Vec<NodeId>,
    /// Global `(let ...)` names bound to this class, in declaration order.
    pub let_names: Vec<String>,
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
        let schema = Schema::collect(egraph);
        let built = egraph.read(|state| Builder::run(&state, &schema))?;
        #[cfg(debug_assertions)]
        built.tripwire.check(egraph, &schema, &built.snapshot)?;
        Ok(built.snapshot)
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

    pub fn nodes_of(&self, class: ClassId) -> &[NodeId] {
        &self.classes[class.index()].nodes
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
enum Column {
    Class(SortId),
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
    /// Parallel to `sorts`, for the debug tripwire's `value_to_class_id`.
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
                if sort.is_eq_sort() {
                    let next = SortId::try_from(schema.sorts.len()).expect("sorts fit in u32");
                    let id = *ids.entry(sort.name().to_string()).or_insert_with(|| {
                        schema.sorts.push(sort.name().to_string());
                        schema.arc_sorts.push(sort.clone());
                        next
                    });
                    Column::Class(id)
                } else if sort.is_container_sort() {
                    Column::Opaque
                } else {
                    match sort.name() {
                        "i64" => Column::Base(BaseKind::I64),
                        "f64" => Column::Base(BaseKind::F64),
                        "bool" => Column::Base(BaseKind::Bool),
                        "BigInt" => Column::Base(BaseKind::BigInt),
                        "String" => Column::Base(BaseKind::Str),
                        _ => Column::Opaque,
                    }
                }
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
}

struct Builder<'s> {
    schema: &'s Schema,
    snapshot: Snapshot,
    classes: FxHashMap<(SortId, Value), ClassId>,
    tripwire: Tripwire,
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
        let children: Vec<Child> = table
            .inputs
            .iter()
            .zip(children)
            .map(|(column, value)| self.decode(state, *column, *value))
            .collect();
        let output = self.decode(state, table.output, output);
        if let Child::Class(class) = output {
            self.snapshot.classes[class.index()].nodes.push(node);
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
        if let Child::Class(class) = self.decode(state, table.output, output) {
            self.snapshot.classes[class.index()]
                .let_names
                .push(table.name.clone());
        }
    }

    fn decode(&mut self, state: &ReadState<'_, '_>, column: Column, value: Value) -> Child {
        match column {
            Column::Class(sort) => Child::Class(self.intern(sort, value)),
            Column::Base(kind) => Child::Lit(decode_base(state, kind, value)),
            Column::Opaque => Child::Opaque(value),
        }
    }

    fn intern(&mut self, sort: SortId, value: Value) -> ClassId {
        let classes = &mut self.snapshot.classes;
        let sorts = &self.schema.sorts;
        let id = *self.classes.entry((sort, value)).or_insert_with(|| {
            let id = ClassId::from_len(classes.len());
            classes.push(Class {
                sort: sorts[sort as usize].clone(),
                nodes: Vec::new(),
                let_names: Vec::new(),
            });
            id
        });
        #[cfg(debug_assertions)]
        if self.sampled < SAMPLE_ROWS {
            self.tripwire.samples.push((id, sort, value));
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
    /// Sampled rows agree with egglog's own class ids — one class id
    /// string per interned class, and no two of our classes sharing one —
    /// every class holds at least one row, and a class whose rows are all
    /// subsumed is a `LayoutTensorOp` (the preamble subsumes an op's `Lit`
    /// without unioning it, and that row is the only carrier of the op's
    /// operand lists; any other sort with no live row means a rule
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
            if class.nodes.is_empty() {
                bail!(
                    "class {index} of sort {} holds no row (let names: {:?})",
                    class.sort,
                    class.let_names
                );
            }
            if class.sort != "LayoutTensorOp"
                && class.nodes.iter().all(|node| snapshot.node(*node).subsumed)
            {
                let ops: Vec<&str> = class.nodes.iter().map(|node| snapshot.op(*node)).collect();
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
    use std::collections::{BTreeSet, HashMap, HashSet};

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

    /// The renaming under test, checked injective in both directions as
    /// it is discovered.
    #[derive(Default)]
    struct Renaming {
        forward: HashMap<ClassId, ser::ClassId>,
        backward: HashMap<ser::ClassId, ClassId>,
    }

    impl Renaming {
        fn link(&mut self, ours: ClassId, theirs: &ser::ClassId, at: &str) {
            if let Some(seen) = self.forward.insert(ours, theirs.clone())
                && &seen != theirs
            {
                panic!("{at}: our class {ours:?} maps to both {seen} and {theirs}");
            }
            if let Some(seen) = self.backward.insert(theirs.clone(), ours)
                && seen != ours
            {
                panic!("{at}: serialized class {theirs} maps to both {seen:?} and {ours:?}");
            }
        }
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

    /// What one fixture's comparison found, for the failure message.
    struct Census {
        rows: usize,
        classes: usize,
        serialized_nodes: usize,
        serialized_classes: usize,
        dummies: usize,
    }

    fn compare(script: &str, egraph: &EGraph) -> Census {
        let snapshot = Snapshot::build(&egraph).expect("snapshot builds");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;

        // Which sort names name e-classes, so the serializer's classes can
        // be split into e-classes and primitive/container pseudo-classes.
        let mut eq_sorts: HashSet<String> = HashSet::new();
        for (_, function) in egraph.functions_iter() {
            let func_type = function.func_type();
            for sort in func_type.input.iter().chain([&func_type.output]) {
                if sort.is_eq_sort() {
                    eq_sorts.insert(sort.name().to_string());
                }
            }
        }

        let mut renaming = Renaming::default();
        // A literal and an undecoded value each stand for one serialized
        // pseudo-class; both correspondences must be functions.
        let mut literals: HashMap<(String, String), ser::ClassId> = HashMap::new();
        let mut opaques: HashMap<Value, ser::ClassId> = HashMap::new();
        let mut column = |renaming: &mut Renaming,
                          child: &Child,
                          theirs: &ser::ClassId,
                          sort: &str,
                          at: &str| {
            let data = serialized
                .class_data
                .get(theirs)
                .unwrap_or_else(|| panic!("{at}: serialized class {theirs} has no class data"));
            assert_eq!(
                data.typ.as_deref(),
                Some(sort),
                "{at}: serialized class {theirs} has the wrong sort"
            );
            match child {
                Child::Class(ours) => {
                    assert!(eq_sorts.contains(sort), "{at}: {sort} is not an eq sort");
                    renaming.link(*ours, theirs, at);
                }
                Child::Lit(literal) => {
                    let spelled = spelling(literal);
                    // The class holds the primitive node for this value.
                    let printed: BTreeSet<&str> = serialized.classes()[theirs]
                        .nodes
                        .iter()
                        .filter(|node| {
                            matches!(
                                egraph.from_node_id(node),
                                egglog::SerializedNode::Primitive(_)
                            )
                        })
                        .map(|node| serialized.nodes[node].op.as_str())
                        .collect();
                    assert!(
                        printed.contains(spelled.as_str()),
                        "{at}: decoded {literal:?} but the serialized class {theirs} prints {printed:?}"
                    );
                    let seen = literals
                        .entry((sort.to_string(), spelled.clone()))
                        .or_insert(theirs.clone());
                    assert_eq!(
                        seen, theirs,
                        "{at}: {sort} literal {spelled} is in two classes"
                    );
                }
                Child::Opaque(value) => {
                    let seen = opaques.entry(*value).or_insert(theirs.clone());
                    assert_eq!(seen, theirs, "{at}: value {value:?} is in two classes");
                }
            }
        };

        // (a) The renaming, seeded row by row: the serializer numbers a
        // table's emitted rows in scan order, which is the order the read
        // API hands them to us.
        for (index, table) in snapshot.tables.iter().enumerate() {
            let id = TableId::from_len(index);
            for (offset, node) in snapshot.rows_of(id).iter().enumerate() {
                let at = &format!("{}[{offset}]", table.name);
                let node_id = egraph.to_node_id(
                    None,
                    egglog::SerializedNode::Function {
                        name: table.name.clone(),
                        offset,
                    },
                );
                let theirs = serialized
                    .nodes
                    .get(&node_id)
                    .unwrap_or_else(|| panic!("{at}: the serializer emitted no row {node_id}"));
                assert_eq!(theirs.op, table.name, "{at}: operator");
                assert_eq!(theirs.subsumed, node.subsumed, "{at}: subsumption");
                assert_eq!(
                    theirs.children.len(),
                    node.children.len(),
                    "{at}: column count"
                );
                column(
                    &mut renaming,
                    &node.output,
                    &theirs.eclass,
                    &table.output,
                    at,
                );
                for (position, (child, their_child)) in
                    node.children.iter().zip(&theirs.children).enumerate()
                {
                    let at = &format!("{at}.{position}");
                    let theirs = &serialized.nodes[their_child].eclass;
                    column(&mut renaming, child, theirs, &table.inputs[position], at);
                }
            }
        }

        // Every row the serializer emitted is a row we read, and no more.
        let their_rows = serialized
            .nodes
            .keys()
            .filter(|node| {
                matches!(
                    egraph.from_node_id(node),
                    egglog::SerializedNode::Function { .. }
                )
            })
            .count();
        assert_eq!(their_rows, snapshot.nodes.len(), "{script}: row count");

        // The renaming is total: every class on either side is named.
        let their_classes: Vec<&ser::ClassId> = serialized
            .class_data
            .iter()
            .filter(|(_, data)| {
                data.typ
                    .as_deref()
                    .is_some_and(|typ| eq_sorts.contains(typ))
            })
            .map(|(class, _)| class)
            .collect();
        for class in &their_classes {
            assert!(
                renaming.backward.contains_key(*class),
                "{script}: serialized class {class} has no counterpart"
            );
        }
        for index in 0..snapshot.classes.len() {
            assert!(
                renaming.forward.contains_key(&ClassId::from_len(index)),
                "{script}: our class {index} has no counterpart"
            );
        }

        let mut dummies = 0;
        for (ours, theirs) in &renaming.forward {
            let class = snapshot.class(*ours);
            // (b) the same rows, by operator and subsumption.
            let mut mine: Vec<(&str, bool)> = snapshot
                .nodes_of(*ours)
                .iter()
                .map(|node| (snapshot.op(*node), snapshot.node(*node).subsumed))
                .collect();
            let mut theirs_rows: Vec<(&str, bool)> = serialized.classes()[theirs]
                .nodes
                .iter()
                .map(|node| {
                    let node = &serialized.nodes[node];
                    (node.op.as_str(), node.subsumed)
                })
                .filter(|(op, _)| {
                    let dummy = *op == "[...]";
                    dummies += usize::from(dummy);
                    !dummy
                })
                .collect();
            mine.sort_unstable();
            theirs_rows.sort_unstable();
            assert_eq!(mine, theirs_rows, "{script}: rows of class {theirs}");

            let data = &serialized.class_data[theirs];
            // (c) the same let names, in the same order.
            let ours_let = class.let_names.join(", ");
            assert_eq!(
                data.extra.get("let").map(String::as_str),
                (!ours_let.is_empty()).then_some(ours_let.as_str()),
                "{script}: let names of class {theirs}"
            );
            // (d) the same sort.
            assert_eq!(
                data.typ.as_deref(),
                Some(class.sort.as_str()),
                "{script}: sort of class {theirs}"
            );
        }

        // A global bound to a non-e-class value would carry a let name we
        // have nowhere to put; none of our programs has one.
        for (class, data) in &serialized.class_data {
            if data.extra.contains_key("let") {
                assert!(
                    renaming.backward.contains_key(class),
                    "{script}: class {class} is let-bound but is not an e-class"
                );
            }
        }

        Census {
            rows: snapshot.nodes.len(),
            classes: snapshot.classes.len(),
            serialized_nodes: serialized.nodes.len(),
            serialized_classes: their_classes.len(),
            dummies,
        }
    }

    /// The snapshot read out of egglog holds exactly what egglog's own
    /// serializer holds — same rows per class, same operators, same
    /// subsumption, same let names, same sorts — under a renaming of the
    /// class ids.
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

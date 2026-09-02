//! Test-only support: hand-author `ExtractedGraph`s at the level under test
//! (the analogue of writing a `.mlir` file by hand for MLIR's analysis tests),
//! plus a fixture path that runs a real egglog script through the extractor.
//!
//! Assignment rule: any case expressible with default-interface ops should be an
//! egg script tested via [`extract_fixture`]; the [`TestGraph`] builder is only
//! for cases *defined by* a non-default `Bufferizable` interface (declared
//! must-share ties, may-share permits, accumulators), which by design have no
//! egglog surface.

pub mod test_ops {
    //! TEST FIXTURE (seed of the future TestRuntime, ruling 2026-08-13):
    //! the reference runtime implements ONLY non-mutating spellings of the
    //! logical ops, so the fused multi-output op lives here purely to
    //! exercise bufferize's multi-destination invariants. The principled
    //! home is a small TestRuntime with simple view/mutation/multi-output
    //! implementations — recorded in the queue, not built yet.

    use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
    use crate::layout_ir::{AliasInfo, Bufferizable, LayoutIrOp, Sharing, ToDps};

    /// `AddMulFusedGeneric(lhs, rhs) -> (add_out, mul_out)`
    ///
    /// Functional form: pure dataflow, conservative [`Bufferizable`] defaults
    /// (every operand read, both results freshly allocated). Elementwise: element
    /// `i` of each input is read before element `i` of either output is written
    /// (op-level all-pairs claim documented by
    /// `bufferizes_to_elementwise_access`).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AddMulFused;

    impl OpSlotNames for AddMulFused {
        fn operand_name(&self, operand: usize) -> String {
            match operand {
                0 => "lhs".to_string(),
                1 => "rhs".to_string(),
                _ => format!("in{operand}"),
            }
        }
    }

    impl BufferTensorIrOp for AddMulFused {
        fn label(&self) -> &str {
            "AddMulFusedGeneric"
        }
    }

    impl Bufferizable for AddMulFused {}

    impl ToDps for AddMulFused {
        fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
            Some(Box::new(AddMulFusedDps))
        }
    }

    impl LayoutIrOp for AddMulFused {}

    /// Destination-passing form of [`AddMulFused`] — two results, so two
    /// destinations, each tied to its own result, spelled slot by slot:
    ///
    /// ```text
    /// AddMulFusedGeneric(lhs: read, rhs: read,
    ///                    dest0: write-only ↔ out0 (add),
    ///                    dest1: write-only ↔ out1 (mul)) -> (out0, out1)
    /// ```
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct AddMulFusedDps;

    impl OpSlotNames for AddMulFusedDps {
        fn operand_name(&self, operand: usize) -> String {
            match operand {
                0 => "lhs".to_string(),
                1 => "rhs".to_string(),
                2 => "dest0".to_string(),
                3 => "dest1".to_string(),
                _ => format!("in{operand}"),
            }
        }
    }

    impl BufferTensorIrOp for AddMulFusedDps {
        fn label(&self) -> &str {
            "AddMulFusedGeneric" // DPS forms keep the IR name; DPS-ness shows in the operands
        }

        fn operand_reads_memory(&self, operand: usize) -> bool {
            match operand {
                0 => true,  // lhs
                1 => true,  // rhs
                2 => false, // dest0: write-only destination
                3 => false, // dest1: write-only destination
                _ => true,  // outside the signature: conservative default
            }
        }
    }

    impl Bufferizable for AddMulFusedDps {
        fn alias_info(&self) -> Vec<AliasInfo> {
            vec![
                AliasInfo {
                    operand: 2,
                    result: 0,
                    sharing: Sharing::Must,
                },
                AliasInfo {
                    operand: 3,
                    result: 1,
                    sharing: Sharing::Must,
                },
            ]
        }
    }

    impl ToDps for AddMulFusedDps {
        fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
            None // already DPS — keeps the rewrite pass idempotent
        }
    }

    impl LayoutIrOp for AddMulFusedDps {}

    // ---------------------------------------------------------------------------
    // Matchers
    // ---------------------------------------------------------------------------
}

use std::collections::HashMap;

use egraph_serialize::ClassId;
use petgraph::graph::NodeIndex;

use crate::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use crate::layout_ir::{
    Access, AliasInfo, BufferInfo, Bufferizable, ExtractedDag, ExtractedEdge, ExtractedGraph,
    ExtractedNode, FreedBy, InputNode, LayoutInfo, LayoutIrOp, LayoutTensorInfo, LogicalInfo,
    OpInput, OpNode, OutputNode, OutputSlot, Sharing,
};

// =============================================================================
// The mock layout (resident-geometry cleanup: core defines NO layout
// vocabulary, so the trivial test-only `L` lives here)
// =============================================================================

/// The trivial TEST layout: transports the layout e-class identity and
/// nothing else. The bufferizer's bound is `Clone + Debug` only (the
/// equality join was dropped — layout equality is enforced in the
/// e-graph); the `PartialEq` derive here is test-assertion convenience,
/// not a bound the planner uses. Core never constructs one outside test
/// support; runtimes bring their own decoded types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MockLayout(pub ClassId);

/// The mock decoded-layout table for a built graph, keyed by VALUE
/// e-class (the [`crate::bufferize::bufferize`] contract): every value
/// maps to the identity of its layout class. Works for hand-built
/// [`TestGraph`]s and real extractions alike.
pub fn mock_layout_table(graph: &ExtractedGraph) -> HashMap<ClassId, MockLayout> {
    let mut table = HashMap::new();
    let mut record = |value: &LayoutTensorInfo| {
        table
            .entry(value.eclass.clone())
            .or_insert_with(|| MockLayout(value.layout.eclass.clone()));
    };
    for node in graph.dag.node_weights() {
        match node {
            ExtractedNode::BufferInput(input) => record(&input.value),
            ExtractedNode::LayoutOp(op) => {
                for output in &op.outputs {
                    record(output);
                }
            }
            ExtractedNode::BufferOutput(_) => {}
        }
    }
    table
}

/// [`crate::bufferize::bufferize`] under the mock table — the test-side
/// one-argument convenience every suite that does not exercise a real
/// decoder plans through.
pub fn bufferize_mock(
    graph: &ExtractedGraph,
) -> anyhow::Result<crate::bufferize::BufferIrGraph<MockLayout>> {
    crate::bufferize::bufferize(graph, &mock_layout_table(graph))
}

// =============================================================================
// Mock ops (interface-defined behaviors the real op set cannot express)
// =============================================================================

/// A configurable op for exercising the analyzer and planner.
///
/// * `reads[i]` — does operand `i` read its buffer?
/// * `in_place_operand` — which operand (if any) declares result 0 as an
///   aliasing value (its in-place candidate; a dest tie, since MockOp writes
///   its result).
#[derive(Debug, Clone, Default)]
pub struct MockOp {
    pub reads: Vec<bool>,
    pub in_place_operand: Option<usize>,
    /// Grants the may-share permit for EVERY operand against the tied result
    /// (the unconditional, trusted permission).
    pub not_conflicting: bool,
}

impl MockOp {
    /// A write-only destination on operand 0 aliasing result 0 (the shape of
    /// every DPS dest operand).
    pub fn write_only_dest() -> Self {
        MockOp {
            reads: vec![false],
            in_place_operand: Some(0),
            ..Default::default()
        }
    }
}

impl OpSlotNames for MockOp {}

impl BufferTensorIrOp for MockOp {
    fn label(&self) -> &str {
        "MockOp"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        self.reads.get(operand).copied().unwrap_or(false)
    }
}

impl Bufferizable for MockOp {
    fn alias_info(&self) -> Vec<AliasInfo> {
        let mut info = Vec::new();
        if let Some(operand) = self.in_place_operand {
            info.push(AliasInfo {
                operand,
                result: 0,
                sharing: Sharing::Must,
            });
            if self.not_conflicting {
                for read in 0..self.reads.len() {
                    info.push(AliasInfo {
                        operand: read,
                        result: 0,
                        sharing: Sharing::May,
                    });
                }
            }
        }
        info
    }
}

impl crate::layout_ir::ToDps for MockOp {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None // mocks declare their interface directly; no DPS rewrite
    }
}

impl LayoutIrOp for MockOp {}

/// A pure view op (à la `tensor.extract_slice`): its single result ALIASES
/// operand 0's storage under the result's own layout — a derived buffer,
/// never interchangeable with the parent. What makes it a VIEW is its
/// declared memory effects, not a tie kind: it writes nothing, so its tie is
/// not a dest tie (seeding never crosses it), and an ADMITTED view folds to
/// nothing in the plan. A REJECTED view repairs like every other tie — the
/// result gets fresh storage initialized by copying the parent's bytes (a
/// view over a copy of the buffer, layout unchanged).
#[derive(Debug, Clone)]
pub struct MockView;

impl OpSlotNames for MockView {}

impl BufferTensorIrOp for MockView {
    fn label(&self) -> &str {
        "MockView"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }
}

impl Bufferizable for MockView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl crate::layout_ir::ToDps for MockView {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for MockView {}

/// [`MockView`] carrying a numeric index map — the Phase-3 fold reads it
/// through `view_index_map` and records it (with the parent's dims) on
/// every consumer's operand descriptor. Same declared memory effects and
/// tie as `MockView`; only the map differs.
#[derive(Debug, Clone)]
pub struct MockViewWithMap {
    pub entries: Vec<crate::index_expr::IotaExpr>,
}

impl OpSlotNames for MockViewWithMap {}

impl BufferTensorIrOp for MockViewWithMap {
    fn label(&self) -> &str {
        "MockViewWithMap"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }
    fn view_index_map(&self, _result: usize) -> Option<Vec<crate::index_expr::IotaExpr>> {
        Some(self.entries.clone())
    }
}

impl Bufferizable for MockViewWithMap {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl crate::layout_ir::ToDps for MockViewWithMap {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for MockViewWithMap {}

/// An alloc-like op (à la `tensor.empty`): its single result is undefined
/// storage, so reading it is never a conflict.
#[derive(Debug, Clone)]
pub struct EmptyOp;

impl OpSlotNames for EmptyOp {}

impl BufferTensorIrOp for EmptyOp {
    fn label(&self) -> &str {
        "Empty"
    }

    fn result_is_undefined(&self, _result: usize) -> bool {
        true
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false
    }
}

impl Bufferizable for EmptyOp {}

impl crate::layout_ir::ToDps for EmptyOp {
    fn to_dps(&self) -> Option<Box<dyn crate::layout_ir::LayoutIrOp>> {
        None
    }
}

impl LayoutIrOp for EmptyOp {}

// =============================================================================
// TestGraph: hand-authored ExtractedGraphs
// =============================================================================

/// Builds a well-formed [`ExtractedGraph`] node by node. Buffer and layout
/// identities are controlled by name so tests can express cohabitation (two
/// values sharing one buffer) and layout (in)equality precisely.
pub struct TestGraph {
    dag: ExtractedDag,
    /// The node producing each value; every operand must have one (asserted).
    producers: HashMap<ClassId, NodeIndex>,
    slots: Vec<OutputSlot>,
    next: u32,
}

impl Default for TestGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TestGraph {
    pub fn new() -> Self {
        TestGraph {
            dag: ExtractedDag::new(),
            producers: HashMap::new(),
            slots: Vec::new(),
            next: 0,
        }
    }

    fn fresh(&mut self) -> u32 {
        self.next += 1;
        self.next
    }

    fn value_info(&self, name: &str, layout: &str) -> LayoutTensorInfo {
        LayoutTensorInfo {
            eclass: ClassId::from(format!("val${name}")),
            label: name.to_string(),
            tooltip: Default::default(),
            shape: None,
            dtype: None,
            dtype_enum: None,
            dims: None,
            element_bits: None,
            logical: LogicalInfo {
                eclass: ClassId::from(format!("logical${name}")),
                label: name.to_string().into(),
                tooltip: Default::default(),
                op: None,
                children: Vec::new(),
            },
            layout: LayoutInfo {
                eclass: ClassId::from(format!("layout${layout}")),
                label: layout.to_string().into(),
                tooltip: Default::default(),
            },
        }
    }

    /// Buffer identities are keyed by `buffer` name: two bindings naming the
    /// same buffer share one `id_eclass` (cohabitation), distinct names get
    /// distinct identities.
    fn buffer_info(
        &mut self,
        buffer: &str,
        access: Option<Access>,
        freed_by: Option<FreedBy>,
    ) -> BufferInfo {
        let n = self.fresh();
        BufferInfo {
            lit: None,
            tensor_eclass: ClassId::from(format!("buftensor${n}")),
            tensor_label: buffer.to_string(),
            tensor_tooltip: Default::default(),
            id_eclass: ClassId::from(format!("buf${buffer}")),
            id_label: buffer.to_string(),
            id_tooltip: Default::default(),
            access,
            freed_by,
        }
    }

    /// A program input: value `name` (with layout `layout`) living in `buffer`.
    pub fn input(&mut self, name: &str, buffer: &str, access: Access, layout: &str) -> ClassId {
        self.input_binding(name, buffer, Some(access), Some(FreedBy::Caller), layout)
    }

    /// The fully-explicit binding builder: `None` models a program that OMITS
    /// a boundary declaration (for input-validation tests — well-formed
    /// programs always declare).
    pub fn input_binding(
        &mut self,
        name: &str,
        buffer: &str,
        access: Option<Access>,
        freed_by: Option<FreedBy>,
        layout: &str,
    ) -> ClassId {
        let value = self.value_info(name, layout);
        let eclass = value.eclass.clone();
        let buffer = self.buffer_info(buffer, access, freed_by);
        let node = self
            .dag
            .add_node(ExtractedNode::BufferInput(Box::new(InputNode {
                value,
                buffer,
            })));
        self.producers.insert(eclass.clone(), node);
        eclass
    }

    /// An op with the given interface. One output value per `(name, layout)`
    /// pair. Adds real dataflow edges from each operand's producer.
    pub fn op(
        &mut self,
        iface: Box<dyn LayoutIrOp>,
        inputs: &[&ClassId],
        outputs: &[(&str, &str)],
    ) -> Vec<ClassId> {
        let n = self.fresh();
        let output_infos: Vec<LayoutTensorInfo> = outputs
            .iter()
            .map(|(name, layout)| self.value_info(name, layout))
            .collect();
        let result_classes: Vec<ClassId> = output_infos
            .iter()
            .map(|info| info.eclass.clone())
            .collect();
        let op_inputs: Vec<OpInput> = inputs
            .iter()
            .enumerate()
            .map(|(index, value)| OpInput {
                port: format!("in{index}"),
                value: (*value).clone(),
            })
            .collect();
        let node = self.dag.add_node(ExtractedNode::LayoutOp(OpNode {
            op: iface,
            provenance: crate::layout_ir::Provenance::Synthesized { id: n },
            inputs: op_inputs,
            outputs: output_infos,
            tooltip: Default::default(),
            heuristic_cost: 1,
        }));
        for (index, value) in inputs.iter().enumerate() {
            let producer = *self
                .producers
                .get(*value)
                .unwrap_or_else(|| panic!("operand {value} has no producer"));
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge {
                    value: (*value).clone(),
                    port: format!("in{index}"),
                },
            );
        }
        for eclass in &result_classes {
            self.producers.insert(eclass.clone(), node);
        }
        result_classes
    }

    /// Pin `value` into `buffer` as the next output slot.
    pub fn output(&mut self, value: &ClassId, buffer: &str) {
        let index = self.slots.len();
        let buffer = self.buffer_info(buffer, Some(Access::ReadWrite), Some(FreedBy::Caller));
        self.slots.push(OutputSlot {
            index,
            value: value.clone(),
            buffer,
        });
    }

    /// Finalize: emit the `BufferOutput` node (with edges from every slot
    /// value's producer, so topological order is correct) and return the graph.
    pub fn build(mut self) -> ExtractedGraph {
        let slots = std::mem::take(&mut self.slots);
        let node = self.dag.add_node(ExtractedNode::BufferOutput(OutputNode {
            eclass: ClassId::from("output$0"),
            label: "output".to_string(),
            tooltip: String::new(),
            slots: slots.clone(),
        }));
        for slot in &slots {
            let producer = *self
                .producers
                .get(&slot.value)
                .unwrap_or_else(|| panic!("output value {} has no producer", slot.value));
            self.dag.add_edge(
                producer,
                node,
                ExtractedEdge {
                    value: slot.value.clone(),
                    port: format!("out {}", slot.index),
                },
            );
        }
        // Re-insert slots into the node weight (cloned above for edge wiring).
        if let ExtractedNode::BufferOutput(output) = &mut self.dag[node] {
            output.slots = slots;
        }
        ExtractedGraph {
            dag: self.dag,
            outputs: vec![node],
        }
    }
}

// The fixture path (real egglog scripts through the real extractor) and
// the run_reference harness moved to `luminal_reference::harness` with the
// reference registry (Step B, ruling 2026-08-17): they default the matcher
// set to the reference ops, which core no longer owns. Core keeps only the
// runtime-neutral pieces: TestGraph/mocks above and harness_search_options.

#[cfg(test)]
mod harness_tests {
    // DEP-WORLD suite (Step B): fixtures and the reference registry come
    // through the luminal_reference dev-dependency, so every luminal type
    // here must come from the `luminal::` build that crate links, never
    // `crate` — the cyclic dev-dependency's two library builds do not
    // unify their types.
    use std::fs;

    use luminal::bufferize;
    use luminal::layout_ir::Access;
    use luminal::test_support::*;
    // Narrowed by the runtime split (PR #425): the fixtures that needed a
    // view/mutating vocabulary — and the entry points that reached for it —
    // moved to the test runtime's own crate. What core still asserts here
    // runs on the reference registry alone.
    use luminal_reference::harness::{extract_fixture, serialize_fixture};

    /// The builder produces a graph the real pipeline accepts end to end.
    #[test]
    fn builder_graph_bufferizes() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let results = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        );
        g.output(&results[0], "B");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");
        // Out-of-place: y in a fresh alloc, then a copy into pinned B.
        let summary = plan.summary();
        assert!(summary.contains("MockOp"), "{summary}");
        assert!(summary.contains("BufferCopy"), "{summary}");
    }

    /// The fixture path runs a real egg script through egglog + the extractor.
    #[test]
    fn fixture_extracts_and_bufferizes() {
        let graph = extract_fixture("boundary_pass_through.egg");
        let plan = bufferize_mock(&graph).expect("bufferizes");
        let summary = plan.summary();
        assert!(summary.contains("ops (0):"), "{summary}");
    }

    /// The canonical WAR hazard, through the REAL pipeline: y = Sqrt(x) lands in
    /// x's buffer via a boundary copy while Exp(x) still reads x. The plan must
    /// carry an Anti edge ordering Exp's read before the copy's write.
    #[test]
    fn war_hazard_gets_anti_edge() {
        use luminal::bufferize::{BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let graph = extract_fixture("boundary_war_hazard.egg");
        let plan = bufferize_mock(&graph).expect("bufferizes");

        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|e| e.weight().kind == EdgeKind::Anti)
            .collect();
        // 1 WAR (Exp's read before the copy overwriting x's buffer) + 2
        // lifetime (each fresh buffer's src-reading copy before its free).
        assert_eq!(
            anti.len(),
            3,
            "expected 1 WAR + 2 lifetime antis:\n{}",
            plan.summary()
        );
        let war: Vec<_> = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .collect();
        assert_eq!(
            war.len(),
            1,
            "expected exactly one WAR anti:\n{}",
            plan.summary()
        );
        let edge = war[0];
        // Source must be the Exp compute node; target the copy into x's buffer.
        match &plan.dag[edge.source()] {
            BufferNode::Compute { op, .. } => assert_eq!(op.label(), "ExpFunctionalGeneric"),
            other => panic!("anti edge source should be ExpFunctionalGeneric, got {other:?}"),
        }
        assert!(matches!(
            &plan.dag[edge.target()],
            BufferNode::BufferCopy { .. }
        ));
    }

    /// Copy-vs-copy WAR: out0 passes x onward into C (copy B->C reads B) while
    /// out1 materializes f(x) into B (copy alloc->B writes B). The two copies
    /// are unordered by dataflow; the plan must order read-of-B first.
    #[test]
    fn copy_reading_buffer_ordered_before_copy_writing_it() {
        use luminal::bufferize::{BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&x, "C"); // copy B -> C (reads B)
        g.output(&y, "B"); // copy alloc -> B (writes B)
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|e| e.weight().kind == EdgeKind::Anti)
            .collect();
        // 1 WAR (the B-reading copy before the B-writing copy) + 1 lifetime
        // (the B-writing copy's src-read before y's buffer is freed).
        assert_eq!(
            anti.len(),
            2,
            "expected 1 WAR + 1 lifetime anti:\n{}",
            plan.summary()
        );
        let war: Vec<_> = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .collect();
        assert_eq!(
            war.len(),
            1,
            "expected exactly one WAR anti:\n{}",
            plan.summary()
        );
        let edge = war[0];
        // Direction: the copy READING B (the B->C pass-onward) must run before
        // the copy WRITING B (the alloc->B overwrite).
        match &plan.dag[edge.source()] {
            BufferNode::BufferCopy { src, .. } => assert_eq!(src, &plan.value_buffer[&x]),
            other => panic!("anti source should be the B-reading copy, got {other:?}"),
        }
        match &plan.dag[edge.target()] {
            BufferNode::BufferCopy { dst, .. } => assert_eq!(dst, &plan.value_buffer[&x]),
            other => panic!("anti target should be the B-writing copy, got {other:?}"),
        }
    }

    /// A buffer both passed through to one slot and copy-overwritten for another
    /// is unsatisfiable (one buffer, two final values). Input-program validation
    /// rejects the program up front — the demanded value y is not among B's
    /// inputs, so it cannot cohabit with the promised pass-through.
    #[test]
    fn passthrough_plus_copy_into_same_buffer_is_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&x, "B"); // pass-through: B's final contents promised to be x
        g.output(&y, "B"); // copy alloc -> B: overwrites them
        let err = bufferize_mock(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("distinct final values"), "{err}");
    }

    /// OUT-OF-PLACE RESOLUTION, accumulator case: op2 accumulates into x
    /// (reads its own destination) but a sibling reader of x forces rejection.
    /// The plan must copy x's contents into the fresh result buffer BEFORE op2,
    /// retarget op2's operand to it, and never write x's pinned buffer.
    #[test]
    fn rejected_accumulator_copies_contents_and_retargets() {
        use luminal::bufferize::{BufferId, BufferNode};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        // sibling reader, unordered with the accumulator -> forces rejection
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true], // accumulator: reads its own destination
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&x],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&y, "C");
        g.output(&r, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        // Find the accumulator's compute node.
        let (acc_idx, acc_reads, acc_writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && plan.value_buffer[&r] == writes[0] =>
                {
                    Some((idx, reads.clone(), writes.clone()))
                }
                _ => None,
            })
            .expect("accumulator node");
        // Retargeted: reads its own fresh result buffer, not x's pinned B.
        assert_eq!(acc_reads[0], acc_writes[0], "{}", plan.summary());
        assert!(matches!(acc_writes[0], BufferId::Allocated(_)));
        // PINNING (critique test i): must NOT share the pinned input buffer.
        assert_ne!(acc_writes[0], plan.value_buffer[&x]);
        // A copy feeds the accumulator: src = x's buffer, dst = the fresh alloc,
        // dataflow-ordered before the op.
        let copy_edge = plan
            .dag
            .edges_directed(acc_idx, petgraph::Direction::Incoming)
            .find(|e| matches!(&plan.dag[e.source()], BufferNode::BufferCopy { .. }))
            .expect("copy feeds the accumulator");
        match &plan.dag[copy_edge.source()] {
            BufferNode::BufferCopy { src, dst, .. } => {
                assert_eq!(src, &plan.value_buffer[&x]);
                assert_eq!(dst, &acc_writes[0]);
            }
            _ => unreachable!(),
        }
    }

    /// OUT-OF-PLACE RESOLUTION, write-only case: a rejected write-only
    /// destination retargets with NO contents copy (poison contents are
    /// irrelevant), and must not share the operand's storage.
    #[test]
    fn rejected_write_only_dest_retargets_without_copy() {
        use luminal::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        // Interior operand: v is another op's result with a sibling reader.
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&x],
            &[("v", "rm")],
        )[0]
        .clone();
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                in_place_operand: None,
                ..Default::default()
            }),
            &[&v],
            &[("y", "rm")],
        )[0]
        .clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false], // write-only destination
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&y, "C");
        g.output(&r, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        let (reads, writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && plan.value_buffer[&r] == writes[0] =>
                {
                    Some((reads.clone(), writes.clone()))
                }
                _ => None,
            })
            .expect("dest-taking node");
        assert_eq!(reads[0], writes[0], "{}", plan.summary());
        // PINNING (critique test ii): must NOT share the interior operand's alloc.
        assert_ne!(writes[0], plan.value_buffer[&v]);
        assert!(matches!(writes[0], BufferId::Allocated(_)));
        // No copy targets the retargeted buffer (write-only: nothing to preserve).
        let copies_into_target = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                matches!(&plan.dag[idx], BufferNode::BufferCopy { dst, .. } if dst == &writes[0])
            })
            .count();
        assert_eq!(copies_into_target, 0, "{}", plan.summary());
    }

    /// CHAINED in-place destinations, now with MULTI-HOP destination seeding:
    /// OpA writes into poison e producing r1; OpB uses r1 as its write-only
    /// destination producing r2, bound to output D. Bottom-up analysis decides
    /// OpB first (union r1~r2), then OpA (union e~r1); the seed walks D's slot
    /// back through BOTH hops to the chain root e, and since both hops are
    /// admitted in place the whole chain binds to D itself — zero allocations,
    /// zero copies, every op computing directly into the output storage.
    #[test]
    fn chained_destinations_collapse_onto_the_output_buffer() {
        use luminal::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r1 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r1", "rm")],
        )[0]
        .clone();
        let r2 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&r1],
            &[("r2", "rm")],
        )[0]
        .clone();
        g.output(&r2, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        assert_eq!(plan.value_buffer[&e], plan.value_buffer[&r1]);
        assert_eq!(plan.value_buffer[&r1], plan.value_buffer[&r2]);
        assert!(
            matches!(plan.value_buffer[&r2], BufferId::Boundary(_)),
            "the chain seeds into the output buffer itself:\n{}",
            plan.summary()
        );
        assert!(
            plan.value_buffer
                .values()
                .all(|b| !matches!(b, BufferId::Allocated(_))),
            "zero allocations:\n{}",
            plan.summary()
        );
        let copies = plan
            .dag
            .node_indices()
            .filter(|&idx| matches!(&plan.dag[idx], BufferNode::BufferCopy { .. }))
            .count();
        assert_eq!(copies, 0, "zero copies:\n{}", plan.summary());
    }

    /// One value bound to TWO output buffers: the chain-root poison can only
    /// live in one of them. The seed goes to the lowest slot (D); the other
    /// slot (E) is served by a copy out of D.
    #[test]
    fn seed_ties_break_to_lowest_slot() {
        use luminal::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        g.output(&r, "E");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r], BufferId::Boundary(_)),
            "r seeds into an output buffer:\n{}",
            plan.summary()
        );
        let copies: Vec<(&BufferId, &BufferId)> = plan
            .dag
            .node_indices()
            .filter_map(|idx| match &plan.dag[idx] {
                BufferNode::BufferCopy { src, dst, .. } => Some((src, dst)),
                _ => None,
            })
            .collect();
        assert_eq!(
            copies.len(),
            1,
            "one copy serves the losing slot:\n{}",
            plan.summary()
        );
        assert_eq!(
            copies[0].0,
            &plan.value_buffer[&r],
            "the copy reads the seeded buffer:\n{}",
            plan.summary()
        );
        assert_ne!(copies[0].0, copies[0].1);
    }

    /// A seed whose write would clobber a value an UNORDERED op still reads is
    /// rejected, and the plan degrades to exactly the unseeded shape: fresh
    /// allocation, boundary copy, and the WAR machinery ordering the copy
    /// after the reader.
    #[test]
    fn rejected_seed_degrades_to_copy_with_war_edge() {
        use luminal::bufferize::{BufferId, BufferNode, EdgeKind};
        use petgraph::visit::EdgeRef;

        let mut g = TestGraph::new();
        let x = g.input("x", "D", Access::ReadWrite, "rm");
        let e = g.op(Box::new(EmptyOp), &[], &[("e", "rm")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&e],
            &[("r", "rm")],
        )[0]
        .clone();
        // An unordered reader of x keeps x's storage (buffer D) live.
        let s = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("s", "rm")],
        )[0]
        .clone();
        g.output(&r, "D"); // seed proposal: e ↦ D — must be rejected
        g.output(&s, "E");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r], BufferId::Allocated(_)),
            "the rejected seed falls back to a fresh allocation:\n{}",
            plan.summary()
        );
        let copies_into_d = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                matches!(&plan.dag[idx], BufferNode::BufferCopy { dst, .. }
                    if dst == &plan.value_buffer[&x])
            })
            .count();
        assert_eq!(
            copies_into_d,
            1,
            "boundary copy restored:\n{}",
            plan.summary()
        );
        // 1 WAR (the reader before the boundary copy) + 2 lifetime (each
        // fresh buffer's src-reading copy before its free).
        let anti: Vec<_> = plan
            .dag
            .edge_references()
            .filter(|edge| edge.weight().kind == EdgeKind::Anti)
            .collect();
        assert_eq!(
            anti.len(),
            3,
            "1 WAR + 2 lifetime antis:\n{}",
            plan.summary()
        );
        let war = anti
            .iter()
            .filter(|e| {
                !matches!(&plan.dag[e.target()], BufferNode::Compute { writes, .. }
                    if writes.is_empty())
            })
            .count();
        assert_eq!(
            war,
            1,
            "reader ordered before the copy:\n{}",
            plan.summary()
        );
    }

    /// RANK 6 + the CRITICAL WAR fix regression: two inputs swapping buffers
    /// (out0 = b into P, out1 = a into Q, with a@P and b@Q) produce two copies
    /// that each read the other's destination. Both anti edges must be added
    /// (judged against the FROZEN dataflow ordering — the executed-miscompile
    /// fix), making the plan a cycle: a loud "unschedulable" error, never a
    /// silent wrong-order schedule.
    #[test]
    fn buffer_swap_fails_loudly_not_silently() {
        let mut g = TestGraph::new();
        let a = g.input("a", "P", Access::ReadWrite, "rm");
        let b = g.input("b", "Q", Access::ReadWrite, "rm");
        g.output(&b, "P"); // copy Q -> P (reads Q, writes P)
        g.output(&a, "Q"); // copy P -> Q (reads P, writes Q)
        let err = bufferize_mock(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unschedulable"), "{err}");
    }

    /// MAJOR review finding, now caught at the door: two DIFFERENT computed
    /// values demanded in the SAME output buffer is a one-buffer-two-values
    /// contradiction. Input-program validation rejects the program outright
    /// (it is not a planning failure — no plan can satisfy it at buffer
    /// granularity), so the WAW contradiction can no longer reach planning
    /// through materialized outputs. The final-bindings check remains as the
    /// trusted kernel's backstop for the pass-through sliver.
    #[test]
    fn two_materialized_values_into_one_buffer_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y1 = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y1", "rm")],
        )[0]
        .clone();
        let y2 = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y2", "rm")],
        )[0]
        .clone();
        g.output(&y1, "D");
        g.output(&y2, "D"); // same buffer, two distinct computed values
        let err = bufferize_mock(&g.build()).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("distinct final values"), "{err}");
    }

    /// PARKED PROGRAM CLASS (Decision 1b, 2026-07-08): mutating one boundary
    /// view of a buffer while another view of the same buffer must survive to
    /// the output demands two distinct final values in one buffer — sound only
    /// under REGION reasoning (the views are disjoint rows), which the planner
    /// deliberately does not do. Input-program validation rejects it up front;
    /// the fixture stays as the acceptance test for future region support.
    #[test]
    fn aliased_views_in_place_fixture_rejected_as_unsupported() {
        let graph = extract_fixture("boundary_aliased_views_in_place.egg");
        let err = bufferize_mock(&graph).unwrap_err();
        assert!(err.to_string().contains("unsupported program"), "{err}");
        assert!(err.to_string().contains("pass-throughs"), "{err}");
    }

    /// RANK 8: materializing into a Read-granted destination is a hard error
    /// (write obligation without write rights)...
    #[test]
    fn copy_into_read_granted_buffer_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        let y = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&x],
            &[("y", "rm")],
        )[0]
        .clone();
        g.output(&y, "B"); // materialize a computed value into the Read buffer
        let err = bufferize_mock(&g.build()).unwrap_err();
        assert!(err.to_string().contains("read-only buffer"), "{err}");
    }

    /// Declaration satisfiability (the last inheritor of the final-bindings
    /// check, now at input validation): two pass-through demands on one
    /// buffer under the SAME layout but with DIFFERENT values are
    /// contradictory — the buffer cannot end holding both.
    #[test]
    fn same_layout_pass_through_contradiction_rejected() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "B", Access::ReadWrite, "rm");
        g.output(&x, "B");
        g.output(&y, "B");
        let err = bufferize_mock(&g.build()).unwrap_err();
        assert!(err.to_string().contains("same layout"), "{err}");
        assert!(err.to_string().contains("invalid input program"), "{err}");
    }

    /// ...while passing the Read buffer's own value through is legal.
    #[test]
    fn passthrough_of_read_granted_buffer_is_legal() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        g.output(&x, "B");
        let plan = bufferize_mock(&g.build()).expect("pass-through is legal");
        assert!(plan.summary().contains("ops (0):"), "{}", plan.summary());
    }

    /// Numeric geometry rides extraction: literal dims and bit widths are
    /// walked off the e-graph terms onto every value info — the surface the
    /// ReferenceRuntime sizes its buffers from.
    #[test]
    fn extraction_carries_numeric_dims_and_bits() {
        use luminal::layout_ir::ExtractedNode;

        // Retargeted off the retired fused-matmul fixture (non-mutating
        // inventory ruling 2026-08-13): any extracted op output carries
        // numeric geometry; the gather fixture's data operand pins it.
        let graph = extract_fixture("boundary_gather.egg");
        let gather_out = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label().starts_with("GatherGeneric") => {
                    Some(op.outputs[0].clone())
                }
                _ => None,
            })
            .expect("gather extracted");
        assert!(gather_out.dims.is_some(), "numeric dims carried");
        assert_eq!(gather_out.element_bits, Some(32));
    }

    /// FIXED ISSUE PROOF (root-caused and fixed 2026-07-29, Austin-approved):
    /// this sound div/mod iota gather program once fired the axis-support
    /// tripwire. Root cause: the subst CoordVar arm instantiated coordinates
    /// with map entries OUTSIDE the coordinate's own range — an out-of-box
    /// instantiation of box-true equations (e.g. a view-space quotient
    /// box-true equal to 1), whose per-representative image unions then
    /// welded false equalities into literal classes. The in-range
    /// instantiation guard on that arm (the well-definedness condition of
    /// subst-of; see the preamble subst contract) removes the corruption;
    /// this test — the original reproducer, formerly the expect-panic pin —
    /// now asserts CLEAN saturation and stands as the fix's proof.
    #[test]
    fn div_mod_iota_gather_with_layout_saturates_cleanly() {
        let body = r#"
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let flat (IntAdd (IntMul (CoordVar out_shape 1) (IntLit 3)) (CoordVar out_shape 0)))
(let value
  (IntAdd
    (IntMul (IntAdd (IntTruncRem (IntTruncDiv flat (IntLit 3)) (IntLit 2)) (IntLit 1)) (IntLit 5))
    (IntAdd (IntTruncRem flat (IntLit 3)) (IntLit 2))))
(let row_coord (IntTruncDiv value (IntLit 5)))
(let col_coord (IntTruncRem value (IntLit 5)))
(let row_iota (LogicalIota row_coord out_shape))
(let col_iota (LogicalIota col_coord out_shape))
(let data_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 5) (IntExprNil)))))
(let data_logical (LogicalTensorInputLit (LogicalIdLit "data") data_shape (F32)))
(let gathered
  (LogicalGather data_logical
    (LogicalTensorCons row_iota (LogicalTensorCons col_iota (LogicalTensorNil)))))
(let data_layout (RightMajorContiguousElementLayoutLit data_shape (bits-of (F32))))
(let data_layout_tensor (LayoutTensorLit data_logical data_layout))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{}", luminal_reference::assembled_program(), body);
        luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &full)
            .expect("sound div/mod gather program saturates cleanly under the subst range guard");
    }

    /// The bool bridge (Austin's design 2026-07-29): decided comparisons
    /// collapse to their literals, undecided indicators stay bits, and the
    /// pad pattern's clamp/mask bounds derive — the fixture's checks and
    /// fail-pins are the proof.
    #[test]
    fn bool_bridge_fixture_holds() {
        let _ = serialize_fixture("bool_bridge_example.egg");
    }

    /// The pinned-plan fixture list, shared by the pin test and the
    /// regenerator below.
    const GOLDEN_SCRIPTS: &[&str] = &[
        "add_mul_fused",
        "basic_program",
        "matmul_fused_example",
        "boundary_aliased_views",
        "boundary_donated_input",
        "boundary_gather",
        "boundary_in_place_mutation",
        "boundary_iota",
        "boundary_pass_through",
        "boundary_scalar",
        "boundary_scatter",
        "boundary_view_feeds_compute",
        "boundary_war_hazard",
        "boundary_write_into_viewed_buffer",
        "transformer",
    ];

    /// Regenerate output/<stem>.bufferized.txt — run explicitly by name:
    /// `cargo test regenerate_golden_plans -- --ignored` (no env vars, by
    /// ruling 2026-08-06; invocation-by-name IS the programmatic opt-in).
    /// Regenerated diffs are REVIEWED, never rubber-stamped — a golden
    /// change must trace to an intended ruling (e.g. 2026-08-06: the
    /// transformer golden had pinned a float-reassociated residual sum the
    /// dtype-gate removed).
    #[test]
    #[ignore = "golden regenerator — run explicitly by name"]
    fn regenerate_golden_plans() {
        for stem in GOLDEN_SCRIPTS {
            let graph = extract_fixture(&format!("{stem}.egg"));
            let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect(stem);
            fs::write(format!("output/{stem}.bufferized.txt"), plan.summary())
                .expect("golden writes");
        }
    }

    /// The zero-input source path (R7): iota extracts with an EMPTY operand
    /// list, and after the DPS rewrite its appended destination is the op's
    /// ONLY operand — the degenerate case of the trailing-destination
    /// convention, proven end to end on the boundary_iota fixture.
    #[test]
    fn iota_extracts_as_zero_input_source() {
        use luminal::layout_ir::ExtractedNode;
        let graph = extract_fixture("boundary_iota.egg");
        let extracted = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "IotaGeneric" => Some(op.clone()),
                _ => None,
            })
            .expect("iota op extracted");
        assert!(extracted.inputs.is_empty(), "a source op has no operands");
        assert_eq!(extracted.outputs.len(), 1);

        let rewritten = luminal::dps::dps_rewrite(&graph);
        let dps = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op)
                    if op.op.label() == "IotaGeneric" && op.op.to_dps().is_none() =>
                {
                    Some(op.clone())
                }
                _ => None,
            })
            .expect("iota rewritten to its DPS form");
        assert_eq!(dps.inputs.len(), 1, "the destination is the only operand");
        assert!(dps.inputs[0].port.starts_with("dest"));
    }

    /// A scatter whose coordinate shapes genuinely disagree with src's
    /// shape survives the gated main stratum untouched (fail-open) and
    /// dies inside the terminal stratum's coordinate-shape-lock closure —
    /// egglog's own merge machinery raises the error.
    #[test]
    fn scatter_with_disagreeing_src_and_coordinate_shapes_dies_in_the_terminal_stratum() {
        let preamble = luminal_reference::assembled_program();
        let script = r#"
(let cache_shape (ShapeLit (IntExprCons (IntLit 6) (IntExprCons (IntLit 4) (IntExprNil)))))
(let cache (LogicalTensorInputLit (LogicalIdLit "cache") cache_shape (F32)))
(let row_shape (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let wide_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let src (LogicalTensorInputLit (LogicalIdLit "src") wide_shape (F32)))
(let position (LogicalTensorInputLit (LogicalIdLit "position") row_shape (Int)))
(let column (LogicalTensorInputLit (LogicalIdLit "column") row_shape (Int)))
(let bad_scatter
  (LogicalScatter cache
    (LogicalTensorCons position (LogicalTensorCons column (LogicalTensorNil)))
    src))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (saturate (run fixpoint-invariants)))
"#;
        let program = format!(
            "{preamble}

{script}"
        );
        let mut egraph = luminal::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the shape lock must collide in the terminal stratum");
        assert!(
            err.to_string().contains("shape-of"),
            "the error should come from the shape-of merge: {err}"
        );
    }

    /// The variable-arity path (R8): gather's rank is DATA read out of the
    /// e-graph by its matcher — the first data-carrying instance. The
    /// extracted op has 1 + rank operands in declared order, and after the
    /// DPS rewrite the destination is the trailing operand.
    #[test]
    fn gather_extracts_with_rank_counted_operands() {
        use luminal::layout_ir::ExtractedNode;
        let graph = extract_fixture("boundary_gather.egg");
        let extracted = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label() == "GatherGeneric" => Some(op.clone()),
                _ => None,
            })
            .expect("gather op extracted");
        assert_eq!(extracted.inputs.len(), 3, "data + one coord per data axis");
        assert_eq!(extracted.inputs[0].port, "data");
        assert_eq!(extracted.inputs[1].port, "coord0");
        assert_eq!(extracted.inputs[2].port, "coord1");

        let rewritten = luminal::dps::dps_rewrite(&graph);
        let dps = rewritten
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op)
                    if op.op.label() == "GatherGeneric" && op.op.to_dps().is_none() =>
                {
                    Some(op.clone())
                }
                _ => None,
            })
            .expect("gather rewritten to its DPS form");
        assert_eq!(dps.inputs.len(), 4, "the destination trails the coords");
        assert_eq!(dps.inputs[3].port, "dest0");
    }

    /// TYPED-BUFFERS PIN, RESPELLED FOR THE CORRECTED CONTRACT
    /// (2026-08-31, correction 1). The serialized `dtype-of` rows are
    /// still readable (op = "dtype-of", children[0] = the logical
    /// argument, the row's own eclass holds the nullary Dtype member —
    /// the bounds-row encoding), but they NO LONGER thread onto plan
    /// buffers: `PlanDtype` left the plan and bufferizer vocabulary
    /// entirely. What they thread onto is the RUNTIME'S OWN `L` — here
    /// `RefLayout { mirror, dtype }` — which the reference decoder folds
    /// at extraction time and which Option B then carries on each
    /// assignment entry.
    ///
    /// The mixed-dtype gather fixture exercises F32 data, an Int boundary
    /// input, an Int interior iota (a planner-allocated buffer), and an
    /// F32 output — so both boundary and allocated buffers must arrive
    /// with a typed carried layout. There is no width/dtype consistency
    /// bail any more (the e-graph enforces dtype consistency); the fact
    /// pinned here is TRANSPORT: every assignment row's `L` carries the
    /// dtype its own backed tensor was declared with.
    #[test]
    fn dtype_rows_reach_the_runtimes_own_layout_type() {
        use luminal::dtype::PlanDtype;
        let egraph = serialize_fixture("boundary_gather.egg");
        let graph = luminal::dps::dps_rewrite(&extract_fixture("boundary_gather.egg"));
        let mut cache = std::collections::HashMap::new();
        let table = luminal::extractor::decoded_layout_table(
            &egraph,
            &graph, // the POST-DPS graph: value-keyed tables cover poisons
            &luminal_reference::ReferenceLayoutDecoder,
            &mut cache,
        )
        .expect("the reference decoder covers every elected value");
        let plan = bufferize::bufferize(&graph, &table).expect("mixed-dtype plan bufferizes");

        let mut by_lit: std::collections::HashMap<i64, PlanDtype> = Default::default();
        let mut allocated_int = 0usize;
        for buffer in plan.buffers.values() {
            let dtype = buffer.layout.dtype.unwrap_or_else(|| {
                panic!("buffer {} carries no dtype fact in its L", buffer.label)
            });
            // TRANSPORT, not derivation: the row on the buffer's carried
            // layout IS the row the decoder put on the BACKED tensor.
            assert_eq!(
                Some(&buffer.layout),
                table.get(&buffer.backs),
                "buffer {} carries its backed tensor's own decoded layout",
                buffer.label
            );
            if let Some(lit) = buffer.lit {
                by_lit.insert(lit, dtype);
            } else if dtype == PlanDtype::Int {
                allocated_int += 1;
            }
        }
        assert_eq!(by_lit.get(&310), Some(&PlanDtype::F32), "embedding table");
        assert_eq!(by_lit.get(&311), Some(&PlanDtype::Int), "token ids");
        assert_eq!(by_lit.get(&312), Some(&PlanDtype::F32), "lookup output");
        assert!(
            allocated_int >= 1,
            "the interior column iota's allocated buffer must be typed Int"
        );
    }

    /// The terminal-stratum checker (user ruling 2026-07-23: every
    /// invariant lives in egglog): a gather whose coordinate shapes are
    /// GENUINELY unequal survives the gated main stratum untouched
    /// (fail-open, no race), then dies inside (run fixpoint-invariants) —
    /// the unguarded shape closure writes both shapes into one :no-merge
    /// slot and egglog's own merge machinery raises the error. Inline
    /// program — the pipeline sweep never runs intentionally ill-formed
    /// scripts.
    #[test]
    fn gather_with_disagreeing_coordinate_shapes_dies_in_the_terminal_stratum() {
        let preamble = luminal_reference::assembled_program();
        let script = r#"
(let three_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let four_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let ids_three (LogicalTensorInputLit (LogicalIdLit "ids_three") three_shape (Int)))
(let ids_four (LogicalTensorInputLit (LogicalIdLit "ids_four") four_shape (Int)))
(let ids_three_layout (RightMajorContiguousElementLayoutLit three_shape (bits-of (Int))))
(let ids_four_layout (RightMajorContiguousElementLayoutLit four_shape (bits-of (Int))))
(let ids_three_lt (LayoutTensorLit ids_three ids_three_layout))
(let ids_four_lt (LayoutTensorLit ids_four ids_four_layout))
(let data_shape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 7) (IntExprNil)))))
(let data (LogicalTensorInputLit (LogicalIdLit "data") data_shape (F32)))
(let data_layout (RightMajorContiguousElementLayoutLit data_shape (bits-of (F32))))
(let data_lt (LayoutTensorLit data data_layout))
(let bad_gather
  (LogicalGather data
    (LogicalTensorCons ids_three
      (LogicalTensorCons ids_four (LogicalTensorNil)))))
(let out_layout (RightMajorContiguousElementLayoutLit three_shape (bits-of (F32))))
(let out_lt (LayoutTensorLit bad_gather out_layout))
(let out_buffer (BufferLit 500))
(set (buffer-access-of out_buffer) (ReadWrite))
(set (buffer-freed-by out_buffer) (CallerFrees))
(let output
  (BufferOutputLit (BufferTensorCons (BufferTensorLit out_lt out_buffer) (BufferTensorNil))))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let program = format!("{preamble}\n\n{script}");
        let mut egraph = luminal::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the shape closure must collide in the terminal stratum");
        assert!(
            err.to_string().contains("shape-of"),
            "the error should come from the shape-of merge: {err}"
        );
    }

    /// The missing-bounds half of the iota gate (user ruling 2026-07-23:
    /// no seeding — the lattice's top stays ABSENCE): the CONSTRUCTION
    /// SITE emits the bounds demand as post-schedule checks, and a check
    /// whose lookup finds nothing fails the program loudly at the
    /// fixpoint. An iota over an expression nothing bounds — an unseeded
    /// IntVar — dies there.
    #[test]
    fn iota_with_no_derivable_bounds_dies_in_the_terminal_stratum() {
        let preamble = luminal_reference::assembled_program();
        let script = r#"
(let mystery_var (IntVar "mystery_var"))
(let unsafe_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let unbounded_iota (LogicalIota mystery_var unsafe_shape))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (saturate (run fixpoint-invariants)))
(check (= ?demanded_lower (lower-bound-of mystery_var)))
(check (= ?demanded_upper (upper-bound-of mystery_var)))
"#;
        let program = format!("{preamble}\n\n{script}");
        let mut egraph = luminal::egglog_snippet::new_egraph();
        let err = egraph
            .parse_and_run_program(None, &program)
            .expect_err("the constructor-site demand must fail on the absent bound");
        assert!(
            err.to_string().contains("lower-bound-of"),
            "the error should cite the absent bounds demand: {err}"
        );
    }

    /// A view is free: the consumer reads the PARENT's buffer directly, and
    /// the view op leaves no node in the plan (folded like a poison producer).
    #[test]
    fn view_reads_parent_buffer_with_zero_plan_nodes() {
        use luminal::bufferize::BufferNode;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        assert_eq!(
            plan.value_buffer[&v],
            plan.value_buffer[&x],
            "the view derives its parent's buffer:\n{}",
            plan.summary()
        );
        for idx in plan.dag.node_indices() {
            if let BufferNode::Compute { op, reads, .. } = &plan.dag[idx] {
                assert_ne!(
                    op.label(),
                    "MockView",
                    "views are folded:\n{}",
                    plan.summary()
                );
                // Storage nodes (alloc/free) read no operands; the invariant
                // under test is about the view's CONSUMER.
                if op.label() == "MockOp" {
                    assert_eq!(
                        reads[0], plan.value_buffer[&x],
                        "the consumer reads the parent buffer directly"
                    );
                }
            }
        }
    }

    /// A view of READ-ONLY storage is legal: it writes nothing, so the
    /// writability veto does not apply (previously any in-place candidate on a
    /// Read-granted buffer was refused — correct only for writers).
    #[test]
    fn view_of_read_only_input_is_admitted() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadOnly, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize_mock(&g.build()).expect("a view writes nothing — Read is fine");
        assert_eq!(plan.value_buffer[&v], plan.value_buffer[&x]);
    }

    /// REGION-BLIND conservatism, pinned (user decision): a writer into a
    /// viewed buffer conflicts with a live, unordered read THROUGH the view —
    /// even though the view's layout differs (the regions might be disjoint;
    /// without the interval oracle we must assume overlap). The optional
    /// writer yields out-of-place; the view, decided in phase 1, stands.
    #[test]
    fn writer_into_viewed_buffer_yields_out_of_place() {
        use luminal::bufferize::BufferId;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        let s = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v],
            &[("s", "rm")],
        )[0]
        .clone();
        let r_w = g.op(Box::new(MockOp::write_only_dest()), &[&x], &[("rW", "rm")])[0].clone();
        g.output(&r_w, "D");
        g.output(&s, "E");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        assert!(
            matches!(plan.value_buffer[&r_w], BufferId::Allocated(_)),
            "the writer yields to a fresh allocation:\n{}",
            plan.summary()
        );
        assert_eq!(plan.value_buffer[&v], plan.value_buffer[&x]);
    }

    /// Chained views resolve link by link to the root storage.
    #[test]
    fn chained_views_bind_through_to_the_root_buffer() {
        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v1 = g.op(Box::new(MockView), &[&x], &[("v1", "row0")])[0].clone();
        let v2 = g.op(Box::new(MockView), &[&v1], &[("v2", "cell0")])[0].clone();
        let r = g.op(
            Box::new(MockOp {
                reads: vec![true],
                ..Default::default()
            }),
            &[&v2],
            &[("r", "rm")],
        )[0]
        .clone();
        g.output(&r, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");
        assert_eq!(plan.value_buffer[&v2], plan.value_buffer[&x]);
        assert_eq!(plan.value_buffer[&v1], plan.value_buffer[&x]);
    }

    /// THE VIEW-OF-INPUT ZERO-COPY PIN (ruling 2026-08-27): a view bound
    /// straight to an output slot on its parent's own buffer returns
    /// zero-copy — zero copies, zero allocations — AND the slot's binding
    /// DISCLOSES the elected layout (the view's fold chain over the input
    /// buffer). The old silent-dense-misread hazard is closed by the layout
    /// field, never by refusing: the caller interprets the returned buffer
    /// under the returned layout.
    #[test]
    fn view_passthrough_to_output_slot() {
        use luminal::bufferize::{BufferId, BufferNode};

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let v = g.op(Box::new(MockView), &[&x], &[("v", "row0")])[0].clone();
        g.output(&v, "B");
        let graph = g.build();
        let plan = bufferize_mock(&graph).expect("bufferizes");

        assert!(
            plan.buffers
                .keys()
                .all(|id| matches!(id, BufferId::Boundary(_)))
        );
        assert!(
            plan.dag
                .node_indices()
                .all(|idx| !matches!(&plan.dag[idx], BufferNode::BufferCopy { .. })),
            "pass-through needs no copy:\n{}",
            plan.summary()
        );
        let slot = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
                _ => None,
            })
            .expect("one output slot");
        assert_eq!(
            slot.buffer,
            plan.value_buffer[&x],
            "the slot is backed by the input buffer:\n{}",
            plan.summary()
        );
        // THE DISCLOSURE (Option B): the binding carries the VIEW value's
        // own elected layout — a different function from its parent's, over
        // the same bytes. That is the whole of it: no hop chain, no dims,
        // no dtype. The caller interprets the returned buffer under it.
        let table = luminal::test_support::mock_layout_table(&graph);
        assert_eq!(&slot.layout, &table[&v], "the view's layout, verbatim");
        assert_ne!(table[&v], table[&x], "the view is not its parent");
    }

    /// ALLOC/FREE PHASE 3, the donated-input fixture end to end: a READ-ONLY
    /// ProgramFrees input (read-then-destroy — Access and FreedBy orthogonal)
    /// gets exactly one BufferFree consuming the donated buffer, and the
    /// certificate's lifetime arms hold: the free is ordered after the
    /// kernel's read (the anti edge), no alloc exists for caller storage,
    /// and the donated buffer backs no output slot.
    #[test]
    fn donated_input_fixture_frees_the_donated_buffer() {
        use luminal::bufferize::{BufferId, BufferNode};

        let graph = extract_fixture("boundary_donated_input.egg");
        let plan = bufferize_mock(&luminal::dps::dps_rewrite(&graph)).expect("bufferizes");

        assert!(
            plan.buffers
                .keys()
                .all(|id| matches!(id, BufferId::Boundary(_))),
            "no allocations:\n{}",
            plan.summary()
        );
        let launch: Vec<BufferId> = plan
            .dag
            .node_weights()
            .filter_map(|node| match node {
                BufferNode::BufferInput { slots } => {
                    Some(slots.iter().map(|slot| slot.buffer.clone()))
                }
                _ => None,
            })
            .flatten()
            .collect();
        assert_eq!(launch.len(), 1, "one caller input:\n{}", plan.summary());

        let mut frees = 0;
        for idx in plan.dag.node_indices() {
            match &plan.dag[idx] {
                BufferNode::BufferCopy { .. } => panic!("no copies:\n{}", plan.summary()),
                BufferNode::Compute { op, reads, .. } => {
                    if op.label() == "BufferFree" {
                        frees += 1;
                        assert_eq!(
                            reads[0],
                            launch[0],
                            "the free consumes the donated input buffer:\n{}",
                            plan.summary()
                        );
                    }
                }
                _ => {}
            }
        }
        assert_eq!(
            frees,
            1,
            "donated storage is freed exactly once:\n{}",
            plan.summary()
        );
    }

    /// COMMITTED-WRITER INTERFERENCE (review-confirmed miscompile, fixed):
    /// two ops each claim the SAME poison value as their write-only
    /// destination; r2 is dead, r1 is the output. Whichever candidate the
    /// bottom-up order decides first is admitted blind (the other's read set
    /// is not yet aliased); the reads-only scan then let the second one
    /// through too, and the seed bound BOTH unordered writers into the
    /// caller's output buffer — a scheduler picking the dead writer last
    /// left r2's bytes where the boundary promised r1. The committed-writer
    /// check (try_in_place step 3, MLIR's write-side interference) must
    /// reject the second candidate: the output value ends up the buffer's
    /// only writer, whichever decision order the toposort produces.
    #[test]
    fn unordered_second_writer_of_shared_destination_is_rejected() {
        use luminal::bufferize::BufferNode;

        let mut g = TestGraph::new();
        let p = g.op(Box::new(EmptyOp), &[], &[("p", "rm")])[0].clone();
        let r2 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&p],
            &[("r2", "rm")],
        )[0]
        .clone();
        let r1 = g.op(
            Box::new(MockOp {
                reads: vec![false],
                in_place_operand: Some(0),
                ..Default::default()
            }),
            &[&p],
            &[("r1", "rm")],
        )[0]
        .clone();
        g.output(&r1, "D");
        let plan = bufferize_mock(&g.build()).expect("bufferizes");

        // The two writers must not share storage: exactly one of the two
        // in-place candidates survives, so r1's buffer has ONE compute writer.
        assert_ne!(
            plan.value_buffer[&r1],
            plan.value_buffer[&r2],
            "unordered writers must not share storage:\n{}",
            plan.summary()
        );
        let writers_of_r1_buffer = plan
            .dag
            .node_indices()
            .filter(|&idx| {
                // A BufferAlloc lists the buffer but installs no binding —
                // it is not a writer in the sense under test.
                matches!(&plan.dag[idx], BufferNode::Compute { reads, writes, .. }
                    if !reads.is_empty() && writes.contains(&plan.value_buffer[&r1]))
            })
            .count();
        assert_eq!(
            writers_of_r1_buffer,
            1,
            "the output value's buffer has exactly one writer:\n{}",
            plan.summary()
        );
    }

    /// Multi-destination DPS (review finding: previously unexercised): an
    /// AddMulFused node run through dps_rewrite gains TWO destinations, each
    /// tied to its own result — the pairs must land in DISTINCT allocations,
    /// with reads[dest_j] == writes[j] for both.
    #[test]
    fn multi_destination_pairs_get_distinct_allocations() {
        use luminal::bufferize::BufferNode;
        use luminal::test_support::test_ops::AddMulFused;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "C", Access::ReadWrite, "rm");
        let results = g.op(
            Box::new(AddMulFused),
            &[&x, &y],
            &[("sum", "rm"), ("prod", "rm")],
        );
        let (sum, prod) = (results[0].clone(), results[1].clone());
        g.output(&sum, "D");
        g.output(&prod, "E");

        let rewritten = luminal::dps::dps_rewrite(&g.build());
        let plan = bufferize_mock(&rewritten).expect("bufferizes");

        // Each (poison, result) pair on its own allocation — never shared.
        assert_ne!(
            plan.value_buffer[&sum],
            plan.value_buffer[&prod],
            "{}",
            plan.summary()
        );
        // The fused compute reads both destinations it writes, positionally.
        let (reads, writes) = plan
            .dag
            .node_indices()
            .find_map(|idx| match &plan.dag[idx] {
                BufferNode::Compute {
                    op, reads, writes, ..
                } if op.label() == "AddMulFusedGeneric" => Some((reads.clone(), writes.clone())),
                _ => None,
            })
            .expect("fused DPS node present");
        assert_eq!(reads.len(), 4, "2 data + 2 destinations");
        assert_eq!(&reads[2], &writes[0]);
        assert_eq!(&reads[3], &writes[1]);
    }

    /// Slot tables render each declared Must tie as one full-width row
    /// (`in ↔ out`) whose single port serves both compass sides — the tied
    /// result's edge leaves the SAME row its destination entered. Pinned on
    /// the multi-tie op because no fixture extracts one: both pairs must
    /// span, and both results must dock at their tie rows' east sides.
    #[test]
    fn slot_tables_render_ties_as_spanning_rows() {
        use luminal::test_support::test_ops::AddMulFused;

        let mut g = TestGraph::new();
        let x = g.input("x", "B", Access::ReadWrite, "rm");
        let y = g.input("y", "C", Access::ReadWrite, "rm");
        let results = g.op(
            Box::new(AddMulFused),
            &[&x, &y],
            &[("sum", "rm"), ("prod", "rm")],
        );
        g.output(&results[0], "D");
        g.output(&results[1], "E");

        let rewritten = luminal::dps::dps_rewrite(&g.build());
        let dot = rewritten.to_dot();
        for span in ["dest0 ↔ out0</TD>", "dest1 ↔ out1</TD>"] {
            assert!(dot.contains(span), "missing tie row {span:?} in:\n{dot}");
        }
        for dock in [":p_dest0:e ->", ":p_dest1:e ->"] {
            assert!(
                dot.contains(dock),
                "tied result not docked at {dock:?} in:\n{dot}"
            );
        }
    }
}

#[cfg(test)]
mod intcoordvar_probe {
    //! Probes for the (IntCoordVar LogicalTensor i64) constructibility
    //! analysis. Preamble is never touched on disk; programs assemble in
    //! memory. Run: cargo test intcoordvar_probe -- --nocapture

    /// Step-by-step runner: prints each step's Ok/Err so every claim is
    /// separately observable.
    fn run_steps(egraph: &mut egglog::EGraph, steps: &[(&str, &str)]) {
        for (label, prog) in steps {
            match egraph.parse_and_run_program(None, prog) {
                Ok(lines) => {
                    println!("X {label}: Ok");
                    for l in lines {
                        println!("X {label} out: {l}");
                    }
                }
                Err(e) => println!("X {label}: Err: {e}"),
            }
        }
    }

    /// Two-sort mutually recursive mini-datatype (a stand-in for
    /// LogicalTensor <-> IndexMap <-> IntCoordVar). Emulates the
    /// self-referential coordinate tag via the name+union route and
    /// observes: cyclic-class closure under congruence, extraction, and
    /// the nonce cost (structurally identical views never merge).
    #[test]
    fn x_probe_cyclic_selftag() {
        let mut egraph = crate::egglog_snippet::new_egraph();
        run_steps(
            &mut egraph,
            &[
                (
                    "setup",
                    r#"
(datatype*
  (PT (PTInput String)
      (PTName String)
      (PTView PT PMap))
  (PMap (PMapLit PCoord))
  (PCoord (PCoordVar PT i64)))
(let data (PTInput "data"))
(let nm (PTName "v1"))
(let v (PTView data (PMapLit (PCoordVar nm 0))))
(union nm v)
"#,
                ),
                (
                    "cyclic-check",
                    "(check (= v (PTView data (PMapLit (PCoordVar v 0)))))",
                ),
                ("extract-v", "(extract v)"),
                ("extract-v-variants", "(extract v 4)"),
                (
                    "nonce-no-merge",
                    r#"
(let n1 (PTName "nonce1"))
(let n2 (PTName "nonce2"))
(let w1 (PTView data (PMapLit (PCoordVar n1 0))))
(let w2 (PTView data (PMapLit (PCoordVar n2 0))))
(union n1 w1)
(union n2 w2)
(fail (check (= w1 w2)))
"#,
                ),
                (
                    "hashcons-baseline",
                    r#"
(let u1 (PTView data (PMapLit (PCoordVar data 0))))
(let u2 (PTView data (PMapLit (PCoordVar data 0))))
(check (= u1 u2))
"#,
                ),
            ],
        );
    }

    /// Today's behavior on the user's own discriminating example: a
    /// 3-vector with slices v[0..2] and v[1..3]. Both slices' coordinate
    /// is the identical term (CoordVar out_shape 0); the VIEW terms stay
    /// distinct; an identically-written view hash-conses (free CSE).
    #[test]
    fn x_probe_slice_views_today() {
        let mut full = luminal_reference::assembled_program().to_string();
        full.push_str(
            r#"
(let vec_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprNil))))
(let v_in (LogicalTensorInputLit (LogicalIdLit "v") vec_shape (F32)))
(let coord (CoordVar out_shape 0))
(let slice_a (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons coord (IntExprNil)) vec_shape) out_shape))
(let slice_b (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons (IntAdd coord (IntLit 1)) (IntExprNil)) vec_shape) out_shape))
(let slice_a2 (LogicalIndexMapApply v_in
  (IndexMapLit (IntExprCons (CoordVar out_shape 0) (IntExprNil)) vec_shape) out_shape))
(let v_layout (RightMajorContiguousElementLayoutLit vec_shape (bits-of (F32))))
(let v_lt (LayoutTensorLit v_in v_layout))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#,
        );
        let mut egraph = crate::egglog_snippet::new_egraph();
        match egraph.parse_and_run_program(None, &full) {
            Ok(_) => println!("X slice-fixture: Ok"),
            Err(e) => {
                println!("X slice-fixture: Err: {e}");
                return;
            }
        }
        run_steps(
            &mut egraph,
            &[
                ("cse-check", "(check (= slice_a slice_a2))"),
                ("distinct-views", "(fail (check (= slice_a slice_b)))"),
                (
                    "shared-coord-term",
                    "(check (= coord (CoordVar out_shape 0)))",
                ),
                (
                    "corruption-pairs",
                    r#"
(fail (check (= (IntLit 0) (IntLit 1))))
(fail (check (= (IntLit 0) (IntLit 2))))
(fail (check (= (IntLit 0) (IntLit 3))))
(fail (check (= (IntLit 0) (IntLit 4))))
(fail (check (= (IntLit 0) (IntLit 5))))
(fail (check (= (IntLit 1) (IntLit 2))))
(fail (check (= (IntLit 1) (IntLit 3))))
(fail (check (= (IntLit 1) (IntLit 4))))
(fail (check (= (IntLit 1) (IntLit 5))))
(fail (check (= (IntLit 2) (IntLit 3))))
(fail (check (= (IntLit 2) (IntLit 4))))
(fail (check (= (IntLit 2) (IntLit 5))))
(fail (check (= (IntLit 3) (IntLit 4))))
(fail (check (= (IntLit 3) (IntLit 5))))
(fail (check (= (IntLit 4) (IntLit 5))))
"#,
                ),
            ],
        );
    }
}

/// The test harness's search budget — the SAME genetic algorithm as the
/// module-level ladder tests, sized for a suite of hundreds of graphs
/// (ruling 2026-08-06: everything in the main tree runs the genetic
/// implementation search — there is no plain-walk bypass). Deterministic
/// (fixed seed); 2 generations × 4 genomes exercises random genomes plus
/// the mutation step without profiling 64 candidates per differential.
pub fn harness_search_options() -> crate::implementation_search::ImplementationSearchOptions {
    crate::implementation_search::ImplementationSearchOptions {
        generations: 2,
        generation_size: 4,
        mutations: 2,
        trials: 1,
        seed: 0,
        search_log: false,
    }
}

#[cfg(test)]
mod stage4b_probes {
    use luminal::dtype::DType;

    /// (History: this module held the Step-4b degenerate-broadcast
    /// unsoundness pin — the stride recovery walks welding the zero
    /// class. Fixed by the testimony landing; the repro is now the LIVE
    /// regression `degenerate_broadcast_runs_clean` below.)
    ///
    /// PINNED KNOWN GAP (Step 4b): a PURE-IDENTITY graph — an input
    /// directly output(), no ops — panics the axis-extent-lock tripwire
    /// natively (input and output bindings share one BufferLit and one
    /// logical value). The model zoo's identity-add idiom (Topic E, yolo)
    /// exists for exactly this; the real fix is binding-level (4d/M4
    /// aliasing/donation design).
    #[test]
    fn pinned_pure_identity_output() {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(2, DType::F32);
        let b = a.output();
        let rt =
            luminal_reference::harness::run_reference(&cx, &[(a.id, vec![1.0f32, 2.0].into())]);
        let got = rt.get_f32(b.id).unwrap();
        assert_eq!(got, &vec![1.0, 2.0]);
    }

    /// Austin's stride-destructuring contract (chain world, 2026-08-04):
    /// live axes yield their stride class, stride-1 axes yield Unit, a
    /// zero slot on a live axis is the DETERMINED broadcast Some(Zero),
    /// and a provably extent-1 slot is None — the free parameter each
    /// consumer resolves for itself.
    #[test]
    fn chain_strides_destructure_contract() {
        use luminal::extractor::{ChainStride, chain_strides};
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil))))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 2) (IntExprCons (CoordVar osh 0) (IntExprNil))) psh) osh))
(let dsh (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 2) (IntExprNil)))))
(let d (RightMajorContiguousElementLayoutLit dsh (bits-of (F32))))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{body}", luminal_reference::assembled_program());
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &full)
            .expect("program runs");
        let serialized = egraph.serialize(egglog::SerializeConfig::default()).egraph;

        let by_let = |name: &str| {
            serialized
                .class_data
                .iter()
                .find(|(_, data)| data.extra.get("let").map(String::as_str) == Some(name))
                .map(|(class, _)| class.clone())
                .unwrap_or_else(|| panic!("let {name} not found"))
        };

        // Parent (2,3) right-major: strides [3, 1].
        let p = chain_strides(&serialized, &by_let("p")).expect("parent destructures");
        assert_eq!(p.len(), 2);
        assert!(matches!(p[0], Some(ChainStride::Expr(_))), "{p:?}");
        assert_eq!(p[1], Some(ChainStride::Unit), "{p:?}");

        // Degenerate (1,2): the extent-1 slot is the FREE parameter.
        let d = chain_strides(&serialized, &by_let("d")).expect("degenerate destructures");
        assert_eq!(
            d[0], None,
            "extent-1 slot must be the consumer's choice: {d:?}"
        );
        assert_eq!(d[1], Some(ChainStride::Unit), "{d:?}");

        // Broadcast view (2,5,3): [3, DETERMINED 0, 1].
        let v_logical = by_let("v");
        let view_layout = serialized
            .nodes
            .iter()
            .find_map(|(_, node)| {
                if node.op != "LayoutTensorLit" {
                    return None;
                }
                let logical = node.children.first()?;
                if serialized.nid_to_cid(logical) != &v_logical {
                    return None;
                }
                // The materializing copy pairs the SAME logical with an
                // RM target layout — skip it; the view's own layout is
                // the composed (non-contiguous) one.
                let layout = serialized.nid_to_cid(node.children.get(1)?).clone();
                let is_rm = serialized.nodes.iter().any(|(nid2, n2)| {
                    n2.op == "RightMajorContiguousElementLayoutLit"
                        && serialized.nid_to_cid(nid2) == &layout
                });
                if is_rm { None } else { Some(layout) }
            })
            .expect("view LayoutTensor exists");
        let v = chain_strides(&serialized, &view_layout).expect("view destructures");
        assert!(matches!(v[0], Some(ChainStride::Expr(_))), "{v:?}");
        assert_eq!(
            v[1],
            Some(ChainStride::Zero),
            "broadcast axis is determined: {v:?}"
        );
        assert_eq!(v[2], Some(ChainStride::Unit), "{v:?}");
    }

    // (The script-corpus gate and the assembled-program dump moved to
    // crates/luminal_reference/tests/corpus.rs with the reference registry
    // in Step B — they run the reference/testruntime assembly.)

    /// REJOIN-DIVERGENCE ROUND DRIVER (2026-08-10): the two sick mini
    /// graphs (full-anatomy gemma3: 5.8GB in 12s; MiniDit: 10+ min in
    /// free-join) bisected to a 5-line movement reproducer — slice two
    /// halves of a rank-3 tensor with a leading extent-1 axis, concat
    /// them back (pad+add), merge. sin/cos/pad/concat/split-merge alone
    /// are all clean (~90ms); this rejoin detonates. The driver runs
    /// the MAIN ruleset one round at a time, printing total tuples and
    /// the top-growing tables per round — the exploding table names the
    /// rule family. Bounded: 40 rounds, bail on a >200k-tuple round.
    /// Run: cargo test --release rejoin_divergence_probe -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic — run explicitly by name (release, bounded)"]
    fn rejoin_divergence_probe() {
        for lead in [1usize, 2usize] {
            eprintln!("[rejoin-probe] ===== lead extent {lead} =====");
            let mut cx = luminal::graph::Graph::new();
            let x = cx.tensor((lead, 8usize), DType::F32);
            let heads = x.split_dims(1, 4);
            let x1 = heads.slice_along(0..2, 2);
            let x2 = heads.slice_along(2..4, 2);
            let _out = x2.concat_along(x1, 2).merge_dims(1, 2).output();
            let (pre, _inputs, _outputs, _post, _labeled) = cx
                .logical
                .bound_parts(&luminal_reference::ReferenceBindings)
                .expect("recorder clean");
            let full = format!("{}\n\n{pre}", luminal_reference::assembled_program());
            let mut egraph = luminal::egglog_snippet::new_egraph();
            egraph
                .parse_and_run_program(None, &full)
                .expect("body loads");
            let sizes = |egraph: &mut egglog::EGraph| -> std::collections::HashMap<String, isize> {
                let out = egraph
                    .parse_and_run_program(None, "(print-size)")
                    .expect("sizes");
                let mut map = std::collections::HashMap::new();
                for chunk in &out {
                    let text = chunk.to_string();
                    // Post-bump engine prints parenthesized pairs:
                    // ((name count) (name count) ...). The old `name: count`
                    // line format is parsed as fallback.
                    for fragment in text.split('(') {
                        let fragment = fragment.trim().trim_end_matches(')');
                        if let Some((name, count)) = fragment.rsplit_once(' ') {
                            if let Ok(count) = count.trim().parse::<isize>() {
                                map.insert(name.trim().to_string(), count);
                            }
                        }
                    }
                    for line in text.lines() {
                        if let Some((name, count)) = line.rsplit_once(": ") {
                            if let Ok(count) = count.trim().parse::<isize>() {
                                map.insert(name.trim().to_string(), count);
                            }
                        }
                    }
                }
                map
            };
            // GROWTH-CHANNEL ACCOUNTING (Austin's root-cause experiment,
            // 2026-08-11): per round, separate the three channels —
            // NODES (spellings: IntAdd table size), CLASSES (new sub-sums:
            // distinct IntAdd e-classes in a serialization), and DEMAND
            // ROWS (subst-demand fan-out) — to name which growth LEADS at
            // ignition. Serialization runs only near/after ignition.
            let channel_counts = |egraph: &mut egglog::EGraph| -> (usize, usize, usize) {
                use egglog::SerializeConfig;
                let serialized = egraph.serialize(SerializeConfig::default()).egraph;
                let mut intadd_nodes = 0usize;
                let mut intadd_classes = std::collections::HashSet::new();
                let mut all_classes = std::collections::HashSet::new();
                for node in serialized.nodes.values() {
                    all_classes.insert(node.eclass.clone());
                    if node.op == "IntAdd" {
                        intadd_nodes += 1;
                        intadd_classes.insert(node.eclass.clone());
                    }
                }
                (intadd_nodes, intadd_classes.len(), all_classes.len())
            };
            let mut previous = sizes(&mut egraph);
            for round in 1..=150 {
                let start = std::time::Instant::now();
                let round_out = egraph
                    .parse_and_run_program(None, "(run 1)")
                    .expect("round runs");
                // Name the firing rules once the mint turns geometric.
                for chunk in &round_out {
                    let egglog::CommandOutput::RunSchedule(report) = chunk else {
                        continue;
                    };
                    let mut rules: Vec<(String, usize)> = report
                        .num_matches_per_rule
                        .iter()
                        .map(|(name, &matches)| (name.to_string(), matches))
                        .collect();
                    rules.sort_by_key(|(_, matches)| std::cmp::Reverse(*matches));
                    let hot: Vec<String> = rules
                        .iter()
                        .take(4)
                        .filter(|(_, matches)| *matches > 50)
                        .map(|(name, matches)| {
                            let flat: String =
                                name.split_whitespace().collect::<Vec<_>>().join(" ");
                            format!("x{matches} {}", flat.chars().take(90).collect::<String>())
                        })
                        .collect();
                    if !hot.is_empty() {
                        eprintln!("[rejoin-probe]   rules: {}", hot.join(" ‖ "));
                    }
                }
                let current = sizes(&mut egraph);
                let total: isize = current.values().sum();
                let mut deltas: Vec<(String, isize)> = current
                    .iter()
                    .map(|(name, &count)| {
                        (
                            name.clone(),
                            count - previous.get(name).copied().unwrap_or(0),
                        )
                    })
                    .filter(|(_, delta)| *delta != 0)
                    .collect();
                deltas.sort_by_key(|(_, delta)| -*delta);
                let grew: isize = deltas.iter().map(|(_, delta)| *delta).sum();
                let top: Vec<String> = deltas
                    .iter()
                    .take(6)
                    .map(|(name, delta)| format!("{name} {delta:+}"))
                    .collect();
                eprintln!(
                    "[rejoin-probe] round {round}: total {total} ({grew:+}) in {:.2}s | {}",
                    start.elapsed().as_secs_f64(),
                    top.join(", ")
                );
                // Channel accounting near ignition: spellings-per-class vs
                // class mint vs demand fan-out.
                if (36..=50).contains(&round) {
                    let (nodes, classes, total_classes) = channel_counts(&mut egraph);
                    let demand_rows = current.get("int-subst-demand").copied().unwrap_or(0);
                    let image_rows = current.get("int-subst-of").copied().unwrap_or(0);
                    eprintln!(
                        "[channels] round {round}: IntAdd nodes {nodes} / classes {classes} \
                     (spellings-per-class {:.2}) | all classes {total_classes} | \
                     int-subst-demand rows {demand_rows} | int-subst-of rows {image_rows}",
                        nodes as f64 / classes.max(1) as f64
                    );
                }
                // Specimen dumps at the pre-ignition and early-geometric
                // rounds: the ACTUAL IntAdd rows being bred (extracted
                // representative terms), for the divergence walkthrough.
                if round == 41 || round == 45 {
                    let dump = egraph
                        .parse_and_run_program(None, "(print-function IntAdd 18)")
                        .expect("dump");
                    eprintln!("[rejoin-probe] --- IntAdd rows @ round {round} ---");
                    for chunk in &dump {
                        for line in chunk.to_string().lines().take(18) {
                            let flat: String =
                                line.split_whitespace().collect::<Vec<_>>().join(" ");
                            eprintln!(
                                "[rejoin-probe]   {}",
                                flat.chars().take(200).collect::<String>()
                            );
                        }
                    }
                }
                if grew > 200_000 {
                    eprintln!("[rejoin-probe] BAIL: runaway round — divergence confirmed");
                    break;
                }
                if grew == 0 {
                    eprintln!("[rejoin-probe] SATURATED at round {round}");
                    break;
                }
                previous = current;
            }
        }
    }

    /// EXTENT-1 SPECIMEN (subst experiment, 2026-08-14): the
    /// composition-rows dossier's batch matmul (1,2,3)x(3,5) — the
    /// extent-1 lead that detonated the UNCOMMITTED derived-rows block.
    /// Committed roads must saturate it; this times the full schedule.
    /// Run: cargo test specimen_1235_full_schedule -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic — run explicitly by name"]
    fn specimen_1235_full_schedule() {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor((1usize, 2usize, 3usize), DType::F32);
        let b = cx.tensor((3usize, 5usize), DType::F32);
        let _out = a.matmul(b).output();
        let (pre, _is, _os, post, _labeled) = cx
            .logical
            .bound_parts(&luminal_reference::ReferenceBindings)
            .expect("recorder clean");
        let program = format!(
            "{}\n\n{pre}{}{post}",
            luminal_reference::assembled_program(),
            luminal_reference::ReferenceBindings::SCHEDULE
        );
        let start = std::time::Instant::now();
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &program)
            .expect("specimen saturates");
        eprintln!(
            "[specimen-1235] full schedule green in {:.2}s",
            start.elapsed().as_secs_f64()
        );
    }

    /// SATURATION A/B REPORT (subst experiment): per specimen, time the
    /// run-schedule ALONE (program pre-loaded) and report e-graph size —
    /// total tuples (print-size sum) and distinct e-classes (serialized).
    /// Run: cargo test --release saturation_ab_report -- --ignored --nocapture
    #[test]
    #[ignore = "measurement — run explicitly by name (release)"]
    fn saturation_ab_report() {
        let specimens: Vec<(&str, String)> = vec![
            ("slice_pad(27x10)", {
                let mut cx = luminal::graph::Graph::new();
                let a = cx.tensor((27usize, 10usize), DType::F32);
                let _out = a
                    .slice((2..6, 7..10))
                    .pad(((1usize, 2usize), (1usize, 0usize)), 0.)
                    .output();
                let (pre, _is, _os, _post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                pre
            }),
            ("batch_matmul(2,3,4)x(4,5)", {
                let mut cx = luminal::graph::Graph::new();
                let a = cx.tensor((2usize, 3usize, 4usize), DType::F32);
                let b = cx.tensor((4usize, 5usize), DType::F32);
                let _out = a.matmul(b).output();
                let (pre, _is, _os, _post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                pre
            }),
            ("specimen(1,2,3)x(3,5)", {
                let mut cx = luminal::graph::Graph::new();
                let a = cx.tensor((1usize, 2usize, 3usize), DType::F32);
                let b = cx.tensor((3usize, 5usize), DType::F32);
                let _out = a.matmul(b).output();
                let (pre, _is, _os, _post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                pre
            }),
            ("rejoin_lead1(1,8)", {
                let mut cx = luminal::graph::Graph::new();
                let x = cx.tensor((1usize, 8usize), DType::F32);
                let heads = x.split_dims(1, 4);
                let x1 = heads.slice_along(0..2, 2);
                let x2 = heads.slice_along(2..4, 2);
                let _out = x2.concat_along(x1, 2).merge_dims(1, 2).output();
                let (pre, _is, _os, _post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                pre
            }),
        ];
        for (name, pre) in &specimens {
            let mut egraph = luminal::egglog_snippet::new_egraph();
            let body = format!("{}\n\n{pre}", luminal_reference::assembled_program());
            egraph
                .parse_and_run_program(None, &body)
                .expect("body loads");
            let start = std::time::Instant::now();
            egraph
                .parse_and_run_program(None, luminal_reference::ReferenceBindings::SCHEDULE)
                .expect("schedule saturates");
            let sat = start.elapsed().as_secs_f64();
            let out = egraph
                .parse_and_run_program(None, "(print-size)")
                .expect("sizes");
            let mut tuples: isize = 0;
            for chunk in &out {
                let text = chunk.to_string();
                for fragment in text.split('(') {
                    let fragment = fragment.trim().trim_end_matches(')');
                    if let Some((_, count)) = fragment.rsplit_once(' ') {
                        if let Ok(count) = count.trim().parse::<isize>() {
                            tuples += count;
                        }
                    }
                }
            }
            use egglog::SerializeConfig;
            let serialized = egraph.serialize(SerializeConfig::default()).egraph;
            let mut classes = std::collections::HashSet::new();
            let mut intadd = 0usize;
            for node in serialized.nodes.values() {
                classes.insert(node.eclass.clone());
                if node.op == "IntAdd" {
                    intadd += 1;
                }
            }
            eprintln!(
                "[sat-ab] {name}: saturation {sat:.3}s | tuples {tuples} | \
                 classes {} | IntAdd nodes {intadd}",
                classes.len()
            );
        }
    }

    /// SATURATION PROFILER (Austin commissioned 2026-08-05): run each
    /// specimen's FULL schedule once and read egglog's own RunReport —
    /// per-rule search+apply times and per-ruleset totals. Suspect under
    /// test: the a9fb4b55 commit (map-range lock + diagonal arms)
    /// roughly doubled movement-shard wall time.
    /// Run: RRX=1 cargo test rrx_profile -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic — run explicitly by name"]
    fn rrx_profile() {
        let specimens: Vec<(&str, String)> = vec![
            ("slice_pad(27x10)", {
                let mut cx = luminal::graph::Graph::new();
                let a = cx.tensor((27usize, 10usize), DType::F32);
                let _out = a
                    .slice((2..6, 7..10))
                    .pad(((1usize, 2usize), (1usize, 0usize)), 0.)
                    .output();
                let (pre, _is, _os, post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                format!(
                    "{pre}{}{post}",
                    luminal_reference::ReferenceBindings::SCHEDULE
                )
            }),
            ("batch_matmul(2,3,4,5)", {
                let mut cx = luminal::graph::Graph::new();
                let a = cx.tensor((2usize, 3usize, 4usize), DType::F32);
                let b = cx.tensor((4usize, 5usize), DType::F32);
                let _out = a.matmul(b).output();
                let (pre, _is, _os, post, _labeled) = cx
                    .logical
                    .bound_parts(&luminal_reference::ReferenceBindings)
                    .expect("recorder clean");
                format!(
                    "{pre}{}{post}",
                    luminal_reference::ReferenceBindings::SCHEDULE
                )
            }),
        ];
        // The fixed floor: parse + declare the assembled preamble alone.
        {
            let preamble = luminal_reference::assembled_program();
            let start = std::time::Instant::now();
            let mut egraph = luminal::egglog_snippet::new_egraph();
            egraph
                .parse_and_run_program(None, preamble)
                .expect("preamble loads");
            eprintln!(
                "[prof] ===== preamble only (parse+declare, no body/schedule): {:.2}s, {} lines =====",
                start.elapsed().as_secs_f64(),
                preamble.lines().count()
            );
        }
        for (name, body) in specimens {
            let full = format!("{}\n\n{body}", luminal_reference::assembled_program());
            let mut egraph = luminal::egglog_snippet::new_egraph();
            let start = std::time::Instant::now();
            let outputs = egraph
                .parse_and_run_program(None, &full)
                .expect("program runs");
            let wall = start.elapsed().as_secs_f64();
            eprintln!("\n[prof] ===== {name}: total wall {wall:.2}s =====");
            for chunk in &outputs {
                let egglog::CommandOutput::RunSchedule(report) = chunk else {
                    continue;
                };
                let mut rulesets: Vec<(String, f64)> = report
                    .search_and_apply_time_per_ruleset
                    .iter()
                    .map(|(name, time)| (name.to_string(), time.as_secs_f64()))
                    .collect();
                rulesets.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (ruleset, secs) in rulesets.iter().take(6) {
                    let label = if ruleset.is_empty() {
                        "(default)"
                    } else {
                        ruleset
                    };
                    let rebuild = report
                        .rebuild_time_per_ruleset
                        .iter()
                        .find(|(name, _)| {
                            name.as_ref() == label || (label == "(default)" && name.is_empty())
                        })
                        .map(|(_, time)| time.as_secs_f64())
                        .unwrap_or(0.0);
                    let merge = report
                        .merge_time_per_ruleset
                        .iter()
                        .find(|(name, _)| {
                            name.as_ref() == label || (label == "(default)" && name.is_empty())
                        })
                        .map(|(_, time)| time.as_secs_f64())
                        .unwrap_or(0.0);
                    eprintln!(
                        "[prof] ruleset {label:<28} search+apply {secs:>7.3}s  rebuild {rebuild:>7.3}s  merge {merge:>7.3}s"
                    );
                }
                let mut rules: Vec<(String, f64, usize)> = report
                    .search_and_apply_time_per_rule
                    .iter()
                    .map(|(name, time)| {
                        let matches = report.num_matches_per_rule.get(name).copied().unwrap_or(0);
                        (name.to_string(), time.as_secs_f64(), matches)
                    })
                    .collect();
                rules.sort_by(|a, b| b.1.total_cmp(&a.1));
                for (rule, secs, matches) in rules.iter().take(15) {
                    let flat: String = rule.split_whitespace().collect::<Vec<_>>().join(" ");
                    let head: String = flat.chars().take(110).collect();
                    eprintln!("[prof] {secs:>7.3}s x{matches:<6} {head}");
                }
            }
        }
    }

    /// G7 MAP-ENTRY RANGE LOCK (Austin ruled 2026-08-05): the
    /// adversary's degenerate bypass — a map reading an extent-1 parent
    /// axis at a 5-extent coordinate — must now DIE LOUDLY in the
    /// invariants stratum instead of composing silently.
    #[test]
    fn map_range_lock_fires_on_degenerate_bypass() {
        let body = r#"
(let psh (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let plog (LogicalTensorInputLit (LogicalIdLit "p") psh (F32)))
(let p (RightMajorContiguousElementLayoutLit psh (bits-of (F32))))
(let plt (LayoutTensorLit plog p))
(let osh (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 4) (IntExprNil)))))
(let v (LogicalIndexMapApply plog (IndexMapLit (IntExprCons (CoordVar osh 1) (IntExprCons (CoordVar osh 0) (IntExprNil))) psh) osh))
(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))
"#;
        let full = format!("{}\n\n{body}", luminal_reference::assembled_program());
        let err = luminal::egglog_snippet::new_egraph()
            .parse_and_run_program(None, &full)
            .expect_err("out-of-range map over a degenerate axis must panic");
        assert!(
            err.to_string().contains("range lock"),
            "wrong failure: {err}"
        );
    }

    /// LIVE REGRESSION (was the pinned Step-4b unsoundness): the
    /// rank-0 -> [1] broadcast that used to weld the zero class (the
    /// recovery walks turned "presentations [1] and [0] are both valid"
    /// into IntExpr equalities 0 ≡ 1 ≡ 32). With recovery deleted and
    /// strided-presentation testimony in its place, it runs end-to-end
    /// with every tripwire live.
    #[test]
    fn degenerate_broadcast_runs_clean() {
        let mut cx = luminal::graph::Graph::new();
        let a = cx.tensor(1, DType::F32);
        let b = (a * 2.0).output();
        let rt = luminal_reference::harness::run_reference(&cx, &[(a.id, vec![0.5f32].into())]);
        let got = rt.get_f32(b.id).unwrap();
        assert!((got[0] - 1.0).abs() < 1e-6, "{got:?}");
    }
}

/// Deep-dive landing 1/6 regression: the subst-of in-range guard.
/// LANDED form = smallest-possible-box interval arm + structural
/// same-extent arm; LEGACY form (reconstructed by reverse surgery) =
/// entry within the coordinate's CURRENT interval. Five standalone
/// egglog scenarios, exact verdicts asserted for both forms.
///
/// On the two-phase scenarios: pins happen ONCE, pre-run (Austin's
/// ruling — buckets are separate egglog programs; there is no
/// operational mid-run pin). The late `(set (upper-bound-of ...))`
/// models a DERIVED tightening arriving late in saturation, plus
/// seed-order sensitivity generally: the legacy guard's verdict depends
/// on WHEN a monotone fact lands; the landed guard's does not. The core
/// defect is fixpoint-level regardless: legacy admits dyn-extent
/// entries that are out-of-box under admissible valuations.
#[cfg(test)]
mod subst_guard_study {
    const LANDED_GUARD: &str = "    (= ?entry_lower (lower-bound-of ?entry))\n    (= ?entry_upper (upper-bound-of ?entry))\n    (>= ?entry_lower (bigint 0))\n    (= ?extent_lower (lower-bound-of ?extent))\n    (<= ?entry_upper (- ?extent_lower (bigint 1)))";

    const LEGACY_GUARD: &str = "    (= ?entry_lower (lower-bound-of ?entry))\n    (= ?entry_upper (upper-bound-of ?entry))\n    (= ?coord_lower (lower-bound-of ?expr))\n    (= ?coord_upper (upper-bound-of ?expr))\n    (>= ?entry_lower ?coord_lower)\n    (<= ?entry_upper ?coord_upper)";

    /// The structural arm's rule text as landed (comment elided; the
    /// unique premise pair suffices as the removal anchor).
    const STRUCTURAL_ARM_ANCHOR: &str = "(rule\n  (\n    (int-subst-demand ?expr ?map)\n    (= ?expr (CoordVar ?source_shape ?axis))\n    (= ?map (IndexMapLit ?entries ?source_shape))\n    (= ?entry (expr-list-nth-from-end ?entries ?axis))\n    (= ?source_shape (ShapeLit ?source_dims))\n    (= ?extent (expr-list-nth-from-end ?source_dims ?axis))\n    (= ?entry (CoordVar ?entry_shape ?entry_axis))\n    (= ?entry_shape (ShapeLit ?entry_dims))\n    (= ?entry_extent (expr-list-nth-from-end ?entry_dims ?entry_axis))\n    (= ?entry_extent ?extent)\n  )\n  ((union (int-subst-of ?expr ?map) ?entry))\n)";

    fn variant(text: &str, name: &str) -> String {
        match name {
            "landed" => {
                assert!(text.contains(LANDED_GUARD), "landed guard text drifted");
                assert!(
                    text.contains(STRUCTURAL_ARM_ANCHOR),
                    "structural arm text drifted"
                );
                text.to_string()
            }
            "legacy" => {
                assert!(text.contains(LANDED_GUARD), "landed guard text drifted");
                let t = text.replacen(LANDED_GUARD, LEGACY_GUARD, 1);
                assert!(
                    t.contains(STRUCTURAL_ARM_ANCHOR),
                    "structural arm text drifted"
                );
                t.replacen(
                    STRUCTURAL_ARM_ANCHOR,
                    "; [study: structural arm removed]",
                    1,
                )
            }
            other => panic!("unknown variant {other}"),
        }
    }

    fn scenarios() -> Vec<(&'static str, String)> {
        let sg1_common = "\
(let sgn (IntVar \"sgn\"))\n\
(set (lower-bound-of sgn) (bigint 1))\n\
(set (upper-bound-of sgn) (bigint 8))\n\
(let sg_cout_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))\n\
(let sg_cout (CoordVar sg_cout_shape 0))\n\
(let sg_entry (IntAdd sg_cout (IntLit 5)))\n\
(let sg_src (ShapeLit (IntExprCons sgn (IntExprNil))))\n\
(let sg_map (IndexMapLit (IntExprCons sg_entry (IntExprNil)) sg_src))\n\
(let sg_coord (CoordVar sg_src 0))\n\
(int-subst-demand sg_coord sg_map)\n\
(run-schedule (saturate (saturate (run)) (run subst-walk)))\n";
        let sg4_common = "\
(let s4n (IntVar \"s4n\"))\n\
(set (lower-bound-of s4n) (bigint 1))\n\
(set (upper-bound-of s4n) (bigint 8))\n\
(let s4_entry (IntLit 5))\n\
(let s4_src (ShapeLit (IntExprCons s4n (IntExprNil))))\n\
(let s4_map (IndexMapLit (IntExprCons s4_entry (IntExprNil)) s4_src))\n\
(let s4_coord (CoordVar s4_src 0))\n\
(int-subst-demand s4_coord s4_map)\n\
(run-schedule (saturate (saturate (run)) (run subst-walk)))\n";
        vec![
            (
                "sg1_admits",
                format!("{sg1_common}(check (= (int-subst-of sg_coord sg_map) sg_entry))\n"),
            ),
            (
                "sg1_tighten",
                format!(
                    "{sg1_common}(set (upper-bound-of sgn) (bigint 1))\n(run-schedule (saturate (saturate (run)) (run subst-walk)))\n"
                ),
            ),
            (
                "sg2_static",
                "\
(let s2_cout_shape (ShapeLit (IntExprCons (IntLit 3) (IntExprNil))))\n\
(let s2_cout (CoordVar s2_cout_shape 0))\n\
(let s2_entry (IntAdd s2_cout (IntLit 1)))\n\
(let s2_src (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))\n\
(let s2_map (IndexMapLit (IntExprCons s2_entry (IntExprNil)) s2_src))\n\
(let s2_coord (CoordVar s2_src 0))\n\
(int-subst-demand s2_coord s2_map)\n\
(run-schedule (saturate (saturate (run)) (run subst-walk)))\n\
(check (= (int-subst-of s2_coord s2_map) s2_entry))\n"
                    .to_string(),
            ),
            (
                "sg3_identity",
                "\
(let s3n (IntVar \"s3n\"))\n\
(set (lower-bound-of s3n) (bigint 1))\n\
(set (upper-bound-of s3n) (bigint 8))\n\
(let s3_entry_shape (ShapeLit (IntExprCons s3n (IntExprCons s3n (IntExprNil)))))\n\
(let s3_entry (CoordVar s3_entry_shape 1))\n\
(let s3_src (ShapeLit (IntExprCons s3n (IntExprNil))))\n\
(let s3_map (IndexMapLit (IntExprCons s3_entry (IntExprNil)) s3_src))\n\
(let s3_coord (CoordVar s3_src 0))\n\
(int-subst-demand s3_coord s3_map)\n\
(run-schedule (saturate (saturate (run)) (run subst-walk)))\n\
(check (= (int-subst-of s3_coord s3_map) s3_entry))\n"
                    .to_string(),
            ),
            (
                "sg4_admits",
                format!(
                    "{s4}(check (= (int-subst-of s4_coord s4_map) s4_entry))\n",
                    s4 = sg4_common
                ),
            ),
            (
                "sg4_tighten",
                format!(
                    "{s4}(set (upper-bound-of s4n) (bigint 1))\n(run-schedule (saturate (saturate (run)) (run subst-walk)))\n",
                    s4 = sg4_common
                ),
            ),
        ]
    }

    fn run_verdict(text: &str) -> &'static str {
        let mut egraph = luminal::egglog_snippet::new_egraph();
        match egraph.parse_and_run_program(None, text) {
            Ok(_) => "ok",
            Err(err) => {
                let e = err.to_string();
                if e.contains("crossed IntExpr bounds") {
                    "panic-crossed-bounds"
                } else if e.contains("distinct integer literals") {
                    "panic-distinct-literals"
                } else if e.contains("Check failed") || e.contains("check failed") {
                    "refused"
                } else {
                    eprintln!("STUDY-ERR detail: {}", &e[..e.len().min(400)]);
                    "other-error"
                }
            }
        }
    }

    #[test]
    fn subst_guard_landed_vs_legacy() {
        // (variant, scenario) -> expected verdict. "refused" = the image
        // correctly did not fire (fail-closed); the legacy admissions and
        // detonations are the documented unsoundness.
        let expected = [
            ("landed", "sg1_admits", "refused"),
            ("landed", "sg1_tighten", "ok"),
            ("landed", "sg2_static", "ok"),
            ("landed", "sg3_identity", "ok"),
            ("landed", "sg4_admits", "refused"),
            ("landed", "sg4_tighten", "ok"),
            ("legacy", "sg1_admits", "ok"),
            ("legacy", "sg1_tighten", "panic-crossed-bounds"),
            ("legacy", "sg2_static", "ok"),
            ("legacy", "sg3_identity", "ok"),
            ("legacy", "sg4_admits", "ok"),
            ("legacy", "sg4_tighten", "panic-crossed-bounds"),
        ];
        let base = luminal_reference::assembled_program();
        for var_name in ["landed", "legacy"] {
            let varied = variant(base, var_name);
            for (scen_name, tail) in scenarios() {
                let verdict = run_verdict(&format!("{varied}\n{tail}"));
                let want = expected
                    .iter()
                    .find(|(v, s, _)| *v == var_name && *s == scen_name)
                    .map(|(_, _, w)| *w)
                    .unwrap();
                eprintln!("STUDY {var_name:>7} | {scen_name:<12} | {verdict}");
                assert_eq!(
                    verdict, want,
                    "guard regression: {var_name}/{scen_name} expected {want}, got {verdict}"
                );
            }
        }
    }

    /// INTERFACE-SURFACE PIN (Stage 1, 2026-08-12): input_specs
    /// returns the PRISTINE label plus geometry; output_named surfaces
    /// through output_specs and the model text; duplicate output names
    /// poison the graph loudly.
    #[test]
    fn interface_specs_report_pristine_labels_and_named_outputs() {
        use luminal::prelude::{DType, Graph};
        let mut cx = Graph::default();
        let a = cx.named_tensor("blocks.0.wq.weight", (2usize, 3usize), DType::F32);
        let b = cx.tensor((2usize, 3usize), DType::F32);
        let _ = (a + b).output_named("logits");

        let inputs = cx.logical.input_specs();
        assert_eq!(inputs.len(), 2);
        assert_eq!(inputs[0].label, "blocks.0.wq.weight");
        assert_eq!(inputs[0].id, a.id, "spec id is the staging key");
        assert_eq!(
            inputs[0]
                .dims
                .iter()
                .map(|d| d.to_usize().unwrap())
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(inputs[0].dtype, DType::F32);
        assert_eq!(
            inputs[1].label, "arg.0",
            "anonymous inputs auto-name in declaration order (Stage 3)"
        );

        let outputs = cx.logical.output_specs();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].label, "logits");
        let text = cx.logical.model_text().expect("records clean");
        assert!(
            text.contains("(LogicalTensorNamed (LogicalIdLit \"logits\"))"),
            "authored output name reaches the IR text"
        );

        let c = cx.tensor((2usize, 3usize), DType::F32);
        let _ = c.output_named("logits");
        assert!(
            cx.logical
                .model_text()
                .unwrap_err()
                .contains("duplicate output name"),
            "second \"logits\" poisons loudly"
        );

        // Stage 3: duplicate INPUT labels poison at the choke point.
        let mut cx2 = Graph::default();
        let _a = cx2.named_tensor("blocks.0.wq.weight", (2usize,), DType::F32);
        let _b = cx2.named_tensor("blocks.0.wq.weight", (2usize,), DType::F32);
        assert!(
            cx2.logical
                .model_text()
                .unwrap_err()
                .contains("duplicate input label"),
            "duplicate input label poisons loudly"
        );
    }

    /// COMPOUND-DIM PROBE (2026-08-12, Austin's challenge: "do we
    /// actually know that to be true?"): a dim of `a + b` — an
    /// arbitrary IntExpr, not an atom — records, saturates, and
    /// EXECUTES under pins. Two spellings (`a + b` and `b + a`) live in
    /// one graph: each pins to [5,5] via interval arithmetic and the
    /// [n,n] collapse unions BOTH with (IntLit 5) — the spellings
    /// unify through the bounds lattice, no ring axioms involved.
    #[test]
    fn compound_dim_extents_record_saturate_and_run() {
        use luminal::prelude::{DType, Graph};
        use luminal::shape::IntExpr;
        let mut cx = Graph::default();
        cx.set_dim('a', 2);
        cx.set_dim('b', 3);
        let ab = IntExpr::from('a') + IntExpr::from('b');
        let ba = IntExpr::from('b') + IntExpr::from('a');
        let x = cx.named_tensor("x", (ab,), DType::F32);
        let y = cx.named_tensor("y", (ba,), DType::F32);
        let doubled = (x + x).output();
        let summed = (y * y).output();
        // MIXED-SPELLING elementwise: a+b meets b+a directly — the
        // frontend accepts via egglog_equal (ruling 2026-08-13) and the
        // egglog side unifies the extents through the pin collapse.
        let mixed = (x + y).output();

        let x_vals = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let y_vals = vec![2.0f32, 3.0, 4.0, 5.0, 6.0];
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, x_vals.clone().into()), (y.id, y_vals.clone().into())],
        );
        let out_x = rt.get_f32(doubled.id).expect("compound-dim output runs");
        let out_y = rt.get_f32(summed.id).expect("second spelling runs");
        for (i, v) in out_x.iter().enumerate() {
            assert_eq!(*v, x_vals[i] * 2.0);
        }
        for (i, v) in out_y.iter().enumerate() {
            assert_eq!(*v, y_vals[i] * y_vals[i]);
        }
        let out_mixed = rt.get_f32(mixed.id).expect("mixed-spelling add runs");
        for (i, v) in out_mixed.iter().enumerate() {
            assert_eq!(*v, x_vals[i] + y_vals[i]);
        }
    }

    /// SHAPE-CONTRACT PIN (squeeze option 3, ruling 2026-08-13): a
    /// symbolic-extent squeeze records unconditionally and the
    /// post-saturation invariant decides per binding — the [1,1] pin
    /// discharges it; a [2,2] pin refuses at saturation.
    #[test]
    fn squeeze_contract_discharges_at_one_and_refuses_otherwise() {
        use luminal::prelude::{DType, Graph};
        let mut cx = Graph::default();
        cx.set_dim('s', 1);
        let x = cx.named_tensor("x", ('s', 3usize), DType::F32);
        let out = (x.squeeze(0) * 2.0).output();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, vec![1.0f32, 2.0, 3.0].into())],
        );
        assert_eq!(rt.get_f32(out.id).unwrap(), &[2.0, 4.0, 6.0]);

        let mut cx = Graph::default();
        cx.set_dim('s', 2);
        let x = cx.named_tensor("x", ('s', 3usize), DType::F32);
        let _ = (x.squeeze(0) * 2.0).output();
        let mut rt = luminal_reference::ReferenceRuntime::load(&cx).expect("records + loads");
        let data: rustc_hash::FxHashMap<_, _> = [(
            x.id,
            luminal::buffer_tensor_ir::TypedBuffer::from(vec![0.0f32; 6]),
        )]
        .into_iter()
        .collect();
        rt.bind_dyn_range('s', 2, 2).expect("bind");
        let err = rt
            .search(
                &data,
                &luminal::implementation_search::ImplementationSearchOptions::default(),
            )
            .expect_err("extent 2 violates the squeeze contract");
        assert!(
            format!("{err:#}").contains("axis extent must be exactly 1"),
            "the labeled door names the squeeze contract: {err:#}"
        );
    }

    /// SYMBOLIC-EXTENT PAD + CONCAT (the dim-grammar widening's
    /// proving test): both record, saturate under pins, and execute.
    #[test]
    fn symbolic_pad_and_concat_record_and_run() {
        use luminal::prelude::{DType, Graph};
        let mut cx = Graph::default();
        cx.set_dim('s', 3);
        cx.set_dim('t', 2);
        let x = cx.named_tensor("x", ('s',), DType::F32);
        let y = cx.named_tensor("y", ('t',), DType::F32);
        let padded = x.pad_along(1, 1, 0, 0.0).output();
        let joined = x.concat_along(y, 0).output();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[
                (x.id, vec![1.0f32, 2.0, 3.0].into()),
                (y.id, vec![10.0f32, 20.0].into()),
            ],
        );
        assert_eq!(rt.get_f32(padded.id).unwrap(), &[0.0, 1.0, 2.0, 3.0, 0.0]);
        assert_eq!(rt.get_f32(joined.id).unwrap(), &[1.0, 2.0, 3.0, 10.0, 20.0]);
    }

    /// UNFOLD WINDOW CONTRACT: a symbolic dim records with the
    /// kernel-fits invariant; the fitting pin runs, the violating pin
    /// refuses with the NAMED door.
    #[test]
    fn unfold_window_contract_discharges_and_names_its_door() {
        use luminal::prelude::{DType, Graph};
        let mut cx = Graph::default();
        cx.set_dim('s', 5);
        let x = cx.named_tensor("x", ('s',), DType::F32);
        let out = x.unfold((3usize,), (1usize,), (1usize,)).sum(1).output();
        let rt = luminal_reference::harness::run_reference(
            &cx,
            &[(x.id, vec![1.0f32, 2.0, 3.0, 4.0, 5.0].into())],
        );
        assert_eq!(rt.get_f32(out.id).unwrap(), &[6.0, 9.0, 12.0]);

        let mut cx = Graph::default();
        cx.set_dim('s', 2);
        let x = cx.named_tensor("x", ('s',), DType::F32);
        let _ = x.unfold((3usize,), (1usize,), (1usize,)).sum(1).output();
        let mut rt = luminal_reference::ReferenceRuntime::load(&cx).expect("records + loads");
        let data: rustc_hash::FxHashMap<_, _> = [(
            x.id,
            luminal::buffer_tensor_ir::TypedBuffer::from(vec![0.0f32; 2]),
        )]
        .into_iter()
        .collect();
        rt.bind_dyn_range('s', 2, 2).expect("bind");
        let err = rt
            .search(
                &data,
                &luminal::implementation_search::ImplementationSearchOptions::default(),
            )
            .expect_err("kernel 3 cannot fit in extent 2");
        assert!(
            format!("{err:#}").contains("unfold window on axis 0"),
            "the labeled door names the unfold contract: {err:#}"
        );
    }
}

/// RING IGNITION BATTERY — the G2 fence's permanent regression suite
/// (Austin ruling 2026-08-27: "put the G2 patch in, and we can add the
/// battery as a regression suite").
///
/// THE RING: the core preamble's Int arithmetic rewrite family —
/// commutativity, the G2/certificate-guarded associativity and
/// distributivity, literal folds, and the bounds lattice whose [n,n]
/// pin-collapse rule folds point-bound classes to literals.
///
/// THE IGNITION PREDICATE (P1–P4), from the Ring Ignition analysis
/// (raw-spelling repro clone + battery logs, 2026-08-27). A spelling
/// ignites the ring when ALL FOUR hold:
///   P1  a NESTED sum/product spelling is present RAW in the egraph
///       (e.g. (s-3)+1 as IntAdd(IntAdd(s, IntMul(-1,3)), 1)) — not
///       pre-folded by the frontend;
///   P2  the inner node's leaves are point-bound ([n,n] seeds), so the
///       pin-collapse rule folds the inner class to a literal;
///   P3  the pinned literal's value COLLIDES with another resident
///       atom's class (a leaf var's pin, a literal operand, or the
///       root's own fold — the value-coincidence D group);
///   P4  an associativity/distributivity rule can regroup THROUGH the
///       collided constant class — each regroup feeds the fold, the
///       fold re-pins, the pin re-collides: the orbit that minted fresh
///       literal atoms every round (pre-fence: 300k–4.4M tuples by
///       r11–r15, battery.log).
/// The G2 fence starves P4: three premises ((= ?x_lower
/// (lower-bound-of ?x)) (= ?x_upper (upper-bound-of ?x)) (< ?x_lower
/// ?x_upper)) on Assoc-IntAdd, Assoc-IntMul, and Distributivity-expand
/// refuse point-bound matched inner classes — a known constant is owned
/// by the fold/pin machinery, and associativity is equivalence
/// discovery for UNKNOWNS.
///
/// EVERY test is HARD-BOUNDED: a round cap, a tuple ceiling checked
/// after every single round, and a per-round wall-clock bail. If the
/// fence is ever weakened, the igniter configs cross their tuple
/// ceilings within ~12 rounds and the tests FAIL LOUDLY (never hang).
/// That is this suite's whole purpose.
///
/// RAW SPELLINGS BY HAND: this tree's frontend simplify folds
/// recorder-built extents ((s-3)+1 arrives as s-2), so recorder-built
/// fixtures would exercise the SHIELD (the fold), not the RING. Every
/// fixture below injects hand-authored egglog text; `drive` asserts the
/// injected spelling reaches the program verbatim.
#[cfg(test)]
mod ring_ignition_battery {
    /// All function-table sizes (`(print-size)`), both engine print
    /// formats parsed.
    fn table_sizes(egraph: &mut egglog::EGraph) -> std::collections::BTreeMap<String, isize> {
        let out = egraph
            .parse_and_run_program(None, "(print-size)")
            .expect("sizes");
        let mut map = std::collections::BTreeMap::new();
        for chunk in &out {
            let text = chunk.to_string();
            for fragment in text.split('(') {
                let fragment = fragment.trim().trim_end_matches(')');
                if let Some((n, c)) = fragment.rsplit_once(' ') {
                    if let Ok(c) = c.trim().parse::<isize>() {
                        map.insert(n.trim().to_string(), c);
                    }
                }
            }
            for line in text.lines() {
                if let Some((n, c)) = line.rsplit_once(": ") {
                    if let Ok(c) = c.trim().parse::<isize>() {
                        map.insert(n.trim().to_string(), c);
                    }
                }
            }
        }
        map
    }

    const RING_ONLY: &str = "(run 1)";
    const RING_AND_SUBST: &str = "(run 1) (run subst-walk 1)";

    /// Hard per-round wall-clock bail: a healthy round on this suite is
    /// milliseconds; one slow round means the ring is burning.
    const ROUND_WALL_SECS: u64 = 30;

    enum Verdict {
        /// Delta hit zero (after round 3) under the ceiling.
        Quiesce { rounds: usize, tuples: isize },
        /// Round cap reached without quiescing — still under the
        /// ceiling (legal only for pinned residual growers).
        NoQuiesce { tuples: isize, last_delta: isize },
    }

    /// Bounded round driver. PANICS (loudly, boundedly) if the tuple
    /// ceiling is crossed or a round exceeds the wall clock — those are
    /// the ignition signatures. Never hangs: at most `max_rounds`
    /// rounds, each individually walled.
    fn drive(
        label: &str,
        body: &str,
        schedule: &str,
        max_rounds: usize,
        tuple_ceiling: isize,
    ) -> Verdict {
        let full = format!("{}\n\n{body}", luminal_reference::assembled_program());
        // Raw-spelling verification: the hand-authored fixture text must
        // reach the egraph verbatim (nothing folded it en route).
        for line in body.lines().map(str::trim).filter(|l| !l.is_empty()) {
            assert!(
                full.contains(line),
                "[{label}] fixture line did not reach the program raw: {line}"
            );
        }
        let mut egraph = luminal::egglog_snippet::new_egraph();
        egraph
            .parse_and_run_program(None, &full)
            .unwrap_or_else(|e| panic!("[{label}] body loads: {e}"));
        let trace = std::env::var_os("G2_BATTERY_TRACE").is_some();
        let mut prev_sizes = table_sizes(&mut egraph);
        let mut prev: isize = prev_sizes.values().sum();
        let mut last_delta = 0isize;
        for round in 1..=max_rounds {
            let start = std::time::Instant::now();
            egraph
                .parse_and_run_program(None, schedule)
                .unwrap_or_else(|e| panic!("[{label}] round {round}: {e}"));
            let sizes = table_sizes(&mut egraph);
            let total: isize = sizes.values().sum();
            last_delta = total - prev;
            if trace {
                let deltas: Vec<String> = sizes
                    .iter()
                    .filter_map(|(n, &c)| {
                        let d = c - prev_sizes.get(n).copied().unwrap_or(0);
                        (d != 0).then(|| format!("{n} {d:+}"))
                    })
                    .collect();
                eprintln!(
                    "[g2-trace:{label}] r{round}: total {total} ({last_delta:+}) | {}",
                    deltas.join(", ")
                );
            }
            prev_sizes = sizes;
            assert!(
                total <= tuple_ceiling,
                "[{label}] RING IGNITION: {total} tuples > ceiling {tuple_ceiling} at round \
                 {round} (last delta {last_delta:+}) — the G2 fence (egglog_preamble.egg, \
                 Assoc-IntAdd / Assoc-IntMul / Distributivity-expand) has been weakened, or a \
                 new rule feeds the pin-collapse orbit"
            );
            assert!(
                start.elapsed().as_secs() <= ROUND_WALL_SECS,
                "[{label}] RING IGNITION (slow round): round {round} took \
                 {:.1}s at {total} tuples — bailing before the hang",
                start.elapsed().as_secs_f64()
            );
            if last_delta == 0 && round > 3 {
                return Verdict::Quiesce {
                    rounds: round,
                    tuples: total,
                };
            }
            prev = total;
        }
        Verdict::NoQuiesce {
            tuples: prev,
            last_delta,
        }
    }

    /// Assert a fixture quiesces within the round cap at or under the
    /// pinned fixed-point ceiling.
    fn assert_quiesce(
        label: &str,
        body: &str,
        schedule: &str,
        max_rounds: usize,
        tuple_ceiling: isize,
    ) {
        match drive(label, body, schedule, max_rounds, tuple_ceiling) {
            Verdict::Quiesce { rounds, tuples } => {
                eprintln!("[g2-battery:{label}] QUIESCE r{rounds}, {tuples} tuples");
            }
            Verdict::NoQuiesce { tuples, last_delta } => panic!(
                "[{label}] did not quiesce in {max_rounds} rounds ({tuples} tuples, last \
                 delta {last_delta:+}) — a quiescing config regressed; suspect the G2 fence \
                 or a new ring rule"
            ),
        }
    }

    const S5: &str = "(set (lower-bound-of (IntVar \"s\")) (bigint 5))\n(set (upper-bound-of (IntVar \"s\")) (bigint 5))\n";
    const T2: &str = "(set (lower-bound-of (IntVar \"t\")) (bigint 2))\n(set (upper-bound-of (IntVar \"t\")) (bigint 2))\n";

    fn seeds(pairs: &[(&str, i64, i64)]) -> String {
        pairs
            .iter()
            .map(|(v, lo, hi)| {
                format!(
                    "(set (lower-bound-of (IntVar \"{v}\")) (bigint {lo}))\n(set (upper-bound-of (IntVar \"{v}\")) (bigint {hi}))\n"
                )
            })
            .collect()
    }

    fn e(expr: &str, seeds: &str) -> String {
        format!("(let e_root {expr})\n{seeds}")
    }

    /// Fixed-point ceiling for the bare-spelling configs: clone
    /// measurements post-fence were 69–203 tuples; main measures in the
    /// same band. Slack is deliberately wide — the pre-fence igniters
    /// cross 300k, so anything under this ceiling is unambiguous.
    const BARE_QUIESCE_CEILING: isize = 2_000;
    const BARE_MAX_ROUNDS: usize = 20;

    fn assert_group(entries: &[(&str, String)]) {
        for (label, body) in entries {
            assert_quiesce(
                label,
                body,
                RING_ONLY,
                BARE_MAX_ROUNDS,
                BARE_QUIESCE_CEILING,
            );
        }
    }

    /// Group A: controls + the original raw window-count spellings.
    /// A3 is THE flagship igniter — (s-3)+1 with s pinned [5,5] ignited
    /// pre-fence at r14 with 2.6M tuples (battery.log 2026-08-27).
    #[test]
    fn group_a_controls_and_flagship_igniter_quiesce() {
        assert_group(&[
            (
                "A1_add_st",
                e(
                    r#"(IntAdd (IntVar "s") (IntVar "t"))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "A2_sub_s3",
                e(
                    r#"(IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3)))"#,
                    S5,
                ),
            ),
            (
                "A3_sub_s3_p1_flagship",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "A4_add_st_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntVar "t")) (IntLit 1))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "A5_sub_st",
                e(
                    r#"(IntAdd (IntVar "s") (IntMul (IntLit -1) (IntVar "t")))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "A6_sub_st_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntVar "t"))) (IntLit 1))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "A7_mul_s2_p1",
                e(
                    r#"(IntAdd (IntMul (IntVar "s") (IntLit 2)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "A8_mul_s2_pt",
                e(
                    r#"(IntAdd (IntMul (IntVar "s") (IntLit 2)) (IntVar "t"))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "A9_div_s2_p1",
                e(
                    r#"(IntAdd (IntTruncDiv (IntVar "s") (IntLit 2)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "A10_win_full",
                e(
                    r#"(IntAdd (IntTruncDiv (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 2)) (IntLit 1))"#,
                    S5,
                ),
            ),
        ]);
    }

    /// Group B: predicate discriminators (which shapes carry P1–P4).
    /// Pre-fence igniters here: B3, B4, B6, B8. B5 is pinned separately
    /// below as the known residual grower.
    #[test]
    fn group_b_predicate_discriminators_quiesce() {
        assert_group(&[
            (
                "B1_add_s3_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit 3)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B2_sublit_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit -3)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B3_sub_s3_pt",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntVar "t"))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "B4_deep",
                e(
                    r#"(IntAdd (IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B6_lit_first",
                e(
                    r#"(IntAdd (IntAdd (IntLit 3) (IntMul (IntLit -1) (IntVar "s"))) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B7_mulvar_p1",
                e(
                    r#"(IntAdd (IntMul (IntVar "s") (IntVar "t")) (IntLit 1))"#,
                    &format!("{S5}{T2}"),
                ),
            ),
            (
                "B8_sub_s1_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 1))) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B9_pad_out",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit 1)) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B10_two_prods",
                e(
                    r#"(IntAdd (IntAdd (IntMul (IntVar "s") (IntLit 2)) (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    S5,
                ),
            ),
            (
                "B11_prodpair",
                e(
                    r#"(IntAdd (IntVar "s") (IntMul (IntLit 2) (IntLit 3)))"#,
                    S5,
                ),
            ),
            (
                "B12_prodpair_p1",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit 2) (IntLit 3))) (IntLit 1))"#,
                    S5,
                ),
            ),
        ]);
    }

    /// B5 (s-(t-3))+1: in the analysis clone this was the one RESIDUAL
    /// GROWER post-fence (NOQ30, 37k tuples — a literal div-fold /
    /// backward-distributivity family). ON MAIN IT QUIESCES (r7, 120
    /// tuples, measured 2026-08-27) — main's tree closes the family the
    /// clone left open. Pinned as quiescing; pre-fence this config
    /// IGNITED at r11 with 474k tuples, so the ceiling discriminates
    /// the fence hard either way.
    #[test]
    fn group_b5_clone_residual_grower_quiesces_on_main() {
        let body = e(
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntAdd (IntVar "t") (IntMul (IntLit -1) (IntLit 3))))) (IntLit 1))"#,
            &format!("{S5}{T2}"),
        );
        assert_quiesce(
            "B5_sub_nest_t",
            &body,
            RING_ONLY,
            BARE_MAX_ROUNDS,
            BARE_QUIESCE_CEILING,
        );
    }

    /// Group C: bounds variants of the flagship igniter — wide ranges
    /// and zero-crossing ranges must not ignite (P2 fails: no pin).
    #[test]
    fn group_c_bounds_variants_quiesce() {
        assert_group(&[
            (
                "C1_ign_wide",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    &seeds(&[("s", 4, 100)]),
                ),
            ),
            (
                "C2_ign_zerox",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    &seeds(&[("s", 1, 100)]),
                ),
            ),
            (
                "C3_ign_zerox0",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    &seeds(&[("s", 0, 10)]),
                ),
            ),
            (
                "C4_ign_nobounds",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    "",
                ),
            ),
        ]);
    }

    /// Groups D/E: value-coincidence discriminators (same spellings,
    /// different point bindings — P3 probes) + the symbolic-kernel
    /// unfold window-count spellings.
    #[test]
    fn group_d_e_value_coincidence_and_symbolic_window_quiesce() {
        let sub_s3_p1 =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#;
        let sub_st_p1 =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntVar "t"))) (IntLit 1))"#;
        let lit_first =
            r#"(IntAdd (IntAdd (IntLit 3) (IntMul (IntLit -1) (IntVar "s"))) (IntLit 1))"#;
        let sub_s1_p1 =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 1))) (IntLit 1))"#;
        let sub_s3_p3 =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 3))"#;
        let sub_s2_p1 =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 2))) (IntLit 1))"#;
        let dividend = r#"(IntAdd (IntVar "d") (IntMul (IntLit -1) (IntAdd (IntMul (IntVar "l") (IntAdd (IntVar "k") (IntMul (IntLit -1) (IntLit 1)))) (IntLit 1))))"#;
        let win_sym = format!(r#"(IntAdd (IntTruncDiv {dividend} (IntLit 2)) (IntLit 1))"#);
        assert_group(&[
            ("D1_s6", e(sub_s3_p1, &seeds(&[("s", 6, 6)]))),
            ("D2_s4", e(sub_s3_p1, &seeds(&[("s", 4, 4)]))),
            ("D3_s8", e(sub_s3_p1, &seeds(&[("s", 8, 8)]))),
            ("D4_s7t3", e(sub_st_p1, &seeds(&[("s", 7, 7), ("t", 3, 3)]))),
            ("D5_s5t4", e(sub_st_p1, &seeds(&[("s", 5, 5), ("t", 4, 4)]))),
            ("D6_s3t2", e(sub_st_p1, &seeds(&[("s", 3, 3), ("t", 2, 2)]))),
            ("D7_B6_s6", e(lit_first, &seeds(&[("s", 6, 6)]))),
            ("D8_B8_s7", e(sub_s1_p1, &seeds(&[("s", 7, 7)]))),
            ("D9_s5", e(sub_s3_p3, &seeds(&[("s", 5, 5)]))),
            ("D10_s9", e(sub_s2_p1, &seeds(&[("s", 9, 9)]))),
            (
                "E1_dividend",
                e(dividend, &seeds(&[("d", 16, 16), ("k", 3, 3), ("l", 2, 2)])),
            ),
            (
                "E2_win_sym",
                e(&win_sym, &seeds(&[("d", 16, 16), ("k", 3, 3), ("l", 2, 2)])),
            ),
            (
                "E3_win_sym",
                e(&win_sym, &seeds(&[("d", 17, 17), ("k", 4, 4), ("l", 1, 1)])),
            ),
        ]);
    }

    /// Group F: predicate boundary predictions, constructed before the
    /// fence landed to falsify the collision predicate (P3 probes with
    /// tail-sum/leaf/operand collisions).
    #[test]
    fn group_f_predicate_boundaries_quiesce() {
        assert_group(&[
            (
                "F1_tail_hits_var",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntVar "t")) (IntLit 1))"#,
                    &seeds(&[("s", 5, 5), ("t", 4, 4)]),
                ),
            ),
            (
                "F2_tail_hits_op",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit 2) (IntLit 3))) (IntLit 1))"#,
                    &seeds(&[("s", 2, 2)]),
                ),
            ),
            (
                "F3_diag_no_prod",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit 2)) (IntLit 3))"#,
                    &seeds(&[("s", 1, 1)]),
                ),
            ),
            (
                "F4_no_coll",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit 2)) (IntLit 3))"#,
                    &seeds(&[("s", 6, 6)]),
                ),
            ),
            (
                "F5_var_is_leaf",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntLit 2)) (IntLit 3))"#,
                    &seeds(&[("s", 2, 2)]),
                ),
            ),
        ]);
    }

    /// Group M: nested-PRODUCT shapes against the Assoc-IntMul fence
    /// premise. Probed 2026-08-27 with that premise ABLATED: none of
    /// these ignite — no known product detonator exists (the premise is
    /// prophylactic symmetry with the add fence; products lack the
    /// like-term/tail-sum collision fuel). Kept as coverage so any
    /// future ring rule that makes products detonable is caught here;
    /// the premise itself is pinned textually by
    /// `g2_fence_text_is_intact` below.
    #[test]
    fn group_m_nested_products_quiesce() {
        assert_group(&[
            (
                "M1_s5_x3_x2",
                e(
                    r#"(IntMul (IntMul (IntVar "s") (IntLit 3)) (IntLit 2))"#,
                    S5,
                ),
            ),
            (
                "M2_s3_x2_x3",
                e(
                    r#"(IntMul (IntMul (IntVar "s") (IntLit 2)) (IntLit 3))"#,
                    &seeds(&[("s", 3, 3)]),
                ),
            ),
            (
                "M3_st_x2",
                e(
                    r#"(IntMul (IntMul (IntVar "s") (IntVar "t")) (IntLit 2))"#,
                    &seeds(&[("s", 3, 3), ("t", 2, 2)]),
                ),
            ),
            (
                "M4_s2_x2_x2",
                e(
                    r#"(IntMul (IntMul (IntVar "s") (IntLit 2)) (IntLit 2))"#,
                    &seeds(&[("s", 2, 2)]),
                ),
            ),
            (
                "M5_s7_x2_x3_nocoll",
                e(
                    r#"(IntMul (IntMul (IntVar "s") (IntLit 2)) (IntLit 3))"#,
                    &seeds(&[("s", 7, 7)]),
                ),
            ),
            (
                "M6_deep_mul",
                e(
                    r#"(IntMul (IntMul (IntMul (IntVar "s") (IntLit 2)) (IntLit 3)) (IntLit 2))"#,
                    &seeds(&[("s", 2, 2)]),
                ),
            ),
            (
                "M7_mul_of_sub",
                e(
                    r#"(IntMul (IntMul (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 2)) (IntLit 3))"#,
                    S5,
                ),
            ),
            (
                "M8_mixed_addroot",
                e(
                    r#"(IntAdd (IntMul (IntMul (IntVar "s") (IntLit 2)) (IntLit 3)) (IntLit 1))"#,
                    &seeds(&[("s", 3, 3)]),
                ),
            ),
            (
                "M9_win_times",
                e(
                    r#"(IntMul (IntMul (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 3)) (IntLit 2))"#,
                    S5,
                ),
            ),
        ]);
    }

    /// THE FENCE TEXT TRIPWIRE (repo precedent: subst_guard_study's
    /// guard anchors). The Assoc-IntMul G2 premise has no known
    /// behavioral detonator, so behavior tests alone cannot catch its
    /// removal — the fence is ALSO pinned textually: each of the three
    /// guarded rules must appear verbatim in the assembled program, and
    /// each rule action exactly once (no unguarded twin beside a
    /// guarded original).
    #[test]
    fn g2_fence_text_is_intact() {
        const ASSOC_INTADD: &str = "(rule\n  (\n    (= ?expr (IntAdd ?inner ?outer))\n    (= ?inner (IntAdd ?lhs ?rhs))\n    (provably-cannot-union-with-zero ?inner)\n    (= ?inner_lower (lower-bound-of ?inner))\n    (= ?inner_upper (upper-bound-of ?inner))\n    (< ?inner_lower ?inner_upper)\n  )\n  ((union ?expr (IntAdd ?lhs (IntAdd ?rhs ?outer))))\n)";
        const ASSOC_INTMUL: &str = "(rule\n  (\n    (= ?expr (IntMul ?inner ?outer))\n    (= ?inner (IntMul ?lhs ?rhs))\n    (provably-cannot-union-with-zero ?inner)\n    (= ?inner_lower (lower-bound-of ?inner))\n    (= ?inner_upper (upper-bound-of ?inner))\n    (< ?inner_lower ?inner_upper)\n  )\n  ((union ?expr (IntMul ?lhs (IntMul ?rhs ?outer))))\n)";
        const DIST_EXPAND: &str = "(rule\n  (\n    (= ?expr (IntMul ?summands ?factor))\n    (= ?summands (IntAdd ?lhs ?rhs))\n    (provably-cannot-union-with-zero ?expr)\n    (provably-cannot-union-with-zero ?lhs)\n    (provably-cannot-union-with-zero ?rhs)\n    (provably-cannot-union-with-zero ?factor)\n    (= ?summands_lower (lower-bound-of ?summands))\n    (= ?summands_upper (upper-bound-of ?summands))\n    (< ?summands_lower ?summands_upper)\n  )\n  ((union ?expr (IntAdd (IntMul ?lhs ?factor) (IntMul ?rhs ?factor))))\n)";
        let program = luminal_reference::assembled_program();
        for (name, rule, action) in [
            (
                "Assoc-IntAdd",
                ASSOC_INTADD,
                "((union ?expr (IntAdd ?lhs (IntAdd ?rhs ?outer))))",
            ),
            (
                "Assoc-IntMul",
                ASSOC_INTMUL,
                "((union ?expr (IntMul ?lhs (IntMul ?rhs ?outer))))",
            ),
            (
                "Distributivity-expand",
                DIST_EXPAND,
                "((union ?expr (IntAdd (IntMul ?lhs ?factor) (IntMul ?rhs ?factor))))",
            ),
        ] {
            assert!(
                program.contains(rule),
                "G2 fence drifted: the guarded {name} rule text is not in the preamble \
                 (egglog_preamble.egg — if the change is intentional, re-prove with this \
                 battery and re-pin)"
            );
            assert_eq!(
                program.matches(action).count(),
                1,
                "G2 fence bypassed: {name}'s action appears more than once — an unguarded \
                 twin rule beside the guarded original"
            );
        }
    }

    /// Group G: mixed static/dynamic bounds — wide var with a pinned
    /// literal tail (the shipping symbolic-dim shape).
    #[test]
    fn group_g_mixed_bounds_quiesce() {
        assert_group(&[
            (
                "G1_wide_s_lit2",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 2))) (IntLit 1))"#,
                    &seeds(&[("s", 4, 100)]),
                ),
            ),
            (
                "G2_wide_s_pt_t",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntVar "t"))) (IntLit 1))"#,
                    &seeds(&[("s", 4, 100), ("t", 2, 2)]),
                ),
            ),
            (
                "G3_wide_s_lit3",
                e(
                    r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#,
                    &seeds(&[("s", 4, 100)]),
                ),
            ),
        ]);
    }

    /// USER-DIM END-TO-END: the raw (s-t)+1 dim spelling inside real
    /// shape/layout context (LogicalTensorInputLit + right-major
    /// layout), hand-authored to mirror the clone recorder's output
    /// byte-for-byte at the extent (this tree's frontend would fold it —
    /// the shield the battery must bypass). Pre-fence this IGNITED at
    /// r15 with 683k tuples (exposure.log).
    #[test]
    fn user_dim_raw_spelling_in_layout_context_quiesces() {
        let body = "\
(let g2ud_shape (ShapeLit (IntExprCons (IntAdd (IntAdd (IntVar \"s\") (IntMul (IntLit -1) (IntVar \"t\"))) (IntLit 1)) (IntExprNil))))
(let g2ud_x (LogicalTensorInputLit (LogicalIdLit \"g2ud_x\") g2ud_shape (F32)))
(let g2ud_layout (RightMajorContiguousElementLayoutLit g2ud_shape (bits-of (F32))))
(set (lower-bound-of (IntVar \"s\")) (bigint 5))
(set (upper-bound-of (IntVar \"s\")) (bigint 5))
(set (lower-bound-of (IntVar \"t\")) (bigint 2))
(set (upper-bound-of (IntVar \"t\")) (bigint 2))
";
        assert_quiesce("user_dim_raw", body, RING_AND_SUBST, 40, 5_000);
    }

    /// SUBST-COMPOSITION (coord entry): the walk substitutes a
    /// crop/flip-style negative-offset map entry into a flat outer
    /// IntAdd, minting the nested spelling AT SATURATION — rewrites can
    /// build the igniter from flat parts, so the fence must hold on
    /// rule-minted (not just seeded) spellings.
    #[test]
    fn subst_composition_minted_nesting_quiesces() {
        let body = r#"
(let ext_e (IntVar "e"))
(set (lower-bound-of ext_e) (bigint 5))
(set (upper-bound-of ext_e) (bigint 5))
(let src_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let out_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let c_src (CoordVar src_shape 0))
(let c_out (CoordVar out_shape 0))
(let entry (IntAdd (IntAdd ext_e (IntMul (IntLit -1) (IntLit 3))) c_out))
(let map (IndexMapLit (IntExprCons entry (IntExprNil)) src_shape))
(let outer_expr (IntAdd c_src (IntLit 1)))
(int-subst-demand outer_expr map)
"#;
        assert_quiesce("subst_comp", body, RING_AND_SUBST, 25, 5_000);
    }

    /// SUBST-COMPOSITION (coord-free entry, pure dim offset e-3): the
    /// walk mints IntAdd(IntAdd(e, -1*3), 1) — the A3 flagship — at
    /// saturation.
    #[test]
    fn subst_composition_coordfree_mints_flagship_quiesces() {
        let body = r#"
(let ext_e (IntVar "e"))
(set (lower-bound-of ext_e) (bigint 5))
(set (upper-bound-of ext_e) (bigint 5))
(let src_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let c_src (CoordVar src_shape 0))
(let entry (IntAdd ext_e (IntMul (IntLit -1) (IntLit 3))))
(let map (IndexMapLit (IntExprCons entry (IntExprNil)) src_shape))
(let outer_expr (IntAdd c_src (IntLit 1)))
(int-subst-demand outer_expr map)
"#;
        assert_quiesce("subst_comp_coordfree", body, RING_AND_SUBST, 25, 5_000);
    }

    /// THE ORIGINAL RAW-UNFOLD DRIVER: a real recorded unfold graph
    /// (this tree's recorder folds the window count — verified below)
    /// PLUS the raw window-count spelling injected beside it, so the
    /// ring sees the raw atom inside full graph/subst-walk context.
    #[test]
    fn unfold_graph_with_raw_window_count_quiesces() {
        use luminal::prelude::{DType, Graph};
        let mut cx = Graph::new();
        cx.set_dim('s', 5);
        let x = cx.named_tensor("x", ('s',), DType::F32);
        let _out = x.unfold((3usize,), (1usize,), (1usize,)).sum(1).output();
        let (pre, _inputs, _outputs, _post, _labeled) = cx
            .logical
            .bound_parts(&luminal_reference::ReferenceBindings)
            .expect("recorder clean");
        let raw_window =
            r#"(IntAdd (IntAdd (IntVar "s") (IntMul (IntLit -1) (IntLit 3))) (IntLit 1))"#;
        // The shield check: this tree's frontend simplify must have
        // folded the recorder's window count — if the raw nested
        // spelling ever starts arriving from the recorder, the fixture
        // double-covers and should be revisited.
        eprintln!(
            "[g2-battery:unfold] recorder emits raw spelling: {}",
            pre.contains(raw_window)
        );
        let body = format!("{pre}\n(let g2_raw_window {raw_window})\n{S5}");
        assert_quiesce("unfold_raw_window", &body, RING_AND_SUBST, 60, 10_000);
    }
}

#[cfg(test)]
mod escape_execution_tests {
    //! THE HAND-BUILT-PLAN FIXTURE — where Option B's carried `L` is load
    //! bearing (correction 7). A live runtime knows every layout before it
    //! calls `bufferize`, so the plan tells it nothing new. These plans
    //! have NO such runtime behind them: they are the surface `load_plan`
    //! accepts, built by hand, with no e-graph, no recorder and no
    //! decoded table anywhere. Under Option B they need no companion
    //! argument, because the boundary/layout knowledge a live runtime
    //! would have had rides the plan itself: `Buffer::layout` sizes every
    //! allocation and `OutputBinding::layout` describes every delivery.
    //!
    //! They also prove the escape path end to end: the fetch returns the
    //! BACKING buffer's bytes plus the elected layout, elements are read
    //! by evaluating that layout, and the executor guard refuses
    //! minted-non-escaping storage backing an output slot. Same dep-world
    //! discipline as `harness_tests`: every luminal type comes from the
    //! `luminal::` build luminal_reference links.
    use egraph_serialize::ClassId;

    use luminal::bufferize::{
        Buffer, BufferEdge, BufferId, BufferIrGraph, BufferNode, EdgeKind, InputBinding,
        OutputBinding, Owner,
    };
    use luminal::layout_ir::{Access, FreedBy};
    use luminal::prelude::petgraph::graph::DiGraph;
    use luminal_reference::{RefLayout, ReferenceRuntime};

    /// A hand-built plan's carried layout. THE ONLY SIZING INPUT: the
    /// executor allocates span-of-layout elements in the layout's dtype —
    /// no dims field, no walk, no vote.
    fn rm_layout(dims: &[i64]) -> RefLayout {
        // Dep-world discipline: `luminal::layouts`, never `crate::layouts` —
        // RefLayout is the plain `luminal` build's MirrorLayout, and the
        // cfg(test) build's types do not unify with it.
        use luminal::layouts::{
            BitWidthTerm, IntExprTerm, MirrorLayout, RightMajorContiguousElementLayout, ShapeTerm,
        };
        RefLayout {
            mirror: MirrorLayout::RightMajor(RightMajorContiguousElementLayout {
                shape: ShapeTerm(dims.iter().map(|&d| IntExprTerm::Lit(d)).collect()),
                width: BitWidthTerm(32),
            }),
            dtype: Some(luminal::dtype::PlanDtype::F32),
        }
    }

    /// PROTOTYPE (Option B): the transpose view's COMPOSED layout — the
    /// `L` the e-graph would mint for the view value (shape `[3,2]`,
    /// reading its dense `[2,3]` parent): element (i,j) at parent flat
    /// j*3 + i, spelled as the strided chain from-end [coord0*3, coord1].
    fn transpose_strided_layout() -> RefLayout {
        use luminal::layouts::{
            BitWidthTerm, IntExprTerm, MirrorLayout, ShapeTerm, StridedElementLayout,
        };
        let coord = |axis_from_end: i64| IntExprTerm::Coord { axis_from_end };
        RefLayout {
            mirror: MirrorLayout::Strided(StridedElementLayout {
                shape: ShapeTerm(vec![IntExprTerm::Lit(3), IntExprTerm::Lit(2)]),
                chain: vec![
                    IntExprTerm::Mul(Box::new(coord(0)), Box::new(IntExprTerm::Lit(3))),
                    coord(1),
                ],
                width: BitWidthTerm(32),
            }),
            dtype: Some(luminal::dtype::PlanDtype::F32),
        }
    }

    /// A minimal escaped-output plan: input x `[2,3]` (BufferLit 7) is
    /// base-copied into minted buffer A (the stand-in for a kernel
    /// producing the parent there), and the output slot binds the
    /// TRANSPOSE VIEW of that parent — backed by A, `freed_by` flipped to
    /// the escape cell.
    fn escaped_plan(freed_by: FreedBy) -> BufferIrGraph<RefLayout> {
        let x = ClassId::from("val$x");
        let v = ClassId::from("val$v");
        let input_id = BufferId::Boundary(ClassId::from("buf$B"));
        let escaped_id = BufferId::Allocated(0);
        let mut buffers = std::collections::HashMap::new();
        buffers.insert(
            input_id.clone(),
            Buffer {
                id: input_id.clone(),
                access: Access::ReadOnly,
                freed_by: FreedBy::Caller,
                owner: Owner::Caller,
                label: "B".to_string(),
                lit: Some(7),
                backs: x.clone(),
                layout: rm_layout(&[2, 3]),
            },
        );
        buffers.insert(
            escaped_id.clone(),
            Buffer {
                id: escaped_id.clone(),
                access: Access::ReadWrite,
                freed_by,
                owner: Owner::System,
                label: "escaped".to_string(),
                lit: None,
                // THE ASSIGNMENT: this storage holds the PARENT's bytes
                // (the base copy landed them), so it backs x and is sized
                // by x's layout — parent-sized, not view-sized.
                backs: x.clone(),
                layout: rm_layout(&[2, 3]),
            },
        );
        let mut dag: DiGraph<BufferNode<RefLayout>, BufferEdge> = DiGraph::new();
        let input = dag.add_node(BufferNode::BufferInput {
            slots: vec![InputBinding {
                value: x.clone(),
                buffer: input_id.clone(),
            }],
        });
        // The copy carries ONLY {src, dst} — a dumb exact-size whole-buffer
        // copy (both buffers are the parent's 6 f32s). Ordering is this
        // fixture's own obligation, discharged by the data edges below.
        let copy = dag.add_node(BufferNode::BufferCopy {
            src: input_id.clone(),
            dst: escaped_id.clone(),
        });
        let out = dag.add_node(BufferNode::BufferOutput {
            slots: vec![OutputBinding {
                index: 0,
                value: v.clone(),
                buffer: escaped_id.clone(),
                // A VIEW OUTPUT, fulfilled STRUCTURALLY: the slot's buffer
                // IS the parent's storage (zero-copy by construction), and
                // the carried layout — the view's composed strided form —
                // is how the caller reads it. No dims, no dtype, no hop
                // chain: the layout is the whole disclosure.
                layout: transpose_strided_layout(),
            }],
        });
        dag.add_edge(
            input,
            copy,
            BufferEdge {
                buffer: input_id,
                port: "in".to_string(),
                kind: EdgeKind::Data,
            },
        );
        dag.add_edge(
            copy,
            out,
            BufferEdge {
                buffer: escaped_id,
                port: "out 0".to_string(),
                kind: EdgeKind::Data,
            },
        );
        let mut value_buffer = std::collections::BTreeMap::new();
        value_buffer.insert(x, BufferId::Allocated(0));
        BufferIrGraph {
            dag,
            buffers,
            value_buffer,
            outputs: vec![out],
        }
    }

    /// PROBE 1, the executed-bytes half — OPTION B RESTRUCTURE: the
    /// escaped slot's fetch returns the backing buffer's bytes plus the
    /// slot's HELD LAYOUT (`binding.layout`, verbatim), and elements are
    /// read by EVALUATING the mirror layout's own expressions — no core
    /// walker (`walk_layout_index` left core; the canonical test-equality
    /// utility lives in the testing crate `test_runtime::test_equality`,
    /// duplicated minimally here because core's own dev-tests cannot
    /// depend on the testing crate without a dev-cycle).
    #[test]
    fn escaped_output_executes_and_reads_through_the_held_layout() {
        let mut rt = ReferenceRuntime::default();
        rt.load_plan(escaped_plan(FreedBy::Caller));
        let staged: Vec<f32> = (0..6).map(|n| n as f32 * 10.0).collect();
        rt.set_data_buffer(7, staged.clone());
        rt.execute().expect("an escaping output executes");

        let (data, binding) = rt.output_slot(0).expect("the universal fetch");
        let bytes = data.as_f32().expect("f32 backing bytes");
        assert_eq!(bytes.len(), 6, "the BACKING buffer is parent-sized");
        // The value's shape comes from the CARRIED LAYOUT's domain — there
        // is no dims field to read.
        use luminal::layouts::{IntExprTerm, MirrorLayout};
        assert_eq!(binding.layout.mirror.literal_extents(), Some(vec![3, 2]));
        let MirrorLayout::Strided(strided) = &binding.layout.mirror else {
            panic!("the disclosed layout is the composed strided form");
        };
        // Evaluate one chain summand at concrete coords (from-end axes).
        fn eval(expr: &IntExprTerm, coords: &[usize]) -> i64 {
            match expr {
                IntExprTerm::Lit(v) => *v,
                IntExprTerm::Coord { axis_from_end } => {
                    let rank = coords.len();
                    coords[rank - 1 - *axis_from_end as usize] as i64
                }
                IntExprTerm::Add(a, b) => eval(a, coords) + eval(b, coords),
                IntExprTerm::Mul(a, b) => eval(a, coords) * eval(b, coords),
                other => panic!("fixture layout uses no {other:?}"),
            }
        }
        for i in 0..3usize {
            for j in 0..2usize {
                let coords = [i, j];
                let flat: i64 = strided.chain.iter().map(|s| eval(s, &coords)).sum();
                assert_eq!(
                    bytes[flat as usize],
                    staged[j * 3 + i],
                    "v[{i},{j}] must be x[{j},{i}]"
                );
            }
        }
        // The layout accessor alone agrees with the fetch.
        let layout = rt.output_layout(0).expect("layout accessor");
        assert_eq!(layout.buffer, BufferId::Allocated(0));
    }

    /// PROBE 5, reference side: a hand-built plan whose output slot is
    /// backed by minted NON-ESCAPING storage (Owner::System +
    /// FreedBy::Program) must be refused loudly at execute — the caller
    /// would receive bytes the program destroys. (The cuda-lite executor
    /// carries the same guard in `device::execute_plan`, ahead of any
    /// device work; it is feature-gated on `device` and compile-checked
    /// here.)
    #[test]
    fn executor_refuses_minted_non_escaping_output_backing() {
        let mut rt = ReferenceRuntime::default();
        rt.load_plan(escaped_plan(FreedBy::Program));
        rt.set_data_buffer(7, vec![0.0f32; 6]);
        let err = rt.execute().expect_err("the escape guard must refuse");
        assert!(
            err.to_string().contains("NON-ESCAPING"),
            "the guard names the violation: {err:#}"
        );
    }

    /// PROBE 5b: DONATED boundary storage (Owner::Caller +
    /// FreedBy::Program) backing an output slot is refused the same way —
    /// the guard keys on FreedBy alone, so donation status cannot slip
    /// past the owner check (validate()'s donated arm forbids exactly
    /// this plan shape; the executor re-checks for loaded plans).
    #[test]
    fn executor_refuses_donated_boundary_output_backing() {
        let mut plan = escaped_plan(FreedBy::Caller);
        let input_id = BufferId::Boundary(ClassId::from("buf$B"));
        plan.buffers.get_mut(&input_id).unwrap().freed_by = FreedBy::Program;
        for node in plan.dag.node_weights_mut() {
            if let BufferNode::BufferOutput { slots } = node {
                slots[0].buffer = input_id.clone();
                slots[0].layout = rm_layout(&[2, 3]);
            }
        }
        let mut rt = ReferenceRuntime::default();
        rt.load_plan(plan);
        rt.set_data_buffer(7, vec![0.0f32; 6]);
        let err = rt
            .execute()
            .expect_err("the escape guard must refuse donated backing");
        assert!(
            err.to_string().contains("NON-ESCAPING") && err.to_string().contains("Caller"),
            "the guard names the violation and the owner: {err:#}"
        );
    }
}

#[cfg(test)]
mod deferred_display_text {
    //! The deferral half of the render ruling (2026-09-01): the info
    //! structs' display text is built on FIRST READ, not during
    //! extraction. These pin both halves of that promise — nothing is
    //! rendered until asked, and asking produces the real text.

    use luminal::layout_ir::ExtractedNode;
    use luminal_reference::harness::extract_fixture;

    /// Extraction leaves the display text UNBUILT, and reading it builds
    /// it. If a search-path caller starts forcing these again, the first
    /// assertion is what notices.
    #[test]
    fn extraction_defers_display_text_until_it_is_read() {
        let graph = extract_fixture("boundary_gather.egg");

        let value = graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) => op.outputs.first(),
                _ => None,
            })
            .expect("the fixture has an op with an output");

        assert!(
            !value.tooltip.is_rendered(),
            "extraction rendered a value tooltip that nobody asked for"
        );
        assert!(
            !value.layout.tooltip.is_rendered(),
            "extraction rendered a layout tooltip that nobody asked for"
        );

        let tooltip = value.tooltip.to_string();
        assert!(
            tooltip.contains(&format!("eclass={}", value.eclass)),
            "the deferred tooltip did not build its real text: {tooltip:?}"
        );
        assert!(value.tooltip.is_rendered());
        assert_eq!(
            tooltip,
            value.tooltip.to_string(),
            "a second read must return the first read's text"
        );
    }

    /// `to_dot` forces EVERY deferred field on a real extracted graph.
    /// That is also the runtime guard on the render memo's recursion
    /// (`render_class_prefer` → `render_node` → `render_class_prefer`):
    /// a `RefCell` borrow held across it is a `BorrowMutError`, and the
    /// depth-16/32 layout tooltips are where it would fire.
    #[test]
    fn to_dot_forces_every_deferred_field() {
        for script in [
            "boundary_gather.egg",
            "boundary_pass_through.egg",
            "boundary_iota.egg",
        ] {
            let graph = extract_fixture(script);
            let dot = graph.to_dot();
            assert!(
                dot.contains("tooltip=\"eclass="),
                "{script}: no rendered tooltip reached the dot output"
            );
            for node in graph.dag.node_weights() {
                if let ExtractedNode::LayoutOp(op) = node {
                    assert!(
                        op.tooltip.is_rendered(),
                        "{script}: to_dot left an op tooltip unrendered"
                    );
                    for output in &op.outputs {
                        assert!(output.tooltip.is_rendered());
                        assert!(output.logical.label.is_rendered());
                        assert!(output.layout.tooltip.is_rendered());
                    }
                }
            }
        }
    }
}

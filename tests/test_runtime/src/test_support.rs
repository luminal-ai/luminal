//! Test-only support: hand-author `ExtractedGraph`s at the level under test
//! (the analogue of writing a `.mlir` file by hand for MLIR's analysis tests),
//! plus a fixture path that runs a real egglog script through the extractor.
//!
//! Assignment rule: any case expressible with default-interface ops should be an
//! egg script tested via the runtime fixture extractor; the [`TestGraph`] builder is only
//! for cases *defined by* a non-default `Bufferizable` interface (declared
//! must-share ties, may-share permits, accumulators), which by design have no
//! egglog surface.

pub mod test_ops {
    //! Test fixtures for the bufferizer's multi-destination invariants.

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

use crate::prelude::egraph_serialize::ClassId;
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
            tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
            shape: None,
            dtype: None,
            dtype_enum: None,
            dims: None,
            element_bits: None,
            logical: LogicalInfo {
                eclass: ClassId::from(format!("logical${name}")),
                label: {
                    let text = name.to_owned();
                    std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(move || text)))
                },
                tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
                op: None,
                children: Vec::new(),
            },
            layout: LayoutInfo {
                eclass: ClassId::from(format!("layout${layout}")),
                label: {
                    let text = layout.to_owned();
                    std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(move || text)))
                },
                tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
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
            tensor_tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
            id_eclass: ClassId::from(format!("buf${buffer}")),
            id_label: buffer.to_string(),
            id_tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
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
            tooltip: std::rc::Rc::new(once_cell::unsync::Lazy::new(Box::new(String::new))),
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

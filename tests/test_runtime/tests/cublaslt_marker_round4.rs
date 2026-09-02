//! Prove-outs: the DPS alias tables, bufferize over ALL FOUR contracts,
//! the B x left-major arm, relu on a symbolic-k op, and the static-pitch
//! OUTPUT layout (creator-rewrite certified).

use luminal::bufferize::BufferId;
use luminal::dtype::DType;
use luminal::graph::Graph;
use luminal::layout_ir::{Bufferizable, ExtractedNode, Sharing};
use test_runtime::cublaslt_marker::{CuDim, CuEpilogue, CublasLt, CublasLtDps, CublasLtForm};

type GraphBuilder = Box<dyn Fn(&mut Graph)>;
type FormProgram = (&'static str, CublasLtForm, GraphBuilder);

const SCHEDULE: &str = "(run-schedule (saturate (saturate (run)) (run subst-walk)) (run materializing-copy-mint) (run layout-tensor-op-metadata) (saturate (run fixpoint-invariants)))";

const PIN: &[&str] = &[
    "LayoutTensorOpCublasLtAccumulateBias",
    "LayoutTensorOpCublasLtBias",
    "LayoutTensorOpCublasLtAccumulate",
    "LayoutTensorOpCublasLt",
    // ROUND 10 (view admission in ELECTION): the sibling site's result is
    // routed to the recorder's boundary value by a transpose VIEW; prefer
    // the view op over materialize/copy wherever both produce a class.
    "LayoutTensorOpIndexMapApplyViewGeneric",
];

fn count_op(s: &luminal::prelude::egraph_serialize::EGraph, op: &str) -> usize {
    s.nodes.values().filter(|n| n.op == op).count()
}

fn count_cublaslt(s: &luminal::prelude::egraph_serialize::EGraph) -> usize {
    s.nodes
        .values()
        .filter(|n| n.op.starts_with("LayoutTensorOpCublasLt"))
        .count()
}

fn cublaslt_in_plan(graph: &luminal::layout_ir::ExtractedGraph) -> Vec<(CublasLt, Vec<String>)> {
    graph
        .dag
        .node_weights()
        .filter_map(|node| match node {
            ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => {
                let concrete = (*op.op).as_any().downcast_ref::<CublasLt>().cloned()?;
                let inputs = op.inputs.iter().map(|i| i.value.to_string()).collect();
                Some((concrete, inputs))
            }
            _ => None,
        })
        .collect()
}

// ===========================================================================
// The DPS alias tables, asserted per contract.
// ===========================================================================
#[test]
fn t4_dps_alias_tables() {
    for form in CublasLtForm::ALL {
        let dps = CublasLtDps {
            op: CublasLt { form, spec: None },
        };
        let aliases = dps.alias_info();
        let dest = form.lit_arity();
        println!("T4 {form:?}: dest at {dest}, aliases {aliases:?}");
        assert_eq!(aliases[0].operand, dest, "Must tie: dest operand");
        assert_eq!(aliases[0].result, 0);
        assert_eq!(aliases[0].sharing, Sharing::Must);
        if form.has_c() {
            // The API's C==D same-buffer in-place accumulate (legal since
            // the C-fold guard makes the C and D layouts identical).
            assert_eq!(aliases.len(), 2);
            assert_eq!(aliases[1].operand, 2, "C sits at contract slot 2");
            assert_eq!(aliases[1].result, 0);
            assert_eq!(aliases[1].sharing, Sharing::May);
        } else {
            assert_eq!(aliases.len(), 1, "no C, no May permit");
        }
    }
}

// ===========================================================================
// dps_rewrite + bufferize over all four elected plans.
// ===========================================================================
#[test]
fn t6a_bufferize_all_four_forms() {
    let programs: [FormProgram; 4] = [
        (
            "base",
            CublasLtForm::Base,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let _ = x.matmul(w).output();
            }),
        ),
        (
            "bias",
            CublasLtForm::Bias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = (x.matmul(w) + b.expand_dim(0, 4usize)).output();
            }),
        ),
        (
            "accumulate",
            CublasLtForm::Accumulate,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let _ = (x.matmul(w) + c).output();
            }),
        ),
        (
            "accumulate-bias",
            CublasLtForm::AccumulateBias,
            Box::new(|cx: &mut Graph| {
                let x = cx.tensor((4usize, 8usize), DType::F32);
                let w = cx.tensor((8usize, 3usize), DType::F32);
                let c = cx.tensor((4usize, 3usize), DType::F32);
                let b = cx.tensor(3usize, DType::F32);
                let _ = ((x.matmul(w) + c) + b.expand_dim(0, 4usize)).output();
            }),
        ),
    ];
    for (name, form, build) in &programs {
        let text = {
            let mut cx = Graph::new();
            build(&mut cx);
            cx.logical
                .bound_program(&test_runtime::TestRuntimeBindings)
                .expect("recorder clean")
                .text
        };
        let (graph, _) = test_runtime::extract_fixture_with_genome(
            &text,
            // The CONTRACT is elected by name; the view op is round-10
            // routing infrastructure (sibling result -> boundary value).
            &[
                form.constructor_name(),
                "LayoutTensorOpIndexMapApplyViewGeneric",
            ],
        );
        let elected = cublaslt_in_plan(&graph);
        assert_eq!(
            elected.len(),
            1,
            "{name}: the {form:?} contract elected by name"
        );
        assert_eq!(elected[0].0.form, *form);
        assert_eq!(
            elected[0].1.len(),
            form.lit_arity(),
            "{name}: Lit arity is the name's constant"
        );

        // THE PIPELINE: functional plan -> DPS rewrite -> bufferize.
        let dps_graph = luminal::dps::dps_rewrite(&graph);
        let dps_op = dps_graph
            .dag
            .node_weights()
            .find_map(|node| match node {
                ExtractedNode::LayoutOp(op) if op.op.label().starts_with("CublasLt") => Some(op),
                _ => None,
            })
            .expect("cublaslt op survives the DPS rewrite");
        assert_eq!(
            dps_op.inputs.len(),
            form.lit_arity() + 1,
            "{name}: DPS operands = contract arity + dest"
        );
        assert_eq!(
            dps_op.op.operand_name(form.lit_arity()),
            "dest",
            "{name}: dest tie present"
        );

        let plan = luminal::test_support::bufferize_mock(&dps_graph)
            .unwrap_or_else(|err| panic!("T6a {name}: bufferizer REFUSED: {err}"));
        let summary = plan.summary();
        let allocs = plan
            .buffers
            .keys()
            .filter(|id| matches!(id, BufferId::Allocated(_)))
            .count();
        println!("T6a {name} ({form:?}): bufferized OK, {allocs} allocs");
        let label = match form {
            CublasLtForm::Base => "CublasLt",
            CublasLtForm::Bias => "CublasLtBias",
            CublasLtForm::Accumulate => "CublasLtAccumulate",
            CublasLtForm::AccumulateBias => "CublasLtAccumulateBias",
        };
        assert!(summary.contains(label), "{name}: kernel in plan\n{summary}");
        // ESCAPE-AND-DISCLOSE RE-PIN (ruling 2026-08-27, supersedes the
        // round-10 recorded cost — the "redundant identity-bytes copy"
        // out of a scratch alloc): the kernel claims the SIBLING's
        // transpose-view tensor as the boundary value, and that view
        // output now ESCAPES. One alloc — the kernel dest, handed to the
        // caller (FreedBy::Caller, no free) — ZERO copies, and the slot
        // is backed by the alloc itself with the weld's (RM-equal)
        // layout disclosed on the binding.
        assert_eq!(
            allocs, 1,
            "{name}: one alloc — the kernel dest, which escapes\n{summary}"
        );
        assert!(
            !summary.contains("BufferCopy"),
            "{name}: zero boundary copies under escape (ruling 2026-08-27)\n{summary}"
        );
        let slot = plan
            .dag
            .node_weights()
            .find_map(|node| match node {
                luminal::bufferize::BufferNode::BufferOutput { slots } => Some(slots[0].clone()),
                _ => None,
            })
            .expect("slot 0");
        assert!(
            matches!(slot.buffer, BufferId::Allocated(_)),
            "{name}: the slot is backed by the escaping kernel alloc\n{summary}"
        );
        assert_eq!(
            plan.buffers[&slot.buffer].freed_by,
            luminal::layout_ir::FreedBy::Caller,
            "{name}: the backing buffer escapes to the caller\n{summary}"
        );
        // THE DISCLOSURE (Option B, corrected contract): the binding
        // carries the slot VALUE's own elected layout `L`, verbatim from
        // the decoded table — for this weld, the transpose view's
        // composed layout. Mock plans transport `MockLayout` (the layout
        // class identity), so the pin here is IDENTITY: the returned L
        // is the value's own table row, and the ASSIGNMENT is queryable
        // (the escaping buffer backs a real tensor whose buffer is the
        // slot's). Element-level walkability of real decoded layouts is
        // pinned in `test_runtime::test_equality`'s own tests.
        let table = luminal::test_support::mock_layout_table(&dps_graph);
        assert_eq!(
            &slot.layout,
            table
                .get(&slot.value)
                .expect("slot value has a mock table row"),
            "{name}: the binding discloses the slot value's own elected layout\n{summary}"
        );
        let backs = plan
            .backed_tensor(&slot.buffer)
            .expect("escaping buffer has an assignment row")
            .clone();
        assert_eq!(
            plan.buffer_of(&backs),
            Some(&slot.buffer),
            "{name}: the assignment is queryable both ways (buffer <-> backed tensor)\n{summary}"
        );
    }
}

/// The donation OBSERVATION: an Accumulate whose C is an INTERMEDIATE
/// (program-freed) value is the shape where the May permit could let the
/// bufferizer place D in C's buffer. Soundness is the assertion; donation
/// is the printed observation.
#[test]
fn t6a_accumulate_intermediate_c_donation_observed() {
    let text = {
        let mut cx = Graph::new();
        let x = cx.tensor((4usize, 8usize), DType::F32);
        let w = cx.tensor((8usize, 3usize), DType::F32);
        let y = cx.tensor((4usize, 3usize), DType::F32);
        let z = cx.tensor((4usize, 3usize), DType::F32);
        let c = y + z; // intermediate C (program-freed once consumed)
                       // Original boundary-flowing spelling (restored under
                       // escape-and-disclose: the view-produced bound output escapes);
                       // this probe's subject is donation.
        let _ = (x.matmul(w) + c).output();
        cx.logical
            .bound_program(&test_runtime::TestRuntimeBindings)
            .expect("recorder clean")
            .text
    };
    let (graph, _) = test_runtime::extract_fixture_with_genome(&text, PIN);
    let elected = cublaslt_in_plan(&graph);
    assert!(
        elected
            .iter()
            .any(|(op, _)| op.form == CublasLtForm::Accumulate),
        "the Accumulate contract elected"
    );
    let plan = luminal::test_support::bufferize_mock(&luminal::dps::dps_rewrite(&graph))
        .unwrap_or_else(|err| panic!("T6a-donation: bufferizer REFUSED: {err}"));
    let allocs = plan
        .buffers
        .keys()
        .filter(|id| matches!(id, BufferId::Allocated(_)))
        .count();
    println!(
        "T6a donation probe: {allocs} alloc(s) (C's intermediate buffer {} donated into D)",
        if allocs == 0 { "WAS" } else { "was NOT" }
    );
}

// ===========================================================================
// The B x LEFT-major arm, field-verified.
// ===========================================================================
#[test]
fn t5_left_major_b_arm_field_verified() {
    let fx = format!(
        r#"(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout_lm (LeftMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt_lm (LayoutTensorLit x_logical x_layout_lm))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt_lm x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    // The LM reading exists with the swapped-COL geometry: storage LM
    // [2,4] IS col [2,4] (ld=2); op(B') needs [k,m]=[4,2] => T.
    let mut lm_reading_found = false;
    for node in s.nodes.values() {
        if node.op != "CublasLtOperandBDescriptor" {
            continue;
        }
        // round-8b: descriptors are (site, lt, operation) — child 2.
        let is_t = node
            .children
            .get(2)
            .and_then(|id| s.nodes.get(id))
            .map(|n| {
                s.nodes
                    .values()
                    .any(|m| m.eclass == n.eclass && m.op == "CublasLtOperationT")
            });
        if is_t == Some(true) {
            lm_reading_found = true;
        }
    }
    assert!(lm_reading_found, "the B x left-major T reading minted");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let elected = cublaslt_in_plan(&graph);
    assert_eq!(elected.len(), 1);
    let (op, inputs) = &elected[0];
    let spec = op.spec.as_ref().expect("spec parses");
    println!(
        "T5 B-LM elected: m={} n={} k={} trans_a={} trans_b={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.trans_a, spec.trans_b, spec.lda, spec.ldb, spec.ldd
    );
    assert_eq!(spec.mnk_lits(), (3, 2, 4));
    assert!(!spec.trans_a);
    if spec.trans_b {
        assert_eq!(spec.ldb, 2, "left-major pitch = m");
    } else {
        assert_eq!(spec.ldb, 4, "the RM materialized-copy reading");
    }
    assert_eq!(spec.lda, 3);
    assert_eq!(spec.ldd, 3);
    assert_eq!(inputs.len(), 2);
}

// ===========================================================================
// relu decoration of the SYMBOLIC-K static-bucket op.
// ===========================================================================
#[test]
fn t6b_relu_decorates_symbolic_k() {
    let fx = format!(
        r#"(let s_var (IntVar "s"))
(set (lower-bound-of s_var) (bigint 2))
(set (upper-bound-of s_var) (bigint 8))
(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons s_var (IntExprCons (IntLit 3) (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons s_var (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout_contig (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout (RightMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt_contig (LayoutTensorLit x_logical x_layout_contig))
(let w_lt (LayoutTensorLit w_logical w_layout))
(let out_lt (LayoutTensorLit out_logical out_layout))
(let x_static (StridedElementLayoutLit a_shape
  (IntAffineExprCons (IntMul (CoordVar a_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar a_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let x_static_lt (LayoutTensorLit x_logical x_static))
(set (injectivity-of x_static_lt) (Injective))
(let zconst (LogicalConstant 0.0))
(let scalar_shape (ShapeLit (IntExprNil)))
(let scalar_map (IndexMapLit (IntExprNil) scalar_shape))
(let zfill (LogicalIndexMapApply zconst scalar_map out_shape))
(let nconst (LogicalConstant -1.0))
(let nfill (LogicalIndexMapApply nconst scalar_map out_shape))
(let oconst (LogicalConstant 1.0))
(let ofill (LogicalIndexMapApply oconst scalar_map out_shape))
(let lt0 (LogicalLessThan out_logical zfill))
(let tc (LogicalCast lt0 (F32)))
(let term1 (LogicalMul tc zfill))
(let nm (LogicalMul tc nfill))
(let p (LogicalAdd nm ofill))
(let sel1 (LogicalCast (LogicalLessThan zfill p) (F32)))
(let sel2 (LogicalCast (LogicalLessThan p zfill) (F32)))
(let sum2 (LogicalAdd sel1 sel2))
(let bsel (LogicalLessThan zfill sum2))
(let u (LogicalCast bsel (F32)))
(let term2 (LogicalMul u out_logical))
(let relu_logical (LogicalAdd term1 term2))
(let relu_lt (LayoutTensorLit relu_logical out_layout))
(let xc_buffer_id (BufferLit 10))
(set (buffer-access-of xc_buffer_id) (ReadOnly))
(set (buffer-freed-by xc_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 12))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let relu_buffer_id (BufferLit 13))
(set (buffer-access-of relu_buffer_id) (ReadWrite))
(set (buffer-freed-by relu_buffer_id) (CallerFrees))
(let xc_buffer_tensor (BufferTensorLit x_lt_contig xc_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt w_buffer_id))
(let relu_buffer_tensor (BufferTensorLit relu_lt relu_buffer_id))
(let output (BufferOutputLit (BufferTensorCons relu_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let base_ops = count_op(&s, "LayoutTensorOpCublasLt");
    let relu_value = count_op(&s, "CublasLtEpilogueRelu");
    println!("T6b symbolic-k relu: {base_ops} base-form enode(s), relu-value minted: {relu_value}");
    assert!(
        relu_value >= 1,
        "the relu field rewrite fired on the symbolic-k op"
    );
    assert!(
        base_ops >= 2,
        "base + relu-decorated (same contract, field differs)"
    );

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let elected = cublaslt_in_plan(&graph);
    assert_eq!(elected.len(), 1);
    let spec = elected[0]
        .0
        .spec
        .as_ref()
        .expect("spec parses with symbolic k");
    println!(
        "  elected: ep={:?} m={} n={} k={} ldb={}",
        spec.epilogue, spec.m, spec.n, spec.k, spec.ldb
    );
    assert_eq!(
        spec.epilogue,
        CuEpilogue::Relu,
        "relu epilogue survives symbolic k"
    );
    assert!(matches!(spec.k, CuDim::Symbolic(_)));
    assert!(
        spec.ldb == 8 || matches!(spec.ldb, CuDim::Symbolic(_)),
        "bucket pitch or symbolic contiguous ld, got {}",
        spec.ldb
    );
    assert_eq!(elected[0].1.len(), 2);
}

// ===========================================================================
// Static-pitch OUTPUT layout: creator-rewrite certified, so the earlier
// refusal is LIFTED. Symbolic n (call-m) routed through A (w left-major so
// lda stays literal) with a bucket-max padded D.
// ===========================================================================
#[test]
fn t6c_strided_output_injectivity_probe() {
    let fx = format!(
        r#"(let s_var (IntVar "s"))
(set (lower-bound-of s_var) (bigint 2))
(set (upper-bound-of s_var) (bigint 8))
(let a_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 4) (IntExprNil)))))
(let b_shape (ShapeLit (IntExprCons (IntLit 4) (IntExprCons s_var (IntExprNil)))))
(let prod_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprCons (IntLit 4) (IntExprNil))))))
(let out_shape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))))
(let x_logical (LogicalTensorInputLit (LogicalIdLit "x") a_shape (F32)))
(let w_logical (LogicalTensorInputLit (LogicalIdLit "w") b_shape (F32)))
(let x_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 2)
      (IntExprCons (CoordVar prod_shape 0) (IntExprNil)))
    a_shape))
(let w_to_prod_map
  (IndexMapLit
    (IntExprCons (CoordVar prod_shape 0)
      (IntExprCons (CoordVar prod_shape 1) (IntExprNil)))
    b_shape))
(let x_applied (LogicalIndexMapApply x_logical x_to_prod_map prod_shape))
(let w_applied (LogicalIndexMapApply w_logical w_to_prod_map prod_shape))
(let out_logical (LogicalReduceSum (LogicalMul x_applied w_applied) 0))
(let x_layout (RightMajorContiguousElementLayoutLit a_shape (bits-of (F32))))
(let w_layout_lm (LeftMajorContiguousElementLayoutLit b_shape (bits-of (F32))))
(let out_layout_contig (RightMajorContiguousElementLayoutLit out_shape (bits-of (F32))))
(let x_lt (LayoutTensorLit x_logical x_layout))
(let w_lt_lm (LayoutTensorLit w_logical w_layout_lm))
(let out_lt_contig (LayoutTensorLit out_logical out_layout_contig))
(let out_static (StridedElementLayoutLit out_shape
  (IntAffineExprCons (IntMul (CoordVar out_shape 1) (IntLit 8))
    (IntAffineExprCons (CoordVar out_shape 0) (IntAffineExprNil)))
  (bits-of (F32))))
(let out_static_lt (LayoutTensorLit out_logical out_static))
(set (injectivity-of out_static_lt) (Injective))
(strided-lists out_static out_shape
  (IntExprCons (IntLit 2) (IntExprCons s_var (IntExprNil)))
  (IntExprCons (IntLit 8) (IntExprCons (IntLit 1) (IntExprNil)))
  (bits-of (F32)))
(let x_buffer_id (BufferLit 10))
(set (buffer-access-of x_buffer_id) (ReadOnly))
(set (buffer-freed-by x_buffer_id) (CallerFrees))
(let w_buffer_id (BufferLit 11))
(set (buffer-access-of w_buffer_id) (ReadOnly))
(set (buffer-freed-by w_buffer_id) (CallerFrees))
(let out_buffer_id (BufferLit 12))
(set (buffer-access-of out_buffer_id) (ReadWrite))
(set (buffer-freed-by out_buffer_id) (CallerFrees))
(let x_buffer_tensor (BufferTensorLit x_lt x_buffer_id))
(let w_buffer_tensor (BufferTensorLit w_lt_lm w_buffer_id))
(let out_buffer_tensor (BufferTensorLit out_lt_contig out_buffer_id))
(let output (BufferOutputLit (BufferTensorCons out_buffer_tensor (BufferTensorNil))))
{SCHEDULE}
"#
    );
    let s = test_runtime::serialize_fixture(&fx);
    let sites = count_op(&s, "CublasLtLogicalMatmulSite");
    let a = count_op(&s, "CublasLtOperandADescriptor");
    let b = count_op(&s, "CublasLtOperandBDescriptor");
    let d = count_op(&s, "CublasLtOutputDDescriptor");
    let ops = count_cublaslt(&s);
    println!("T6c strided output: {sites} site(s), readings a={a} b={b} d={d}, {ops} op enode(s)");
    // ROUND-10 RE-PIN: original + transpose-sandwich sibling; the padded
    // D is read through the sibling's composed view of the creator's
    // authored (strided-lists) pitched layout.
    assert_eq!(sites, 2, "the logical site pair marks");
    assert!(
        a >= 1,
        "A reading mints via the left-major arm (symbolic cols)"
    );
    assert!(b >= 1, "B reading mints");
    assert!(
        d >= 1,
        "the padded D reading mints against the creator facts"
    );
    assert!(ops >= 1, "creator-certified strided output ASSEMBLES");
    println!("  VERDICT: creator-rewrite strided output ASSEMBLES (refusal LIFTED)");

    let (graph, _) = test_runtime::extract_fixture_with_genome(&fx, PIN);
    let elected = cublaslt_in_plan(&graph);
    assert_eq!(elected.len(), 1);
    let spec = elected[0].0.spec.as_ref().expect("spec parses");
    println!(
        "  elected: m={} n={} k={} lda={} ldb={} ldd={}",
        spec.m, spec.n, spec.k, spec.lda, spec.ldb, spec.ldd
    );
    assert!(
        matches!(spec.m, CuDim::Symbolic(_)),
        "call m = logical n = symbolic"
    );
    assert_eq!(spec.k, 4);
    assert_eq!(spec.ldb, 4, "B = x RM contiguous, literal");
    assert!(
        spec.ldd == 8 || matches!(spec.ldd, CuDim::Symbolic(_)),
        "D ld is the bucket pitch or the symbolic contiguous extent, got {}",
        spec.ldd
    );
    assert_eq!(elected[0].1.len(), 2, "base contract Lit [a, b]");
}

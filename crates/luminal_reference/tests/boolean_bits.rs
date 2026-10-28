use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

fn run(g: &Graph, inputs: &[(GraphTensor, TypedBuffer)]) -> ReferenceRuntime {
    let data = FxHashMap::from_iter(inputs.iter().map(|(x, v)| (x.id, v.clone())));
    let mut rt = ReferenceRuntime::load(g).unwrap();
    rt.search(
        &Default::default(),
        &Default::default(),
        &data,
        &harness_search_options(),
    )
    .unwrap();
    for (x, v) in inputs {
        rt.set_data(x.id, v.clone());
    }
    rt.execute().unwrap();
    rt
}

#[test]
fn byte_encodings_and_boolean_algebra() {
    let mut g = Graph::new();
    let x = g.tensor(vec![256], DType::U8);
    let bits = x.bits_to_bools();
    assert_eq!(bits.dims(), vec![IntExpr::from(256), IntExpr::from(8)]);
    let packed = bits.bools_to_bits(DType::U8);
    let inverted = bits.logical_not().bools_to_bits(DType::U8);
    let zeros = bits
        .logical_and(bits.logical_not())
        .bools_to_bits(DType::U8);
    let ones = bits.logical_or(bits.logical_not()).bools_to_bits(DType::U8);
    let data: Vec<u8> = (0..=255).collect();
    let rt = run(&g, &[(x, TypedBuffer::U8(data.clone()))]);
    assert_eq!(*rt.get_u8(packed.id).unwrap(), data);
    assert_eq!(
        *rt.get_u8(inverted.id).unwrap(),
        data.iter().map(|x| !x).collect::<Vec<_>>()
    );
    assert_eq!(*rt.get_u8(zeros.id).unwrap(), vec![0; 256]);
    assert_eq!(*rt.get_u8(ones.id).unwrap(), vec![255; 256]);
}

fn run_forced(
    g: &Graph,
    inputs: &[(GraphTensor, TypedBuffer)],
    excluded: &[&str],
    required: &[&str],
) -> ReferenceRuntime {
    let data = FxHashMap::from_iter(inputs.iter().map(|(x, v)| (x.id, v.clone())));
    let mut rt = ReferenceRuntime::load(g).unwrap();
    rt.with_ops(
        luminal_reference::reference_allow_list()
            .into_iter()
            .filter(|op| !excluded.contains(op))
            .collect(),
    )
    .unwrap();
    let outcome = rt
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .unwrap();
    let labels = outcome
        .best_plan
        .dag
        .node_weights()
        .filter_map(|node| match node {
            luminal::bufferize::BufferNode::Compute { op, .. } => Some(op.label()),
            _ => None,
        })
        .collect::<Vec<_>>();
    for expected in required {
        if expected.ends_with("View") {
            let materializer = expected.replace("View", "Generic");
            assert!(
                !labels.contains(&materializer.as_str()),
                "view route materialized: {labels:?}"
            );
            assert!(
                outcome.best_plan.dag.node_weights().any(|node| match node {
                    luminal::bufferize::BufferNode::Compute { operand_info, .. } => operand_info
                        .iter()
                        .any(|s| s.layout != outcome.best_plan.buffers[&s.buffer].layout),
                    _ => false,
                }),
                "view did not reach a consumer descriptor"
            );
        } else {
            assert!(labels.contains(expected), "missing {expected}: {labels:?}");
        }
    }
    for (x, v) in inputs {
        rt.set_data(x.id, v.clone());
    }
    rt.execute().unwrap();
    rt
}

#[test]
fn forced_unpack_view_and_materializer_agree() {
    for view in [false, true] {
        let mut g = Graph::new();
        let x = g.tensor(vec![3], DType::U8);
        let bits = x.bits_to_bools().logical_not();
        let out = bits.bools_to_bits(DType::U8);
        let excluded = if view {
            "LayoutTensorOpBitsToBoolsGeneric"
        } else {
            "LayoutTensorOpBitsToBoolsView"
        };
        let required = if view {
            "BitsToBoolsView"
        } else {
            "BitsToBoolsGeneric"
        };
        let rt = run_forced(
            &g,
            &[(x, TypedBuffer::U8(vec![0, 0x81, 0xff]))],
            &[excluded],
            &[required],
        );
        assert_eq!(*rt.get_u8(out.id).unwrap(), vec![255, 0x7e, 0]);
    }
}

#[test]
fn forced_pack_view_and_materializer_agree() {
    for view in [false, true] {
        let mut g = Graph::new();
        let x = g.tensor(vec![2, 8], DType::Bool);
        let out = x.bools_to_bits(DType::U8).cast(DType::Int);
        let excluded = if view {
            "LayoutTensorOpBoolsToBitsGeneric"
        } else {
            "LayoutTensorOpBoolsToBitsView"
        };
        let required = if view {
            "BoolsToBitsView"
        } else {
            "BoolsToBitsGeneric"
        };
        let data = vec![1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0];
        let rt = run_forced(
            &g,
            &[(x, TypedBuffer::bool8(data).unwrap())],
            &[excluded],
            &[required],
        );
        assert_eq!(*rt.get_i32(out.id).unwrap(), vec![0x81, 0x42]);
    }
}

#[test]
fn signed_and_wide_encodings() {
    let mut g = Graph::new();
    let a = g.tensor(vec![4], DType::I64);
    let b = g.tensor(vec![4], DType::U64);
    let aa = vec![i64::MIN, -1, 0, i64::MAX];
    let bb = vec![0, 1u64 << 63, u64::MAX, (1u64 << 54) + 3];
    let aout = a.bits_to_bools().logical_not().bools_to_bits(DType::I64);
    let bout = b.bits_to_bools().bools_to_bits(DType::U64);
    let rt = run(
        &g,
        &[
            (a, TypedBuffer::I64(aa.clone())),
            (b, TypedBuffer::U64(bb.clone())),
        ],
    );
    assert_eq!(
        *rt.get_i64(aout.id).unwrap(),
        aa.iter().map(|x| !x).collect::<Vec<_>>()
    );
    assert_eq!(*rt.get_u64(bout.id).unwrap(), bb);
}

#[test]
fn float_payloads_and_signed_zero_round_trip_without_arithmetic() {
    let mut g = Graph::new();
    let x = g.tensor(vec![6], DType::F32);
    let y = g.tensor(vec![6], DType::F64);
    let out = x.bits_to_bools().bools_to_bits(DType::F32);
    let yout = y.bits_to_bools().bools_to_bits(DType::F64);
    let codes32 = [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 1];
    let codes64 = [
        0,
        1u64 << 63,
        0x7ff0000000000000,
        0xfff0000000000000,
        0x7ff8000000001234,
        1,
    ];
    let rt = run(
        &g,
        &[
            (x, TypedBuffer::F32(codes32.map(f32::from_bits).to_vec())),
            (y, TypedBuffer::F64(codes64.map(f64::from_bits).to_vec())),
        ],
    );
    assert_eq!(
        rt.get_f32(out.id)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        codes32
    );
    assert_eq!(
        rt.get_f64(yout.id)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        codes64
    );
}

#[test]
fn transpose_bit_axis_requires_materialization_and_keeps_logical_order() {
    let mut g = Graph::new();
    let x = g.tensor(vec![8, 2], DType::U8);
    // [8,2,8] -> [8,8,2], then pack pairs of original columns into U16.
    let bits = x.bits_to_bools().permute(vec![0, 2, 1]).merge_dims(1, 2);
    let out = bits.bools_to_bits(DType::I16);
    let values = (0..16u8).collect::<Vec<_>>();
    let rt = run(&g, &[(x, TypedBuffer::U8(values.clone()))]);
    let expected = values
        .as_chunks::<2>()
        .0
        .iter()
        .map(|v| {
            let mut code = 0u16;
            for bit in 0..8 {
                code |= (((v[0] >> bit) & 1) as u16) << (2 * bit);
                code |= (((v[1] >> bit) & 1) as u16) << (2 * bit + 1);
            }
            code as i16
        })
        .collect::<Vec<_>>();
    assert_eq!(*rt.get_i16(out.id).unwrap(), expected);
}

#[test]
fn scalar_unpack_and_empty_last_axis() {
    let mut g = Graph::new();
    let x = g.tensor(Vec::<usize>::new(), DType::U8);
    let y = g.tensor(vec![2, 0], DType::Bool);
    let out = x.bits_to_bools().bools_to_bits(DType::U8);
    let empty = y.bools_to_bits(DType::U8);
    assert_eq!(out.dims(), vec![IntExpr::from(1)]);
    let rt = run(
        &g,
        &[
            (x, TypedBuffer::U8(vec![0xa5])),
            (y, TypedBuffer::bool8(vec![]).unwrap()),
        ],
    );
    assert_eq!(*rt.get_u8(out.id).unwrap(), vec![0xa5]);
    assert!(rt.get_u8(empty.id).unwrap().is_empty());
}

#[test]
#[should_panic(expected = "divisible")]
fn packing_rejects_partial_group() {
    let mut g = Graph::new();
    g.tensor(vec![7], DType::Bool).bools_to_bits(DType::U8);
}
#[test]
#[should_panic(expected = "last axis")]
fn packing_rejects_scalar() {
    let mut g = Graph::new();
    g.tensor(Vec::<usize>::new(), DType::Bool)
        .bools_to_bits(DType::U8);
}
#[test]
#[should_panic(expected = "Bool input")]
fn boolean_algebra_rejects_integer_truthiness() {
    let mut g = Graph::new();
    g.tensor(vec![1], DType::Int).logical_not();
}

#[test]
fn byte_boolean_kernels_enforce_legal_codes() {
    use luminal_reference::{ReferenceKernelCtx, kernels, ops};
    let mut ctx = ReferenceKernelCtx {
        operands: vec![
            TypedBuffer::Bool8(vec![0, 0, 1, 1]),
            TypedBuffer::Bool8(vec![0, 1, 0, 1]),
        ],
        operand_dims: vec![vec![4], vec![4]],
        dests: vec![TypedBuffer::Bool8(vec![0; 4])],
        dims: Default::default(),
    };
    for (op, expected) in [
        (
            &ops::AndDps as &dyn luminal::buffer_tensor_ir::BufferTensorIrOp,
            vec![0, 0, 0, 1],
        ),
        (
            &ops::OrDps as &dyn luminal::buffer_tensor_ir::BufferTensorIrOp,
            vec![0, 1, 1, 1],
        ),
        (
            &ops::NotDps as &dyn luminal::buffer_tensor_ir::BufferTensorIrOp,
            vec![1, 1, 0, 0],
        ),
    ] {
        (kernels::kernel_for(op).unwrap().execute)(op, &mut ctx).unwrap();
        assert_eq!(*ctx.dests[0].as_bool8().unwrap(), expected);
    }
    ctx.operands[0] = TypedBuffer::Bool8(vec![0, 0xff, 0, 0]);
    assert!((kernels::kernel_for(&ops::NotDps).unwrap().execute)(&ops::NotDps, &mut ctx).is_err());
    // Packing a byte Boolean validates all eight bits, rather than testing truthiness.
    let op = ops::BoolsToBitsDps { width: 8 };
    ctx.operands = vec![TypedBuffer::Bool8(vec![1, 1, 0, 0, 0, 0, 0, 0])];
    ctx.dests = vec![TypedBuffer::Bool8(vec![0])];
    assert!((kernels::kernel_for(&op).unwrap().execute)(&op, &mut ctx).is_err());
    ctx.operands[0] = TypedBuffer::Bool8(vec![1, 0, 0, 0, 0, 0, 0, 0]);
    (kernels::kernel_for(&op).unwrap().execute)(&op, &mut ctx).unwrap();
    assert_eq!(*ctx.dests[0].as_bool8().unwrap(), vec![1]);
}

#[test]
fn selected_bit_plan_serializes_and_executes_without_search() {
    for materialize in [false, true] {
        let mut g = Graph::new();
        let x = g.tensor(vec![3], DType::U8);
        let out = x.bits_to_bools().logical_not().bools_to_bits(DType::U8);
        let excluded = if materialize {
            "LayoutTensorOpBitsToBoolsView"
        } else {
            "LayoutTensorOpBitsToBoolsGeneric"
        };
        let leader = run_forced(&g, &[(x, TypedBuffer::U8(vec![1, 2, 3]))], &[excluded], &[]);
        let output = luminal_reference::ReferenceBindings::leaves(&g.logical).outputs()[0].buffer;
        let artifact = leader.serialize_compiled(&[x.id], &[output]).unwrap();
        let mut follower = ReferenceRuntime::default();
        follower
            .deserialize_compiled(&artifact, &[x.id], &[out.id], 1 << 20)
            .unwrap();
        follower.set_data(x.id, TypedBuffer::U8(vec![0xff, 0x81, 0]));
        follower.execute().unwrap();
        assert_eq!(*follower.get_u8(out.id).unwrap(), vec![0, 0x7e, 255]);
    }
}

#[test]
fn dynamic_outer_extent_reuses_bit_view_plan() {
    let mut g = Graph::new();
    let n = IntExpr::from('n');
    let x = g.tensor(vec![n, 2.into()], DType::U8);
    let out = x.bits_to_bools().logical_not().bools_to_bits(DType::U8);
    let mut rt = ReferenceRuntime::load(&g).unwrap();
    rt.with_ops(
        luminal_reference::reference_allow_list()
            .into_iter()
            .filter(|x| *x != "LayoutTensorOpBitsToBoolsGeneric")
            .collect(),
    )
    .unwrap();
    let bounds = SymbolBounds::from_ranges([(Symbol::from('n'), (0, 3))]).unwrap();
    let profile = FxHashMap::from_iter([(Symbol::from('n'), 2usize)]);
    rt.search(
        &bounds,
        &profile,
        &FxHashMap::from_iter([(x.id, TypedBuffer::U8(vec![1, 2, 3, 4]))]),
        &harness_search_options(),
    )
    .unwrap();
    for rows in [0, 1, 3, 2] {
        let values = (0..rows * 2).map(|v| v as u8).collect::<Vec<_>>();
        rt.set_dim('n', rows);
        rt.set_data(x.id, TypedBuffer::U8(values.clone()));
        rt.execute().unwrap();
        assert_eq!(
            *rt.get_u8(out.id).unwrap(),
            values.iter().map(|v| !v).collect::<Vec<_>>()
        );
    }
}

#[test]
fn packing_same_bits_into_equal_width_dtypes_keeps_distinct_values() {
    for materialize in [false, true] {
        let mut g = Graph::new();
        let x = g.tensor(vec![3], DType::U64);
        let bits = x.bits_to_bools();
        let unsigned = bits.bools_to_bits(DType::U64);
        let signed = bits.bools_to_bits(DType::I64);
        let floating = bits.bools_to_bits(DType::F64);
        let values = vec![u64::MAX, 1 << 63, (1 << 54) + 3];
        let excluded = if materialize {
            "LayoutTensorOpBoolsToBitsView"
        } else {
            "LayoutTensorOpBoolsToBitsGeneric"
        };
        let rt = run_forced(
            &g,
            &[(x, TypedBuffer::U64(values.clone()))],
            &[excluded],
            &[],
        );
        assert_eq!(*rt.get_u64(unsigned.id).unwrap(), values);
        assert_eq!(
            *rt.get_i64(signed.id).unwrap(),
            values.iter().map(|v| *v as i64).collect::<Vec<_>>()
        );
        assert_eq!(
            rt.get_f64(floating.id)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values
        );
    }
}

#[test]
fn noncontiguous_bit_input_packs_directly_into_its_destination() {
    let script = r#"
(let s (ShapeLit (IntExprCons (IntLit 16) (IntExprNil))))
(let input (LogicalTensorInputLit (LogicalIdLit "strided-bits") s (Bool)))
(let il (StridedElementLayoutLit s
 (IntAffineExprCons (IntMul (CoordVar s 0) (IntLit 2)) (IntAffineExprNil)) (BitWidthLit 1)))
(let it (LayoutTensorLit input il))
(let out (LogicalBoolsToBits input (U8)))
(let os (ShapeLit (IntExprCons (IntLit 2) (IntExprNil))))
(let ot (LayoutTensorLit out (RightMajorContiguousElementLayoutLit os (BitWidthLit 8))))
(let ib (BufferLit 0))
(set (buffer-access-of ib) (ReadOnly))
(set (buffer-freed-by ib) (CallerFrees))
(let ob (BufferLit 1))
(set (buffer-access-of ob) (ReadWrite))
(set (buffer-freed-by ob) (CallerFrees))
(let bi (BufferInputLit (BufferTensorCons (BufferTensorLit it ib) (BufferTensorNil))))
(let bo (BufferOutputLit (BufferTensorCons (BufferTensorLit ot ob) (BufferTensorNil))))
"#;
    let mut eg = luminal::egglog_snippet::new_egraph();
    eg.parse_and_run_program(
        None,
        &format!(
            "{}{}{}",
            luminal_reference::assembled_program(),
            script,
            luminal_reference::ReferenceBindings::SCHEDULE
        ),
    )
    .unwrap();
    let serialized = eg
        .serialize(luminal::prelude::egglog::SerializeConfig::default())
        .egraph;
    let extracted = luminal_reference::extract_layout_ir(&serialized)
        .unwrap()
        .unwrap();
    let dps = luminal::dps::dps_rewrite(&extracted);
    let view = luminal::egglog_utils::eclass::EGraphView::new(
        &serialized,
        luminal_reference::decoder_registry(),
    );
    let layouts = luminal::layouts::decode_layout_table(
        &view,
        &dps,
        "strided-bit test",
        &mut luminal::layouts::LayoutDecodeCache::new(),
    )
    .unwrap();
    let plan = luminal::bufferize::bufferize(&dps, &layouts).unwrap();
    assert!(plan.dag.node_weights().any(|n|matches!(n,luminal::bufferize::BufferNode::Compute{op,..} if op.label()=="BoolsToBitsGeneric")));
    assert!(
        !plan
            .dag
            .node_weights()
            .any(|n| matches!(n, luminal::bufferize::BufferNode::BufferCopy { .. }))
    );
    let expected = [0x81u8, 0x42];
    let mut data = vec![1u8; 31];
    for i in 0..16 {
        data[2 * i] = (expected[i / 8] >> (i % 8)) & 1;
    }
    let mut rt = ReferenceRuntime::default();
    rt.load_plan(plan, Default::default());
    rt.set_data_buffer(0, TypedBuffer::bool8(data).unwrap());
    rt.execute().unwrap();
    assert_eq!(*rt.get_buffer(1).unwrap().as_u8().unwrap(), expected);
}

#[test]
fn numeric_producer_storage_matches_its_layout_before_bit_observation() {
    let mut g = Graph::new();
    let x = g.tensor(vec![2, 3], DType::F32);
    let y = g.tensor(vec![3, 2], DType::F32);
    let out = (x.permute(vec![1, 0]) * y)
        .bits_to_bools()
        .bools_to_bits(DType::F32);
    let rt = run_forced(
        &g,
        &[
            (x, TypedBuffer::F32(vec![1., 2., 3., 4., 5., 6.])),
            (y, TypedBuffer::F32(vec![10., 20., 30., 40., 50., 60.])),
        ],
        &["LayoutTensorOpBitsToBoolsGeneric"],
        &[],
    );
    assert_eq!(
        *rt.get_f32(out.id).unwrap(),
        vec![10., 80., 60., 200., 150., 360.]
    );
}

/// A zero-element tensor's bit view, sliced into its IEEE fields and
/// reduced the way the encoding classifiers read it, executes to an empty
/// result. The field slice's strided layout used to size its storage at a
/// negative reach (`1 + Σ (extent−1)*stride` over a zero extent: −9 for the
/// F32 significand of (1,0,3)), which failed the search's allocation
/// pruning with "out of range integral type conversion attempted".
#[test]
fn sliced_bit_view_of_a_zero_element_tensor_is_empty() {
    for (shape, dtype, mantissa, end) in [
        (vec![1, 0, 3], DType::F32, 23, 31),
        (vec![0], DType::F64, 52, 63),
        (vec![0, 1, 3], DType::F16, 10, 15),
    ] {
        let mut g = Graph::new();
        let x = g.tensor(shape.clone(), dtype);
        let bits = x.bits_to_bools();
        let axis = bits.rank() - 1;
        let exponent = bits
            .slice_along(mantissa..end, axis)
            .cast(DType::Int)
            .sum(vec![axis]);
        let fraction = bits
            .slice_along(0..mantissa, axis)
            .cast(DType::Int)
            .sum(vec![axis]);
        let zero = g.constant_i32(0).expand_rhs(x.dims());
        let out = exponent
            .gt(zero)
            .logical_and(fraction.gt(zero))
            .cast(DType::Int);
        assert_eq!(
            out.dims(),
            shape.iter().map(|&d| IntExpr::from(d)).collect::<Vec<_>>()
        );
        let empty = match dtype {
            DType::F32 => TypedBuffer::F32(vec![]),
            DType::F64 => TypedBuffer::F64(vec![]),
            DType::F16 => TypedBuffer::F16(vec![]),
            _ => unreachable!(),
        };
        let rt = run(&g, &[(x, empty)]);
        assert!(
            rt.get_i32(out.id).unwrap().is_empty(),
            "{dtype:?} {shape:?}"
        );
    }
}

/// A value slice of a zero-element tensor (`x[:, :1]` for `x` of shape
/// (0, 2)) is the same strided view over an empty domain and executes too.
#[test]
fn slice_of_a_zero_element_tensor_is_empty() {
    let mut g = Graph::new();
    let x = g.tensor(vec![0, 2], DType::F32);
    let out = x.slice_along(0..1, 1) * 2.0;
    assert_eq!(out.dims(), vec![IntExpr::from(0), IntExpr::from(1)]);
    let rt = run(&g, &[(x, TypedBuffer::F32(vec![]))]);
    assert!(rt.get_f32(out.id).unwrap().is_empty());
}

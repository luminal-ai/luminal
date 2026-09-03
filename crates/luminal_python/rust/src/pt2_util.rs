use luminal::prelude::*;

fn same_dim(lhs: IntExpr, rhs: IntExpr) -> bool {
    lhs == rhs || lhs.simplify() == rhs.simplify() || lhs.egglog_equal(rhs)
}

/// Binary operation type.
#[derive(Clone, Copy)]
pub enum BinaryOp {
    Add,
    Mul,
    Sub,
    Div,
}

/// Reduction operation type.
#[derive(Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Mean,
    Max,
    Min,
    Prod,
}

/// Normalize a potentially negative dimension index.
pub fn normalize_dim(dim: i64, ndim: usize) -> usize {
    if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    }
}

pub fn normalize_slice_bound(bound: IntExpr, dim_size: IntExpr) -> IntExpr {
    match bound.as_num() {
        Some(n) if n < 0 => (dim_size + IntExpr::from(n as i32)).simplify(),
        _ => bound,
    }
}

/// Broadcast two tensors following NumPy broadcasting rules.
/// Right-aligns dims, unsqueezes shorter, expands size-1 dims.
pub fn broadcast_binary(mut a: GraphTensor, mut b: GraphTensor) -> (GraphTensor, GraphTensor) {
    let a_ndim = a.legacy_tracker_ref().len();
    let b_ndim = b.legacy_tracker_ref().len();

    // Right-align: unsqueeze the shorter tensor on the left
    if a_ndim < b_ndim {
        for _ in 0..(b_ndim - a_ndim) {
            a = a.unsqueeze(0);
        }
    } else if b_ndim < a_ndim {
        for _ in 0..(a_ndim - b_ndim) {
            b = b.unsqueeze(0);
        }
    }

    // Now both have same ndim. Expand size-1 dims to match.
    let ndim = a.legacy_tracker_ref().len();
    for i in 0..ndim {
        let a_dim = a.legacy_tracker_ref().dims[i];
        let b_dim = b.legacy_tracker_ref().dims[i];

        if same_dim(a_dim, b_dim) {
            continue;
        }

        if a_dim.to_usize() == Some(1) {
            a.legacy_tracker_mut().dims[i] = b_dim;
            a.legacy_tracker_mut().strides[i] = IntExpr::from(0usize);
        } else if b_dim.to_usize() == Some(1) {
            b.legacy_tracker_mut().dims[i] = a_dim;
            b.legacy_tracker_mut().strides[i] = IntExpr::from(0usize);
        }
    }

    (a, b)
}

/// Ensure two tensors have the same dtype, casting Int->F32 or Bool->F32 if needed.
pub fn ensure_same_dtype(a: GraphTensor, b: GraphTensor) -> (GraphTensor, GraphTensor) {
    if a.dtype == b.dtype {
        return (a, b);
    }
    // Promotion lattice mirroring torch's type-promotion rules, as
    // implemented by `torch.promote_types` (documented under "Type
    // promotion",
    // https://docs.pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc):
    // wider wins, floats
    // beat ints, and a mixed f16/bf16 pair promotes to f32 — verifiable
    // directly: promote_types(int32, int64) == int64,
    // promote_types(float16, bfloat16) == float32. The old table promoted
    // (Int, I64) to Int, silently truncating the 64-bit side.
    let target = match (a.dtype, b.dtype) {
        (DType::F64, _) | (_, DType::F64) => DType::F64,
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        (DType::F16, DType::Bf16) | (DType::Bf16, DType::F16) => DType::F32,
        (DType::F16, _) | (_, DType::F16) => DType::F16,
        (DType::Bf16, _) | (_, DType::Bf16) => DType::Bf16,
        (DType::I64, _) | (_, DType::I64) => DType::I64,
        (DType::Int, _) | (_, DType::Int) => DType::Int,
        _ => DType::F32,
    };
    let a = if a.dtype != target { a.cast(target) } else { a };
    let b = if b.dtype != target { b.cast(target) } else { b };
    (a, b)
}

/// Reshape a GraphTensor by replacing its ShapeTracker (view-only, no new
/// node). LEGACY: this is the torch-FX translator's own pipeline; the
/// wholesale tracker replacement goes through the A2 escape hatch and
/// clears any recorder view handle (the recorder path is not maintained
/// by this translator).
pub fn reshape_tensor(t: GraphTensor, shape: Vec<IntExpr>) -> GraphTensor {
    let mut out = t;
    *out.legacy_tracker_mut() = ShapeTracker::new(shape);
    out.logical_view = None;
    out
}

/// Resolve -1 in a reshape target shape.
pub fn resolve_neg1_dim(target: &[i64], current_dims: &[IntExpr]) -> Vec<IntExpr> {
    let mut neg1_idx = None;
    let mut known_product: i64 = 1;
    let mut result = Vec::with_capacity(target.len());

    for (i, &s) in target.iter().enumerate() {
        if s == -1 {
            neg1_idx = Some(i);
            result.push(IntExpr::from(0usize)); // placeholder
        } else {
            known_product *= s;
            result.push(IntExpr::from(s as usize));
        }
    }

    if let Some(idx) = neg1_idx {
        result[idx] = match current_dims
            .iter()
            .map(|d| d.to_usize())
            .collect::<Option<Vec<_>>>()
        {
            Some(vs) => IntExpr::from(vs.iter().product::<usize>() / known_product as usize),
            None => {
                crate::dim_arith::product_of_dims(current_dims.iter().copied())
                    / IntExpr::from(known_product as usize)
            }
        };
    }

    result
}

/// Resolve -1 in a reshape target shape that contains IntExpr values.
pub fn resolve_neg1_dim_exprs(
    target: &[IntExpr],
    current_dims: &[IntExpr],
) -> Vec<IntExpr> {
    let neg1_expr = IntExpr::from(-1i32);
    let neg1_idx = target.iter().position(|e| *e == neg1_expr);

    if let Some(idx) = neg1_idx {
        let mut result = target.to_vec();

        let mut input_concrete: i64 = 1;
        let mut input_symbolic: Vec<IntExpr> = Vec::new();
        for d in current_dims {
            if let Some(v) = d.to_usize() {
                input_concrete *= v as i64;
            } else {
                input_symbolic.push(*d);
            }
        }

        let mut target_concrete: i64 = 1;
        let mut target_symbolic: Vec<IntExpr> = Vec::new();
        for (i, e) in target.iter().enumerate() {
            if i == idx {
                continue;
            }
            if let Some(v) = e.to_usize() {
                target_concrete *= v as i64;
            } else {
                target_symbolic.push(*e);
            }
        }

        for ts in &target_symbolic {
            if let Some(pos) = input_symbolic.iter().position(|is| is == ts) {
                input_symbolic.remove(pos);
            }
        }

        if input_symbolic.is_empty() {
            result[idx] = IntExpr::from((input_concrete / target_concrete) as usize);
        } else {
            let mut operands: Vec<IntExpr> = Vec::with_capacity(input_symbolic.len() + 1);
            operands.push(IntExpr::from(
                (input_concrete / target_concrete) as usize,
            ));
            operands.extend(input_symbolic.iter().copied());
            result[idx] = crate::dim_arith::product_of_dims(operands);
        }

        result
    } else {
        target.to_vec()
    }
}

/// Map a PT2 dtype code to luminal `DType`. Panics for variants the IR
/// doesn't model as first-class types (the complex family, unsupported
/// float8 variants, and uint16) and for unknown codes. The common narrow
/// integers map to native-width HLIR dtypes without widening.
pub fn torch_dtype_int_to_luminal(dtype: u32) -> DType {
    let t = crate::torch_dtype::TorchDType::from_code(dtype)
        .unwrap_or_else(|c| panic!("torch_dtype_int_to_luminal: unknown PT2 dtype code {c}"));
    DType::try_from(t).unwrap_or_else(|t| {
        panic!(
            "torch_dtype_int_to_luminal: {} isn't a first-class luminal IR type",
            t.name()
        )
    })
}


/// LOCAL FOSSIL of the destroyed ShapeTracker::expand (Austin 2026-07-31):
/// PyTorch-broadcast on tracker state, kept only until this translator's
/// rework onto explicit logical views.
pub fn tracker_expand(tracker: &mut ShapeTracker, new_shape: impl luminal::prelude::ToShape) {
    let new_shape = new_shape.to_shape();
    assert!(
        new_shape.len() >= tracker.len(),
        "Cannot expand from {} dims to {} dims",
        tracker.len(),
        new_shape.len()
    );
    while tracker.len() < new_shape.len() {
        tracker.expand_dim(0, 1);
    }
    for (axis, ((size, dim), stride)) in new_shape
        .into_iter()
        .zip(&mut tracker.dims)
        .zip(&mut tracker.strides)
        .enumerate()
    {
        if *dim == size {
            continue;
        }
        if dim.to_usize() == Some(1) {
            *dim = size;
            *stride = 0.into();
        } else {
            let (dim_simplified, size_simplified) = (dim.simplify(), size.simplify());
            if dim_simplified == size_simplified {
                *dim = size;
            } else {
                panic!(
                    "Cannot expand dim {axis} from {dim} to {size} \
                     (simplified: {dim_simplified} vs {size_simplified})",
                );
            }
        }
    }
}

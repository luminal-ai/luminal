use luminal::prelude::*;

fn same_dim(lhs: Expression, rhs: Expression) -> bool {
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

/// Broadcast two tensors following NumPy broadcasting rules.
/// Right-aligns dims, unsqueezes shorter, expands size-1 dims.
pub fn broadcast_binary(mut a: GraphTensor, mut b: GraphTensor) -> (GraphTensor, GraphTensor) {
    let a_ndim = a.shape.len();
    let b_ndim = b.shape.len();

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
    let ndim = a.shape.len();
    for i in 0..ndim {
        let a_dim = a.shape.dims[i];
        let b_dim = b.shape.dims[i];

        if same_dim(a_dim, b_dim) {
            continue;
        }

        if a_dim.to_usize() == Some(1) {
            a.shape.dims[i] = b_dim;
            a.shape.strides[i] = Expression::from(0usize);
        } else if b_dim.to_usize() == Some(1) {
            b.shape.dims[i] = a_dim;
            b.shape.strides[i] = Expression::from(0usize);
        }
    }

    (a, b)
}

/// Ensure two tensors have the same dtype, casting Int->F32 or Bool->F32 if needed.
pub fn ensure_same_dtype(a: GraphTensor, b: GraphTensor) -> (GraphTensor, GraphTensor) {
    if a.dtype == b.dtype {
        return (a, b);
    }
    let target = match (a.dtype, b.dtype) {
        (DType::F32, _) | (_, DType::F32) => DType::F32,
        (DType::Int, _) | (_, DType::Int) => DType::Int,
        _ => DType::F32,
    };
    let a = if a.dtype != target { a.cast(target) } else { a };
    let b = if b.dtype != target { b.cast(target) } else { b };
    (a, b)
}

/// Reshape a GraphTensor by replacing its ShapeTracker (view-only, no new node).
pub fn reshape_tensor(t: GraphTensor, shape: Vec<Expression>) -> GraphTensor {
    let new_shape = ShapeTracker::new(shape);
    GraphTensor {
        id: t.id,
        graph_ref: t.graph_ref,
        shape: new_shape,
        dtype: t.dtype,
    }
}

/// Resolve -1 in a reshape target shape.
pub fn resolve_neg1_dim(target: &[i64], current_dims: &[Expression]) -> Vec<Expression> {
    let mut neg1_idx = None;
    let mut known_product: i64 = 1;
    let mut result = Vec::with_capacity(target.len());

    for (i, &s) in target.iter().enumerate() {
        if s == -1 {
            neg1_idx = Some(i);
            result.push(Expression::from(0usize)); // placeholder
        } else {
            known_product *= s;
            result.push(Expression::from(s as usize));
        }
    }

    if let Some(idx) = neg1_idx {
        let mut total = Expression::from(1usize);
        for d in current_dims {
            total *= *d;
        }
        if let (Some(total_val), Some(_)) = (
            {
                let mut t = 1i64;
                let mut all_concrete = true;
                for d in current_dims {
                    if let Some(v) = d.to_usize() {
                        t *= v as i64;
                    } else {
                        all_concrete = false;
                    }
                }
                if all_concrete { Some(t) } else { None }
            },
            Some(known_product),
        ) {
            result[idx] = Expression::from((total_val / known_product) as usize);
        } else {
            result[idx] = total / Expression::from(known_product as usize);
        }
    }

    result
}

/// Resolve -1 in a reshape target shape that contains Expression values.
pub fn resolve_neg1_dim_exprs(
    target: &[Expression],
    current_dims: &[Expression],
) -> Vec<Expression> {
    let neg1_expr = Expression::from(-1i32);
    let neg1_idx = target.iter().position(|e| *e == neg1_expr);

    if let Some(idx) = neg1_idx {
        let mut result = target.to_vec();

        let mut input_concrete: i64 = 1;
        let mut input_symbolic: Vec<Expression> = Vec::new();
        for d in current_dims {
            if let Some(v) = d.to_usize() {
                input_concrete *= v as i64;
            } else {
                input_symbolic.push(*d);
            }
        }

        let mut target_concrete: i64 = 1;
        let mut target_symbolic: Vec<Expression> = Vec::new();
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
            result[idx] = Expression::from((input_concrete / target_concrete) as usize);
        } else {
            let mut expr = Expression::from((input_concrete / target_concrete) as usize);
            for s in &input_symbolic {
                expr *= *s;
            }
            result[idx] = expr;
        }

        result
    } else {
        target.to_vec()
    }
}

/// Map a PT2 dtype code to luminal `DType`. Panics for variants the IR
/// doesn't model as first-class types (narrow ints `Byte` / `Char` /
/// `Short`, the complex family, the float8 family) and for unknown
/// codes — better to fail loudly at the translator boundary than to
/// silently widen and lie about the user's dtype.
pub fn torch_dtype_int_to_luminal(dtype: u32) -> DType {
    let t = crate::torch_dtype::TorchDType::from_code(dtype)
        .unwrap_or_else(|c| panic!("torch_dtype_int_to_luminal: unknown PT2 dtype code {c}"));
    match t {
        crate::torch_dtype::TorchDType::Byte
        | crate::torch_dtype::TorchDType::Char
        | crate::torch_dtype::TorchDType::Short => panic!(
            "torch_dtype_int_to_luminal: PT2 dtype {} (code {}) isn't a first-class \
             IR type yet — cast to torch.int32 at the call site, or wait for the \
             narrower-int IR follow-up.",
            t.name(),
            t.code(),
        ),
        other => DType::try_from(other).unwrap_or_else(|t| {
            panic!(
                "torch_dtype_int_to_luminal: {} isn't a first-class luminal IR type",
                t.name()
            )
        }),
    }
}

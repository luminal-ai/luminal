use luminal::prelude::*;

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

        if a_dim == b_dim {
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

/// Map torch dtype integer (PT2 format) to luminal DType.
/// PT2 numbering: 1=uint8, 2=int8, 3=int16, 4=int32, 5=int64, 6=float16, 7=float32, 8=float64, 12=bool, 13=bfloat16
pub fn torch_dtype_int_to_luminal(dtype: u32) -> DType {
    match dtype {
        6 => DType::F16,
        7 => DType::F32,
        8 => DType::F32, // float64 → F32 (no F64 in luminal)
        13 => DType::Bf16,
        12 => DType::Bool,
        1..=5 => DType::Int, // uint8, int8, int16, int32, int64
        _ => DType::F32,
    }
}

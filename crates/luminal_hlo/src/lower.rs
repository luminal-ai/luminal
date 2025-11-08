use crate::ast::{Attr, Operation};
use crate::parser::{parse_output_shape_from_op, parse_tensor_shape};

use anyhow::{bail, ensure, Result};
use std::collections::{BTreeSet, HashMap, HashSet};

use luminal::prelude::*;
use luminal_nn::Conv2D;

type LowerFn = fn(&Operation, &mut Graph, &mut HashMap<String, GraphTensor>) -> Result<()>;

pub fn lower_op(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    match lookup(op.name.as_str()) {
        Some(f) => f(op, g, env),
        None => bail!("unsupported op {}", op.name),
    }
}

fn lookup(op: &str) -> Option<LowerFn> {
    let f = match op {
        // Unary
        "stablehlo.abs" => lower_unary_abs,
        "stablehlo.negate" => lower_unary_negate,
        "stablehlo.sqrt" => lower_unary_sqrt,
        "stablehlo.rsqrt" => lower_unary_rsqrt,
        "stablehlo.log" => lower_unary_log,
        "stablehlo.exponential" => lower_unary_exp,
        "stablehlo.convert" => lower_unary_convert,
        "stablehlo.transpose" => lower_transpose,
        "stablehlo.slice" => lower_slice,
        "stablehlo.not" => lower_unary_not,
        // Binary
        "stablehlo.add" => lower_bin_add,
        "stablehlo.subtract" => lower_bin_sub,
        "stablehlo.multiply" => lower_bin_mul,
        "stablehlo.divide" => lower_bin_div,
        "stablehlo.remainder" => lower_bin_rem,
        "stablehlo.maximum" => lower_bin_max,
        "stablehlo.minimum" => lower_bin_min,
        "stablehlo.compare" => lower_bin_compare,
        "stablehlo.dot_general" => lower_bin_dot_general,
        // Ternary
        "stablehlo.select" => lower_select,
        // Movement
        "stablehlo.reshape" => lower_reshape,
        "stablehlo.broadcast_in_dim" => lower_broadcast_in_dim,
        "stablehlo.concatenate" => lower_concatenate,
        // Constant
        "stablehlo.constant" => lower_constant,
        // Reduce
        "stablehlo.reduce" => lower_reduce,
        "stablehlo.reduce_window" => lower_reduce_window,
        // Convolution
        "stablehlo.convolution" => lower_convolution,
        // Other
        "stablehlo.iota" => lower_iota,
        // Return
        "return" => lower_return,
        _ => return None,
    };
    Some(f)
}

fn binary_with_numpy_broadcast<F>(mut a: GraphTensor, mut b: GraphTensor, f: F) -> GraphTensor
where
    F: Fn(GraphTensor, GraphTensor) -> GraphTensor,
{
    if a.shape.dims().is_empty() && !b.shape.dims().is_empty() {
        for &dim in b.shape.dims().iter() {
            a = a.expand_dim(0, dim);
        }
    } else if b.shape.dims().is_empty() && !a.shape.dims().is_empty() {
        for &dim in a.shape.dims().iter() {
            b = b.expand_dim(0, dim);
        }
    }
    f(a, b)
}

// Unary
fn lower_unary_abs(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.abs());
    Ok(())
}
fn lower_unary_negate(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), -x);
    Ok(())
}
fn lower_unary_sqrt(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.sqrt());
    Ok(())
}
fn lower_unary_log(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.log());
    Ok(())
}
fn lower_unary_exp(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), x.exp());
    Ok(())
}
fn lower_unary_rsqrt(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    env.insert(op.result_name.clone(), 1.0 / x.sqrt());
    Ok(())
}
fn lower_unary_convert(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    // NOTE: Everything is in f32, so this is a no-op
    env.insert(op.result_name.clone(), env[&op.operands[0]]);
    Ok(())
}

// Binary
fn lower_bin_add(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l + r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_sub(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l - r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_mul(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l * r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_div(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l / r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_rem(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l % r);
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_max(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.maximum(r));
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_min(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];
    let y = binary_with_numpy_broadcast(a, b, |l, r| l.minimum(r));
    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_bin_compare(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let lhs = env[&op.operands[0]];
    let mut rhs = env[&op.operands[1]];
    if lhs.shape.dims().len() != rhs.shape.dims().len() {
        rhs = rhs.expand(lhs.shape);
    }
    let comparison_direction = op.attributes.get("comparison_direction");
    match comparison_direction {
        Some(Attr::Id(s)) => {
            let y = match s.as_str() {
                "NE" => lhs.ne(rhs),
                "GT" => lhs.gt(rhs),
                "GE" => lhs.ge(rhs),
                "LT" => lhs.lt(rhs),
                "LE" => lhs.le(rhs),
                "EQ" => lhs.eq(rhs),
                _ => bail!("compare: invalid comparison_direction"),
            };
            env.insert(op.result_name.clone(), y);
        }
        _ => bail!("compare: missing compare_type"),
    };

    Ok(())
}

pub fn lower_bin_dot_general(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let a = env[&op.operands[0]];
    let b = env[&op.operands[1]];

    let lhs_contracting_dims = match op.attributes.get("lhs_contracting_dims") {
        Some(Attr::IntVec(v)) => v.clone(),
        _ => bail!("dot_general: missing lhs_contracting_dims"),
    };
    let rhs_contracting_dims = match op.attributes.get("rhs_contracting_dims") {
        Some(Attr::IntVec(v)) => v.clone(),
        _ => bail!("dot_general: missing rhs_contracting_dims"),
    };
    let lhs_batching_dims: Vec<usize> = match op.attributes.get("lhs_batching_dims") {
        Some(Attr::IntVec(v)) => v.clone(),
        _ => vec![],
    };
    let rhs_batching_dims: Vec<usize> = match op.attributes.get("rhs_batching_dims") {
        Some(Attr::IntVec(v)) => v.clone(),
        _ => vec![],
    };

    let a_shape: Vec<usize> = a
        .shape
        .dims()
        .iter()
        .map(|x| x.to_usize().unwrap())
        .collect();
    let b_shape: Vec<usize> = b
        .shape
        .dims()
        .iter()
        .map(|x| x.to_usize().unwrap())
        .collect();

    let a_rank = a_shape.len();
    let b_rank = b_shape.len();

    let is_unique = |v: &[usize]| {
        let mut h = HashSet::with_capacity(v.len());
        v.iter().all(|x| h.insert(*x))
    };

    let in_range = |all_len: usize, v: &[usize]| v.iter().all(|&d| d < all_len);

    if lhs_batching_dims.len() != rhs_batching_dims.len() {
        bail!(
            "dot_general: (C1) batch dims length mismatch: lhs={} rhs={}",
            lhs_batching_dims.len(),
            rhs_batching_dims.len()
        );
    }
    if lhs_contracting_dims.len() != rhs_contracting_dims.len() {
        bail!(
            "dot_general: (C2) contracting dims length mismatch: lhs={} rhs={}",
            lhs_contracting_dims.len(),
            rhs_contracting_dims.len()
        );
    }

    {
        let mut lhs_all = lhs_batching_dims.clone();
        lhs_all.extend(&lhs_contracting_dims);
        if !is_unique(&lhs_all) {
            bail!("dot_general: (C3) lhs batching+contracting must be unique");
        }

        let mut rhs_all = rhs_batching_dims.clone();
        rhs_all.extend(&rhs_contracting_dims);
        if !is_unique(&rhs_all) {
            bail!("dot_general: (C4) rhs batching+contracting must be unique");
        }
    }

    if !in_range(a_rank, &lhs_batching_dims) {
        bail!("dot_general: (C5) lhs batching out of range");
    }
    if !in_range(a_rank, &lhs_contracting_dims) {
        bail!("dot_general: (C6) lhs contracting out of range");
    }
    if !in_range(b_rank, &rhs_batching_dims) {
        bail!("dot_general: (C7) rhs batching out of range");
    }
    if !in_range(b_rank, &rhs_contracting_dims) {
        bail!("dot_general: (C8) rhs contracting out of range");
    }

    for (ld, rd) in lhs_batching_dims.iter().zip(rhs_batching_dims.iter()) {
        if a_shape[*ld] != b_shape[*rd] {
            bail!("dot_general: (C9) batch dim size mismatch: lhs axis {} (size {}) vs rhs axis {} (size {})",
                  ld, a_shape[*ld], rd, b_shape[*rd]);
        }
    }

    for (ld, rd) in lhs_contracting_dims.iter().zip(rhs_contracting_dims.iter()) {
        if a_shape[*ld] != b_shape[*rd] {
            bail!("dot_general: (C10) contracting size mismatch: lhs axis {} (size {}) vs rhs axis {} (size {})",
                  ld, a_shape[*ld], rd, b_shape[*rd]);
        }
    }

    let contains = |set: &[usize], x: usize| set.iter().any(|&d| d == x);

    let lhs_free: Vec<usize> = (0..a_rank)
        .filter(|d| !contains(&lhs_batching_dims, *d) && !contains(&lhs_contracting_dims, *d))
        .collect();
    let rhs_free: Vec<usize> = (0..b_rank)
        .filter(|d| !contains(&rhs_batching_dims, *d) && !contains(&rhs_contracting_dims, *d))
        .collect();

    let lhs_perm: Vec<usize> = lhs_batching_dims
        .iter()
        .chain(lhs_free.iter())
        .chain(lhs_contracting_dims.iter())
        .copied()
        .collect();

    let rhs_perm: Vec<usize> = rhs_batching_dims
        .iter()
        .chain(rhs_contracting_dims.iter())
        .chain(rhs_free.iter())
        .copied()
        .collect();

    let a_t = a.permute(lhs_perm);
    let b_t = b.permute(rhs_perm);

    let a_t_shape: Vec<usize> = a_t
        .shape
        .dims()
        .iter()
        .map(|x| x.to_usize().unwrap())
        .collect();
    let b_t_shape: Vec<usize> = b_t
        .shape
        .dims()
        .iter()
        .map(|x| x.to_usize().unwrap())
        .collect();

    let b_len = lhs_batching_dims.len();
    let k_len = lhs_contracting_dims.len();

    let a_b = &a_t_shape[0..b_len];
    let a_l = &a_t_shape[b_len..a_t_shape.len() - k_len];
    let a_k = &a_t_shape[a_t_shape.len() - k_len..];

    let b_b = &b_t_shape[0..b_len];
    let b_k = &b_t_shape[b_len..b_len + k_len];
    let b_r = &b_t_shape[b_len + k_len..];

    for i in 0..b_len {
        if a_b[i] != b_b[i] {
            bail!(
                "dot_general: batch size mismatch after permute at {}: lhs={} rhs={}",
                i,
                a_b[i],
                b_b[i]
            );
        }
    }

    let product = |xs: &[usize]| xs.iter().copied().fold(1usize, |p, x| p.saturating_mul(x));

    if product(a_k) != product(b_k) {
        bail!(
            "dot_general: contracting product mismatch after permute: lhs={} rhs={}",
            product(a_k),
            product(b_k)
        );
    }

    if b_len == 0 {
        let m = product(a_l);
        let n = product(b_r);
        let k = product(a_k);

        let a_2d = a_t.reshape(&[m, k]);
        let b_2d = b_t.reshape(&[k, n]);
        let c_2d = a_2d.matmul(b_2d);

        // Output shape is [L..., R...] in that order
        let mut out_shape = Vec::with_capacity(a_l.len() + b_r.len());
        out_shape.extend_from_slice(a_l);
        out_shape.extend_from_slice(b_r);

        let out = c_2d.reshape(out_shape);
        env.insert(op.result_name.clone(), out);
        return Ok(());
    }

    let mut target_shape = Vec::with_capacity(a_b.len() + a_l.len() + a_k.len() + b_r.len());
    target_shape.extend_from_slice(a_b);
    target_shape.extend_from_slice(a_l);
    target_shape.extend_from_slice(a_k);
    target_shape.extend_from_slice(b_r);

    let a_broadcast_dims: Vec<usize> = (0..(b_len + a_l.len() + a_k.len())).collect();

    let mut b_broadcast_dims = Vec::with_capacity(b_len + b_k.len() + b_r.len());
    b_broadcast_dims.extend(0..b_len);

    let k_start = b_len + a_l.len();
    b_broadcast_dims.extend(k_start..k_start + a_k.len());

    let r_start = k_start + a_k.len();
    b_broadcast_dims.extend(r_start..r_start + b_r.len());

    let a_bc = broadcast_in_dim_like(a_t, &target_shape, &a_broadcast_dims)?;
    let b_bc = broadcast_in_dim_like(b_t, &target_shape, &b_broadcast_dims)?;

    let prod_el = a_bc * b_bc;
    let k_axes: Vec<usize> = {
        let start = b_len + a_l.len();
        (start..start + a_k.len()).collect()
    };

    let out_blr = prod_el.sum(k_axes);

    env.insert(op.result_name.clone(), out_blr);
    Ok(())
}

// Ternary
fn lower_select(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let pred = env[&op.operands[0]];
    let on_true = env[&op.operands[1]];
    let on_false = env[&op.operands[2]];

    let out = on_false + pred * (on_true - on_false);
    env.insert(op.result_name.clone(), out);
    Ok(())
}

// Movement
fn lower_reshape(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let shape = parse_output_shape_from_op(&op.result_type_src);
    env.insert(op.result_name.clone(), x.reshape(shape));
    Ok(())
}
fn lower_transpose(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let dims: Vec<usize> = match op.attributes.get("dims") {
        Some(Attr::IntVec(v)) => v.iter().map(|&u| u as usize).collect(),
        _ => bail!("broadcast_in_dim missing 'dims' attribute"),
    };
    env.insert(op.result_name.clone(), x.permute(dims));
    Ok(())
}

fn broadcast_axis(x: &mut GraphTensor, axis: usize, new_len: impl Into<Expression>) {
    let p = x.shape.indexes[axis];
    x.shape.dims[p] = new_len.into();
    x.shape.fake[p] = true;
}

pub fn broadcast_in_dim_like(
    mut x: GraphTensor,
    out_shape: &[usize],
    dims: &[usize],
) -> Result<GraphTensor> {
    let r_out = out_shape.len();
    let r_in = x.shape.dims().len();

    if r_in == 0 {
        let ok_dims = dims.is_empty() || (dims.len() == r_out && dims.iter().copied().eq(0..r_out));
        ensure!(
            ok_dims,
            "broadcast_in_dim: scalar expects dims == [] or 0..r_out-1; got {:?}",
            dims
        );

        // Repeatedly insert axes and broadcast each to out_shape
        for ax in 0..r_out {
            x = x.expand_dim(ax, 1);
            broadcast_axis(&mut x, ax, out_shape[ax]);
        }
        return Ok(x);
    }

    ensure!(
        dims.len() == r_in,
        "broadcast_in_dim_like: dims len {} != input rank {}",
        dims.len(),
        r_in
    );
    ensure!(
        dims.windows(2).all(|w| w[0] < w[1]),
        "broadcast_in_dim_like: dims must be strictly increasing"
    );
    ensure!(
        dims.iter().all(|&d| d < r_out),
        "broadcast_in_dim_like: dims entries must be < out rank"
    );

    // Insert the missing axes (those not mentioned in dims) as size-1
    let dims_set: BTreeSet<_> = dims.iter().copied().collect();
    for out_ax in 0..r_out {
        if !dims_set.contains(&out_ax) {
            x = x.expand_dim(out_ax, 1);
        }
    }

    for out_ax in 0..r_out {
        let want = out_shape[out_ax];
        let cur = x.shape.dims()[out_ax];
        if cur == want {
            continue;
        }
        ensure!(
            cur == 1,
            "broadcast_in_dim_like: axis {} has length {}, cannot broadcast to {}",
            out_ax,
            cur,
            want
        );
        broadcast_axis(&mut x, out_ax, want);
    }

    Ok(x)
}

fn lower_broadcast_in_dim(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];

    let dims: Vec<usize> = match op.attributes.get("dims") {
        Some(Attr::IntVec(v)) => v.iter().map(|&u| u as usize).collect(),
        _ => bail!("broadcast_in_dim missing 'dims' attribute"),
    };

    let out_shape: Vec<usize> = parse_output_shape_from_op(&op.result_type_src);
    let y = broadcast_in_dim_like(x, &out_shape, &dims)?;

    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn lower_concatenate(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let mut tensors = Vec::new();
    for i in 0..op.operands.len() {
        let a = env[&op.operands[i]];
        tensors.push(a);
    }
    let dim = match op.attributes.get("dim") {
        Some(Attr::Int(i)) => *i as usize,
        _ => 0usize,
    };
    let mut y = tensors[0];
    for i in 1..tensors.len() {
        y = y.concat_along(tensors[i], dim);
    }
    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn lower_slice(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];

    let start = match op.attributes.get("start_indices") {
        Some(Attr::IntVec(v)) => v,
        _ => bail!("slice missing 'start_indices' attribute"),
    };

    let end = match op.attributes.get("end_indices") {
        Some(Attr::IntVec(v)) => v,
        _ => bail!("slice missing 'end_indices' attribute"),
    };

    let mut y = x;
    match start.len() {
        1 => {
            y = y.slice(start[0]..end[0]);
        }
        2 => {
            y = y.slice((start[0]..end[0], start[1]..end[1]));
        }
        3 => {
            y = y.slice((start[0]..end[0], start[1]..end[1], start[2]..end[2]));
        }
        4 => {
            y = y.slice((
                start[0]..end[0],
                start[1]..end[1],
                start[2]..end[2],
                start[3]..end[3],
            ));
        }
        _ => bail!("slice: unsupported number of dimensions"),
    };

    env.insert(op.result_name.clone(), y);
    Ok(())
}
fn lower_unary_not(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let zero_tensor = g.constant(0.0).expand(x.shape).retrieve();
    let y = x.eq(zero_tensor);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

// Constant
fn lower_constant(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let v = match op.attributes.get("dense") {
        Some(Attr::Float(f)) => *f as f32,
        Some(Attr::Int(i)) => *i as f32,
        _ => bail!("constant missing 'dense' literal"),
    };
    let t = g.constant(v).retrieve();
    env.insert(op.result_name.clone(), t);
    Ok(())
}

// Reduce
fn lower_reduce(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let init = env[&op.operands[1]];

    let mut dims = match (op.attributes.get("dimensions"), op.attributes.get("dims")) {
        (Some(Attr::IntVec(v)), _) | (_, Some(Attr::IntVec(v))) => v.clone(),
        _ => vec![],
    };

    // NOTE: Luminal sum() op does not support unsorted dimensions
    dims.sort();

    match op.attributes.get("apply") {
        Some(Attr::Id(s)) if s == "stablehlo.add" => {
            let y = x.sum(dims);
            env.insert(op.result_name.clone(), y);
            Ok(())
        }
        Some(Attr::Id(s)) if s == "stablehlo.or" => {
            let zero_tensor = g.constant(0.0).expand(x.shape).retrieve();
            let x_bool = x.ne(zero_tensor);
            let reduced = x_bool.max(dims);
            let y = reduced.maximum(init.expand(reduced.shape));

            env.insert(op.result_name.clone(), y);
            Ok(())
        }
        Some(Attr::Id(s)) if s == "stablehlo.maximum" => {
            let reduced = x.max(dims);
            let init_b = init.expand(reduced.shape);
            let y = reduced.maximum(init_b);

            env.insert(op.result_name.clone(), y);
            Ok(())
        }
        other => bail!("unsupported reduce.apply: {:?}", other),
    }
}

// Convolution
fn lower_convolution(
    op: &Operation,
    g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    let x = env[&op.operands[0]];
    let w = env[&op.operands[1]];

    // dim_numbers
    let (input_t, kernel_t, output_t) = match op.attributes.get("dim_numbers") {
        Some(Attr::DimNumbers {
            input,
            kernel,
            output,
        }) => (input.clone(), kernel.clone(), output.clone()),
        _ => bail!("convolution: missing dim_numbers"),
    };

    // Only supports canonical NCHW x OIHW -> NCHW
    let is_nchw = input_t == vec!["b", "f", "0", "1"]
        && kernel_t == vec!["o", "i", "0", "1"]
        && output_t == vec!["b", "f", "0", "1"];
    if !is_nchw {
        bail!("[lower_convolution] Only NCHW dim_numbers supported");
    }

    // window: pad / stride / dilations
    let pads: Vec<(usize, usize)> = match op.attributes.get("window_pad") {
        Some(Attr::PadPairs(p)) => p.clone(),
        _ => vec![(0, 0), (0, 0)],
    };
    let stride = match op.attributes.get("stride") {
        Some(Attr::IntVec(v)) if v.len() >= 2 => (v[0], v[1]),
        _ => (1, 1),
    };
    let base_dilations = match op.attributes.get("base_dilations") {
        Some(Attr::IntVec(v)) if v.len() >= 2 => (v[0], v[1]),
        _ => (1, 1),
    };
    let window_dilations = match op.attributes.get("window_dilations") {
        Some(Attr::IntVec(v)) if v.len() >= 2 => (v[0], v[1]),
        _ => (1, 1),
    };

    // groups
    let bg = match op.attributes.get("batch_group_count") {
        Some(Attr::Int(i)) => *i as usize,
        _ => 1,
    };
    let fg = match op.attributes.get("feature_group_count") {
        Some(Attr::Int(i)) => *i as usize,
        _ => 1,
    };
    if bg != 1 {
        bail!("[lower_convolution] batch_group_count != 1 NYI");
    }
    if fg != 1 {
        bail!("[lower_convolution] feature_group_count != 1 NYI");
    }

    // padding and dilation
    if base_dilations != (1, 1) {
        bail!("[lower_convolution] base dilation NYI");
    }

    let x_dims = x
        .shape
        .dims()
        .iter()
        .map(|d| d.to_usize().unwrap())
        .collect::<Vec<_>>();
    let w_dims = w
        .shape
        .dims()
        .iter()
        .map(|d| d.to_usize().unwrap())
        .collect::<Vec<_>>();

    let ch_in = x_dims[1];
    let ch_out = w_dims[0];
    let k_h = w_dims[2];
    let k_w = w_dims[3];

    let (pt, pb) = pads.get(0).copied().unwrap_or((0, 0));
    let (pl, pr) = pads.get(1).copied().unwrap_or((0, 0));
    let x_padded = if pt | pb | pl | pr != 0 {
        pad_nchw(g, &x, pt, pb, pl, pr)?
    } else {
        x.clone()
    };

    let conv = Conv2D::new(
        ch_in,
        ch_out,
        (k_h, k_w),
        stride,
        window_dilations,
        false,
        g,
    );

    // TODO: Hack to invalidate the weight tensor in the
    // graph before aliasing it to the Conv2D weight.
    w.set([0.0]);
    env.insert(op.operands[1].clone(), conv.weight.clone());

    let y = conv.forward(x_padded);
    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn pad_nchw(
    g: &mut Graph,
    x: &GraphTensor,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
) -> Result<GraphTensor> {
    // x shape: [N, C, H, W]
    let dims = x
        .shape
        .dims()
        .iter()
        .map(|d| d.to_usize().unwrap())
        .collect::<Vec<_>>();
    ensure!(dims.len() == 4, "pad_nchw expects NCHW rank-4 tensor");
    let (n, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);

    let h_padded = h + pad_top + pad_bottom;
    let w_padded = w + pad_left + pad_right;

    let mut zeros = |shape: (usize, usize, usize, usize)| -> GraphTensor {
        let t = g.named_tensor("Zeros", (shape.0, shape.1, shape.2, shape.3));
        t.set([0.0]);
        t
    };

    let left = if pad_left > 0 {
        zeros((n, c, h, pad_left))
    } else {
        x.clone()
    };
    let right = if pad_right > 0 {
        zeros((n, c, h, pad_right))
    } else {
        x.clone()
    };
    let x_hpad = if pad_left > 0 || pad_right > 0 {
        left.concat_along(x.clone(), 3).concat_along(right, 3)
    } else {
        x.clone()
    };

    let top = if pad_top > 0 {
        zeros((n, c, pad_top, w_padded))
    } else {
        x_hpad.clone()
    };
    let bottom = if pad_bottom > 0 {
        zeros((n, c, pad_bottom, w_padded))
    } else {
        x_hpad.clone()
    };

    let y = if pad_top > 0 || pad_bottom > 0 {
        top.concat_along(x_hpad.clone(), 2).concat_along(bottom, 2)
    } else {
        x_hpad
    };

    ensure!(
        y.shape
            .dims()
            .iter()
            .map(|d| d.to_usize().unwrap())
            .collect::<Vec<_>>()
            == vec![n, c, h_padded, w_padded],
        "pad_nchw produced unexpected shape"
    );
    Ok(y)
}

fn lower_reduce_window(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    if op.operands.len() < 1 {
        bail!("reduce_window: expected at least 1 input");
    }
    let mut y = env[&op.operands[0]];

    if op.operands.len() != 2 {
        bail!("reduce_window: currently supports exactly one input + one init_value");
    }

    let apply = match op.attributes.get("apply") {
        Some(Attr::Id(s)) => s.as_str(),
        _ => bail!("reduce_window: missing/unknown reducer body; expected maximum"),
    };
    if apply != "stablehlo.maximum" {
        bail!(
            "reduce_window: only `stablehlo.maximum` is supported right now, got {}",
            apply
        );
    }

    let rank = y.shape.dims().len();
    if rank != 4 {
        bail!(
            "reduce_window: only rank-4 NCHW tensors are supported (got rank={})",
            rank
        );
    }

    let win_dims = get_usize_vec(&op, "window_dimensions")?;
    if win_dims.len() != 4 {
        bail!("reduce_window: window_dimensions must be length 4 for NCHW");
    }
    let win_strides = get_usize_vec_default(&op, "window_strides", 4, 1)?;
    let base_dils = get_usize_vec_default(&op, "base_dilations", 4, 1)?;
    let win_dils = get_usize_vec_default(&op, "window_dilations", 4, 1)?;
    let pads = get_pad_pairs_default(&op, 4)?;

    if win_dims[0] != 1 || win_dims[1] != 1 {
        bail!("reduce_window: only spatial pooling supported; expected window_dimensions[0..2) == [1,1]");
    }
    if base_dils.iter().any(|&v| v != 1) {
        bail!("reduce_window: base_dilations not supported for pooling");
    }

    let kh = win_dims[2];
    let kw = win_dims[3];
    let sh = win_strides[2];
    let sw = win_strides[3];
    let dh = win_dils[2];
    let dw = win_dils[3];
    let (ph_lo, ph_hi) = pads[2];
    let (pw_lo, pw_hi) = pads[3];

    if ph_lo != 0 || ph_hi != 0 || pw_lo != 0 || pw_hi != 0 {
        y = y.pad(&[(0, 0), (0, 0), (ph_lo, ph_hi), (pw_lo, pw_hi)]);
    }

    y = y.pool_last_dim(kw as usize, sw as usize, dw as usize);
    let last_axis = y.shape.dims().len() - 1;
    y = y.max(last_axis);

    y = y.permute(&[0, 1, 3, 2]); // N, C, W, H
    y = y.pool_last_dim(kh as usize, sh as usize, dh as usize);
    let last_axis = y.shape.dims().len() - 1;
    y = y.max(last_axis);
    y = y.permute(&[0, 1, 3, 2]);

    env.insert(op.result_name.clone(), y);
    Ok(())
}

fn get_usize_vec(op: &Operation, key: &str) -> Result<Vec<usize>> {
    match op.attributes.get(key) {
        Some(Attr::IntVec(v)) => Ok(v.clone()),
        other => bail!("reduce_window: missing `{}` (got {:?})", key, other),
    }
}

fn get_usize_vec_default(op: &Operation, key: &str, n: usize, fill: usize) -> Result<Vec<usize>> {
    let v = match op.attributes.get(key) {
        Some(Attr::IntVec(v)) => v.clone(),
        None => vec![fill; n],
        other => bail!("reduce_window: bad `{}` (got {:?})", key, other),
    };
    if v.len() != n {
        bail!("reduce_window: `{}` length must be {}", key, n);
    }
    Ok(v)
}

fn get_pad_pairs_default(op: &Operation, n: usize) -> Result<Vec<(usize, usize)>> {
    Ok(match op.attributes.get("padding") {
        Some(Attr::PadPairs(p)) => {
            if p.len() != n {
                bail!("reduce_window: padding must have {} pairs", n);
            }
            p.clone()
        }
        None => vec![(0, 0); n],
        other => bail!("reduce_window: bad `padding` (got {:?})", other),
    })
}

fn lower_iota(op: &Operation, g: &mut Graph, env: &mut HashMap<String, GraphTensor>) -> Result<()> {
    let iota_dim = match op.attributes.get("dim") {
        Some(Attr::Int(i)) => *i as usize,
        _ => 0usize,
    };
    let shape = parse_tensor_shape(&op.result_type_src);
    let iota_tensor = g.arange(shape[iota_dim]) + 1.0;
    let iota_tensor = iota_tensor.expand(shape);
    env.insert(op.result_name.clone(), iota_tensor);
    Ok(())
}

// return
fn lower_return(
    op: &Operation,
    _g: &mut Graph,
    env: &mut HashMap<String, GraphTensor>,
) -> Result<()> {
    if let Some(src) = op.operands.get(0) {
        let y = env[src].retrieve();
        env.insert(src.clone(), y);
    }
    Ok(())
}

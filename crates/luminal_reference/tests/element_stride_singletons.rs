//! LUM-1147: `max_pool1d(x.unsqueeze(0), 2, return_indices=True)` on a 2x6
//! input, built the way the PyTorch translator lowers it: a 2-D pool over
//! [1, 2, 1, 6] (pad, unfold, slice, squeeze, stable argsort, gather), whose
//! rank-8 views carry up to five singleton axes. Composing every
//! singleton-coefficient spelling of a parent grew the saturated e-graph to
//! 834k nodes and the reference post-pass to about 30 minutes.

use luminal::prelude::*;
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;
use std::time::{Duration, Instant};

/// The translator's `pool_windows`: pad, unfold the spatial axes, slice the
/// window axes to the output shape, then drop the prefix window axes.
fn pool_windows(
    x: GraphTensor,
    kernel: [usize; 2],
    stride: [usize; 2],
    output_shape: &[usize],
    fill: GraphTensor,
) -> GraphTensor {
    let rank = x.rank();
    let first_spatial = rank - 2;
    let padded = x.pad_with(vec![(IntExpr::from(0), IntExpr::from(0)); rank], fill);
    let mut full_kernel = vec![IntExpr::from(1); rank];
    let mut full_stride = vec![IntExpr::from(1); rank];
    for s in 0..2 {
        full_kernel[first_spatial + s] = IntExpr::from(kernel[s]);
        full_stride[first_spatial + s] = IntExpr::from(stride[s]);
    }
    let mut windows = padded.unfold(full_kernel, full_stride, vec![IntExpr::from(1); rank]);
    for (axis, &extent) in output_shape.iter().enumerate().skip(first_spatial) {
        windows = windows.slice_along(IntExpr::from(0)..IntExpr::from(extent), axis);
    }
    for axis in (0..first_spatial).rev() {
        windows = windows.squeeze(rank + axis);
    }
    windows
}

#[test]
fn max_pool_with_indices_over_singleton_axes_searches_in_seconds() {
    let mut cx = Graph::new();
    let input = cx.tensor(vec![2usize, 6], DType::F32);
    let x = input.expand_dim(0, 1usize).expand_dim(2, 1usize);
    let out_shape = [1usize, 2, 1, 3];
    let lowest = cx.constant_f32(f32::NEG_INFINITY);
    let mut candidates = pool_windows(x, [1, 2], [1, 2], &out_shape, lowest);
    let positions = cx
        .iota(
            vec![
                IntExpr::from(1usize),
                IntExpr::from(2usize),
                IntExpr::from(1usize),
                IntExpr::from(6usize),
            ],
            |c| c[2] * 6 + c[3],
        )
        .cast(DType::I64);
    let zero = cx.constant_i64(0);
    let mut indices = pool_windows(positions, [1, 2], [1, 2], &out_shape, zero);
    // The translator's `select_pool_max`.
    let output_rank = 4;
    while candidates.rank() > output_rank + 1 {
        let last = candidates.rank();
        candidates = candidates.merge_dims(last - 2, last - 1);
        indices = indices.merge_dims(last - 2, last - 1);
    }
    let selected = candidates
        .stable_argsort(output_rank, true)
        .slice_along(0..1, output_rank)
        .squeeze(output_rank);
    let dims = selected.dims();
    let mut coordinates: Vec<GraphTensor> = (0..output_rank)
        .map(|axis| {
            let mut position = cx.arange(dims[axis]);
            for (other, size) in dims.iter().copied().enumerate() {
                if other != axis {
                    position = position.expand_dim(other, size);
                }
            }
            position
        })
        .collect();
    coordinates.push(selected);
    let values = candidates.gather(&coordinates).squeeze(2);
    let picked = indices.gather(&coordinates).squeeze(2);

    let data = vec![
        3.5, -0.0, 2.25, -7.0, 0.5, 2.25, 1.0, -2.5, 8.0, 0.25, -1.0, 4.0,
    ];
    let mut runtime = ReferenceRuntime::load(&cx).expect("load");
    let mut inputs: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    inputs.insert(input.id, TypedBuffer::F32(data.clone()));
    let start = Instant::now();
    let outcome = runtime
        .search(
            &SymbolBounds::default(),
            &FxHashMap::default(),
            &inputs,
            &harness_search_options(),
        )
        .expect("search");
    let elapsed = start.elapsed();
    // Seconds in a debug build; the regression took about 30 minutes.
    assert!(
        elapsed < Duration::from_secs(60),
        "search took {elapsed:?}: {}",
        outcome.timings.summary()
    );

    runtime.set_data(input.id, TypedBuffer::F32(data));
    runtime.execute().expect("execute");
    assert_eq!(
        runtime.get_f32(values.id).expect("values"),
        &vec![3.5, 2.25, 2.25, 1.0, 8.0, 4.0]
    );
    assert_eq!(
        runtime.get_i64(picked.id).expect("indices"),
        &vec![0, 2, 5, 0, 2, 5]
    );
}

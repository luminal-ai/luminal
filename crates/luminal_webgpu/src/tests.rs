use crate::{
    kernel::{lower_expression_for_webgpu, lower_scalar_expression_for_webgpu},
    runtime::WebGpuRuntime,
};
use luminal::prelude::*;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "Length mismatch: got {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        let rel_err = diff / e.abs().max(1.0);
        assert!(
            rel_err < tolerance,
            "Mismatch at index {i}: got {a}, expected {e}, rel_err={rel_err}"
        );
    }
}

#[test]
fn dynamic_const_codegen_uses_params_buffer() {
    let expr = (Expression::from('a') * 2 + Expression::from('z')).simplify();
    let code = lower_expression_for_webgpu(&expr, "idx");

    assert!(
        !code.contains("const_"),
        "dynamic symbols should be lowered via params buffer, got: {code}"
    );
    assert!(
        code.contains("params.dims["),
        "expected generated kernel expression to reference params buffer, got: {code}"
    );
}

#[test]
fn scalar_codegen_treats_z_as_stride_unit() {
    let expr = (Expression::from('a') * Expression::from('z')).simplify();
    let code = lower_scalar_expression_for_webgpu(&expr);

    assert!(
        code.contains("params.dims["),
        "dynamic symbols should still lower through params, got: {code}"
    );
    assert!(
        !code.contains("idx"),
        "scalar stride expressions should substitute z with one, got: {code}"
    );
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn webgpu_simple_add_runs() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a + b).output();

    cx.build_search_space::<WebGpuRuntime>(CompileOptions::default());
    let mut rt = WebGpuRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(
        rt,
        CompileOptions {
            limit: 5,
            ..CompileOptions::default()
        },
    );
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(output), &[6.0, 8.0, 10.0, 12.0], 0.001);
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn webgpu_generic_matmul_runs() {
    let mut cx = Graph::default();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((3, 2));
    let output = a.matmul(b).output();

    cx.build_search_space::<WebGpuRuntime>(CompileOptions::default());
    let mut rt = WebGpuRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    rt.set_data(b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    rt = cx.search(
        rt,
        CompileOptions {
            limit: 10,
            ..CompileOptions::default()
        },
    );
    rt.execute(&cx.dyn_map);

    assert_close(&rt.get_f32(output), &[58.0, 64.0, 139.0, 154.0], 0.001);
    assert!(
        rt.contains_matmul(),
        "expected WebGPU search to select a matmul kernel, got {:?}",
        rt.debug_kernel_ops()
    );
}

#[test]
#[ignore = "requires a WebGPU adapter"]
fn webgpu_large_reduce_dispatch_chunks() {
    const ROWS: usize = 128_256;

    let mut cx = Graph::default();
    let input = cx.tensor((ROWS, 2));
    let output = input.sum(1).output();

    cx.build_search_space::<WebGpuRuntime>(CompileOptions::default());
    let mut rt = WebGpuRuntime::initialize(());
    let data = vec![1.0f32; ROWS * 2];
    rt.set_data(input, data);
    rt = cx.search(
        rt,
        CompileOptions {
            limit: 5,
            ..CompileOptions::default()
        },
    );
    rt.execute(&cx.dyn_map);

    let result = rt.get_f32(output);
    assert_eq!(result.len(), ROWS);
    assert_close(&result[..8], &[2.0; 8], 0.001);
    assert_close(&result[ROWS - 8..], &[2.0; 8], 0.001);
}

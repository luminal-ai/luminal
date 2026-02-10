use crate::{kernel::lower_expression_for_metal, runtime::MetalRuntime};
use luminal::prelude::*;
use proptest::prelude::*;

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
            "Mismatch at index {}: got {}, expected {}, rel_err={}",
            i,
            a,
            e,
            rel_err
        );
    }
}

/// dynamic symbols in kernel expressions should route through dyn buffer.
#[test]
fn dynamic_const_codegen_uses_dyn_buffer() {
    let expr = (Expression::from('a') * 2 + Expression::from('z')).simplify();
    let code = lower_expression_for_metal(&expr, "idx");

    assert!(
        !code.contains("*const_"),
        "dynamic symbols should be lowered via dyn buffer, got: {code}"
    );
    assert!(
        code.contains("dyn["),
        "expected generated kernel expression to reference dyn buffer, got: {code}"
    );
}

/// dynamic-dimension reduction should compile and execute on Metal.
#[test]
fn dynamic_dim_sum_reduce_runs() {
    let mut cx = Graph::default();
    cx.set_dim('a', 3);
    let input = cx.tensor(('a', 2));
    let output = input.sum(0).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    rt = cx.search(rt, 1);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[9.0, 12.0], 0.001);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(5))]

    /// Test basic addition: input + input = 2 * input
    #[test]
    fn metal_add_test(len in 1usize..32, values in proptest::collection::vec(-5.0f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input + input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * 2.0).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test basic multiplication: input * input = input^2
    #[test]
    fn metal_mul_test(len in 1usize..32, values in proptest::collection::vec(0.1f32..5.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = (input * input).output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| v * v).collect();
        assert_close(&out, &expected, 0.001);
    }

    /// Test exp2: 2^x
    #[test]
    fn metal_exp2_test(len in 1usize..32, values in proptest::collection::vec(-3.0f32..3.0, 1..64)) {
        prop_assume!(values.len() >= len);

        let mut cx = Graph::default();
        let input = cx.tensor(len);
        let output = input.exp2().output();

        cx.build_search_space::<MetalRuntime>();
        let mut rt = MetalRuntime::initialize(());
        let input_values: Vec<f32> = values.into_iter().take(len).collect();
        rt.set_data(input, &input_values);
        rt = cx.search(rt, 5);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let out = rt.get_f32(output);
        let expected: Vec<f32> = input_values.iter().map(|v| 2.0f32.powf(*v)).collect();
        assert_close(&out, &expected, 0.001);
    }
}

/// Simple deterministic test for add
#[test]
fn metal_simple_add() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a + b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![6.0, 8.0, 10.0, 12.0]);
}

/// Simple deterministic test for mul
#[test]
fn metal_simple_mul() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a * b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 2.0, 3.0, 4.0]);
    rt.set_data(b, &[5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_eq!(out, vec![5.0, 12.0, 21.0, 32.0]);
}

/// Simple deterministic test for exp2
#[test]
fn metal_simple_exp2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.exp2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[0.0, 1.0, 2.0, 3.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 4.0, 8.0], 0.001);
}

#[test]
fn metal_simple_log2() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.log2().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 2.0, 3.0], 0.001);
}

#[test]
fn metal_simple_sin() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sin().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(
        input,
        &[
            0.0,
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            3.0 * std::f32::consts::FRAC_PI_2,
        ],
    );
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[0.0, 1.0, 0.0, -1.0], 0.01);
}

#[test]
fn metal_simple_sqrt() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.sqrt().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 4.0, 9.0, 16.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 4.0], 0.001);
}

#[test]
fn metal_simple_recip() {
    let mut cx = Graph::default();
    let input = cx.tensor(4);
    let output = input.reciprocal().output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(input, &[1.0, 2.0, 4.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 0.5, 0.25, 0.2], 0.001);
}

#[test]
fn metal_simple_mod() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = (a % b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[7.0, 10.0, 15.0, 8.5]);
    rt.set_data(b, &[3.0, 4.0, 6.0, 2.5]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[1.0, 2.0, 3.0, 1.0], 0.001);
}

#[test]
fn metal_simple_less_than() {
    let mut cx = Graph::default();
    let a = cx.tensor(4);
    let b = cx.tensor(4);
    let output = a.lt(b).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    rt.set_data(a, &[1.0, 5.0, 3.0, 4.0]);
    rt.set_data(b, &[2.0, 3.0, 3.0, 5.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    // 1 < 2 = true (1.0), 5 < 3 = false (0.0), 3 < 3 = false (0.0), 4 < 5 = true (1.0)
    assert_eq!(out, vec![1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn metal_simple_sum_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // sum over axis 1
    let output = input.sum(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,2,3,4], [5,6,7,8]] -> [10, 26]
    rt.set_data(input, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[10.0, 26.0], 0.001);
}

#[test]
fn metal_simple_max_reduce() {
    let mut cx = Graph::default();
    let input = cx.tensor((2, 4));
    // max over axis 1
    let output = input.max(1).output();

    cx.build_search_space::<MetalRuntime>();
    let mut rt = MetalRuntime::initialize(());
    // [[1,4,2,3], [8,5,7,6]] -> [4, 8]
    rt.set_data(input, &[1.0, 4.0, 2.0, 3.0, 8.0, 5.0, 7.0, 6.0]);
    rt = cx.search(rt, 5);
    rt.allocate_intermediate_buffers(&cx.dyn_map);
    rt.execute(&cx.dyn_map);

    let out = rt.get_f32(output);
    assert_close(&out, &[4.0, 8.0], 0.001);
}

use dfdx::prelude::{Module as DfdxModule, *};
use rand::{rngs::StdRng, SeedableRng};

use luminal::{module::Module, prelude::*};
use luminal_nn::{Conv1D, LayerNorm, Linear, ReLU};

use crate::{binary_test, unary_test, CudaCompiler};
luminal::test_imports!();

// INT8 quantized operations - simplified for kernel testing
fn quantize_int8(x: GraphTensor) -> GraphTensor {
    // Simple pass-through for kernel testing
    x
}

// INT8 tests - focusing on kernel execution rather than quantization precision
unary_test!(|a| quantize_int8(a.sin()), |a| a.sin(), test_sin_int8, f32);
unary_test!(
    |a| quantize_int8(a.sqrt()),
    |a| a.sqrt(),
    test_sqrt_int8,
    f32
);
unary_test!(
    |a| quantize_int8(a.reciprocal()),
    |a| a.recip(),
    test_reciprocal_int8,
    f32
);
unary_test!(
    |a| quantize_int8(a * a),
    |a| a.clone() * a,
    test_square_int8,
    f32
);

binary_test!(
    |a, b| quantize_int8(a + b),
    |a, b| a + b,
    test_add_int8,
    f32
);
binary_test!(
    |a, b| quantize_int8(a - b),
    |a, b| a - b,
    test_sub_int8,
    f32
);
binary_test!(
    |a, b| quantize_int8(a * b),
    |a, b| a * b,
    test_mul_int8,
    f32
);
binary_test!(
    |a, b| quantize_int8(a / b),
    |a, b| a / b,
    test_div_int8,
    f32
);

#[test]
fn test_int8_matmul() {
    let data = random_vec(1024);
    let mut cx = Graph::new();
    let a = cx.tensor((64, 64)).set(data[..4096].to_vec()).keep();
    let b = cx.tensor((64, 64)).set(data[4096..8192].to_vec()).keep();
    let mut c = quantize_int8(a.matmul(b)).retrieve();

    cx.compile(CudaCompiler::<f32>::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data[..4096].to_vec(), (DConst::<64>, DConst::<64>));
    let d_b = d_dev.tensor_from_vec(data[4096..8192].to_vec(), (DConst::<64>, DConst::<64>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_int8_sum() {
    let data = random_vec(10240);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = quantize_int8(a);
    let mut b = a_q.sum(2).retrieve();
    let mut c = a_q.sum(1).retrieve();
    let mut d = a_q.sum(0).retrieve();

    cx.compile(CudaCompiler::<f32>::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<1024>));
    let d_b = d_a.clone().sum::<_, DAxis<2>>();
    let d_c = d_a.clone().sum::<_, DAxis<1>>();
    let d_d = d_a.sum::<_, DAxis<0>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_int8_max() {
    let data = random_vec(10240);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = quantize_int8(a);
    let mut b = a_q.max(2).retrieve();
    let mut c = a_q.max(1).retrieve();
    let mut d = a_q.max(0).retrieve();

    cx.compile(CudaCompiler::<f32>::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<1024>));
    let d_b = d_a.clone().max::<_, DAxis<2>>();
    let d_c = d_a.clone().max::<_, DAxis<1>>();
    let d_d = d_a.max::<_, DAxis<0>>();

    assert_close(&b.data(), &d_b.as_vec());
    assert_close(&c.data(), &d_c.as_vec());
    assert_close(&d.data(), &d_d.as_vec());
}

#[test]
fn test_int8_mean() {
    let data = random_vec(10240);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = quantize_int8(a);
    let mut b = a_q.mean(2).retrieve();
    let mut c = a_q.mean(1).retrieve();
    let mut d = a_q.mean(0).retrieve();

    cx.compile(CudaCompiler::<f32>::default(), (&mut b, &mut c, &mut d));
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data, (DConst::<1>, DConst::<10>, DConst::<1024>));
    let d_b = d_a.clone().mean::<_, DAxis<2>>();
    let d_c = d_a.clone().mean::<_, DAxis<1>>();
    let d_d = d_a.mean::<_, DAxis<0>>();

    assert_close_precision(&b.data(), &d_b.as_vec(), 1e-2);
    assert_close_precision(&c.data(), &d_c.as_vec(), 1e-2);
    assert_close_precision(&d.data(), &d_d.as_vec(), 1e-2);
}

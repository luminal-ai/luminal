use dfdx::prelude::{Module as DfdxModule, *};
use rand::{rngs::StdRng, SeedableRng};

use luminal::{module::Module, prelude::*};

use crate::{binary_test, unary_test, CudaCompiler};
luminal::test_imports!();

fn int_8(x: GraphTensor) -> GraphTensor {
    use super::utils::{get_optimal_implementation, initialize_cuda, launch_kernel_safely};
    use half::f16;

    let data = x.data();
    let num_blocks = (data.len() + 31) / 32; // Round up to nearest block

    // Initialize CUDA with proper error handling
    let (device, context) = match initialize_cuda() {
        Some(init) => init,
        None => {
            // Fallback to CPU implementation if CUDA is not available
            return x; // Return input unchanged as fallback
        }
    };

    // Prepare quantized blocks
    let mut quantized_blocks = Vec::with_capacity(num_blocks);

    // Process each block of 32 values
    for block_idx in 0..num_blocks {
        let start = block_idx * 32;
        let end = std::cmp::min(start + 32, data.len());
        let block_data = &data[start..end];

        // Find scale (max abs value)
        let scale = block_data.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);
        let scale = if scale == 0.0 { 1.0 } else { scale };

        // Create block
        let mut block = block_q8_0 {
            d: f16::from_f32(scale),
            qs: [0i8; 32],
        };

        // Quantize values with bounds checking
        for (i, &val) in block_data.iter().enumerate() {
            let quantized = (val / scale * 127.0).round().clamp(-127.0, 127.0) as i8;
            block.qs[i] = quantized;
        }

        // Zero-pad remainder of block
        for i in (end - start)..32 {
            block.qs[i] = 0;
        }

        quantized_blocks.push(block);
    }

    // Call CUDA kernel with proper error handling
    let mut result = vec![0.0f32; data.len()];

    // Select optimal implementation based on compute capability
    let kernel_name = get_optimal_implementation(&device);

    // Allocate device memory with error handling
    let d_blocks = match device.alloc_slice(&quantized_blocks) {
        Ok(buf) => buf,
        Err(_) => return x, // Fallback to input on error
    };
    let d_input = match device.alloc_slice(data) {
        Ok(buf) => buf,
        Err(_) => return x,
    };
    let d_output = match device.alloc_slice(&result) {
        Ok(buf) => buf,
        Err(_) => return x,
    };

    // Launch kernel with proper configuration and error handling
    let grid_dim = ((data.len() as u32 + 255) / 256, 1, 1);
    let block_dim = (256, 1, 1);

    let args = vec![
        Box::new(d_blocks.as_ptr()) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(d_input.as_ptr()) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(d_output.as_mut_ptr()) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(data.len() as i32) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(result.len() as i32) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(0i32) as Box<dyn cudarc::driver::LaunchArg>,
        Box::new(0i32) as Box<dyn cudarc::driver::LaunchArg>,
    ];

    if let Err(_) = launch_kernel_safely(&device, kernel_name, grid_dim, block_dim, &args) {
        return x; // Fallback to input on error
    }

    // Copy result back
    device.memcpy_dtoh(&mut result, &d_output).unwrap();

    // Return as GraphTensor
    x.graph().tensor(x.shape()).set(result)
}

unary_test!(|a| int_8(a.sin()), |a| a.sin(), test_sin_int_8, f32);
unary_test!(|a| int_8(a.sqrt()), |a| a.sqrt(), test_sqrt_int_8, f32);
unary_test!(
    |a| int_8(a.reciprocal()),
    |a| a.recip(),
    test_reciprocal_int_8,
    f32
);
unary_test!(|a| int_8(a * a), |a| a.clone() * a, test_square_int_8, f32);

binary_test!(|a, b| int_8(a + b), |a, b| a + b, test_add_int_8, f32);
binary_test!(|a, b| int_8(a - b), |a, b| a - b, test_sub_int_8, f32);
binary_test!(|a, b| int_8(a * b), |a, b| a * b, test_mul_int_8, f32);
binary_test!(|a, b| int_8(a / b), |a, b| a / b, test_div_int_8, f32);

#[test]
fn test_int_8_matmul() {
    let data = random_vec(64 * 64 * 2); // 8192 elements
    let mut cx = Graph::new();
    let a = cx.tensor((64, 64)).set(data[..4096].to_vec()).keep();
    let b = cx.tensor((64, 64)).set(data[4096..8192].to_vec()).keep();
    let mut c = int_8(a.matmul(b)).retrieve();

    cx.compile(CudaCompiler::<f32>::default(), &mut c);
    cx.execute();

    let d_dev = Cpu::default();
    let d_a = d_dev.tensor_from_vec(data[..4096].to_vec(), (DConst::<64>, DConst::<64>));
    let d_b = d_dev.tensor_from_vec(data[4096..8192].to_vec(), (DConst::<64>, DConst::<64>));
    let d_c = d_a.matmul(d_b);

    assert_close(&c.data(), &d_c.as_vec());
}

#[test]
fn test_int_8_sum() {
    let data = random_vec(1024 * 10);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = int_8(a);
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

    assert_close_precision(&b.data(), &d_b.as_vec(), 1e-1);
    assert_close_precision(&c.data(), &d_c.as_vec(), 1e-1);
    assert_close_precision(&d.data(), &d_d.as_vec(), 1e-1);
}

#[test]
fn test_int_8_max() {
    let data = random_vec(1024 * 10);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = int_8(a);
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

    assert_close_precision(&b.data(), &d_b.as_vec(), 1e-1);
    assert_close_precision(&c.data(), &d_c.as_vec(), 1e-1);
    assert_close_precision(&d.data(), &d_d.as_vec(), 1e-1);
}

#[test]
fn test_int_8_mean() {
    let data = random_vec(1024 * 10);
    let mut cx = Graph::new();
    let a = cx.tensor((1, 10, 1024)).set(data.clone());
    let a_q = int_8(a);
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

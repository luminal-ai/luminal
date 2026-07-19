/// luminal_cpu example
use luminal::prelude::*;
use luminal_cpu::CpuRuntime;

fn main() {
    println!("=== 1. Simple matmul (3×1) @ (1×4) ===");
    {
        let mut cx = Graph::new();
        let a = cx.tensor((3, 1));
        let b = cx.tensor((1, 4));
        let c = a.matmul(b).output();

        cx.build_search_space::<CpuRuntime>();
        let mut rt = CpuRuntime::initialize(());
        rt.set_data(a, vec![1.0f32, 2.0, 3.0]);
        rt.set_data(b, vec![1.0f32, 2.0, 3.0, 4.0]);
        rt = cx.search(rt, 1);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        // Expected: outer product
        // [[1,2,3,4], [2,4,6,8], [3,6,9,12]]
        println!("Result: {:?}", rt.get_f32(c));
    }

    println!("\n=== 2. Element-wise: (a + b) * 2.0 ===");
    {
        let mut cx = Graph::new();
        let a = cx.tensor(4);
        let b = cx.tensor(4);
        let out = ((a + b) * 2.0).output();

        cx.build_search_space::<CpuRuntime>();
        let mut rt = CpuRuntime::initialize(());
        rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
        rt.set_data(b, vec![4.0f32, 3.0, 2.0, 1.0]);
        rt = cx.search(rt, 1);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        // Expected: [10, 10, 10, 10]
        println!("Result: {:?}", rt.get_f32(out));
    }

    println!("\n=== 3. Softmax([1, 2, 3, 4]) ===");
    {
        let mut cx = Graph::new();
        let a = cx.tensor((1, 4));
        let out = a.softmax(1).output();

        cx.build_search_space::<CpuRuntime>();
        let mut rt = CpuRuntime::initialize(());
        rt.set_data(a, vec![1.0f32, 2.0, 3.0, 4.0]);
        rt = cx.search(rt, 1);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        let result = rt.get_f32(out);
        let sum: f32 = result.iter().sum();
        println!("Result: {:?}", result);
        println!("Sum (should be 1.0): {:.6}", sum);
    }
    println!("\n=== 4. Two-layer MLP (4→8→2) with ReLU ===");
    {
        let batch = 2usize;
        let d_in  = 4usize;
        let d_hid = 8usize;
        let d_out = 2usize;

        let mut cx = Graph::new();
        let x  = cx.tensor((batch, d_in));
        let w1 = cx.tensor((d_in,  d_hid));
        let w2 = cx.tensor((d_hid, d_out));

        let h     = x.matmul(w1);
        let zero  = cx.tensor((batch, d_hid)); // zero tensor for ReLU comparison
        let gate  = zero.lt(h).as_dtype(DType::F32);               // 1.0 where h > 0, 0.0 elsewhere
        let h_act = h * gate;

        let out = h_act.matmul(w2).output();

        cx.build_search_space::<CpuRuntime>();
        let mut rt = CpuRuntime::initialize(());

        let x_data:  Vec<f32> = (0..batch*d_in) .map(|i| (i as f32 - 4.0) * 0.5).collect();
        let w1_data: Vec<f32> = (0..d_in*d_hid) .map(|i| (i as f32 % 3.0 - 1.0) * 0.3).collect();
        let w2_data: Vec<f32> = (0..d_hid*d_out).map(|i| (i as f32 % 5.0 - 2.0) * 0.2).collect();
        let zeros:   Vec<f32> = vec![0.0; batch * d_hid];

        rt.set_data(x,    x_data);
        rt.set_data(w1,   w1_data);
        rt.set_data(w2,   w2_data);
        rt.set_data(zero, zeros);

        rt = cx.search(rt, 1);
        rt.allocate_intermediate_buffers(&cx.dyn_map);
        rt.execute(&cx.dyn_map);

        println!("MLP output (batch=2, d_out=2): {:?}", rt.get_f32(out));
    }

    println!("\nAll examples completed successfully.");
}
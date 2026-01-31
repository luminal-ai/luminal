use luminal::prelude::*;
use luminal_cuda::{cudarc::driver::CudaContext, runtime::CudaRuntime};

fn main() {
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();
    let mut cx = Graph::new();
    let x = cx.tensor((3, 1));

    let y = ((x * x) + x).output();
    cx.build_search_space::<CudaRuntime>();
    let mut rt = CudaRuntime::initialize(stream);
    rt.set_data(x, vec![1.0, 2.0, 3.0]); // set data before search to enable profiling
    rt = cx.search(rt, 3);
    rt.set_data(x, vec![1.0, 2.0, 3.0]);
    rt.execute(&cx.dyn_map);
    println!("y: {:?}", rt.get_f32(y));
}

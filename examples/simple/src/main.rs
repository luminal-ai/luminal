// glibc malloc degrades into an allocating livelock inside
// nvrtcCompileProgram after heavy search heap churn (hundreds of
// thousands of compiles). jemalloc built with unprefixed symbols
// interposes malloc for the whole process, including dlopened CUDA
// libraries like libnvrtc — a Rust-only global allocator would not.
#[global_allocator]
static ALLOC: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

use luminal::prelude::*;

fn main() {
    // Create compute graph
    let mut cx = Graph::new();
    let a = cx.tensor((3, 1));
    let b = cx.tensor((1, 4));

    let c = a.matmul(b).output();

    display_graph(&cx);

    // Compile
    cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
    let mut rt = cx.search(ReferenceRuntime::default(), CompileOptions::default());

    // Set input tensors
    rt.set_data(a, vec![1.0, 2.0, 3.0]);
    rt.set_data(b, vec![1.0, 2.0, 3.0, 3.0]);

    // Run
    rt.execute(&cx.dyn_map);

    // Get output tensor
    println!("Result: {:?}", rt.get_f32(c));
}

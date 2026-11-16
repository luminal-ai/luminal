//! A lowered program stands alone. Written out, read back and handed to a
//! device that never saw the plan, it computes what the runtime that
//! searched it computes, and runs no compiler to do so. That holds for both
//! ways of running it: replayed as a captured graph, and issued step by
//! step on a stream.
#![cfg(feature = "device")]
mod support;
use luminal::{dtype::DType, graph::Graph, shape::IntExpr};
use luminal_cuda_lite::{
    CudaRuntime,
    device::CudaDevice,
    harness_search_options,
    lowered::{HostCall, KernelCode, LoweredProgram, Step},
    stream::StreamExecutable,
};
use support::{Allocation, TestTransfers};

#[test]
fn a_program_read_back_from_bytes_runs_with_no_plan_and_no_compiler() {
    let mut g = Graph::new();
    let x = g.tensor(vec![IntExpr::from('a'), 8.into()], DType::F32);
    let w = g.tensor(vec![8, 4], DType::F32);
    let b = g.tensor(vec![IntExpr::from('a'), 4.into()], DType::F32);
    let out = (x.matmul(w) * b).relu();
    let mut rt = CudaRuntime::load(&g).unwrap();
    rt.search(
        &luminal::shape::SymbolBounds::from_ranges([('a'.into(), (1, 6))]).unwrap(),
        &[('a'.into(), 3)].into_iter().collect(),
        &Default::default(),
        &harness_search_options(),
    )
    .unwrap();

    // The program as bytes. A kernel that is still source is refused by
    // name; compiled, the program is written out and read back unchanged.
    let mut program = rt.lower().unwrap();
    let refusal = program.ensure_portable().unwrap_err().to_string();
    assert!(refusal.contains("is still source"), "{refusal}");
    program.compile_kernels().unwrap();
    program.ensure_portable().unwrap();
    let bytes = serde_json::to_vec(&program).unwrap();
    let read_back: LoweredProgram = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(serde_json::to_vec(&read_back).unwrap(), bytes);
    // A second copy, to run step by step on a stream.
    let on_a_stream: LoweredProgram = serde_json::from_slice(&bytes).unwrap();
    assert!(
        read_back
            .kernels
            .iter()
            .all(|code| matches!(code, KernelCode::Ptx(_)))
    );
    // The matmul is a host call, and it is plain data.
    assert!(
        read_back.steps.iter().any(
            |step| matches!(step, Step::Host(host) if matches!(host.call, HostCall::CublasLt(_)))
        ),
        "the program holds no cuBLASLt call: {:?}",
        read_back
            .steps
            .iter()
            .map(|step| match step {
                Step::Kernel(kernel) => kernel.label.clone(),
                Step::Copy(_) => "copy".to_string(),
                Step::Host(host) => host.label.clone(),
            })
            .collect::<Vec<_>>()
    );

    // A device that never saw the plan.
    let device = CudaDevice::new(0).unwrap();
    let mut executable = device.executable();
    executable.install_lowered(read_back).unwrap();
    let installed = executable.program().unwrap();
    let home = |lit: i64| {
        installed
            .buffers
            .values()
            .find(|record| record.lit == Some(lit))
            .and_then(|record| record.slice)
            .unwrap()
    };
    let inputs = [x, w, b].map(|tensor| (tensor, home(rt.input_buffer(tensor.id).unwrap())));
    let output = installed
        .output(rt.output_slot_index(out.id).unwrap())
        .unwrap()
        .clone();
    let output_home = installed.slice(&output.buffer).unwrap();
    let output_size = installed
        .buffer(&output.buffer)
        .unwrap()
        .size
        .clone()
        .unwrap();
    let mut arena = Allocation::new(device.stream().clone(), installed.arena_bytes).unwrap();
    let mut staging = executable
        .allocate_staging(installed.staging_bytes)
        .unwrap();

    // The same program issued on a stream: no graph, and scratch memory
    // that is allocated for each call and freed after it.
    let mut stream_executable = StreamExecutable::new(&device, on_a_stream).unwrap();
    assert_eq!(stream_executable.stats().kernel_compilations, 0);
    let other_stream = device.stream().clone();
    let raw_stream = other_stream.cu_stream() as usize as u64;

    let mut runtime_arena = support::allocate_arena(&rt).unwrap();
    let mut runtime_staging = rt.allocate_staging().unwrap();
    // The last call's dimensions, input bytes by arena offset, and result.
    let mut last = None;
    for n in [3usize, 1, 6, 2] {
        let dims: luminal::shape::DynMap = [('a'.into(), n)].into_iter().collect();
        let data = [
            (0..n * 8).map(|i| i as f32 * 0.5 - 3.0).collect::<Vec<_>>(),
            (0..8 * 4).map(|i| (i % 7) as f32 - 2.5).collect::<Vec<_>>(),
            (0..n * 4).map(|i| 1.0 - (i % 3) as f32).collect::<Vec<_>>(),
        ];

        // What the runtime that searched the program computes.
        rt.set_dim('a', n);
        for ((tensor, _), values) in inputs.iter().zip(&data) {
            rt.upload(&mut runtime_arena, tensor.id, values.clone())
                .unwrap();
        }
        rt.execute(runtime_arena.arena(), &mut runtime_staging)
            .unwrap();
        let range = rt.output_arena_range(out.id).unwrap();
        let expected = runtime_arena.read(range.offset, range.bytes).unwrap();
        assert!(expected.iter().any(|byte| *byte != 0));

        // The program that was read back, on its own device and arena.
        for ((_, slice), values) in inputs.iter().zip(&data) {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            arena.write(slice.offset, &bytes).unwrap();
        }
        executable
            .execute(arena.arena(), &mut staging, &dims)
            .unwrap();
        let written = arena
            .read(output_home.offset, output_size.eval(&dims).unwrap())
            .unwrap();
        assert_eq!(written, expected, "a={n}");

        // Issued on a stream, in an arena of this call's own.
        let mut scratch = device
            .allocate_on(raw_stream, stream_executable.program().arena_bytes)
            .unwrap();
        for ((_, slice), values) in inputs.iter().zip(&data) {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_ne_bytes()).collect();
            scratch.write(slice.offset, &bytes).unwrap();
        }
        stream_executable
            .run(raw_stream, scratch.arena(), &dims, &Default::default())
            .unwrap();
        let mut on_stream = vec![0u8; output_size.eval(&dims).unwrap()];
        scratch.read(output_home.offset, &mut on_stream).unwrap();
        assert_eq!(on_stream, expected, "a={n}, issued on a stream");

        let staged: Vec<(usize, Vec<u8>)> = inputs
            .iter()
            .zip(&data)
            .map(|((_, slice), values)| {
                (
                    slice.offset,
                    values.iter().flat_map(|v| v.to_ne_bytes()).collect(),
                )
            })
            .collect();
        last = Some((dims, staged, expected));
    }
    // Nothing was compiled to run it, either way.
    assert_eq!(executable.stats().kernel_compilations, 0);
    assert_eq!(stream_executable.stats().kernel_compilations, 0);
    assert_eq!(stream_executable.stats().launches, 4);

    // A loaded program is handed to another thread and run there, on a
    // device handle, a stream and an arena of that thread's own. The
    // kernels it loaded here are the ones it launches there.
    let (dims, staged, expected) = last.expect("the program ran");
    let output_offset = output_home.offset;
    let output_bytes = expected.len();
    let (elsewhere, launches) = std::thread::spawn(move || -> anyhow::Result<(Vec<u8>, u64)> {
        let device = CudaDevice::new(0)?;
        let raw_stream = device.stream().cu_stream() as usize as u64;
        let mut scratch =
            device.allocate_on(raw_stream, stream_executable.program().arena_bytes)?;
        for (offset, bytes) in &staged {
            scratch.write(*offset, bytes)?;
        }
        stream_executable.run(raw_stream, scratch.arena(), &dims, &Default::default())?;
        let mut written = vec![0u8; output_bytes];
        scratch.read(output_offset, &mut written)?;
        let stats = stream_executable.stats();
        anyhow::ensure!(
            stats.kernel_compilations == 0,
            "a kernel was compiled again"
        );
        Ok((written, stats.launches))
    })
    .join()
    .expect("the other thread")
    .unwrap();
    assert_eq!(elsewhere, expected, "run on another thread");
    assert_eq!(launches, 5);
}

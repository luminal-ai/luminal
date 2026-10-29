#![cfg(feature = "device")]
//! Exercise the generated packing kernel directly, including Bool8, which is
//! available in logical IR but not as a separate frontend DType.

use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg, sys};
use luminal::dtype::PlanDtype;
use luminal::layouts::{
    BitWidthTerm, DecodedLayout, IntExprTerm, RightMajorContiguousElementLayout, ShapeTerm,
};
use luminal_cuda_lite::kernels::{CodegenCtx, KernelOp};
use luminal_cuda_lite::ops::bools_to_bits::BoolsToBitsDps;

fn pack_codes(target: PlanDtype, codes: &[u64]) -> Result<Vec<u8>, DriverError> {
    let width = usize::try_from(target.egglog_bits()).unwrap();
    let input: Vec<u8> = codes
        .iter()
        .flat_map(|code| (0..width).map(move |lane| ((code >> lane) & 1) as u8))
        .collect();
    let layout = |n, dtype, width| {
        DecodedLayout::of(
            RightMajorContiguousElementLayout {
                shape: ShapeTerm(vec![IntExprTerm::Lit(n)]),
                width: BitWidthTerm(width),
            },
            Some(dtype),
        )
    };
    let ctx = CodegenCtx {
        operand_dims: vec![vec![input.len().into()], vec![codes.len().into()]],
        operand_dtypes: vec![PlanDtype::Bool, target],
        dest_dims: vec![vec![codes.len().into()]],
        dest_dtypes: vec![target],
        operand_layouts: vec![
            layout(input.len() as i64, PlanDtype::Bool, 1),
            layout(codes.len() as i64, target, width as i64),
        ],
    };
    let source = BoolsToBitsDps {
        width: width as i64,
    }
    .codegen(&ctx)
    .expect("generate packing kernel");
    let ptx = cudarc::nvrtc::compile_ptx(&source[0].source).expect("compile packing kernel");
    let context = CudaContext::new(0).expect("CUDA device 0");
    let stream = context.default_stream();
    let module = context.load_module(ptx).expect("load packing kernel");
    let function = module.load_function("k").expect("packing entry point");
    let a = stream.clone_htod(&input)?;
    let mut out = stream.alloc_zeros::<u8>(
        codes.len() * luminal_cuda_lite::host_buffer::dtype_bytes(target).unwrap(),
    )?;
    let params = stream.clone_htod(&[0i64])?;
    let mut launch = stream.launch_builder(&function);
    launch.arg(&a).arg(&mut out).arg(&params);
    // SAFETY: allocations cover every coordinate of these dense, static shapes
    // and stay live until synchronization and readback finish.
    unsafe { launch.launch(LaunchConfig::for_num_elems(codes.len() as u32)) }?;
    stream.synchronize()?;
    stream.clone_dtoh(&out)
}

#[test]
fn bool8_packing_accepts_zero_and_one() {
    assert_eq!(pack_codes(PlanDtype::Bool8, &[0, 1]).unwrap(), [0, 1]);
}

#[test]
fn numeric_packing_preserves_unrestricted_encodings() {
    let codes: Vec<u64> = (0..=255).collect();
    assert_eq!(
        pack_codes(PlanDtype::U8, &codes).unwrap(),
        (0..=255).collect::<Vec<u8>>()
    );
    for target in [PlanDtype::UInt64, PlanDtype::Int64, PlanDtype::F64] {
        let codes = [0, 1, 1 << 63, u64::MAX];
        let expected: Vec<u8> = codes.iter().flat_map(|code| code.to_le_bytes()).collect();
        assert_eq!(pack_codes(target, &codes).unwrap(), expected);
    }
}

#[test]
fn bool8_packing_rejects_non_boolean_encodings() {
    const CHILD_CODE: &str = "LUMINAL_BOOL8_PACK_TEST_CODE";
    if let Ok(code) = std::env::var(CHILD_CODE) {
        let code = code.parse().expect("child encoding");
        let result = pack_codes(PlanDtype::Bool8, &[code]);
        let rejected = matches!(
            result,
            Err(DriverError(
                sys::CUresult::CUDA_ERROR_ILLEGAL_INSTRUCTION
                    | sys::CUresult::CUDA_ERROR_LAUNCH_FAILED
            ))
        );
        if !rejected {
            eprintln!("Bool8 code {code:#04x} must trap, got {result:?}");
        }
        // A CUDA trap poisons the process's primary context. Keep it isolated
        // from other tests and exit without reusing that context.
        std::process::exit(if rejected { 0 } else { 1 });
    }
    for code in [2, 3, 128, 255] {
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "bool8_packing_rejects_non_boolean_encodings",
                "--nocapture",
            ])
            .env(CHILD_CODE, code.to_string())
            .output()
            .expect("launch isolated invalid-encoding test");
        assert!(
            output.status.success(),
            "Bool8 code {code:#04x}: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

//! Native BufferIR Metal execution. The portable, decomposed kernel inventory
//! and its search machinery are shared with CUDA Lite (without cuBLASLt or a
//! CUDA device). MSL emission changes only the kernel ABI and scalar dialect;
//! operation matching and layout selection still happen entirely in egglog.
use anyhow::{Result, ensure};
use luminal::shape::Symbol;
use luminal_cuda_lite::{kernels::KernelSource, symbolic};

/// The shared generic kernels use one independent thread per result element,
/// ordinary scalar C expressions, and an input/output/parameter pointer ABI.
/// Reject custom launch geometry and unknown dialect constructs explicitly.
/// This is source emission for an already selected op, not an IR rewrite.
pub fn metal_source(kernel: &KernelSource, schema: &[Symbol]) -> Result<String> {
    ensure!(
        kernel.launch.is_none(),
        "Metal portable kernels require linear launch geometry"
    );
    let source = kernel
        .source
        .trim()
        .strip_prefix("extern \"C\" __global__ void k(")
        .ok_or_else(|| anyhow::anyhow!("unsupported portable kernel signature"))?;
    let (signature, body) = source
        .split_once(") {")
        .ok_or_else(|| anyhow::anyhow!("malformed kernel signature"))?;
    let mut arguments = vec![];
    for (i, arg) in signature.split(',').enumerate() {
        ensure!(
            arg.contains('*'),
            "portable kernel arguments must be pointers"
        );
        arguments.push(format!("device {} [[buffer({i})]]", arg.trim()));
    }
    arguments.push("uint thread_index [[thread_position_in_grid]]".into());
    let index = "(unsigned long long)blockIdx.x * blockDim.x + threadIdx.x";
    ensure!(
        body.contains(index),
        "portable kernel must use the linear thread index"
    );
    let body = body
        .replace(index, "thread_index")
        .replace("__uint_as_float(", "as_type<float>(");
    ensure!(
        !body.contains("__") && !body.contains("blockIdx") && !body.contains("threadIdx"),
        "unsupported CUDA construct in portable kernel"
    );
    let mut source = String::from(
        "#include <metal_stdlib>\nusing namespace metal;\n#define expf exp\n#define exp2f exp2\n#define log2f log2\n#define sinf sin\n#define sqrtf sqrt\n#define fmaxf fmax\n#define fminf fmin\n#define fmodf fmod\n",
    );
    source.push_str(&symbolic::CUDA_HELPERS.replace("__device__", "inline"));
    for (i, s) in schema.iter().enumerate() {
        source.push_str(&format!(
            "#define {} params[{i}]\n",
            symbolic::variable(&s.to_string())
        ));
    }
    source.push_str(&format!("kernel void k({}) {{{body}", arguments.join(", ")));
    // MSL has no double type. CUDA's unsuffixed scientific literals must be
    // emitted as float literals; integer index arithmetic remains 64-bit.
    static FLOATS: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)\b(?:[0-9]+\.[0-9]*|[0-9]+e[+-]?[0-9]+)(?:e[+-]?[0-9]+)?f?")
            .unwrap()
    });
    let source = source.replace("long long", "long").replace("LL", "L");
    Ok(FLOATS
        .replace_all(&source, |caps: &regex::Captures<'_>| {
            let literal = &caps[0];
            if literal.ends_with(['f', 'F']) {
                literal.to_string()
            } else {
                format!("{literal}f")
            }
        })
        .into_owned())
}

#[cfg(target_os = "macos")]
mod runtime;
#[cfg(target_os = "macos")]
pub use runtime::MetalRuntime;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn emits_metal_abi_and_rejects_foreign_kernels() {
        let k=KernelSource::plain("extern \"C\" __global__ void k(const float* a, float* out, const long long* params) { unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; if(i<2) out[i]=a[i]; }".into(),2usize.into());
        let source = metal_source(&k, &['q'.into()]).unwrap();
        assert!(source.contains("device const float* a [[buffer(0)]]"));
        assert!(source.contains("thread_position_in_grid"));
        assert!(!source.contains("__global__"));
        let foreign = KernelSource::plain("something else".into(), 1usize.into());
        assert!(metal_source(&foreign, &[]).is_err());
    }
}

//! ComputeAttnMask — fused op that computes the paged attention mask from indptrs.
//!
//! This op exists so the indptr tensors (qo_indptr, kv_indptr) are visible in the
//! same e-graph chunk as the attention pattern, letting the FlashInfer egglog rule
//! capture them directly.
//!
//! Inputs (3): q_pos (s,) Int, qo_indptr (r,) Int, kv_indptr (r,) Int.
//! Output: mask (s, c) F32 where mask[i, j] = 0.0 (attend) or -1e10 (block).

use std::sync::Arc;

use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{EXPRESSION, OP_KIND},
        extract_expr,
    },
    op::{EgglogOp, HLIROp, LLIROp},
    prelude::*,
};

use crate::{
    cudarc::driver::{CudaStream, result},
    host::{DeviceBuffer, HostOp},
};

/// Computes the paged attention mask from indptr arrays.
///
/// The mask encodes both request-membership and causality:
/// `mask[i, j] = 0.0` if query `i` and context `j` belong to the same request AND
/// context `j`'s local position is `<= q_pos[i]`; `-1e10` otherwise.
#[derive(Debug, Default)]
pub struct ComputeAttnMask {
    pub s_dim: Expression,
    pub c_dim: Expression,
}

impl std::fmt::Display for ComputeAttnMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ComputeAttnMask(s={}, c={})", self.s_dim, self.c_dim)
    }
}

impl HLIROp for ComputeAttnMask {
    fn to_egglog(&self, inputs: &[(NodeIndex, String)]) -> String {
        format!(
            "(Op (ComputeAttnMask {} {}) (ICons {} (ICons {} (ICons {} (INil)))))",
            self.s_dim.to_egglog(),
            self.c_dim.to_egglog(),
            inputs[0].1, // q_pos
            inputs[1].1, // qo_indptr
            inputs[2].1, // kv_indptr
        )
    }
}

impl EgglogOp for ComputeAttnMask {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "ComputeAttnMask",
            &[("s_dim", EXPRESSION), ("c_dim", EXPRESSION)],
        )
    }

    fn n_inputs(&self) -> usize {
        3
    }

    fn rewrites(&self) -> Vec<Rule> {
        // No rewrites — inserted directly by model code.
        vec![]
    }

    fn extract<'a>(
        &'a self,
        egraph: &'a luminal::egglog_utils::SerializedEGraph,
        kind_children: &[&'a ENodeId],
        input_enodes: Vec<&'a ENodeId>,
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let s_dim = extract_expr(egraph, kind_children[0], expr_cache).unwrap();
        let c_dim = extract_expr(egraph, kind_children[1], expr_cache).unwrap();
        let op = Self { s_dim, c_dim };
        let llir_op = LLIROp::new::<dyn HostOp>(Box::new(op) as Box<dyn HostOp>);
        (llir_op, input_enodes)
    }

    fn cleanup(&self) -> bool {
        false
    }
}

impl HostOp for ComputeAttnMask {
    fn execute(
        &self,
        stream: &Arc<CudaStream>,
        self_node: NodeIndex,
        inputs: &[NodeIndex],
        buffers: &FxHashMap<NodeIndex, DeviceBuffer>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> anyhow::Result<()> {
        if inputs.len() < 3 {
            anyhow::bail!(
                "ComputeAttnMask expects 3 inputs (q_pos, qo_indptr, kv_indptr), got {}",
                inputs.len()
            );
        }

        let s = self
            .s_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("ComputeAttnMask s_dim unresolved"))?;
        let c = self
            .c_dim
            .exec(dyn_map)
            .ok_or_else(|| anyhow::anyhow!("ComputeAttnMask c_dim unresolved"))?;
        let r = *dyn_map
            .get(&'r')
            .ok_or_else(|| anyhow::anyhow!("ComputeAttnMask requires dynamic dim 'r'"))?;

        let get_buf = |name: &str, node: NodeIndex| -> anyhow::Result<DeviceBuffer> {
            buffers.get(&node).copied().ok_or_else(|| {
                anyhow::anyhow!("ComputeAttnMask missing {name} buffer for {node:?}")
            })
        };

        let q_pos_buf = get_buf("q_pos", inputs[0])?;
        let qo_indptr_buf = get_buf("qo_indptr", inputs[1])?;
        let kv_indptr_buf = get_buf("kv_indptr", inputs[2])?;
        let out_buf = get_buf("output", self_node)?;

        let q_pos = dtoh_i32(stream, q_pos_buf.ptr(), s)?;
        let qo_indptr = dtoh_i32(stream, qo_indptr_buf.ptr(), r)?;
        let kv_indptr = dtoh_i32(stream, kv_indptr_buf.ptr(), r)?;

        let mut mask = vec![-1e10f32; s * c];
        for i in 0..s {
            let q_req = indptr_to_request(&qo_indptr, i as i32);
            for j in 0..c {
                let c_req = indptr_to_request(&kv_indptr, j as i32);
                if q_req == c_req && q_req >= 0 {
                    let c_local = j as i32 - kv_indptr[c_req as usize];
                    if c_local <= q_pos[i] {
                        mask[i * c + j] = 0.0;
                    }
                }
            }
        }

        let mask_bytes =
            unsafe { std::slice::from_raw_parts(mask.as_ptr() as *const u8, mask.len() * 4) };
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyHtoD_v2(
                out_buf.ptr(),
                mask_bytes.as_ptr() as *const std::ffi::c_void,
                mask_bytes.len(),
            );
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                anyhow::bail!("ComputeAttnMask cuMemcpyHtoD failed: {res:?}");
            }
        }

        Ok(())
    }

    fn output_size(&self) -> Expression {
        self.s_dim * self.c_dim
    }

    fn output_bytes(&self) -> Expression {
        self.output_size() * 4
    }

    fn stats_name(&self) -> Option<&'static str> {
        Some("ComputeAttnMask")
    }
}

fn dtoh_i32(stream: &Arc<CudaStream>, dev_ptr: u64, len: usize) -> anyhow::Result<Vec<i32>> {
    let mut host = vec![0u8; len * std::mem::size_of::<i32>()];
    unsafe {
        result::memcpy_dtoh_async(&mut host, dev_ptr, stream.cu_stream())?;
    }
    stream.synchronize()?;
    let v = unsafe {
        let mut bytes = std::mem::ManuallyDrop::new(host);
        Vec::from_raw_parts(bytes.as_mut_ptr() as *mut i32, len, len)
    };
    Ok(v)
}

/// Given an indptr array `[0, a, b, ...]`, find which segment `idx` belongs to.
/// Returns `count(indptr[i] <= idx) - 1`.
fn indptr_to_request(indptr: &[i32], idx: i32) -> i32 {
    indptr.iter().filter(|&&v| v <= idx).count() as i32 - 1
}

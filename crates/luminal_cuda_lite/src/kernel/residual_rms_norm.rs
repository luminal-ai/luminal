//! Rounded residual addition and RMS normalization, with bounded output views for both storage dtypes.
use super::*;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, F64, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::*,
};

#[derive(Debug, Clone, Default)]
pub struct ResidualRMSNorm(pub rms_norm::RMSNormKernel);
impl KernelOp for ResidualRMSNorm {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        self.0.compile_impl(true, None, stream, cache)
    }
    fn output_size(&self) -> Expression {
        self.0.rows * self.0.cols * (4 + 2 * (self.0.dtype.bits() / 8))
    }
    fn output_bytes(&self) -> Expression {
        self.output_size()
    }
    fn output_dtype(&self) -> DType {
        DType::U8
    }
    fn kernel_name(&self) -> &'static str {
        "ResidualRMSNorm"
    }
}
impl EgglogOp for ResidualRMSNorm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaResidualRMSNorm",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("size", EXPRESSION),
                ("eps", F64),
                ("dtype", DTYPE),
                ("input_dtype", DTYPE),
                ("threads", EXPRESSION),
            ],
        )
    }
    fn n_inputs(&self) -> usize {
        3
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = vec![Rule::raw(r#"
        (rule (
            (= ?norm (Op (FusedRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads)
                (ICons ?residual (ICons ?weight (INil)))))
            (= ?residual (Op (Cast ?size (F32)) (ICons ?sum (INil))))
            (= ?sum (Op (Add (ECons ?rows (ECons ?cols (ENil)))
                (ECons (MMul (MIter) ?cols) (ECons (MIter) (ENil)))
                (ECons (MMul (MIter) ?cols) (ECons (MIter) (ENil)))
                (ECons (MMul (MIter) ?cols) (ECons (MIter) (ENil))))
                (ICons ?left (ICons ?right (INil)))))
            (= ?left (Op (Cast ?size ?dt) (ICons ?x (INil))))
            (= ?right (Op (Cast ?size ?dt) (ICons ?y (INil))))
            (= (dtype ?x) (F32)) (= (dtype ?y) (F32))
            (= (dtype ?weight) (F32)) (= (dtype ?sum) ?dt)
            (!= ?dt (F32))
        ) (
            (let ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads)
                (ICons ?x (ICons ?weight (ICons ?y (INil))))))
            ; Keep narrow producer outputs reachable. Requiring the widened
            ; sources can duplicate a shared matrix product whose other users
            ; already consume its rounded 16-bit result.
            (let ?narrow (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt ?dt ?threads)
                (ICons ?left (ICons ?weight (ICons ?right (INil))))))
            (union ?packed ?narrow)
            (let ?res (Op (CudaBufferView (MNum 0) ?size (F32)) (ICons ?packed (INil))))
            (let ?normalized (Op (CudaBufferView (MMul ?size (MNum 4)) ?size ?dt) (ICons ?packed (INil))))
            (let ?sum_view (Op (CudaBufferView (MMul ?size (MNum 6)) ?size ?dt) (ICons ?packed (INil))))
            (union ?residual ?res) (union ?norm ?normalized) (union ?sum ?sum_view)
            (set (dtype ?sum_view) ?dt)
            (set (dtype ?packed) (U8)) (set (dtype ?res) (F32)) (set (dtype ?normalized) ?dt)
        ) :ruleset kernel_fuse_late :name "rounded residual and rmsnorm with shared result views")
        "#.to_string())];
        rules.extend([128,256,512].map(|threads| Rule::raw(format!(r#"
            (rule ((= ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt ?input_dt (MNum 1024)) ?inputs)))
                ((union ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt ?input_dt (MNum {threads})) ?inputs)))
                :ruleset kernel_fuse_late :name "residual rmsnorm {threads} threads")
        "#))));
        for side in [0, 1] {
            let (inputs, replacement) = if side == 0 {
                (
                    "(ICons ?rounded (ICons ?weight (ICons ?other (INil))))",
                    "(ICons ?raw (ICons ?weight (ICons ?other (INil))))",
                )
            } else {
                (
                    "(ICons ?other (ICons ?weight (ICons ?rounded (INil))))",
                    "(ICons ?other (ICons ?weight (ICons ?raw (INil))))",
                )
            };
            rules.push(Rule::raw(format!(r#"
            (rule (
                (= ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads) {inputs}))
                (= ?rounded (Op (Cast ?size (F32)) (ICons ?low (INil))))
                (= ?low (Op (Cast ?size ?dt) (ICons ?raw (INil))))
                (= (dtype ?raw) (F32))
            ) (
                (union ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads) {replacement}))
                ; Temporary fast-path policy: this kernel already rounds each
                ; input to dt before addition. Keep its narrow-storage family
                ; available when other consumers share the rounded buffer.
                (subsume (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads) {inputs}))
            ) :ruleset kernel_fuse_late :name "absorb residual norm input rounding {side}")
            "#)));
        }
        rules
    }
    fn extract<'a>(
        &'a self,
        eg: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let mut expr = |i| extract_expr(eg, children[i], expressions).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self(rms_norm::RMSNormKernel {
                rows: expr(0),
                cols: expr(1).to_usize().unwrap(),
                eps: eg.enodes[children[3]]
                    .0
                    .replace('"', "")
                    .parse::<f64>()
                    .unwrap() as f32,
                dtype: extract_dtype(eg, children[4]),
                input_dtype: extract_dtype(eg, children[5]),
                threads: expr(6).to_usize().unwrap(),
            }))),
            inputs,
        )
    }
}

/// Fuse a broadcast bias before one input's mandatory low-precision rounding.
/// Keeping the unbiased product as an input preserves its independently tuned
/// implementation and avoids forcing a different library epilogue algorithm.
#[derive(Debug, Clone, Default)]
pub struct BiasResidualRMSNorm {
    pub norm: ResidualRMSNorm,
    pub side: usize,
    pub bias_dtype: DType,
}
impl KernelOp for BiasResidualRMSNorm {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<Symbol, CudaSlice<u8>>,
    ) {
        self.norm
            .0
            .compile_impl(true, Some((self.side, self.bias_dtype)), stream, cache)
    }
    fn output_size(&self) -> Expression {
        self.norm.output_size()
    }
    fn output_bytes(&self) -> Expression {
        self.output_size()
    }
    fn output_dtype(&self) -> DType {
        DType::U8
    }
    fn kernel_name(&self) -> &'static str {
        "BiasResidualRMSNorm"
    }
}
impl EgglogOp for BiasResidualRMSNorm {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaBiasResidualRMSNorm",
            &[
                ("rows", EXPRESSION),
                ("cols", EXPRESSION),
                ("size", EXPRESSION),
                ("eps", F64),
                ("dtype", DTYPE),
                ("input_dtype", DTYPE),
                ("threads", EXPRESSION),
                ("side", EXPRESSION),
                ("bias_dtype", DTYPE),
            ],
        )
    }
    fn n_inputs(&self) -> usize {
        4
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn rewrites(&self) -> Vec<Rule> {
        let mut rules = Vec::new();
        for side in [0, 1] {
            let (inputs, replacement) = if side == 0 {
                (
                    "(ICons ?biased (ICons ?weight (ICons ?other (INil))))",
                    "(ICons ?raw (ICons ?weight (ICons ?other (ICons ?bias (INil)))))",
                )
            } else {
                (
                    "(ICons ?other (ICons ?weight (ICons ?biased (INil))))",
                    "(ICons ?other (ICons ?weight (ICons ?raw (ICons ?bias (INil)))))",
                )
            };
            rules.push(Rule::raw(format!(r#"
            (rule (
                (= ?packed (Op (CudaResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads) {inputs}))
                (= ?biased (Op (Add (ECons ?rows (ECons ?cols (ENil)))
                    (ECons (MMul (MIter) ?cols) (ECons (MIter) (ENil)))
                    (ECons (MNum 0) (ECons (MIter) (ENil)))
                    (ECons (MMul (MIter) ?cols) (ECons (MIter) (ENil))))
                    (ICons ?raw (ICons ?bias (INil)))))
                (= (dtype ?raw) (F32)) (= (dtype ?bias) (F32))
            ) (
                (union ?packed (Op (CudaBiasResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads (MNum {side}) (F32)) {replacement}))
            ) :ruleset kernel_fuse_late :name "broadcast bias residual norm input {side}")
            "#)));
        }
        rules.push(Rule::raw(r#"
        (rule (
            (= ?packed (Op (CudaBiasResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads ?side (F32))
                (ICons ?x (ICons ?weight (ICons ?r (ICons ?bias (INil)))))))
            (= ?bias (Op (Cast ?cols (F32)) (ICons ?low (INil))))
            (= (dtype ?low) ?dt)
        ) (
            (union ?packed (Op (CudaBiasResidualRMSNorm ?rows ?cols ?size ?eps ?dt (F32) ?threads ?side ?dt)
                (ICons ?x (ICons ?weight (ICons ?r (ICons ?low (INil)))))))
        ) :ruleset kernel_fuse_late :name "read residual norm bias in its stored dtype")
        "#.to_string()));
        rules
    }
    fn extract<'a>(
        &'a self,
        eg: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let mut expr = |i| extract_expr(eg, children[i], expressions).unwrap();
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                norm: ResidualRMSNorm(rms_norm::RMSNormKernel {
                    rows: expr(0),
                    cols: expr(1).to_usize().unwrap(),
                    eps: eg.enodes[children[3]]
                        .0
                        .replace('"', "")
                        .parse::<f64>()
                        .unwrap() as f32,
                    dtype: extract_dtype(eg, children[4]),
                    input_dtype: extract_dtype(eg, children[5]),
                    threads: expr(6).to_usize().unwrap(),
                }),
                side: expr(7).to_usize().unwrap(),
                bias_dtype: extract_dtype(eg, children[8]),
            })),
            inputs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg};
    use half::{bf16, f16};
    fn round(x: f32, dt: DType) -> f32 {
        if dt == DType::Bf16 {
            bf16::from_f32(x).to_f32()
        } else {
            f16::from_f32(x).to_f32()
        }
    }
    #[test]
    fn bias_residual_norm_matches_separate_f32_bias_add() {
        use cudarc::driver::DevicePtr;
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut cache = FxHashMap::default();
        for dt in [DType::Bf16, DType::F16] {
            for cols in [129, 136, 2880] {
                for rows in [1, 4, 64] {
                    let x = (0..rows * cols)
                        .map(|i| (i % 41) as f32 / 19. - 1.)
                        .collect::<Vec<_>>();
                    let r = (0..rows * cols)
                        .map(|i| (i % 29) as f32 / 23. - 0.6)
                        .collect::<Vec<_>>();
                    for bias_dt in [DType::F32, dt] {
                        let b = (0..cols)
                            .map(|i| (i % 13) as f32 / 17. - 0.3)
                            .map(|v| if bias_dt == dt { round(v, dt) } else { v })
                            .collect::<Vec<_>>();
                        let bias_bytes = if bias_dt == DType::F32 {
                            bytemuck::cast_slice(&b).to_vec()
                        } else {
                            b.iter()
                                .flat_map(|&v| {
                                    if dt == DType::Bf16 {
                                        bf16::from_f32(v).to_bits().to_le_bytes()
                                    } else {
                                        f16::from_f32(v).to_bits().to_le_bytes()
                                    }
                                })
                                .collect()
                        };
                        let bg = stream.clone_htod(&bias_bytes).unwrap();
                        let wg = stream.clone_htod(&vec![1f32; cols]).unwrap();
                        for side in [0, 1] {
                            let norm = ResidualRMSNorm(rms_norm::RMSNormKernel {
                                rows: rows.into(),
                                cols,
                                eps: 1e-5,
                                dtype: dt,
                                input_dtype: DType::F32,
                                threads: 1024,
                            });
                            let fused = BiasResidualRMSNorm {
                                norm: norm.clone(),
                                side,
                                bias_dtype: bias_dt,
                            };
                            let mut outputs = vec![];
                            for fused_path in [false, true] {
                                let add = |v: &[f32]| {
                                    v.iter()
                                        .enumerate()
                                        .map(|(i, &v)| v + b[i % cols])
                                        .collect::<Vec<_>>()
                                };
                                let xv = if !fused_path && side == 0 {
                                    add(&x)
                                } else {
                                    x.clone()
                                };
                                let rv = if !fused_path && side == 1 {
                                    add(&r)
                                } else {
                                    r.clone()
                                };
                                let xg = stream.clone_htod(&xv).unwrap();
                                let rg = stream.clone_htod(&rv).unwrap();
                                let output = stream.alloc_zeros::<u8>(rows * cols * 8).unwrap();
                                let (f, _, _, _, _, _, _) = if fused_path {
                                    fused.compile(&stream, &mut cache)
                                } else {
                                    norm.compile(&stream, &mut cache)
                                };
                                let (op, xp, wp, rp, bp) = (
                                    output.device_ptr(&stream).0,
                                    xg.device_ptr(&stream).0,
                                    wg.device_ptr(&stream).0,
                                    rg.device_ptr(&stream).0,
                                    bg.device_ptr(&stream).0,
                                );
                                let mut launch = stream.launch_builder(&f);
                                launch.arg(&op).arg(&xp).arg(&wp).arg(&rp);
                                if fused_path {
                                    launch.arg(&bp);
                                }
                                unsafe {
                                    launch
                                        .launch(LaunchConfig {
                                            grid_dim: (rows as u32, 1, 1),
                                            block_dim: (1024, 1, 1),
                                            shared_mem_bytes: 0,
                                        })
                                        .unwrap();
                                }
                                outputs.push(stream.clone_dtoh(&output).unwrap());
                            }
                            assert_eq!(
                                outputs[0], outputs[1],
                                "{dt:?} {bias_dt:?} C{rows} cols{cols} side{side}"
                            );
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn paired_residual_add_preserves_all_low_precision_encodings() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut cache = FxHashMap::default();
        for dt in [DType::Bf16, DType::F16] {
            let x = (0..=u16::MAX)
                .map(|bits| {
                    if dt == DType::Bf16 {
                        bf16::from_bits(bits).to_f32()
                    } else {
                        f16::from_bits(bits).to_f32()
                    }
                })
                .collect::<Vec<_>>();
            // Every encoding appears in both vector lanes. Exercise zero,
            // cancellation, underflow, overflow and a second permuted encoding.
            for mode in 0..4 {
                let r = x
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| match mode {
                        0 => -0.0,
                        1 => -v,
                        2 => v,
                        _ => x[(i * 40503 + 17) & 65535],
                    })
                    .collect::<Vec<_>>();
                let xg = stream.clone_htod(&x).unwrap();
                let rg = stream.clone_htod(&r).unwrap();
                let wg = stream.clone_htod(&vec![1.0f32; 256]).unwrap();
                let output = stream.alloc_zeros::<u8>(x.len() * 8).unwrap();
                let (xp, rp, wp, op) = (
                    xg.device_ptr(&stream).0,
                    rg.device_ptr(&stream).0,
                    wg.device_ptr(&stream).0,
                    output.device_ptr(&stream).0,
                );
                let (f, _, _, _, _, _, _) = ResidualRMSNorm(rms_norm::RMSNormKernel {
                    rows: 256.into(),
                    cols: 256,
                    eps: 1e-5,
                    dtype: dt,
                    input_dtype: DType::F32,
                    threads: 128,
                })
                .compile(&stream, &mut cache);
                unsafe {
                    stream
                        .launch_builder(&f)
                        .arg(&op)
                        .arg(&xp)
                        .arg(&wp)
                        .arg(&rp)
                        .launch(LaunchConfig {
                            grid_dim: (256, 1, 1),
                            block_dim: (128, 1, 1),
                            shared_mem_bytes: 0,
                        })
                        .unwrap();
                }
                let bytes = stream.clone_dtoh(&output).unwrap();
                for (i, bytes) in bytes[..x.len() * 4].as_chunks::<4>().0.iter().enumerate() {
                    let got = f32::from_le_bytes(*bytes);
                    let expected = round(x[i] + r[i], dt);
                    assert!(
                        if expected.is_nan() {
                            got.is_nan()
                        } else {
                            got.to_bits() == expected.to_bits()
                        },
                        "{dt:?} mode {mode}, encoding {i:#x}: {got:?} != {expected:?}"
                    );
                }
            }
        }
    }
    #[test]
    fn residual_rmsnorm_matches_rounded_add_and_existing_normalization() {
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut cache = FxHashMap::default();
        for dt in [DType::Bf16, DType::F16] {
            for cols in [129, 136, 2880] {
                for rows in [1, 2, 4, 8, 16, 32, 64] {
                    let x = (0..rows * cols)
                        .map(|i| (i as f32 * 0.173).sin() * 2.003)
                        .collect::<Vec<_>>();
                    let r = (0..rows * cols)
                        .map(|i| (i as f32 * 0.037).cos() * 1.013)
                        .collect::<Vec<_>>();
                    let w = (0..cols)
                        .map(|i| (i % 19) as f32 / 13. - 0.4)
                        .collect::<Vec<_>>();
                    let sum = x
                        .iter()
                        .zip(&r)
                        .map(|(&x, &r)| round(round(x, dt) + round(r, dt), dt))
                        .collect::<Vec<_>>();
                    for storage in [DType::F32, dt] {
                        let encode = |values: &[f32]| -> Vec<u8> {
                            values
                                .iter()
                                .flat_map(|&v| match storage {
                                    DType::F32 => v.to_le_bytes().to_vec(),
                                    DType::Bf16 => {
                                        bf16::from_f32(v).to_bits().to_le_bytes().to_vec()
                                    }
                                    DType::F16 => f16::from_f32(v).to_bits().to_le_bytes().to_vec(),
                                    _ => unreachable!(),
                                })
                                .collect()
                        };
                        let xg = stream.clone_htod(&encode(&x)).unwrap();
                        let rg = stream.clone_htod(&encode(&r)).unwrap();
                        let wg = stream.clone_htod(&w).unwrap();
                        let sg = stream.clone_htod(&sum).unwrap();
                        let output = stream.alloc_zeros::<u8>(rows * cols * 8 + 64).unwrap();
                        let baseline = stream.alloc_zeros::<u8>(rows * cols * 2).unwrap();
                        let (xp, rp, wp, sp, op, bp) = (
                            xg.device_ptr(&stream).0,
                            rg.device_ptr(&stream).0,
                            wg.device_ptr(&stream).0,
                            sg.device_ptr(&stream).0,
                            output.device_ptr(&stream).0,
                            baseline.device_ptr(&stream).0,
                        );
                        for threads in [128, 256, 512, 1024] {
                            let norm = rms_norm::RMSNormKernel {
                                rows: rows.into(),
                                cols,
                                eps: 1e-5,
                                dtype: dt,
                                input_dtype: DType::F32,
                                threads,
                            };
                            let (f, _, _, _, _, _, _) = ResidualRMSNorm(rms_norm::RMSNormKernel {
                                input_dtype: storage,
                                ..norm.clone()
                            })
                            .compile(&stream, &mut cache);
                            let (b, _, _, _, _, _, _) = norm.compile(&stream, &mut cache);
                            let config = LaunchConfig {
                                grid_dim: (rows as u32, 1, 1),
                                block_dim: (threads as u32, 1, 1),
                                shared_mem_bytes: 0,
                            };
                            unsafe {
                                stream
                                    .launch_builder(&f)
                                    .arg(&op)
                                    .arg(&xp)
                                    .arg(&wp)
                                    .arg(&rp)
                                    .launch(config)
                                    .unwrap();
                                stream
                                    .launch_builder(&b)
                                    .arg(&bp)
                                    .arg(&sp)
                                    .arg(&wp)
                                    .launch(config)
                                    .unwrap();
                            }
                            let got = stream.clone_dtoh(&output).unwrap();
                            let got_sum = got[..rows * cols * 4]
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .map(|v| f32::from_le_bytes(*v))
                                .collect::<Vec<_>>();
                            assert_eq!(
                                got_sum, sum,
                                "rounded residual {dt:?}, C{rows}, cols{cols}, threads{threads}"
                            );
                            assert_eq!(
                                &got[rows * cols * 4..rows * cols * 6],
                                stream.clone_dtoh(&baseline).unwrap(),
                                "normalization {dt:?}, C{rows}, cols{cols}, threads{threads}"
                            );
                            let low = got[rows * cols * 6..rows * cols * 8]
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .map(|bits| {
                                    let bits = u16::from_le_bytes(*bits);
                                    if dt == DType::Bf16 {
                                        bf16::from_bits(bits).to_f32()
                                    } else {
                                        f16::from_bits(bits).to_f32()
                                    }
                                })
                                .collect::<Vec<_>>();
                            assert_eq!(low, sum, "narrow residual output");
                            assert!(got[rows * cols * 8..].iter().all(|v| *v == 0));
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn residual_rmsnorm_rewrite_keeps_one_shared_packed_class() {
        for dt in [DType::Bf16, DType::F16] {
            let mut graph = Graph::new();
            graph.set_dim('s', 4);
            let x = graph.tensor(('s', 136));
            let r = graph.tensor(('s', 136));
            let w = graph.tensor(136);
            let xr = x.cast(dt).cast(DType::F32);
            let rr = r.cast(dt).cast(DType::F32);
            let residual = (xr.cast(dt) + rr.cast(dt)).cast(DType::F32);
            residual.output();
            rms_norm::fused_rms_norm(residual.cast(dt), w, 1e-5).output();
            graph.build_search_space::<crate::runtime::CudaRuntime>(CompileOptions::default());
            let eg = graph.egraph().unwrap();
            let kinds = eg
                .enodes
                .iter()
                .filter(|(_, (op, _))| op == "CudaResidualRMSNorm")
                .collect::<Vec<_>>();
            assert_eq!(
                kinds.len(),
                8,
                "blocks and both storage dtypes must be exposed: {dt:?}"
            );
            let packed = eg
                .enodes
                .iter()
                .filter(|(_, (op, ch))| {
                    op == "Op"
                        && eg.eclasses[&ch[0]]
                            .1
                            .iter()
                            .any(|n| eg.enodes[n].0 == "CudaResidualRMSNorm")
                })
                .map(|(n, _)| eg.node_to_class[n].clone())
                .collect::<FxHashSet<_>>();
            assert_eq!(
                packed.len(),
                1,
                "both output views must share the same packed choice"
            );
        }
    }
}

#[cfg(test)]
#[test]
#[ignore = "GPU timing diagnostic; run on an otherwise idle device"]
fn benchmark_residual_rmsnorm_region() {
    use cudarc::driver::{DevicePtr, LaunchConfig, PushKernelArg, sys::CUevent_flags};
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut cache = FxHashMap::default();
    let source = r#"#include <cuda_bf16.h>
    extern "C" __global__ void add(float* out, const float* x, const float* r, int n) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<n) out[i]=float(__float2bfloat16(float(__float2bfloat16(x[i]))+float(__float2bfloat16(r[i]))));
    }"#;
    let module = stream
        .context()
        .load_module(
            crate::compile_module_image_for_current_device(stream.context(), source).unwrap(),
        )
        .unwrap();
    let add = module.load_function("add").unwrap();
    let flush = stream.alloc_zeros::<u8>(128 * 1024 * 1024).unwrap();
    for rows in [1usize, 2, 4, 8, 16, 32, 64] {
        let cols = 2880;
        let n = (rows * cols) as i32;
        let x = stream
            .clone_htod(&(0..n).map(|i| (i as f32 * 0.17).sin()).collect::<Vec<_>>())
            .unwrap();
        let r = stream
            .clone_htod(&(0..n).map(|i| (i as f32 * 0.31).cos()).collect::<Vec<_>>())
            .unwrap();
        let w = stream.clone_htod(&vec![1.0f32; cols]).unwrap();
        let sum = stream.alloc_zeros::<f32>(n as usize).unwrap();
        let norm = stream.alloc_zeros::<u8>(n as usize * 2).unwrap();
        let packed = stream.alloc_zeros::<u8>(n as usize * 8).unwrap();
        let (xp, rp, wp, sp, np, pp) = (
            x.device_ptr(&stream).0,
            r.device_ptr(&stream).0,
            w.device_ptr(&stream).0,
            sum.device_ptr(&stream).0,
            norm.device_ptr(&stream).0,
            packed.device_ptr(&stream).0,
        );
        for threads in [128, 256, 512, 1024] {
            let op = rms_norm::RMSNormKernel {
                rows: rows.into(),
                cols,
                eps: 1e-5,
                dtype: DType::Bf16,
                input_dtype: DType::F32,
                threads,
            };
            let (base, _, _, _, _, _, _) = op.compile(&stream, &mut cache);
            let (fused, _, _, _, _, _, _) = ResidualRMSNorm(op).compile(&stream, &mut cache);
            let mut graphs = Vec::new();
            for combine in [false, true] {
                stream.synchronize().unwrap();
                CudaGraphHandle::begin_standalone_capture(&stream).unwrap();
                for _ in 0..64 {
                    unsafe {
                        let config = LaunchConfig {
                            grid_dim: (rows as u32, 1, 1),
                            block_dim: (threads as u32, 1, 1),
                            shared_mem_bytes: 0,
                        };
                        if combine {
                            stream
                                .launch_builder(&fused)
                                .arg(&pp)
                                .arg(&xp)
                                .arg(&wp)
                                .arg(&rp)
                                .launch(config)
                                .unwrap();
                        } else {
                            stream
                                .launch_builder(&add)
                                .arg(&sp)
                                .arg(&xp)
                                .arg(&rp)
                                .arg(&n)
                                .launch(LaunchConfig {
                                    grid_dim: ((n as u32).div_ceil(256), 1, 1),
                                    block_dim: (256, 1, 1),
                                    shared_mem_bytes: 0,
                                })
                                .unwrap();
                            stream
                                .launch_builder(&base)
                                .arg(&np)
                                .arg(&sp)
                                .arg(&wp)
                                .launch(config)
                                .unwrap();
                        }
                    }
                }
                let graph = CudaGraphHandle::end_standalone_capture(&stream).unwrap();
                graphs.push((graph.instantiate().unwrap(), graph));
            }
            for trial in 0..24 {
                for index in if trial % 2 == 0 { [0, 1] } else { [1, 0] } {
                    // Restore cold L2 before each complete region graph.
                    unsafe {
                        cudarc::driver::result::memset_d8_async(
                            flush.device_ptr(&stream).0,
                            trial as u8,
                            flush.len(),
                            stream.cu_stream(),
                        )
                        .unwrap();
                    }
                    let start = stream
                        .context()
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    let end = stream
                        .context()
                        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    start.record(&stream).unwrap();
                    graphs[index].0.launch(&stream).unwrap();
                    end.record(&stream).unwrap();
                    end.synchronize().unwrap();
                    eprintln!(
                        "residual_norm_bench,{rows},{threads},{index},{trial},{:.6}",
                        start.elapsed_ms(&end).unwrap() * 1000. / 64.
                    );
                }
            }
        }
    }
}

#[cfg(test)]
#[test]
fn residual_rmsnorm_extracted_graph_reuses_shared_outputs_at_new_widths() {
    use luminal::{egglog_utils::LlirExtractor, search::unroll_packed_llir};
    use rand::SeedableRng;
    let mut graph = Graph::new();
    graph.set_dim('s', 4);
    let x = graph.tensor(('s', 136)).persist();
    let r = graph.tensor(('s', 136)).persist();
    let w = graph.tensor(136).persist();
    let bias = graph.tensor(136).as_dtype(DType::Bf16).persist();
    let xr = (x + bias.cast(DType::F32).expand_lhs(&x.dims()[..1]))
        .cast(DType::Bf16)
        .cast(DType::F32);
    let rr = r.cast(DType::Bf16).cast(DType::F32);
    let low_sum = (xr.cast(DType::Bf16) + rr.cast(DType::Bf16)).output();
    let sum = low_sum.cast(DType::F32);
    let residual = sum.output();
    let downstream = (sum + x).output();
    let norm = rms_norm::fused_rms_norm(sum.cast(DType::Bf16), w, 1e-5).output();
    graph.build_search_space::<crate::runtime::CudaRuntime>(CompileOptions::default());
    let space = graph.search_space().unwrap();
    let eg = &space.buckets[0].egraph;
    let mut extractor = LlirExtractor::new(eg, &space.ops);
    let mut rng = rand::rngs::StdRng::seed_from_u64(20260913);
    let mut choices = luminal::egglog_utils::random_initial_choice(eg, &mut rng);
    // Exercise the new equivalent spelling explicitly in this regression test.
    for (class, node) in &mut choices {
        if let Some(view) = eg.eclasses[*class].1.iter().find(|n| {
            let (op, ch) = &eg.enodes[*n];
            op == "Op"
                && eg.eclasses[&ch[0]].1.iter().any(|kind| {
                    matches!(
                        eg.enodes[kind].0.as_str(),
                        "CudaBufferView" | "CudaBufferViewRead"
                    )
                })
        }) {
            *node = view;
        }
    }
    for (class, node) in &mut choices {
        if let Some(fused) = eg.eclasses[*class].1.iter().find(|n| {
            let (op, ch) = &eg.enodes[*n];
            op == "Op"
                && eg.eclasses[&ch[0]].1.iter().any(|k| {
                    let (op, ch) = &eg.enodes[k];
                    op == "CudaBiasResidualRMSNorm"
                        && eg.eclasses[&ch[8]]
                            .1
                            .iter()
                            .any(|dt| eg.enodes[dt].0 == "Bf16")
                        && eg.eclasses[&ch[5]]
                            .1
                            .iter()
                            .any(|dt| eg.enodes[dt].0 == "F32")
                })
        }) {
            *node = fused;
        }
    }
    let genome = extractor.index_choice_set(&choices);
    let llir = unroll_packed_llir(extractor.extract_indexed_packed(&genome, &[]));
    assert_eq!(
        llir.node_weights()
            .filter(|op| op
                .to_dialect::<dyn KernelOp>()
                .is_some_and(|k| k.kernel_name() == "BiasResidualRMSNorm"))
            .count(),
        1
    );
    assert!(
        !llir
            .node_weights()
            .any(|op| format!("{op:?}").contains("KernelCast")),
        "fused F32 inputs must absorb redundant rounding conversions"
    );
    let stream = CudaContext::new(0).unwrap().new_stream().unwrap();
    let mut runtime = crate::runtime::CudaRuntime::initialize(stream);
    runtime.set_data(w, vec![1f32; 136]);
    let bv = (0..136)
        .map(|i| half::bf16::from_f32((i % 13) as f32 / 17. - 0.3))
        .collect::<Vec<_>>();
    runtime.set_data(bias, bv.clone());
    runtime.set_data(x, vec![0f32; 4 * 136]);
    runtime.set_data(r, vec![0f32; 4 * 136]);
    runtime.load_llir(&llir);
    for rows in [4usize, 1, 2, 8, 16, 32, 64, 2] {
        graph.set_dim('s', rows);
        let xv = (0..rows * 136)
            .map(|i| ((i * 13 % 19) as f32 - 9.) / 19.)
            .collect::<Vec<_>>();
        let rv = (0..rows * 136)
            .map(|i| ((i * 17 % 23) as f32 - 11.) / 23.)
            .collect::<Vec<_>>();
        runtime.set_data(x, xv.clone());
        runtime.set_data(r, rv.clone());
        runtime.execute(&graph.dyn_map);
        let round = |x| half::bf16::from_f32(x).to_f32();
        let expected = xv
            .iter()
            .zip(&rv)
            .enumerate()
            .map(|(i, (&x, &r))| round(round(x + bv[i % 136].to_f32()) + round(r)))
            .collect::<Vec<_>>();
        assert_eq!(runtime.get_f32(residual), expected);
        assert_eq!(
            runtime
                .get_bf16(low_sum)
                .iter()
                .map(|x| x.to_f32())
                .collect::<Vec<_>>(),
            expected
        );
        assert_eq!(
            runtime.get_f32(downstream),
            expected
                .iter()
                .zip(&xv)
                .map(|(&sum, &x)| sum + x)
                .collect::<Vec<_>>()
        );
        let norm = runtime.get_bf16(norm);
        for (values, actual) in expected
            .as_chunks::<136>()
            .0
            .iter()
            .zip(norm.as_chunks::<136>().0.iter())
        {
            let inv = (values.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / 136.
                + 1e-5f32 as f64)
                .sqrt()
                .recip();
            for (&v, &a) in values.iter().zip(actual) {
                let e = v as f64 * inv;
                assert!((a.to_f32() as f64 - e).abs() <= 0.008 * e.abs().max(1e-6));
            }
        }
    }
}

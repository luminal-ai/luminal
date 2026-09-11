//! Cooperative SIMT matrix products, with optional rounded bias/promotion.
//! Layout proofs and all launch choices live in egglog. The plain variant is
//! shared by Lite; full CUDA also registers the bias variant of the same code.

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr,
    },
    op::*,
    prelude::*,
};

use crate::{
    compile_module_image_for_current_device, cuda_dtype,
    kernel::{
        KernelOp,
        hlir::{dtype_includes, generate_dyn_dims_defines},
    },
};

/// (input rows per tile, output columns per tile, warps per column).
/// The widest row tiles remain alternatives; resource validation and measured
/// search decide their suitability rather than a shape-specific Rust policy.
pub fn thin_matmul_choices() -> impl Iterator<Item = (usize, usize, usize)> {
    [1, 2, 4, 8, 16, 32, 64].into_iter().flat_map(|m| {
        [1, 2, 4, 8].into_iter().flat_map(move |n| {
            [1, 2, 4, 8]
                .into_iter()
                .filter_map(move |s| (n * s <= 8).then_some((m, n, s)))
        })
    })
}

#[derive(Default, Clone)]
pub struct KernelThinMatmul<const BIAS: bool, const MIXED: bool = false> {
    m: Expression,
    n: Expression,
    k: Expression,
    dtype: DType,
    output_dtype: DType,
    round_dtype: DType,
    tile_m: usize,
    tile_n: usize,
    warps_per_column: usize,
}

// Keep legacy program fingerprints stable when only new alternatives are added.
impl<const BIAS: bool, const MIXED: bool> std::fmt::Debug for KernelThinMatmul<BIAS, MIXED> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("KernelThinMatmul");
        out.field("m", &self.m)
            .field("n", &self.n)
            .field("k", &self.k)
            .field("dtype", &self.dtype)
            .field("output_dtype", &self.output_dtype);
        if MIXED {
            out.field("round_dtype", &self.round_dtype);
        }
        out.field("tile_m", &self.tile_m)
            .field("tile_n", &self.tile_n)
            .field("warps_per_column", &self.warps_per_column)
            .finish()
    }
}

impl<const BIAS: bool, const MIXED: bool> KernelThinMatmul<BIAS, MIXED> {
    fn name() -> &'static str {
        if MIXED {
            if BIAS {
                "KernelMixedAffine"
            } else {
                "KernelMixedMatmul"
            }
        } else if BIAS {
            "KernelThinMatmulBias"
        } else {
            "KernelThinMatmul"
        }
    }
}

impl<const BIAS: bool, const MIXED: bool> EgglogOp for KernelThinMatmul<BIAS, MIXED> {
    fn sort(&self) -> SortDef {
        let mut fields = vec![
            ("m", EXPRESSION),
            ("n", EXPRESSION),
            ("k", EXPRESSION),
            ("dtype", DTYPE),
            ("output_dtype", DTYPE),
            ("tile_m", EXPRESSION),
            ("tile_n", EXPRESSION),
            ("warps_per_column", EXPRESSION),
        ];
        if MIXED {
            fields.push(("round_dtype", DTYPE));
        }
        sort(OP_KIND, Self::name(), &fields)
    }
    fn n_inputs(&self) -> usize {
        if BIAS { 3 } else { 2 }
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn egglog_declarations(&self) -> Vec<String> {
        vec!["(relation thin_matmul_output_extent (IR Expression))".to_string()]
    }
    fn rewrites(&self) -> Vec<Rule> {
        if MIXED {
            return self.mixed_rewrites();
        }
        let name = Self::name();
        let mut rules = vec![if BIAS {
            Rule::raw(
                r#"(rule
                (
                    (= ?sum (Op (KernelThinMatmul ?m ?n ?k ?dt ?dt
                        (MNum 1) (MNum 8) (MNum 1))
                        (ICons ?x (ICons ?w (INil)))))
                    (= ?out (Op (Add (ECons ?m (ECons ?n (ENil)))
                        ?out_strides (ECons (MNum 0) (ECons (MIter) (ENil))) ?out_strides)
                        (ICons ?sum (ICons ?bias (INil)))))
                    (= ?out_strides (ECons (MMul (MIter) ?n) (ECons (MIter) (ENil))))
                    (= (dtype ?bias) ?dt)
                    (= (dtype ?out) ?dt)
                )
                (
                    (let ?fused (Op (KernelThinMatmulBias ?m ?n ?k ?dt ?dt
                        (MNum 1) (MNum 8) (MNum 1))
                        (ICons ?x (ICons ?w (ICons ?bias (INil))))))
                    (union ?out ?fused)
                    (set (dtype ?fused) ?dt)
                    (thin_matmul_output_extent ?fused (MMul ?m ?n))
                )
                :ruleset matmul_backend
                :name "thin matmul rounded column bias"
            )"#,
            )
        } else {
            Rule::raw(
                r#"(rule
                (
                    (= ?sum (Op (GenericMatmul
                        (ECons ?m (ECons ?n (ENil))) ?mul_shape ?k
                        (ECons (MMul (MIter) ?k) (ECons (MNum 0) (ECons (MIter) (ENil))))
                        (ECons (MNum 0) (ECons (MMul (MIter) ?k) (ECons (MIter) (ENil))))
                        ?sum_in_stride (MIter) ?sum_out_stride ?dt)
                        (ICons ?x (ICons ?w (INil)))))
                    (generic_matmul_exact_2d ?sum ?m ?n ?k ?dt)
                    (low_precision_matmul_dtype ?dt)
                    (= (dtype ?x) ?dt)
                    (= (dtype ?w) ?dt)
                )
                (
                    (let ?kernel (Op (KernelThinMatmul ?m ?n ?k ?dt ?dt
                        (MNum 1) (MNum 8) (MNum 1))
                        (ICons ?x (ICons ?w (INil)))))
                    (union ?sum ?kernel)
                    (set (dtype ?kernel) ?dt)
                    (thin_matmul_output_extent ?kernel (MMul ?m ?n))
                )
                :ruleset matmul_backend
                :name "cooperative thin matmul contiguous row-major transposed-weight layout"
            )"#,
            )
        }];
        // Keep the evaluated extent in a relation: expr folding can subsume
        // the raw MMul node, so matching its spelling loses static promotions.
        // Equality with the Cast extent also rejects casts of a partial view.
        rules.push(Rule::raw(format!(
            r#"(rule
            (
                (= ?product (Op ({name} ?m ?n ?k ?dt ?dt
                    (MNum 1) (MNum 8) (MNum 1)) ?inputs))
                (thin_matmul_output_extent ?product ?elements)
                (= ?out (Op (Cast ?elements (F32)) (ICons ?product (INil))))
            )
            (
                (let ?promoted (Op ({name} ?m ?n ?k ?dt (F32)
                    (MNum 1) (MNum 8) (MNum 1)) ?inputs))
                (union ?out ?promoted)
                (set (dtype ?promoted) (F32))
            )
            :ruleset matmul_backend
            :name "{name} promotion preserves low-precision output rounding"
        )"#
        )));
        rules.extend(thin_matmul_choices().filter(|&v| v != (1, 8, 1)).map(|(m,n,s)| {
            Rule::raw(format!(r#"(rule
                ((= ?out (Op ({name} ?m ?n ?k ?dt ?out_dt (MNum 1) (MNum 8) (MNum 1)) ?inputs)))
                (
                    (let ?tuned (Op ({name} ?m ?n ?k ?dt ?out_dt (MNum {m}) (MNum {n}) (MNum {s})) ?inputs))
                    (union ?out ?tuned)
                    (set (dtype ?tuned) ?out_dt)
                )
                :ruleset matmul_backend
                :name "{name} tile {m} by {n} with {s} warps per column"
            )"#))
        }));
        rules
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _lists: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let mut expression = |i: usize| extract_expr(egraph, children[i], expressions).unwrap();
        let op = Self {
            m: expression(0),
            n: expression(1),
            k: expression(2),
            dtype: extract_dtype(egraph, children[3]),
            output_dtype: extract_dtype(egraph, children[4]),
            round_dtype: extract_dtype(egraph, children[if MIXED { 8 } else { 3 }]),
            tile_m: expression(5).to_usize().expect("constant tile_m"),
            tile_n: expression(6).to_usize().expect("constant tile_n"),
            warps_per_column: expression(7).to_usize().expect("constant warps_per_column"),
        };
        (LLIROp::new::<dyn KernelOp>(Box::new(op)), inputs)
    }
}

impl<const BIAS: bool, const MIXED: bool> KernelOp for KernelThinMatmul<BIAS, MIXED> {
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
        assert!(matches!(self.dtype, DType::Bf16 | DType::F16));
        assert!(self.output_dtype == self.dtype || self.output_dtype == DType::F32);
        let (tm, tn, s) = (self.tile_m, self.tile_n, self.warps_per_column);
        assert!(thin_matmul_choices().any(|v| v == (tm, tn, s)));
        let vars = self.all_dyn_vars();
        let (defines, _) = generate_dyn_dims_defines(&vars);
        let dims = if vars.is_empty() {
            ""
        } else {
            ", const int* dyn_dims"
        };
        let includes = dtype_includes(&[self.dtype, self.output_dtype]);
        let ty = cuda_dtype(self.dtype);
        let out_ty = cuda_dtype(self.output_dtype);
        let (m, n, k) = (self.m.to_kernel(), self.n.to_kernel(), self.k.to_kernel());
        let bias = if BIAS {
            format!(", const {}* bias", if MIXED { "float" } else { ty })
        } else {
            String::new()
        };
        let epilogue = if MIXED {
            let round_ty = cuda_dtype(self.round_dtype);
            let value = if BIAS { "sum + bias[col]" } else { "sum" };
            format!("out[row * ({n}) + col] = ({out_ty})(({round_ty})({value}));")
        } else if BIAS {
            format!(
                "{ty} rounded = ({ty})sum; rounded = ({ty})((float)rounded + (float)bias[col]); out[row * ({n}) + col] = ({out_ty})rounded;"
            )
        } else {
            format!("out[row * ({n}) + col] = ({out_ty})(({ty})sum);")
        };
        let reduction = if s > 1 {
            format!(
                r#"
            __shared__ float partial[{tm} * {tn} * {s}];
            #pragma unroll
            for (int t=0;t<{tm};t++) if (lane==0) partial[t*{tn}*{s}+warp]=acc[t];
            __syncthreads();
            if (split==0 && lane==0 && col<({n})) {{
                #pragma unroll
                for (int t=0;t<{tm};t++) {{
                    long long row=first_row+t;
                    if (row<({m})) {{
                        float sum=partial[t*{tn}*{s}+warp];
                        #pragma unroll
                        for (int p=1;p<{s};p++) sum+=partial[t*{tn}*{s}+warp+p];
                        {epilogue}
                    }}
                }}
            }}"#
            )
        } else {
            format!(
                r#"
            if (lane==0 && col<({n})) {{
                #pragma unroll
                for (int t=0;t<{tm};t++) {{
                    long long row=first_row+t;
                    if (row<({m})) {{ float sum=acc[t]; {epilogue} }}
                }}
            }}"#
            )
        };
        let name = if MIXED {
            if BIAS {
                "mixed_affine_k"
            } else {
                "mixed_matmul_k"
            }
        } else if BIAS {
            "thin_matmul_bias_k"
        } else {
            "thin_matmul_k"
        };
        let source = format!(
            r#"{includes}
{defines}
extern "C" __global__ void {name}({out_ty}* out, const {ty}* x, const {ty}* w{bias}{dims}) {{
    int lane=threadIdx.x%32, warp=threadIdx.x/32, split=warp%{s};
    long long col=(long long)blockIdx.x*{tn}+warp/{s};
    long long first_row=(long long)blockIdx.y*{tm};
    float acc[{tm}]={{}};
    if (col<({n})) {{
        if (({k})%8==0 && (((unsigned long long)x | (unsigned long long)w)&15)==0) {{
            for (long long j=split*32+lane;j<({k})/8;j+=32*{s}) {{
                uint4 weight=((const uint4*)(w+col*({k})))[j];
                #pragma unroll
                for (int t=0;t<{tm};t++) if (first_row+t<({m})) {{
                    uint4 input=((const uint4*)(x+(first_row+t)*({k})))[j];
                    #pragma unroll
                    for (int v=0;v<8;v++) acc[t]+=(float)(({ty}*)&weight)[v]*(float)(({ty}*)&input)[v];
                }}
            }}
        }} else {{
            for (long long j=split*32+lane;j<({k});j+=32*{s}) {{
                float weight=(float)w[col*({k})+j];
                #pragma unroll
                for (int t=0;t<{tm};t++) if (first_row+t<({m})) acc[t]+=weight*(float)x[(first_row+t)*({k})+j];
            }}
        }}
    }}
    #pragma unroll
    for (int t=0;t<{tm};t++) {{
        #pragma unroll
        for (int d=16;d;d/=2) acc[t]+=__shfl_down_sync(0xffffffff,acc[t],d);
    }}
    {reduction}
}}"#
        );
        let (module, function) = if let Some((module, function)) = cache.get(&source) {
            (module.clone(), function.clone())
        } else {
            let image = compile_module_image_for_current_device(stream.context(), &source).unwrap();
            let module = stream.context().load_module(image).unwrap();
            let function = module.load_function(name).unwrap();
            cache.insert(source.clone(), (module.clone(), function.clone()));
            (module, function)
        };
        (
            function,
            module,
            source,
            (
                self.n.ceil_div(tn).max(1),
                self.m.ceil_div(tm).max(1),
                1.into(),
            ),
            ((32 * tn * s).into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }
    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        for dim in [self.m, self.n, self.k] {
            dim.collect_dyn_vars_into(vars);
        }
    }
    fn output_size(&self) -> Expression {
        self.m * self.n
    }
    fn output_bytes(&self) -> Expression {
        (self.output_size() * self.output_dtype.bits()).ceil_div(8)
    }
    fn output_dtype(&self) -> DType {
        self.output_dtype
    }
    fn bytes_loaded(&self) -> Expression {
        ((self.m * self.k + self.n * self.k) * self.dtype.bits()
            + if BIAS {
                self.n * if MIXED { 32 } else { self.dtype.bits() }
            } else {
                0.into()
            })
        .ceil_div(8)
    }
    fn bytes_stored(&self) -> Expression {
        self.output_bytes()
    }
    fn flops(&self) -> Expression {
        self.m * self.n * self.k * 2 + if BIAS { self.m * self.n } else { 0.into() }
    }
    fn kernel_name(&self) -> &'static str {
        if MIXED {
            if BIAS { "MixedAffine" } else { "MixedMatmul" }
        } else if BIAS {
            "ThinMatmulBias"
        } else {
            "ThinMatmul"
        }
    }
}

impl<const BIAS: bool, const MIXED: bool> KernelThinMatmul<BIAS, MIXED> {
    // The storage, accumulation and final rounding contracts are independent:
    // widened operands are exact, bias is added in F32, and a final low-precision
    // round must survive any subsequent promotion to F32.
    fn mixed_rewrites(&self) -> Vec<Rule> {
        let name = Self::name();
        let mut rules = vec![Rule::raw(if BIAS {
            r#"(rule (
                (= ?sum (Op (KernelMixedMatmul ?m ?n ?k ?dt (F32)
                    (MNum 1) (MNum 8) (MNum 1) (F32))
                    (ICons ?x (ICons ?w (INil)))))
                (= ?out (Op (Add (ECons ?m (ECons ?n (ENil)))
                    ?strides (ECons (MNum 0) (ECons (MIter) (ENil))) ?strides)
                    (ICons ?sum (ICons ?bias (INil)))))
                (= ?strides (ECons (MMul (MIter) ?n) (ECons (MIter) (ENil))))
                (= (dtype ?bias) (F32)) (= (dtype ?out) (F32))
            ) (
                (let ?fused (Op (KernelMixedAffine ?m ?n ?k ?dt (F32)
                    (MNum 1) (MNum 8) (MNum 1) (F32))
                    (ICons ?x (ICons ?w (ICons ?bias (INil))))))
                (union ?out ?fused) (set (dtype ?fused) (F32))
                (thin_matmul_output_extent ?fused (MMul ?m ?n))
            ) :ruleset matmul_backend :name "mixed affine F32 bias before final rounding")"#
        } else {
            r#"(rule (
                (= ?sum (Op (GenericMatmul
                    (ECons ?m (ECons ?n (ENil))) ?mul_shape ?k
                    (ECons (MMul (MIter) ?k) (ECons (MNum 0) (ECons (MIter) (ENil))))
                    (ECons (MNum 0) (ECons (MMul (MIter) ?k) (ECons (MIter) (ENil))))
                    ?sum_in_stride (MIter) ?sum_out_stride (F32))
                    (ICons ?xc (ICons ?wc (INil)))))
                (generic_matmul_exact_2d ?sum ?m ?n ?k (F32))
                (= ?xc (Op (Cast ?xs (F32)) (ICons ?x (INil))))
                (= ?wc (Op (Cast ?ws (F32)) (ICons ?w (INil))))
                (= (dtype ?x) ?dt) (= (dtype ?w) ?dt)
                (low_precision_matmul_dtype ?dt)
            ) (
                (let ?kernel (Op (KernelMixedMatmul ?m ?n ?k ?dt (F32)
                    (MNum 1) (MNum 8) (MNum 1) (F32))
                    (ICons ?x (ICons ?w (INil)))))
                (union ?sum ?kernel) (set (dtype ?kernel) (F32))
                (thin_matmul_output_extent ?kernel (MMul ?m ?n))
            ) :ruleset matmul_backend :name "mixed matmul exact widened operands")"#
        })];
        rules.push(Rule::raw(format!(
            r#"(rule (
            (= ?product (Op ({name} ?m ?n ?k ?dt (F32)
                (MNum 1) (MNum 8) (MNum 1) (F32)) ?inputs))
            (thin_matmul_output_extent ?product ?size)
            (= ?out (Op (Cast ?size ?dt) (ICons ?product (INil))))
        ) (
            (let ?narrow (Op ({name} ?m ?n ?k ?dt ?dt
                (MNum 1) (MNum 8) (MNum 1) ?dt) ?inputs))
            (union ?out ?narrow) (set (dtype ?narrow) ?dt)
            (thin_matmul_output_extent ?narrow ?size)
        ) :ruleset matmul_backend :name "{name} final output rounding")"#
        )));
        rules.push(Rule::raw(format!(
            r#"(rule (
            (= ?product (Op ({name} ?m ?n ?k ?dt ?dt
                (MNum 1) (MNum 8) (MNum 1) ?dt) ?inputs))
            (thin_matmul_output_extent ?product ?size)
            (= ?out (Op (Cast ?size (F32)) (ICons ?product (INil))))
        ) (
            (let ?promoted (Op ({name} ?m ?n ?k ?dt (F32)
                (MNum 1) (MNum 8) (MNum 1) ?dt) ?inputs))
            (union ?out ?promoted) (set (dtype ?promoted) (F32))
        ) :ruleset matmul_backend :name "{name} promotion preserves final rounding")"#
        )));
        rules.extend(thin_matmul_choices().filter(|&v| v != (1, 8, 1)).map(|(m,n,s)| {
            Rule::raw(format!(r#"(rule
                ((= ?out (Op ({name} ?m ?n ?k ?dt ?out_dt (MNum 1) (MNum 8) (MNum 1) ?round) ?inputs)))
                (
                    (let ?tuned (Op ({name} ?m ?n ?k ?dt ?out_dt (MNum {m}) (MNum {n}) (MNum {s}) ?round) ?inputs))
                    (union ?out ?tuned) (set (dtype ?tuned) ?out_dt)
                ) :ruleset matmul_backend :name "{name} tile {m} by {n} with {s} warps per column")"#))
        }));
        rules
    }
}

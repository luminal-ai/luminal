//! External implementations must survive extraction, DPS, and buffer-plan
//! cloning without an entry in a backend-owned dispatch table.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::egglog_snippet::EgglogSnippet;
use luminal::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};
use luminal::prelude::FxHashMap;
use luminal_cuda_lite::kernels::{CodegenCtx, KernelSource};
use luminal_cuda_lite::ops::add::{AddFunctionalDps, AddFunctionalMatcher};
use luminal_cuda_lite::{
    CudaOpInterface, CudaRuntime, KernelOp, RegisteredOp, as_host_op, as_kernel_op,
    cuda_registry_without_cublaslt, harness_search_options,
};

#[derive(Debug, Clone)]
struct ExternalAdd {
    dps: bool,
    marker: u32,
}

impl OpSlotNames for ExternalAdd {}

impl BufferTensorIrOp for ExternalAdd {
    fn label(&self) -> &str {
        "ExternalAdd"
    }

    fn operand_reads_memory(&self, operand: usize) -> bool {
        !self.dps || operand < 2
    }

    fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
        self.dps
            .then(|| CudaOpInterface::kernel::<Self>() as &dyn std::any::Any)
    }
}

impl Bufferizable for ExternalAdd {
    fn alias_info(&self) -> Vec<AliasInfo> {
        if self.dps {
            AddFunctionalDps.alias_info()
        } else {
            vec![]
        }
    }
}

impl ToDps for ExternalAdd {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        (!self.dps).then(|| {
            Box::new(Self {
                dps: true,
                marker: self.marker,
            }) as Box<dyn LayoutIrOp>
        })
    }
}

impl LayoutIrOp for ExternalAdd {}

impl KernelOp for ExternalAdd {
    fn codegen(&self, ctx: &CodegenCtx) -> anyhow::Result<Vec<KernelSource>> {
        let mut launches = AddFunctionalDps.codegen(ctx)?;
        for launch in &mut launches {
            launch.launch = Some(luminal_cuda_lite::kernels::KernelLaunch::linear(
                launch.n.clone(),
                128,
            ));
            launch
                .source
                .push_str(&format!("\n// external {}\n", self.marker));
        }
        Ok(launches)
    }
}

#[derive(Debug)]
struct ExternalAddMatcher;

impl OpMatcher for ExternalAddMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpExternalAdd"
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        AddFunctionalMatcher.metadata_slots()
    }

    fn snippets(&self) -> Vec<EgglogSnippet> {
        // Reuse the add semantics/layout premises under an independent
        // constructor, proving claims do not depend on built-in labels.
        static SNIPPETS: std::sync::OnceLock<
            Vec<(luminal::egglog_snippet::SpliceCategory, String)>,
        > = std::sync::OnceLock::new();
        SNIPPETS
            .get_or_init(|| {
                AddFunctionalMatcher
                    .snippets()
                    .into_iter()
                    .map(|snippet| {
                        (
                            snippet.category,
                            snippet.text.replace(
                                AddFunctionalMatcher.egglog_constructor(),
                                self.egglog_constructor(),
                            ),
                        )
                    })
                    .collect()
            })
            .iter()
            .map(|(category, text)| EgglogSnippet {
                category: *category,
                text,
            })
            .collect()
    }

    fn extract(&self, _site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(ExternalAdd {
            dps: false,
            marker: 73,
        })
    }
}

#[test]
fn external_kernel_is_claimed_bufferized_cloned_and_executed() {
    let mut graph = luminal::graph::Graph::new();
    let a = graph.tensor((2, 3), DType::F32);
    let b = graph.tensor((2, 3), DType::F32);
    let out = (a + b).output();
    let mut registry = cuda_registry_without_cublaslt();
    registry.retain(|entry| entry.constructor() != AddFunctionalMatcher.egglog_constructor());
    registry.push(RegisteredOp::new(
        Box::new(ExternalAddMatcher),
        Box::new(ExternalAdd {
            dps: false,
            marker: 0,
        }),
    ));
    let mut runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
    assert!(
        runtime
            .active_allow_list()
            .contains(&ExternalAddMatcher.egglog_constructor())
    );
    let data: FxHashMap<_, _> = [
        (a.id, vec![1.0f32, 2., 3., 4., 5., 6.].into()),
        (b.id, vec![10.0f32, 20., 30., 40., 50., 60.].into()),
    ]
    .into_iter()
    .collect();
    runtime.search(&data, &harness_search_options()).unwrap();
    let plan = runtime.plan().unwrap().clone();
    let mut found = false;
    for node in plan.dag.node_weights() {
        if let BufferNode::Compute {
            op,
            operand_info,
            result_info,
            ..
        } = node
            && let Some(external) = op.as_any().downcast_ref::<ExternalAdd>()
        {
            found = true;
            assert!(external.dps);
            let ctx = CodegenCtx::from_descriptors(op.label(), operand_info, result_info).unwrap();
            let sources = as_kernel_op(op.as_ref()).unwrap().codegen(&ctx).unwrap();
            assert!(sources[0].source.contains("// external 73"));
            assert!(as_host_op(op.as_ref()).is_none());
        }
    }
    assert!(found, "the external operation must survive bufferization");
    #[cfg(feature = "device")]
    {
        for (id, buffer) in data {
            runtime.set_data(id, buffer);
        }
        runtime.execute().unwrap();
        assert_eq!(
            runtime.get_f32(out.id).unwrap(),
            vec![11., 22., 33., 44., 55., 66.]
        );
    }
    #[cfg(not(feature = "device"))]
    let _ = out;
}

#[test]
fn a_familiar_label_without_cuda_traits_is_not_claimed() {
    let graph = luminal::graph::Graph::new();
    let registry = vec![RegisteredOp::new(
        Box::new(luminal_reference::ops::AddFunctionalMatcher),
        Box::new(luminal_reference::ops::AddFunctional),
    )];
    let runtime = CudaRuntime::load_with_registry(&graph, registry).unwrap();
    assert!(runtime.active_allow_list().is_empty());
}

#[test]
fn all_cublaslt_dps_forms_keep_the_host_interface_when_cloned() {
    use luminal_cuda_lite::ops::cublaslt::{CublasLt, CublasLtForm};
    for form in CublasLtForm::ALL {
        let functional = CublasLt { form, spec: None };
        let dps = functional.to_dps().unwrap();
        let cloned = dps.clone_bt_box();
        let host = as_host_op(cloned.as_ref()).expect("cuBLASLt DPS implements HostOp");
        assert_eq!(host.label(), functional.label());
        assert!(as_kernel_op(cloned.as_ref()).is_none());
    }
}

#[cfg(feature = "device")]
mod host_graphs {
    use super::*;
    use cudarc::driver::{CudaFunction, CudaModule, LaunchConfig, PushKernelArg};
    use luminal_cuda_lite::host::{CaptureCtx, HostOp, HostOpContext, PreparedHostOp};
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    static RECORDS: AtomicUsize = AtomicUsize::new(0);
    static DROPPED: AtomicUsize = AtomicUsize::new(0);
    #[derive(Debug, Clone)]
    struct ExternalHostAdd {
        dps: bool,
    }
    impl OpSlotNames for ExternalHostAdd {}
    impl BufferTensorIrOp for ExternalHostAdd {
        fn label(&self) -> &str {
            "ExternalHostAdd"
        }
        fn operand_reads_memory(&self, i: usize) -> bool {
            !self.dps || i < 2
        }
        fn runtime_interface(&self) -> Option<&dyn std::any::Any> {
            self.dps
                .then(|| CudaOpInterface::host::<Self>() as &dyn std::any::Any)
        }
    }
    impl Bufferizable for ExternalHostAdd {
        fn alias_info(&self) -> Vec<AliasInfo> {
            if self.dps {
                AddFunctionalDps.alias_info()
            } else {
                vec![]
            }
        }
    }
    impl ToDps for ExternalHostAdd {
        fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
            (!self.dps).then(|| Box::new(Self { dps: true }) as Box<dyn LayoutIrOp>)
        }
    }
    impl LayoutIrOp for ExternalHostAdd {}
    struct Prepared {
        dim: usize,
        _module: Arc<CudaModule>,
        function: CudaFunction,
        inputs: [u64; 2],
        dest: u64,
        n: u64,
        repeats: usize,
    }
    impl Drop for Prepared {
        fn drop(&mut self) {
            DROPPED.fetch_or(1 << self.dim, Ordering::SeqCst);
        }
    }
    impl PreparedHostOp for Prepared {
        unsafe fn record(&self, ctx: &CaptureCtx<'_>) -> anyhow::Result<()> {
            RECORDS.fetch_add(1, Ordering::SeqCst);
            for _ in 0..self.repeats {
                let mut launch = ctx.stream().launch_builder(&self.function);
                launch
                    .arg(&self.inputs[0])
                    .arg(&self.inputs[1])
                    .arg(&self.dest)
                    .arg(&self.n);
                unsafe { launch.launch(LaunchConfig::for_num_elems(self.n as u32)) }?;
            }
            assert!(self.dim != 7, "injected recording panic");
            Ok(())
        }
    }
    impl HostOp for ExternalHostAdd {
        unsafe fn prepare(
            &self,
            ctx: &HostOpContext<'_>,
        ) -> anyhow::Result<Box<dyn PreparedHostOp>> {
            let a = ctx.dims[&'a'.into()];
            anyhow::ensure!(a != 6, "injected preparation failure");
            let ptx = cudarc::nvrtc::compile_ptx(
                r#"extern "C" __global__ void add(const float* a,const float* b,float* out,unsigned long long n){unsigned long long i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=a[i]+b[i];}"#,
            )?;
            let module = ctx.stream.context().load_module(ptx)?;
            let function = module.load_function("add")?;
            Ok(Box::new(Prepared {
                dim: a,
                _module: module,
                function,
                inputs: [ctx.inputs[0].ptr, ctx.inputs[1].ptr],
                dest: ctx.dest.ptr,
                n: (ctx.dest.bytes / 4) as u64,
                repeats: if a.is_multiple_of(2) { 2 } else { 1 },
            }))
        }
    }
    #[derive(Debug)]
    struct Matcher;
    impl OpMatcher for Matcher {
        fn egglog_constructor(&self) -> &'static str {
            "LayoutTensorOpExternalHostAdd"
        }
        fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
            AddFunctionalMatcher.metadata_slots()
        }
        fn snippets(&self) -> Vec<EgglogSnippet> {
            static SNIPPETS: std::sync::OnceLock<
                Vec<(luminal::egglog_snippet::SpliceCategory, String)>,
            > = std::sync::OnceLock::new();
            SNIPPETS
                .get_or_init(|| {
                    AddFunctionalMatcher
                        .snippets()
                        .into_iter()
                        .map(|s| {
                            (
                                s.category,
                                s.text.replace(
                                    AddFunctionalMatcher.egglog_constructor(),
                                    self.egglog_constructor(),
                                ),
                            )
                        })
                        .collect()
                })
                .iter()
                .map(|(category, text)| EgglogSnippet {
                    category: *category,
                    text,
                })
                .collect()
        }
        fn extract(&self, _: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
            Box::new(ExternalHostAdd { dps: false })
        }
    }
    #[test]
    fn external_host_topology_changes_cache_graphs_and_recover_from_errors() {
        let mut g = luminal::graph::Graph::new();
        let a = g.tensor(('a', 2), DType::F32);
        let b = g.tensor(('a', 2), DType::F32);
        let out = (a + b).output();
        let mut registry = cuda_registry_without_cublaslt();
        registry.retain(|e| e.constructor() != AddFunctionalMatcher.egglog_constructor());
        registry.push(RegisteredOp::new(
            Box::new(Matcher),
            Box::new(ExternalHostAdd { dps: false }),
        ));
        let mut rt = CudaRuntime::load_with_registry(&g, registry).unwrap();
        rt.bind_dyn_range('a', 2, 30).unwrap();
        rt.search(&Default::default(), &harness_search_options())
            .unwrap();
        let initial = RECORDS.load(Ordering::SeqCst);
        for n in [3, 4, 3, 4, 4] {
            rt.set_dim('a', n);
            rt.set_data(a.id, vec![n as f32; n * 2]);
            rt.set_data(b.id, vec![1f32; n * 2]);
            rt.execute().unwrap();
            assert_eq!(rt.get_f32(out.id).unwrap(), vec![n as f32 + 1.; n * 2]);
        }
        let stats = rt.graph_stats().unwrap();
        assert_eq!(stats.host_captures, 2);
        assert_eq!(stats.instantiations, 2);
        assert_eq!(stats.graph_cache_hits, 2);
        assert_eq!(
            RECORDS.load(Ordering::SeqCst) - initial,
            2,
            "record runs only during capture"
        );
        // More signatures than the capture cache holds. The parent source
        // graphs still refer to the initial 3/4 captures, so those preparations
        // (including their CUDA modules) must survive cache eviction.
        for n in 9..=20 {
            rt.set_dim('a', n);
            rt.set_data(a.id, vec![n as f32; n * 2]);
            rt.set_data(b.id, vec![1f32; n * 2]);
            rt.execute().unwrap();
            assert_eq!(rt.get_f32(out.id).unwrap(), vec![n as f32 + 1.; n * 2]);
        }
        assert_eq!(DROPPED.load(Ordering::SeqCst) & ((1 << 3) | (1 << 4)), 0);
        rt.set_data(a.id, vec![4f32; 8]);
        rt.set_data(b.id, vec![1f32; 8]);
        rt.set_dim('a', 6);
        assert!(
            format!("{:#}", rt.execute().unwrap_err()).contains("injected preparation failure")
        );
        rt.set_dim('a', 4);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![5f32; 8]);
        let before = rt.graph_stats().unwrap().instantiations;
        rt.set_dim('a', 7);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| rt.execute())).is_err());
        rt.set_dim('a', 4);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![5f32; 8]);
        assert_eq!(rt.graph_stats().unwrap().instantiations, before + 1);
        let captures = rt.graph_stats().unwrap().host_captures;
        rt.set_dim("external_mode", 1);
        rt.execute().unwrap();
        assert_eq!(
            rt.graph_stats().unwrap().host_captures,
            captures + 1,
            "default capture_dims covers newly supplied dimensions too"
        );
        drop(rt);
        assert_eq!(
            DROPPED.load(Ordering::SeqCst) & ((1 << 3) | (1 << 4)),
            (1 << 3) | (1 << 4)
        );
    }
}

#[cfg(feature = "device")]
#[test]
fn external_kernel_launch_geometry_updates_without_reinstantiation() {
    let mut graph = luminal::graph::Graph::new();
    let a = graph.tensor('a', DType::F32);
    let b = graph.tensor('a', DType::F32);
    let out = (a + b).output();
    let mut registry = cuda_registry_without_cublaslt();
    registry.retain(|e| e.constructor() != AddFunctionalMatcher.egglog_constructor());
    registry.push(RegisteredOp::new(
        Box::new(ExternalAddMatcher),
        Box::new(ExternalAdd {
            dps: false,
            marker: 0,
        }),
    ));
    let mut rt = CudaRuntime::load_with_registry(&graph, registry).unwrap();
    rt.bind_dyn_range('a', 0, 1025).unwrap();
    rt.search(&Default::default(), &harness_search_options())
        .unwrap();
    for n in [0, 1025, 2, 0, 257] {
        rt.set_dim('a', n);
        rt.set_data(a.id, vec![2f32; n]);
        rt.set_data(b.id, vec![3f32; n]);
        rt.execute().unwrap();
        assert_eq!(rt.get_f32(out.id).unwrap(), vec![5f32; n]);
    }
    let stats = rt.graph_stats().unwrap();
    assert_eq!(stats.instantiations, 1);
    assert_eq!(stats.kernel_compilations, 1);
    assert!(stats.node_updates >= 4);
}

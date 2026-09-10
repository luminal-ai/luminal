//! External implementations must survive extraction, DPS, and buffer-plan
//! cloning without an entry in a backend-owned dispatch table.

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::bufferize::BufferNode;
use luminal::dtype::DType;
use luminal::egglog_snippet::EgglogSnippet;
use luminal::layout_ir::{AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, ToDps};
use luminal::prelude::FxHashMap;
use luminal_metal::kernels::{CodegenCtx, KernelSource};
use luminal_metal::ops::add::{AddFunctionalDps, AddFunctionalMatcher};
use luminal_metal::{
    KernelOp, MetalOpInterface, MetalRuntime, RegisteredOp, as_kernel_op, harness_search_options,
    metal_registry,
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
            .then(|| MetalOpInterface::kernel::<Self>() as &dyn std::any::Any)
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
            launch.launch = Some(luminal_metal::kernels::KernelLaunch::linear(
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
    let mut registry = metal_registry();
    registry.retain(|entry| entry.constructor() != AddFunctionalMatcher.egglog_constructor());
    registry.push(RegisteredOp::new(
        Box::new(ExternalAddMatcher),
        Box::new(ExternalAdd {
            dps: false,
            marker: 0,
        }),
    ));
    let mut runtime = MetalRuntime::load_with_registry(&graph, registry).unwrap();
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
        }
    }
    assert!(found, "the external operation must survive bufferization");
    #[cfg(target_os = "macos")]
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
    #[cfg(not(target_os = "macos"))]
    let _ = out;
}

#[test]
fn a_familiar_label_without_metal_traits_is_not_claimed() {
    let graph = luminal::graph::Graph::new();
    let registry = vec![RegisteredOp::new(
        Box::new(luminal_reference::ops::AddFunctionalMatcher),
        Box::new(luminal_reference::ops::AddFunctional),
    )];
    let runtime = MetalRuntime::load_with_registry(&graph, registry).unwrap();
    assert!(runtime.active_allow_list().is_empty());
}

#[cfg(target_os = "macos")]
#[test]
fn external_kernel_launch_geometry_updates_reuse_compiled_pipeline() {
    let mut graph = luminal::graph::Graph::new();
    let a = graph.tensor('a', DType::F32);
    let b = graph.tensor('a', DType::F32);
    let out = (a + b).output();
    let mut registry = metal_registry();
    registry.retain(|e| e.constructor() != AddFunctionalMatcher.egglog_constructor());
    registry.push(RegisteredOp::new(
        Box::new(ExternalAddMatcher),
        Box::new(ExternalAdd {
            dps: false,
            marker: 0,
        }),
    ));
    let mut rt = MetalRuntime::load_with_registry(&graph, registry).unwrap();
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
    assert_eq!(stats.kernel_compilations, 1);
}

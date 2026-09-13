//! Bounded contiguous byte views, including projections of packed kernel results.
use super::*;
use luminal::{
    egglog_utils::{
        api::{Rule, SortDef, sort},
        base::{DTYPE, ELIST, EXPRESSION, OP_KIND},
        extract_dtype, extract_expr, extract_expr_list,
    },
    op::*,
};

#[derive(Debug, Clone, Default)]
pub struct BufferView {
    pub offset: Expression,
    pub size: Expression,
    pub dtype: DType,
}

impl KernelOp for BufferView {
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
        // The compiler interface retains a function for resource metadata. This
        // function is never launched: capture emits an empty dependency node.
        let source = "extern \"C\" __global__ void buffer_view_marker() {}".to_string();
        let (module, function) = cache
            .entry(source.clone())
            .or_insert_with(|| {
                let image =
                    crate::compile_module_image_for_current_device(stream.context(), &source)
                        .unwrap();
                let module = stream.context().load_module(image).unwrap();
                let function = module.load_function("buffer_view_marker").unwrap();
                (module, function)
            })
            .clone();
        (
            function,
            module,
            source,
            (1.into(), 1.into(), 1.into()),
            (1.into(), 1.into(), 1.into()),
            0.into(),
            FxHashMap::default(),
        )
    }
    fn output_size(&self) -> Expression {
        self.size
    }
    fn output_dtype(&self) -> DType {
        self.dtype
    }
    fn output_bytes(&self) -> Expression {
        (self.size * self.dtype.bits()).ceil_div(8)
    }
    fn output_aliases_input(&self) -> Option<usize> {
        Some(0)
    }
    fn output_view_range(&self) -> Option<(Expression, Expression)> {
        Some((self.offset, self.output_bytes()))
    }
    fn mutates_aliased_input(&self) -> bool {
        false
    }
    fn collect_dyn_vars_into(&self, vars: &mut FxHashSet<Symbol>) {
        self.offset.collect_dyn_vars_into(vars);
        self.size.collect_dyn_vars_into(vars);
    }
    fn build_params(
        &self,
        _: &Arc<CudaStream>,
        _: u64,
        _: &[u64],
        _: &[CudaSlice<u8>],
        _: u64,
    ) -> Vec<u64> {
        vec![]
    }
    fn kernel_parameter_count(&self, _: usize, _: bool) -> usize {
        0
    }
    fn kernel_name(&self) -> &'static str {
        "BufferView"
    }
}

impl EgglogOp for BufferView {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaBufferView",
            &[
                ("offset_bytes", EXPRESSION),
                ("size", EXPRESSION),
                ("dtype", DTYPE),
            ],
        )
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn egglog_declarations(&self) -> Vec<String> {
        vec![crate::kernel::other_ops::SCATTER_ALIAS_DECLARATION.to_string()]
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw("(rule ((= ?view (Op (CudaBufferView ?offset ?size ?dt) (ICons ?input (INil))))) ((cuda-scatter-alias ?view) (set (dtype ?view) ?dt)) :ruleset post_cleanup)".to_string())]
    }
    fn extract<'a>(
        &'a self,
        egraph: &'a SerializedEGraph,
        kind_children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        _: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(Self {
                offset: extract_expr(egraph, kind_children[0], expressions).unwrap(),
                size: extract_expr(egraph, kind_children[1], expressions).unwrap(),
                dtype: extract_dtype(egraph, kind_children[2]),
            })),
            inputs,
        )
    }
}

/// Preserve reading a shared result when late pointwise fusion subsumes the
/// original FusionStart(FusionEnd(...)) boundary. Lowering is the same pure
/// region input; the separate constructor keeps this equivalent read available.
#[derive(Debug, Default)]
pub struct BufferViewRead;
impl EgglogOp for BufferViewRead {
    fn sort(&self) -> SortDef {
        sort(
            OP_KIND,
            "CudaBufferViewRead",
            &[("shape", ELIST), ("strides", ELIST), ("dtype", DTYPE)],
        )
    }
    fn n_inputs(&self) -> usize {
        1
    }
    fn cleanup(&self) -> bool {
        false
    }
    fn egglog_declarations(&self) -> Vec<String> {
        vec![crate::kernel::other_ops::SCATTER_ALIAS_DECLARATION.to_string()]
    }
    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            r#"
        (rule (
            (= ?view (Op (CudaBufferView ?offset ?size ?dt) ?storage))
            (= ?read (Op (FusionStart ?shape ?strides ?dt) (ICons ?view (INil))))
        ) (
            (union ?read (Op (CudaBufferViewRead ?shape ?strides ?dt) (ICons ?view (INil))))
        ) :ruleset kernel_fuse_late :name "retain shared view reads before pointwise inlining")
        (rule ((= ?read (Op (CudaBufferViewRead ?shape ?strides ?dt) ?input)))
            ((cuda-scatter-alias ?read)) :ruleset post_cleanup)
        "#
            .to_string(),
        )]
    }
    fn extract<'a>(
        &'a self,
        eg: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        inputs: Vec<&'a ENodeId>,
        lists: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expressions: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        (
            LLIROp::new::<dyn KernelOp>(Box::new(fusion::markers::FusionStart {
                shape: extract_expr_list(eg, children[0], lists, expressions).unwrap(),
                strides: extract_expr_list(eg, children[1], lists, expressions).unwrap(),
                dtype: extract_dtype(eg, children[2]),
            })),
            inputs,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{host::DeviceBuffer, runtime::CudaRuntime};
    use cudarc::driver::DevicePtr;

    #[derive(Debug, Clone)]
    struct ViewCustom(BufferView);
    impl CustomOp for ViewCustom {
        fn to_llir_op(&self) -> LLIROp {
            LLIROp::new::<dyn KernelOp>(Box::new(self.0.clone()))
        }
    }
    #[test]
    fn buffer_view_logical_bounds_and_mirrors() {
        let bytes = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let owner = DeviceBuffer::new(0x1000, 8)
            .with_capacity(64)
            .with_host_bytes(&bytes)
            .with_input_format("packed");
        let view = owner.subview(2, 4).unwrap();
        assert_eq!((view.ptr(), view.len(), view.capacity()), (0x1002, 4, 4));
        assert_eq!(view.host_bytes(), Some(&bytes[2..6]));
        assert_eq!(view.input_format(), None);
        assert!(owner.subview(7, 2).is_err());
        assert!(owner.subview(9, 0).is_err());
        assert!(owner.subview(usize::MAX, 1).is_err());
        assert!(DeviceBuffer::new(u64::MAX - 1, 8).subview(2, 1).is_err());
        assert_eq!(owner.subview(8, 0).unwrap().len(), 0);
        assert_eq!(view.subview(1, 2).unwrap().host_bytes(), Some(&bytes[3..5]));
    }

    #[test]
    fn buffer_view_rebinding_reads_live_owner_and_rejects_shrinking_bounds() {
        let mut graph = Graph::new();
        graph.set_dim('n', 16);
        let input = graph.tensor('n').persist();
        let view = graph
            .custom_op(
                ViewCustom(BufferView {
                    offset: 16.into(),
                    size: 4.into(),
                    dtype: DType::F32,
                }),
                input.id,
                4,
                DType::F32,
            )
            .output();
        let stream = CudaContext::new(0).unwrap().default_stream();
        let mut runtime = CudaRuntime::initialize(stream.clone());
        runtime.set_data(input, vec![1.0f32; 16]);
        runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
        runtime.execute(&graph.dyn_map);
        assert_eq!(runtime.get_f32(view), vec![1.0; 4]);
        let bytes = [7.0f32; 16]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let replacement = stream.clone_htod(&bytes).unwrap();
        runtime.set_buffer(input, replacement);
        // No execute: cached launch bindings still point to the old allocation.
        assert_eq!(runtime.get_f32(view), vec![7.0; 4]);
        runtime.execute(&graph.dyn_map);
        assert_eq!(runtime.get_f32(view), vec![7.0; 4]);
        graph.set_dim('n', 6);
        runtime.set_data(input, vec![2.0f32; 6]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                runtime.execute(&graph.dyn_map);
            }))
            .is_err()
        );
    }

    #[test]
    fn buffer_view_dynamic_shared_outputs_and_external_copy() {
        let mut graph = Graph::new();
        graph.set_dim('s', 4);
        graph.set_dim('k', 3);
        let input = graph.tensor(64).persist();
        let owner = input + input;
        let view = graph.custom_op(
            ViewCustom(BufferView {
                offset: Expression::from('k') * 4,
                size: 's'.into(),
                dtype: DType::F32,
            }),
            owner.id,
            's',
            DType::F32,
        );
        let first = view.output();
        let doubled = (view + view).output();
        let tail = graph
            .custom_op(
                ViewCustom(BufferView {
                    offset: 4.into(),
                    size: Expression::from('s') - 1,
                    dtype: DType::F32,
                }),
                view.id,
                Expression::from('s') - 1,
                DType::F32,
            )
            .output();
        let context = CudaContext::new(0).unwrap();
        let stream = context.default_stream();
        let mut runtime = CudaRuntime::initialize(stream.clone());
        runtime.set_data(input, (0..64).map(|i| i as f32).collect::<Vec<_>>());
        runtime = graph.compile(runtime, CompileOptions::default().search_graph_limit(1));
        let external = stream.alloc_zeros::<u8>(64 * 4).unwrap();
        for (iteration, (offset, len)) in [(3, 4), (8, 16), (2, 2), (30, 8), (1, 1), (4, 4)]
            .into_iter()
            .enumerate()
        {
            graph.set_dim('k', offset);
            graph.set_dim('s', len);
            let values = (0..64)
                .map(|i| (i + iteration * 100) as f32)
                .collect::<Vec<_>>();
            runtime.set_data(input, values.clone());
            if iteration == 1 {
                unsafe {
                    runtime.set_output_device_ptr(
                        first,
                        external.device_ptr(&stream).0,
                        external.len(),
                    );
                }
            }
            if iteration == 4 {
                runtime.clear_output_device_ptr(first);
            }
            runtime.execute(&graph.dyn_map);
            let expected = values[offset..offset + len]
                .iter()
                .map(|x| x * 2.)
                .collect::<Vec<_>>();
            assert_eq!(runtime.get_f32(first), expected);
            assert_eq!(
                runtime.get_f32(doubled),
                expected.iter().map(|x| x * 2.).collect::<Vec<_>>()
            );
            assert_eq!(runtime.get_f32(tail), expected[1..]);
            if (1..4).contains(&iteration) {
                let bytes = stream.clone_dtoh(&external).unwrap();
                let got = bytes[..len * 4]
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|x| f32::from_le_bytes(*x))
                    .collect::<Vec<_>>();
                assert_eq!(got, expected);
                assert!(!runtime.output_is_zero_copy(first));
            }
        }
        let owned = runtime.remove_buffer(first);
        assert_eq!(owned.len(), 4 * 4);
        runtime.execute(&graph.dyn_map);
        assert_eq!(runtime.get_f32(first), vec![1008., 1010., 1012., 1014.]);
    }
}

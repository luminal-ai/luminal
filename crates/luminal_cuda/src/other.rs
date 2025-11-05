use std::{marker::PhantomData, sync::Arc};

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use itertools::Itertools;
use luminal::prelude::{petgraph::visit::EdgeRef, *};
use rustc_hash::FxHashMap;

use crate::{
    binary::CudaSub,
    compile_and_load_kernel, constant,
    prim::{CudaAdd, CudaContiguous, CudaCopyFromDevice, CudaCopyToDevice, CudaSumReduce},
    CudaData, CudaFloat,
};

#[derive(Clone)]
pub struct CudaARange<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    pub size: Expression,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}
crate::debug_type!(CudaARange);

impl<T: CudaFloat> CudaARange<T> {
    pub fn new(
        device: Arc<CudaContext>,
        size: Expression,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, int n_elements) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        out[idx] = ({type_name})idx;
    }}
}}"
        );
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            size,
            _phantom: Default::default(),
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaARange<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let n_elements = self
            .size
            .exec(unsafe { self.dyn_map.as_ref().unwrap() })
            .unwrap();
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(n_elements).unwrap();
        let mut launch_args = stream.launch_builder(&self.function);
        let n_elements_int = n_elements as i32;
        launch_args.arg(&mut out);
        launch_args.arg(&n_elements_int);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(n_elements as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Debug, Default)]
pub struct ARangeCompiler<T: CudaFloat>(PhantomData<T>);

impl<T: CudaFloat> Compiler for ARangeCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, _: To) {
        let dev = crate::shared_cuda_context(0);
        // TODO: Make sure this actually checks the shape transformations to ensure pooling happens
        let contig_one = constant::<T>(1.);
        let contig1 = unary::<CudaContiguous<T>>(contig_one.clone());
        let sum_reduce =
            unary::<CudaSumReduce<T>>(unary::<CudaContiguous<T>>(unary::<CudaContiguous<T>>(
                unary::<CudaContiguous<T>>(contig1.clone()),
            )));
        let sub = binary::<CudaSub<T>>(sum_reduce.clone(), constant::<T>(1.));
        let mut s1 = sub.clone().search(graph);
        let neg_one = constant::<T>(-1.);
        let add = binary::<CudaAdd<T>>(sum_reduce, neg_one.clone());
        let mut s2 = add.clone().search(graph);

        while s1.next_match() || s2.next_match() {
            let s = if s1.matched { &s1 } else { &s2 };
            let arange_amount = {
                let sh = graph
                    .edges_connecting(s.get(&contig_one), s.get(&contig1))
                    .next()
                    .unwrap()
                    .weight()
                    .as_data()
                    .unwrap()
                    .2;
                sh.dims[sh.indexes[sh.len() - 1]]
            };
            let arange_op = graph
                .add_op(CudaARange::<T>::new(
                    dev.clone(),
                    arange_amount.into(),
                    &graph.dyn_map,
                ))
                .finish();
            let fin = if s1.matched {
                s1.get(&sub)
            } else {
                s2.get(&add)
            };
            move_outgoing_edge(fin, arange_op, graph);
            graph.remove_node(fin);
            s.try_delete();
        }
    }
}

// Sometimes CopyTo -> CopyFrom and CopyFrom -> CopyTo patterns remain, so let's clean them up
#[derive(Debug, Default)]
pub struct CopyCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for CopyCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        for (first, second) in graph
            .edge_indices()
            .filter_map(|e| graph.edge_endpoints(e))
            .filter(|(a, b)| {
                (graph
                    .node_weight(*a)
                    .unwrap()
                    .as_any()
                    .is::<CudaCopyToDevice<T>>()
                    && graph
                        .node_weight(*b)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>())
                    || (graph
                        .node_weight(*a)
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && graph
                            .node_weight(*b)
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>())
            })
            .unique_by(|n| n.0)
            .unique_by(|n| n.1)
            .collect::<Vec<_>>()
        {
            if graph
                .edges_directed(first, petgraph::Direction::Outgoing)
                .filter(|e| graph.contains_node(e.target()))
                .filter(|e| {
                    !graph
                        .node_weight(e.target())
                        .unwrap()
                        .as_any()
                        .is::<CudaCopyFromDevice<T>>()
                        && !graph
                            .node_weight(e.target())
                            .unwrap()
                            .as_any()
                            .is::<CudaCopyToDevice<T>>()
                })
                .count()
                > 0
                || graph.no_delete.contains(&first)
            {
                continue;
            }
            let source = graph.get_sources(first)[0];
            move_outgoing_edge(second, source.0, graph);
            remap(second, source.0, &mut ids, graph);
            graph.remove_node(second);
            for dest in graph
                .get_dests(first)
                .iter()
                .map(|(i, _)| *i)
                .collect::<Vec<_>>()
            {
                move_outgoing_edge(dest, source.0, graph);
                remap(dest, source.0, &mut ids, graph);
                graph.remove_node(dest);
            }
            graph.remove_node(first);
        }
    }
}

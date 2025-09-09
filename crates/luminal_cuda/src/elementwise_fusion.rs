use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{any::Any, fmt::Debug, iter::once, marker::PhantomData, mem::size_of, sync::Arc};

use itertools::Itertools;
use luminal::prelude::{
    petgraph::{visit::EdgeRef, Direction},
    *,
};

use crate::{
    compile_and_load_kernel, expr_to_cuda_string, get_buffer_from_tensor, prim::CudaConstant,
    CudaData, CudaFloat,
};

use super::{input_dyn_dims, render_dyn_dim_inputs};

#[derive(Default, Debug)]
pub struct ElementwiseFusionCompiler<T>(PhantomData<T>);

fn get_inputs(node: NodeIndex, graph: &Graph) -> Vec<(NodeIndex, u8, ShapeTracker)> {
    graph
        .edges_directed(node, Direction::Incoming)
        .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
        .sorted_by_key(|(_, i)| i.0)
        .map(|(a, (_, b, c))| (a, b, c))
        .collect()
}

// Check if we stack the views, does more than one view exist for one of a set of given inputs
fn is_more_than_one_view(
    subexpressions: &[(String, ShapeTracker)],
    subexp_indexes: &[usize],
) -> bool {
    let intermediate_match = Regex::new(r"intermediate(\d+)").unwrap();
    let mut subexp_views = subexpressions
        .iter()
        .map(|(_, sh)| vec![*sh])
        .collect::<Vec<_>>();
    for i in (0..subexp_views.len()).rev() {
        for capture in intermediate_match.captures_iter(&subexpressions[i].0) {
            let index = capture.get(1).unwrap().as_str().parse::<usize>().unwrap();
            if subexp_views[index].len() == 1 {
                let v = subexp_views[i].clone();
                subexp_views[index].extend(v);
            }
        }
    }
    if !subexpressions
        .iter()
        .positions(|(s, _)| {
            subexp_indexes
                .iter()
                .any(|i| s.contains(&format!("input{i}")))
        })
        .map(|subexp_index| &subexp_views[subexp_index])
        .all_equal()
    {
        return true;
    }
    false
}

impl<T: CudaFloat> Compiler for ElementwiseFusionCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let device = CudaContext::new(0).unwrap();
        // Track fused ops to compile later
        let mut fused_ops = FxHashSet::default();

        let mut matched = true;
        let mut elementwise_ops = FxHashMap::default();
        for op in graph.node_indices().collect::<Vec<_>>() {
            if let Some(exp) = graph.node_custom::<String, _>(op, "elementwise", ()) {
                elementwise_ops.insert(op, exp);
            }
        }
        let mut intermediate_regexes = FxHashMap::default();
        let mut input_regexes = FxHashMap::default();
        let mut n_fused_ops = 0;
        while matched {
            matched = false;
            for edge in graph.edge_indices().collect::<Vec<_>>() {
                let Some((a, b)) = graph.edge_endpoints(edge) else {
                    continue;
                };
                if graph.no_delete.contains(&a)
                    || graph.no_delete.contains(&b)
                    || (!graph.check_node_type::<CudaConstant<T>>(a)
                        && graph
                            .edges_directed(a, Direction::Outgoing)
                            .filter(|e| e.target() != b)
                            .count()
                            > 0)
                {
                    continue; // A is not a constant and is feeding into some other node
                }
                let (Some(expression_a), Some(expression_b)) =
                    (elementwise_ops.get(&a), elementwise_ops.get(&b))
                else {
                    continue;
                };
                // a and b are elementwise ops
                // Make sure all edges from a to b share the same shape
                if !graph
                    .edges_connecting(a, b)
                    .map(|e| e.weight().as_data().unwrap().2)
                    .all_equal()
                {
                    continue;
                }
                // Check if there are more than one view of this input. If so, we can't merge
                let mut subexpressions_b = graph
                    .try_get_op::<FusedElementwiseOp<T>>(b)
                    .map(|o| o.subexpressions.clone())
                    .unwrap_or_else(|| vec![(expression_b.clone(), ShapeTracker::new(()))]);
                let a_to_b_indexes = graph
                    .edges_connecting(a, b)
                    .map(|e| e.weight().as_data().unwrap().0 as usize)
                    .sorted()
                    .collect::<Vec<_>>();
                if is_more_than_one_view(&subexpressions_b, &a_to_b_indexes) {
                    continue;
                }
                matched = true;
                let a_inputs = get_inputs(a, graph);
                let mut b_inputs = get_inputs(b, graph);
                let (_, _, connecting_shape) = b_inputs.remove(*a_to_b_indexes.last().unwrap());
                for i in a_to_b_indexes.iter().take(a_to_b_indexes.len() - 1).rev() {
                    b_inputs.remove(*i);
                }
                // Get subexpressions
                let mut subexpressions_a = graph
                    .try_get_op::<FusedElementwiseOp<T>>(a)
                    .map(|o| o.subexpressions.clone())
                    .unwrap_or_else(|| vec![(expression_a.clone(), ShapeTracker::new(()))]);
                subexpressions_a.last_mut().unwrap().1 = connecting_shape;
                // Re-reference b intermediates
                for i in (0..subexpressions_b.len()).rev() {
                    let re = if let Some(r) = intermediate_regexes.get(&i) {
                        r
                    } else {
                        intermediate_regexes.insert(
                            i,
                            Regex::new(&format!(r"intermediate{i}([^0-9]|$)")).unwrap(),
                        );
                        intermediate_regexes.get(&i).unwrap()
                    };
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = re
                            .replace_all(
                                exp,
                                format!("intermediate{}$1", i + subexpressions_a.len()),
                            )
                            .to_string();
                    }
                }
                // Re-reference b inputs to a
                for index in &a_to_b_indexes {
                    let re = if let Some(r) = input_regexes.get(index) {
                        r
                    } else {
                        input_regexes.insert(
                            *index,
                            Regex::new(&format!(r"input{index}([^0-9]|$)")).unwrap(),
                        );
                        input_regexes.get(index).unwrap()
                    };
                    for (exp, _) in subexpressions_b.iter_mut() {
                        *exp = re
                            .replace_all(
                                exp,
                                format!("intermediate{}$1", subexpressions_a.len() - 1),
                            )
                            .to_string();
                    }
                }
                // Re-reference b inputs
                for (sub_factor, index) in a_to_b_indexes.iter().enumerate() {
                    for i in (*index - sub_factor + 1)..(b_inputs.len() + a_to_b_indexes.len()) {
                        let re = if let Some(r) = input_regexes.get(&i) {
                            r
                        } else {
                            input_regexes
                                .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                            input_regexes.get(&i).unwrap()
                        };
                        for (exp, _) in subexpressions_b.iter_mut() {
                            *exp = re.replace_all(exp, format!("input{}$1", i - 1)).to_string();
                        }
                    }
                }
                // Combine inputs for a and b
                for i in (0..a_inputs.len()).rev() {
                    // Re-reference the a inputs
                    let re = if let Some(r) = input_regexes.get(&i) {
                        r
                    } else {
                        input_regexes
                            .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                        input_regexes.get(&i).unwrap()
                    };
                    for (exp, _) in subexpressions_a.iter_mut() {
                        *exp = re
                            .replace_all(exp, format!("input{}$1", i + b_inputs.len()))
                            .to_string();
                    }
                }
                b_inputs.extend(a_inputs);
                // a intermediates should remain valid
                // Combine subexpressions
                for subexp in subexpressions_a.into_iter().rev() {
                    subexpressions_b.insert(0, subexp);
                }
                // Create new fused op
                let output_buffer_sizes =
                    if let Some(o) = graph.try_get_op::<FusedElementwiseOp<T>>(b) {
                        o.output_buffer_sizes.clone()
                    } else {
                        vec![
                            graph
                                .edges_directed(b, Direction::Incoming)
                                .filter_map(|e| e.weight().as_data().map(|i| i.2.n_elements()))
                                .reduce(|acc, e| acc.max(e))
                                .unwrap()
                                * size_of::<T>(),
                        ]
                    };
                let new_op = graph
                    .add_op(FusedElementwiseOp::<T> {
                        kernel: None,
                        dyn_map: &graph.dyn_map,
                        dyn_chars: vec![],
                        subexpressions: subexpressions_b.clone(),
                        device: device.clone(),
                        output_buffer_sizes,
                        _phantom: Default::default(),
                    })
                    .finish();
                // Add edges to new op
                move_outgoing_edge(b, new_op, graph);
                for (i, (node, output, shape)) in b_inputs.into_iter().enumerate() {
                    graph.add_edge(
                        node,
                        new_op,
                        Dependency::Data {
                            input_order: i as u8,
                            output_order: output,
                            shape,
                        },
                    );
                }
                graph.remove_node(b);
                graph.safe_remove_node(a, 0);
                // Keep track of the fused op so we can compile it later
                fused_ops.remove(&a);
                fused_ops.remove(&b);
                fused_ops.insert(new_op);
                elementwise_ops.remove(&a);
                elementwise_ops.remove(&b);
                elementwise_ops.insert(new_op, String::new());
                if !graph.contains_node(a) {
                    remap(a, new_op, &mut ids, graph);
                }
                remap(b, new_op, &mut ids, graph);
                n_fused_ops += 1;
            }
        }
        // Compile all the kernels we placed
        let type_name = T::type_name();
        let intermediate_match = Regex::new(r"intermediate(\d+)([^0-9]|$)").unwrap();
        let mut bar = None;
        if debug() {
            println!("Fusing {n_fused_ops} ops into {} ops...", fused_ops.len());
            let b = ProgressBar::new(fused_ops.len() as u64);
            b.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.bright.blue/white}] {pos:>7}/{len:7}",
                )
                .unwrap()
                .progress_chars("##-"),
            );
            bar = Some(b);
        };
        for fused_op in fused_ops {
            let inputs = graph
                .edges_directed(fused_op, Direction::Incoming)
                .flat_map(|e| e.weight().as_data())
                .sorted_by_key(|(i, _, _)| *i)
                .map(|(_, _, sh)| sh)
                .collect::<Vec<_>>();
            let op = graph.get_op_mut::<FusedElementwiseOp<T>>(fused_op);
            // Stack index expressions and replace them in the subexpressions
            // Track all shapes used, will pull dyn dims from these
            let shapes_used = op
                .subexpressions
                .iter()
                .map(|(_, s)| *s)
                .chain(inputs.clone())
                .collect::<Vec<_>>();
            // Track the views of each subexpression by going in reverse order and appending the current subexpression's views to the referenced subexpression
            let mut subexp_views = op
                .subexpressions
                .iter()
                .map(|(_, sh)| vec![*sh]) // Start with the current view for this subexpression
                .collect::<Vec<_>>();
            for i in (0..subexp_views.len() - 1).rev() {
                for capture in intermediate_match.captures_iter(&op.subexpressions[i].0) {
                    let index = capture.get(1).unwrap().as_str().parse::<usize>().unwrap();
                    if subexp_views[index].len() == 1 {
                        let v = subexp_views[i].clone();
                        subexp_views[index].extend(v);
                    } else {
                        assert_eq!(subexp_views[index][1..], subexp_views[i][..]);
                    }
                }
            }
            // Stack views for each input by going to the first subexpression that uses it and combining it's stacked shape with the input's shape
            let stacked_shapes: Vec<Vec<ShapeTracker>> = inputs
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    // Find the first subexpression that uses this input
                    let re = if let Some(r) = input_regexes.get(&i) {
                        r
                    } else {
                        input_regexes
                            .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                        input_regexes.get(&i).unwrap()
                    };
                    let using_subexp = op
                        .subexpressions
                        .iter()
                        .position(|(s, _)| re.is_match(s))
                        .unwrap();

                    once(*s)
                        .chain(
                            subexp_views[using_subexp]
                                .iter()
                                .copied()
                                .filter(|s| !s.is_empty()),
                        )
                        .collect()
                })
                .collect();
            // Stack index expressions
            let stacked_index_expressions_partial = stacked_shapes
                .iter()
                .map(|s| {
                    s.iter()
                        .rev()
                        .take(s.len() - 1)
                        .fold(Expression::from('z'), |acc, inp| {
                            inp.index_expression().substitute('z', acc)
                        })
                })
                .collect::<Vec<_>>();
            let stacked_index_expressions = stacked_index_expressions_partial
                .iter()
                .cloned()
                .zip(&stacked_shapes)
                .map(|(partial, sh)| sh[0].index_expression().substitute('z', partial))
                .collect::<Vec<_>>();
            let stacked_valid_expressions = stacked_index_expressions_partial
                .iter()
                .cloned()
                .zip(&stacked_shapes)
                .map(|(partial, sh)| sh[0].valid_expression().substitute('z', partial))
                .collect::<Vec<_>>();

            // Replace in subexpressions
            let n_subexpressions = op.subexpressions.len();
            for (i, ((subexp, _), stacked_shapes)) in
                op.subexpressions.iter_mut().zip(subexp_views).enumerate()
            {
                // Index
                for (i, (ind_exp, val_exp)) in stacked_index_expressions
                    .iter()
                    .zip(&stacked_valid_expressions)
                    .enumerate()
                {
                    let re = if let Some(r) = input_regexes.get(&i) {
                        r
                    } else {
                        input_regexes
                            .insert(i, Regex::new(&format!(r"input{i}([^0-9]|$)")).unwrap());
                        input_regexes.get(&i).unwrap()
                    };
                    *subexp = re
                        .replace_all(
                            subexp,
                            &if *val_exp != true {
                                format!(
                                    "({} != 0 ? (float)input{i}[{}] : 0.0)$1",
                                    expr_to_cuda_string(val_exp),
                                    expr_to_cuda_string(ind_exp)
                                )
                            } else {
                                format!("(float)input{i}[{}]$1", expr_to_cuda_string(ind_exp))
                            },
                        )
                        .to_string();
                }
                // Valid (not on last subexpression)
                if i != n_subexpressions - 1 {
                    let val_exp = stacked_shapes
                        .iter()
                        .rev()
                        .fold(
                            (Expression::from(true), Expression::from('z')),
                            |(_, ind_acc), inp| {
                                (
                                    inp.valid_expression().substitute('z', ind_acc.clone()),
                                    inp.index_expression().substitute('z', ind_acc),
                                )
                            },
                        )
                        .0;
                    if val_exp != true {
                        *subexp = format!(
                            "(({} != 0) ? {subexp} : 0.0)",
                            expr_to_cuda_string(&val_exp)
                        );
                    }
                }
            }

            let (dyn_chars, rendered) = render_dyn_dim_inputs(&shapes_used);
            let kernel = format!(
                "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({} {type_name}* out, const int n_elements{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elements) {{
        {}
        out[idx] = ({type_name})({});
    }}
}}",
                (0..inputs.len())
                    .map(|inp_ind| format!("const {type_name}* input{inp_ind},"))
                    .collect::<Vec<_>>()
                    .join(" "),
                op.subexpressions
                    .iter()
                    .take(op.subexpressions.len() - 1)
                    .enumerate()
                    .map(|(i, (subexp, _))| format!("float intermediate{i} = {subexp};"))
                    .join("\n        "),
                op.subexpressions.last().unwrap().0
            );
            op.kernel = Some(compile_and_load_kernel(kernel, &device));
            op.dyn_chars = dyn_chars;

            if let Some(bar) = &bar {
                bar.inc(1);
            }
        }
        if let Some(bar) = bar {
            bar.finish();
        }
    }
}

#[derive(Clone)]
pub struct FusedElementwiseOp<T> {
    kernel: Option<CudaFunction>,
    dyn_map: *const FxHashMap<char, usize>,
    dyn_chars: Vec<char>,
    subexpressions: Vec<(String, ShapeTracker)>,
    device: Arc<CudaContext>,
    output_buffer_sizes: Vec<Expression>,
    _phantom: PhantomData<T>,
}
impl<T> Debug for FusedElementwiseOp<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FusedElementwiseOp")
    }
}

impl<T: CudaFloat> Operator for FusedElementwiseOp<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let dyn_map = unsafe { self.dyn_map.as_ref().unwrap() };
        let out_size =
            self.output_buffer_sizes[0].exec(dyn_map).unwrap() / std::mem::size_of::<T>();
        let out_size_int = out_size as i32;
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(out_size).unwrap();

        let mut launch_args = stream.launch_builder(self.kernel.as_ref().unwrap());

        for (buf, _) in &tensors {
            launch_args.arg(get_buffer_from_tensor::<T>(buf));
        }
        launch_args.arg(&mut out);
        launch_args.arg(&out_size_int);
        input_dyn_dims(&mut launch_args, &self.dyn_chars, self.dyn_map);

        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(out_size as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::<String>::default());
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use luminal::{
        prelude::{binary::F32Pow, *},
        tests::{assert_close, assert_close_precision, random_vec, random_vec_rng},
    };
    use luminal_nn::*;
    use rand::{rngs::StdRng, SeedableRng};

    use crate::CudaCompiler;

    #[test]
    fn test_fusion_simple() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let inp = cx.tensor(5).set(random_vec_rng(10, &mut rng));
        let mut out = inp.exp2().cos().sqrt().retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }
    #[test]
    fn test_fusion_binary() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let a = cx.tensor(5).set(random_vec_rng(10, &mut rng));
        let b = cx.tensor(5).set(random_vec_rng(10, &mut rng));
        let mut out = (a.exp2() + b.cos()).retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_subexpression_complex() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("a", 10).set(random_vec(10)).keep();
        let b = cx.named_tensor("b", 10).set(random_vec(10)).keep();
        let d = cx.named_tensor("d", 10).set(random_vec(10)).keep();
        let mut out = ((a.exp2() - b.sin()).sin() * 3.4).lt(d).retrieve();

        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_slicing_padding() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let inp = random_vec_rng(10, &mut rng);
        let a = cx.named_tensor("a", (2, 5)).set(inp);
        let mut padded = a
            .slice((..Expression::from(1), ..))
            .cos()
            .pad(((0, 1), (0, 0)))
            .exp2()
            .retrieve();
        cx.execute();
        let unopt_out = padded.data();
        padded.drop();

        cx.compile(
            <(GenericCompiler, CudaCompiler<f16>)>::default(),
            &mut padded,
        );
        cx.execute();

        assert_close(&padded.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_subexpression() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        let data = random_vec_rng(10, &mut rng);
        let a = cx.tensor((2, 5)).set(data);
        let mut out = (a.sqrt().exp() + a.sqrt().sin()).retrieve();
        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f32>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rope_emb() {
        let mut cx = Graph::new();
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        let freqs = (cx.arange(HEAD_DIM / 2) * 2.0) / (HEAD_DIM as f32);
        let freqs = 1000000_f32.pow(freqs);
        let pos = cx.arange(SEQ) + Expression::from(0);
        let mut emb = pos
            .expand_dim(1, 1)
            .matmul(freqs.expand_dim(0, 1))
            .retrieve();

        cx.execute();
        let unopt_out = emb.data();
        emb.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut emb);
        cx.execute();
        assert_close(&emb.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rotate() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        const HEAD_DIM_OVER_2: usize = HEAD_DIM / 2;
        let a = cx
            .tensor((SEQ, HEAD_DIM))
            .set(random_vec_rng(SEQ * HEAD_DIM, &mut rng))
            .keep();
        let b = cx
            .tensor((SEQ, HEAD_DIM_OVER_2, 1))
            .set(random_vec_rng(SEQ * (HEAD_DIM) / 2, &mut rng))
            .keep();
        // Split input into evens and odds
        let split = a.reshape((SEQ, HEAD_DIM / 2, 2));
        let x0 = split.slice((.., .., ..Expression::from(1)));
        let x1 = split.slice((.., .., Expression::from(1)..));

        let x0_out = x0 * b - x1 * b.cos();
        let x1_out = x0 + x1;

        // Combine back into output
        let mut out = x0_out
            .concat_along(x1_out, 2)
            .reshape((SEQ, HEAD_DIM))
            .retrieve();
        cx.execute();

        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();
        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_rope_full() {
        let mut cx = Graph::new();
        let mut rng = StdRng::seed_from_u64(0);
        const BATCH: usize = 1;
        const N_HEADS: usize = 8;
        const SEQ: usize = 2;
        const HEAD_DIM: usize = 4;
        let a = cx
            .named_tensor("a", (BATCH, N_HEADS, SEQ, HEAD_DIM))
            .set(random_vec_rng(BATCH * N_HEADS * SEQ * HEAD_DIM, &mut rng))
            .keep();
        let freqs = (cx.arange(HEAD_DIM / 2) * 2.0) / (HEAD_DIM as f32);
        let freqs = 1000000_f32.pow(freqs);
        let pos = cx.arange(SEQ) + 0;
        let emb = pos.expand_dim(1, 1).matmul(freqs.expand_dim(0, SEQ));
        // Split input into evens and odds
        let split = a.reshape((BATCH, N_HEADS, SEQ, HEAD_DIM / 2, 2));
        let x0 = split.slice((.., .., .., .., ..1)).contiguous();
        let x1 = split.slice((.., .., .., .., 1..)).contiguous();

        // Apply sin and cos embeddings
        let x0_out = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
        let x1_out = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

        // Combine back into output
        let mut out = x0_out
            .concat_along(x1_out, 4)
            .reshape((BATCH, N_HEADS, SEQ, HEAD_DIM))
            .retrieve();
        cx.execute();
        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close(&out.data(), &unopt_out);
    }

    #[test]
    fn test_fusion_transformer() {
        pub const HIDDEN_DIM: usize = 128;
        pub const N_HEADS: usize = 2;
        pub const N_KV_HEADS: usize = 2;
        pub const MLP_DIM: usize = 256;
        pub const NUM_LAYERS: usize = 2;
        pub const SEQ_LEN: usize = 65;
        pub const N_ATTENTION_GROUPS: usize = N_HEADS / N_KV_HEADS;
        pub const HEAD_DIM: usize = HIDDEN_DIM / N_HEADS;
        pub const ATTN_PROJ_DIM: usize = HEAD_DIM * N_KV_HEADS;
        pub type KVCache = (GraphTensor, GraphTensor);

        pub struct Mlp {
            pub gate_proj: Linear, // hidden -> intermediate
            pub down_proj: Linear, // intermediate -> hidden
            pub up_proj: Linear,   // hidden -> intermediate
        }

        impl Module<GraphTensor> for Mlp {
            type Output = GraphTensor;

            fn forward(&self, input: GraphTensor) -> Self::Output {
                let gate = self.gate_proj.forward(input).swish();
                let up = self.up_proj.forward(input) * gate;
                self.down_proj.forward(up)
            }
        }

        impl Mlp {
            pub fn new(hidden: usize, intermediate: usize, cx: &mut Graph) -> Self {
                Self {
                    gate_proj: Linear::new_permuted(hidden, intermediate, false, cx),
                    down_proj: Linear::new_permuted(intermediate, hidden, false, cx),
                    up_proj: Linear::new_permuted(hidden, intermediate, false, cx),
                }
            }
        }

        impl SerializeModule for Mlp {
            fn serialize(&self, s: &mut Serializer) {
                s.module("ffn_gate", &self.gate_proj);
                s.module("ffn_up", &self.up_proj);
                s.module("ffn_down", &self.down_proj);
            }
        }

        fn apply_rotary_embeddings_ggml(input: GraphTensor, prev_seq: Expression) -> GraphTensor {
            assert_eq!(input.shape.len(), 4); // batch, n_heads, seq, head_dim
            let (batch, n_heads, seq, head_dim) = input.dims4();
            // Get freqs
            let freqs =
                (input.graph().arange(head_dim / 2) * 2.0) / (head_dim.to_usize().unwrap() as f32);
            let freqs = 500_000_f32.pow(freqs);
            let pos = input.graph().arange(seq) + prev_seq;
            let emb = pos.expand_dim(1, 1).matmul(freqs.expand_dim(0, seq));

            // Split input into evens and odds
            let split = input.reshape((batch, n_heads, seq, head_dim / 2, 2));
            let x0 = split.slice((.., .., .., .., ..1));
            let x1 = split.slice((.., .., .., .., 1..));

            // Apply sin and cos embeddings
            let x0_out = x0 * emb.cos().expand(x0.shape) - x1 * emb.sin().expand(x1.shape);
            let x1_out = x0 * emb.sin().expand(x0.shape) + x1 * emb.cos().expand(x1.shape);

            // Combine back into output
            x0_out.concat_along(x1_out, 4).reshape(input.shape)
        }

        pub struct SelfAttention {
            pub q_proj: GraphTensor, // Hidden -> hidden
            pub k_proj: GraphTensor, // Proj dim -> hidden
            pub v_proj: GraphTensor, // Proj dim -> hidden
            pub o_proj: GraphTensor, // Hidden -> hidden
        }

        impl Module<(GraphTensor, KVCache)> for SelfAttention {
            type Output = (GraphTensor, KVCache);
            fn forward(&self, (x, (k_cache, v_cache)): (GraphTensor, KVCache)) -> Self::Output {
                // x: batch, seq, hidden
                let (batch, seq, _) = x.dims3();
                let (_, _, prev_seq, _) = k_cache.dims4();
                // Apply the Projections
                let queries = x
                    .matmul(self.q_proj.permute((1, 0)))
                    .reshape((batch, seq, N_HEADS, HEAD_DIM))
                    .permute((0, 2, 1, 3));

                let keys = x
                    .matmul(self.k_proj.permute((1, 0)))
                    .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
                    .permute((0, 2, 1, 3));

                let values = x
                    .matmul(self.v_proj.permute((1, 0)))
                    .reshape((batch, seq, N_KV_HEADS, HEAD_DIM))
                    .permute((0, 2, 1, 3));

                // Rotary embed queries and keys
                let queries = apply_rotary_embeddings_ggml(queries, prev_seq);
                let keys = apply_rotary_embeddings_ggml(keys, prev_seq);

                // Add KV cache
                let keys = k_cache.concat_along(keys, 2);
                let values = v_cache.concat_along(values, 2);

                // Repeat the KV States for Grouped-Query Attention
                let repeated_keys = keys.expand_dim(2, N_ATTENTION_GROUPS);
                let repeated_values = values.expand_dim(2, N_ATTENTION_GROUPS);

                // Calculate attention weights
                let mut attention_weights = queries
                    .reshape((batch, N_KV_HEADS, N_ATTENTION_GROUPS, seq, HEAD_DIM)) // Split query heads into groups
                    .matmul(repeated_keys.permute((0, 1, 2, 4, 3)))
                    / (HEAD_DIM as f32).sqrt();

                let attention_mask = self.k_proj.graph().triu(seq, 1) * f16::MIN.to_f32();
                attention_weights += attention_mask
                    .pad(((0, 0), (prev_seq, 0)))
                    .expand_dim(0, batch)
                    .expand_dim(1, N_KV_HEADS)
                    .expand_dim(2, N_ATTENTION_GROUPS);

                // Calculate final outputs
                let output = attention_weights
                    .softmax(4)
                    // Apply distribution to values
                    .matmul(repeated_values)
                    // Merge heads
                    .permute((0, 3, 1, 2, 4))
                    .reshape((batch, seq, HIDDEN_DIM));
                let output = output
                    // Apply output projection
                    .matmul(self.o_proj.permute((1, 0)));
                (output, (keys.contiguous(), values.contiguous())) // Cache needs to be contiguous for transferring to another graph
            }
        }

        impl SelfAttention {
            pub fn new(cx: &mut Graph) -> Self {
                Self {
                    q_proj: cx.named_tensor("Q Proj", (HIDDEN_DIM, HIDDEN_DIM)),
                    k_proj: cx.named_tensor("K Proj", (ATTN_PROJ_DIM, HIDDEN_DIM)),
                    v_proj: cx.named_tensor("V Proj", (ATTN_PROJ_DIM, HIDDEN_DIM)),
                    o_proj: cx.named_tensor("O Proj", (HIDDEN_DIM, HIDDEN_DIM)),
                }
            }

            fn initialize(self) -> Self {
                self.k_proj.set(random_vec(
                    self.k_proj.shape.n_elements().to_usize().unwrap(),
                ));
                self.o_proj.set(random_vec(
                    self.o_proj.shape.n_elements().to_usize().unwrap(),
                ));
                self.v_proj.set(random_vec(
                    self.v_proj.shape.n_elements().to_usize().unwrap(),
                ));
                self.q_proj.set(random_vec(
                    self.q_proj.shape.n_elements().to_usize().unwrap(),
                ));
                self
            }
        }

        impl SerializeModule for SelfAttention {
            fn serialize(&self, s: &mut Serializer) {
                s.tensor("attn_q/weight", self.q_proj);
                s.tensor("attn_v/weight", self.v_proj);
                s.tensor("attn_k/weight", self.k_proj);
                s.tensor("attn_output/weight", self.o_proj);
            }
        }

        pub struct TransformerBlock {
            pub attention: SelfAttention,
            pub attention_norm: LayerNorm,
            pub feed_forward: Mlp,
            pub feed_forward_norm: LayerNorm,
        }

        impl Module<(GraphTensor, KVCache)> for TransformerBlock {
            type Output = (GraphTensor, KVCache);
            fn forward(&self, (mut x, cache): (GraphTensor, KVCache)) -> Self::Output {
                // Attention
                let (y, cache) = self
                    .attention
                    .forward((self.attention_norm.forward(x), cache));

                // Residual Addition
                x += y;

                // Feed Forward
                let y = self.feed_forward.forward(self.feed_forward_norm.forward(x));

                // Residual Addition
                (x + y, cache)
            }
        }

        impl TransformerBlock {
            pub fn new(cx: &mut Graph) -> Self {
                Self {
                    attention: SelfAttention::new(cx),
                    attention_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
                    feed_forward: Mlp::new(HIDDEN_DIM, MLP_DIM, cx),
                    feed_forward_norm: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
                }
            }

            fn initialize(mut self) -> Self {
                self.attention_norm = self.attention_norm.initialize();
                self.feed_forward_norm = self.feed_forward_norm.initialize();
                self.attention = self.attention.initialize();
                self.feed_forward.down_proj = self.feed_forward.down_proj.init_rand();
                self.feed_forward.up_proj = self.feed_forward.up_proj.init_rand();
                self.feed_forward.gate_proj = self.feed_forward.gate_proj.init_rand();
                self
            }
        }

        pub struct Llama {
            // Transformer layers
            pub layers: Vec<TransformerBlock>,
            // Norm + LM head
            pub head: LayerNorm,
        }

        impl Module<(GraphTensor, &[KVCache])> for Llama {
            type Output = (GraphTensor, Vec<KVCache>);
            fn forward(&self, (mut x, cache): (GraphTensor, &[KVCache])) -> Self::Output {
                // Run through layers and collect new caches
                let mut new_caches = vec![];
                let mut new_cache;
                for (i, layer) in self.layers.iter().enumerate() {
                    (x, new_cache) = layer.forward((x, cache[i]));
                    new_caches.push(new_cache);
                }
                // Run through last norm and output projection
                (self.head.forward(x), new_caches)
            }
        }

        impl Llama {
            pub fn new(cx: &mut Graph) -> Self {
                Self {
                    head: LayerNorm::new(HIDDEN_DIM, true, false, false, 1e-5, cx),
                    layers: (0..NUM_LAYERS).map(|_| TransformerBlock::new(cx)).collect(),
                }
            }

            fn initialize(mut self) -> Self {
                self.head = self.head.initialize();
                self.layers = self.layers.into_iter().map(|l| l.initialize()).collect();
                self
            }
        }

        let mut cx = Graph::new();
        let model = Llama::new(&mut cx).initialize();
        let caches = (0..NUM_LAYERS)
            .map(|_| {
                (
                    cx.tensor((1, N_KV_HEADS, 'p', HEAD_DIM)).set_dyn(
                        random_vec(SEQ_LEN * N_KV_HEADS * HEAD_DIM),
                        (1, N_KV_HEADS, SEQ_LEN, HEAD_DIM),
                    ),
                    cx.tensor((1, N_KV_HEADS, 'p', HEAD_DIM)).set_dyn(
                        random_vec(SEQ_LEN * N_KV_HEADS * HEAD_DIM),
                        (1, N_KV_HEADS, SEQ_LEN, HEAD_DIM),
                    ),
                )
            })
            .collect::<Vec<_>>();
        let input = cx
            .tensor((1, 's', HIDDEN_DIM))
            .set_dyn(random_vec(2 * HIDDEN_DIM), (1, 2, HIDDEN_DIM));
        let (mut out, _) = model.forward((input, &caches));
        out.retrieve();
        cx.execute();

        let unopt_out = out.data();
        out.drop();

        cx.compile(<(GenericCompiler, CudaCompiler<f16>)>::default(), &mut out);
        cx.execute();

        assert_close_precision(&out.data(), &unopt_out, 1e-2);
    }
}

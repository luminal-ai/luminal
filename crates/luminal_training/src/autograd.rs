use std::any::TypeId;

use itertools::Itertools;
use petgraph::{algo::toposort, visit::EdgeRef, Direction};
use rustc_hash::{FxHashMap, FxHashSet};

use luminal::{
    op::{
        Add, Contiguous, Exp2, Function, LessThan, Log2, MaxReduce, Mod, Mul, Recip, Sin, Sqrt,
        SumReduce,
    },
    prelude::{tinyvec::ArrayVec, *},
};

#[derive(Clone, Debug)]
pub struct Autograd(Vec<NodeIndex>, NodeIndex);

impl Autograd {
    pub fn new<W: ToIds>(params: W, loss: GraphTensor) -> Self {
        Self(params.to_ids(), loss.id)
    }
}

// Run dfs with a starting stack and record all encountered nodes in a set
fn build_dfs_set(
    stack: &mut Vec<NodeIndex>,
    graph: &StorageGraph,
    direction: Direction,
) -> FxHashSet<NodeIndex> {
    let mut set = FxHashSet::default();
    while let Some(n) = stack.pop() {
        if !set.contains(&n) {
            set.insert(n);
            stack.extend(
                graph
                    .edges_directed(n, direction)
                    .filter(|e| !e.weight().is_schedule())
                    .map(|e| match direction {
                        Direction::Incoming => e.source(),
                        Direction::Outgoing => e.target(),
                    }),
            );
        }
    }
    set
}

impl Compiler for Autograd {
    type Output = Vec<(NodeIndex, ShapeTracker)>;
    fn compile<T: ToIdsMut>(&self, graph: &mut Graph, _: T) -> Vec<(NodeIndex, ShapeTracker)> {
        let Autograd(params, loss) = self;
        // Build up valid set for nodes we want to pay attention to (everything outside of this set doesn't matter)
        let forward_set = build_dfs_set(&mut params.clone(), graph, Direction::Outgoing);
        let backward_set = build_dfs_set(&mut vec![*loss], graph, Direction::Incoming);
        let valid_set: FxHashSet<_> = forward_set.intersection(&backward_set).copied().collect();

        // We have the last loss node, now let's backprop through everything to get the gradient graph
        let mut grads = FxHashMap::default();
        // Add loss gradient
        grads.insert(
            *loss,
            (
                graph.constant(1.0).id,
                ShapeTracker::new(()), // Assume scalar loss for now
            ),
        );
        let weight_set = params.iter().copied().collect::<FxHashSet<_>>();
        for fwd_node in toposort(&graph.graph, None).unwrap().into_iter().rev() {
            if !valid_set.contains(&fwd_node) {
                continue;
            }
            // Check if the node is undifferentiable
            let graph_ref: *mut Graph = graph;
            let op = graph.node_weight(fwd_node).unwrap().as_any().type_id();
            if op == TypeId::of::<Function>() {
                continue;
            }
            if op == TypeId::of::<Mod>() || op == TypeId::of::<LessThan>() {
                assert!(
                    !weight_set.contains(&fwd_node),
                    "{fwd_node:?} is marked as a weight but is undifferentiable: {:?}",
                    graph.node_weight(fwd_node).unwrap()
                );
                continue;
            }

            // Differentiate through fwd_node to get gradients for it's sources
            // Get input tensors
            let inps = graph
                .edges_directed(fwd_node, Direction::Incoming)
                .filter_map(|e| e.weight().as_data().map(|i| (e.source(), i)))
                .sorted_by_key(|(_, (a, _, _))| *a)
                .map(|(node, (_, _, sh))| GraphTensor::from_id(node, sh, graph_ref))
                .collect::<Vec<_>>();
            let mut prev_grad = {
                let (id, sh) = grads[&fwd_node];
                GraphTensor::from_id(id, sh, graph_ref)
            };
            if op == TypeId::of::<Add>() {
                // f(a, b) = a + b
                // df/da = 1
                if valid_set.contains(&inps[0].id) {
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
                // df/db = 1
                if valid_set.contains(&inps[1].id) {
                    add_grad(prev_grad, inps[1], graph, &mut grads);
                }
            } else if op == TypeId::of::<Mul>() {
                // f(a, b) = a * b
                // df/da = b
                if valid_set.contains(&inps[0].id) {
                    add_grad(inps[1] * prev_grad, inps[0], graph, &mut grads);
                }
                // df/db = a
                if valid_set.contains(&inps[1].id) {
                    add_grad(inps[0] * prev_grad, inps[1], graph, &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<SumReduce>(fwd_node)
                .cloned()
            {
                // f(x) = sum_reduce(x)
                // f'(x) = 1
                if valid_set.contains(&inps[0].id) {
                    prev_grad
                        .shape
                        .expand_dim(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
            } else if let Some(op) = unsafe { graph_ref.as_ref().unwrap() } // Needed to get around multiple borrows
                .try_get_op::<MaxReduce>(fwd_node)
                .cloned()
            {
                // f(x) = max_reduce(x)
                // f'(x) = x == max_reduce(x)
                if valid_set.contains(&inps[0].id) {
                    // fwd_nod is already max_reduce(x)
                    prev_grad
                        .shape
                        .expand_dim(op.0, inps[0].shape.dims[inps[0].shape.indexes[op.0]]);
                    let reduced = GraphTensor::from_id(fwd_node, prev_grad.shape, graph_ref);
                    let grad = inps[0].eq(reduced) * prev_grad;
                    add_grad(grad, inps[0], graph, &mut grads);
                }
            } else if op == TypeId::of::<Contiguous>() {
                if valid_set.contains(&inps[0].id) {
                    add_grad(prev_grad, inps[0], graph, &mut grads);
                }
            } else {
                if !valid_set.contains(&inps[0].id) {
                    continue;
                }
                let local_grad = if op == TypeId::of::<Log2>() {
                    // f(x) = log2(x)
                    // f'(x) = 1 / (x * ln(2))
                    1.0 / (inps[0] * 2_f32.ln())
                } else if op == TypeId::of::<Exp2>() {
                    // f(x) = exp2(x)
                    // f'(x) = exp2(x) * ln(2)
                    inps[0].exp2() * 2_f32.ln()
                } else if op == TypeId::of::<Sin>() {
                    // f(x) = sin(x)
                    // f'(x) = cos(x)
                    inps[0].cos()
                } else if op == TypeId::of::<Sqrt>() {
                    // f(x) = sqrt(x)
                    // f'(x) = 1 / (2 * sqrt(x))
                    1.0 / (2.0 * inps[0].sqrt())
                } else if op == TypeId::of::<Recip>() {
                    // f(x) = 1 / x
                    // f'(x) = -1 / x**2
                    -1.0 / (inps[0] * inps[0])
                } else {
                    unreachable!()
                };
                add_grad(local_grad * prev_grad, inps[0], graph, &mut grads);
            }
        }

        // Create a gradient array to match 1-1 with the weight array passed in
        self.0.iter().map(|weight| grads[weight]).collect()
    }
}

fn add_grad(
    mut grad: GraphTensor,
    fwd: GraphTensor,
    graph: &mut Graph,
    grad_map: &mut FxHashMap<NodeIndex, (NodeIndex, ShapeTracker)>,
) {
    // Reshape gradient to match the shape of the input source (before the input was reshaped)
    // Undo permutes
    let mut new_indexes = ArrayVec::new();
    new_indexes.resize(fwd.shape.len(), 0);
    for i in 0..fwd.shape.len() {
        new_indexes[fwd.shape.indexes[i]] = grad.shape.indexes[i];
    }
    grad.shape.indexes = new_indexes;

    // Undo expands (sum reduce)
    let mut min_idx_removed: Option<usize> = None;
    let mut min_idx_removed_amount: Option<usize> = None;
    // TODO: the rev() may no longer be required
    for i in fwd.shape.indexes.into_iter().rev() {
        if fwd.shape.fake[i] {
            min_idx_removed = if let Some(prev) = min_idx_removed {
                Some(prev.min(i))
            } else {
                Some(i)
            };
            let min_idx_removed = min_idx_removed.unwrap_or_default();
            let i_diff = if i > min_idx_removed {
                min_idx_removed_amount.unwrap_or_default()
            } else {
                0
            };
            grad.id = graph
                .add_op(SumReduce(i - i_diff))
                .input(grad.id, 0, grad.shape)
                .finish();
            grad.shape.remove_dim(i - i_diff);
            grad.shape = grad.shape.contiguous();
            min_idx_removed_amount = if let Some(prev) = min_idx_removed_amount {
                Some(prev + 1)
            } else {
                Some(1)
            };
        }
    }

    // Check to see if a reshape was done here. If so, we may need to assert grad shape is contiguous or insert a contiguous call
    if let Some((_, _, mut pre_fwd_shape)) = graph.get_sources(fwd.id).first() {
        if let Some(SumReduce(dim)) = graph.try_get_op(fwd.id) {
            pre_fwd_shape.remove_dim(*dim);
        } else if let Some(MaxReduce(dim)) = graph.try_get_op(fwd.id) {
            pre_fwd_shape.remove_dim(*dim);
        }
        if grad.shape.dims() != pre_fwd_shape.dims() {
            if !grad.shape.is_contiguous() {
                grad = grad.contiguous();
            }
            grad.shape = pre_fwd_shape.contiguous();
        }
    }

    if let Some((existing_grad_node, existing_grad_shape)) = grad_map.get(&fwd.id).copied() {
        let grad = GraphTensor::from_id(grad.id, grad.shape, graph);
        let existing_grad = GraphTensor::from_id(existing_grad_node, existing_grad_shape, graph);
        let new_grad = grad + existing_grad;
        grad_map.insert(fwd.id, (new_grad.id, new_grad.shape));
    } else {
        grad_map.insert(fwd.id, (grad.id, grad.shape));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dfdx::nn::Module as DModule;
    use luminal::prelude::Module as LModule;
    luminal::test_imports!();

    fn get_vec(grad: (NodeIndex, ShapeTracker), cx: &mut Graph) -> Vec<f32> {
        GraphTensor::from_id(grad.0, grad.1, cx).data()
    }

    #[test]
    fn test_autograd_max_reduce() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("Input", 2).set([10., 5.]);
        let b = a.max(0);

        let grads = cx.compile(Autograd::new(a, b), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let d_a = dev.tensor([10., 5.]);
        let d_b = d_a.trace(Gradients::leaky()).max();
        let d_grads = d_b.backward();

        assert_exact(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_matmul() {
        let mut cx = Graph::new();
        let a = cx.named_tensor("A", (2, 2)).set([[2., 4.], [3., 1.]]);
        let input = cx.named_tensor("Input", 2).set([10., 5.]);
        let output = (input.matmul(a)).sum(0);

        let grads = cx.compile(Autograd::new(a, output), ());
        cx.keep_tensors(&grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let w1 = dev.tensor([[2., 4.], [3., 1.]]);
        let inp = dev.tensor([10., 5.]);
        let out = inp.trace(Gradients::leaky()).matmul(w1.clone()).sum();
        let d_grads = out.backward();

        assert_exact(&get_vec(grads[0], &mut cx), &d_grads.get(&w1).as_vec());
    }

    #[test]
    fn test_autograd_mlp() {
        let mut cx = Graph::new();
        let model = (
            luminal_nn::Linear::new(2, 2, false, &mut cx),
            luminal_nn::ReLU,
            luminal_nn::Linear::new(2, 1, false, &mut cx),
        );
        model.0.weight.set([[2., 4.], [3., 1.]]);
        model.2.weight.set([[6.], [5.]]);
        let input = cx.named_tensor("Input", 2).set([10., 5.]);
        let output = model.forward(input).sum(0);

        let mut grads = cx.compile(Autograd::new(params(model), output), ());
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), &mut grads);
        cx.execute();

        let dev = dfdx::prelude::Cpu::default();
        let mut d_model = dev.build_module::<(
            dfdx::nn::builders::UnbiasedLinear<2, 2>,
            dfdx::nn::builders::ReLU,
            dfdx::nn::builders::UnbiasedLinear<2, 1>,
        ), f32>();
        d_model.0.weight = dev.tensor([[2., 4.], [3., 1.]]).permute();
        d_model.2.weight = dev.tensor([[6.], [5.]]).permute();
        let inp = dev.tensor([10., 5.]);
        let out = d_model.forward(inp.trace(Gradients::leaky())).sum();
        let d_grads = out.backward();

        assert_exact(
            &get_vec(grads[0], &mut cx),
            &d_grads.get(&d_model.0.weight).permute().as_vec(),
        );
        assert_exact(
            &get_vec(grads[1], &mut cx),
            &d_grads.get(&d_model.2.weight).as_vec(),
        );
    }

    #[test]
    fn test_autograd_layer_norm() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([-1., 2., 3.]);
        let mut b = a.layer_norm(0, 1e-5).max(0).retrieve();

        let grads = cx.compile(Autograd::new(a, b), &mut b);
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), &mut b);
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([-1., 2., 3.]);
        let d_b = d_a.trace(Gradients::leaky()).normalize(1e-5).max();
        assert_close(&b.data(), &d_b.as_vec());
        let d_grads = d_b.backward();
        assert_close(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_softmax() {
        let mut cx = Graph::new();
        let a = cx.tensor(3).set([-1., 2., 3.]);
        let mut b = a.softmax(0).max(0).retrieve();

        let mut grads = cx.compile(Autograd::new(a, b), &mut b);
        cx.keep_tensors(&grads);
        cx.compile(GenericCompiler::default(), (&mut grads, &mut b));
        cx.execute();

        let d_dev = Cpu::default();
        let d_a = d_dev.tensor([-1., 2., 3.]);
        let d_b = d_a.trace(Gradients::leaky()).softmax().max();
        assert_close(&b.data(), &d_b.as_vec());
        let d_grads = d_b.backward();
        assert_close(&get_vec(grads[0], &mut cx), &d_grads.get(&d_a).as_vec());
    }

    #[test]
    fn test_autograd_transformer() {
        let mut cx = Graph::new();
        let model = luminal_nn::TransformerEncoderBlock::new(3, 4, 1, &mut cx);
        model
            .attention
            .w_k
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .attention
            .w_q
            .weight
            .set(vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.]);
        model
            .attention
            .w_v
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.]);
        model
            .attention
            .w_o
            .weight
            .set(vec![1., 22., 3., 1., 2., 3., 1., 2., 3.]);
        model
            .ff
            .0
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.]);
        model
            .ff
            .2
            .weight
            .set(vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.]);

        let a = cx.tensor((2, 3)).set([[-1., 2., 3.], [3., 3., -1.]]);
        let target = cx.tensor((2, 3)).set([[0., 1., 0.], [0., 0., 1.]]);
        let out = model.forward(a);
        let mut loss = crate::cross_entropy_with_logits_loss(out, target).retrieve();

        let mut model_params = params(&model);
        let mut grads = cx.compile(
            Autograd::new((&model_params, a), loss),
            (&mut model_params, &mut loss),
        );
        cx.keep_tensors(&grads);
        cx.compile(
            GenericCompiler::default(),
            (&mut model_params, &mut grads, &mut loss),
        );
        cx.execute();

        let d_dev = Cpu::default();
        let mut d_model = d_dev
            .build_module::<dfdx::nn::modules::builders::TransformerEncoderBlock<3, 1, 4>, f32>();
        d_model.self_attn.w_k.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_v.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_q.bias.copy_from(&[0.0, 0.0, 0.0]);
        d_model.self_attn.w_o.bias.copy_from(&[0., 0., 0.]);
        d_model.self_attn.w_o.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (DConst::<3>, DConst::<3>),
            )
            .permute();
        d_model.self_attn.w_k.weight = d_dev
            .tensor_from_vec(
                vec![1., 22., 3., 1., 2., 3., 1., 2., 3.],
                (DConst::<3>, DConst::<3>),
            )
            .permute();
        d_model.self_attn.w_q.weight = d_dev
            .tensor_from_vec(
                vec![3., 2., 3., 1.3, 2., 3., 3., 2., 3.],
                (DConst::<3>, DConst::<3>),
            )
            .permute();
        d_model.self_attn.w_v.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3.],
                (DConst::<3>, DConst::<3>),
            )
            .permute();
        d_model.ff.0 .0.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 11., 2., 3.],
                (DConst::<3>, DConst::<4>),
            )
            .permute();
        d_model.ff.0 .0.bias = d_dev.tensor_from_vec(vec![0., 0., 0., 0.], (DConst::<4>,));
        d_model.ff.0 .2.weight = d_dev
            .tensor_from_vec(
                vec![-1., 12., 3., -1., 2., -3., 11., 2., 3., 3., -1., 2.],
                (DConst::<4>, DConst::<3>),
            )
            .permute();
        d_model.ff.0 .2.bias = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
        d_model.norm1.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
        d_model.norm2.gamma = d_dev.tensor_from_vec(vec![1., 1., 1.], (DConst::<3>,));
        d_model.norm1.epsilon = 1e-5;
        d_model.norm2.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
        d_model.norm1.beta = d_dev.tensor_from_vec(vec![0., 0., 0.], (DConst::<3>,));
        d_model.norm2.epsilon = 1e-5;
        let d_a = d_dev.tensor_from_vec(vec![-1., 2., 3., 3., 3., -1.], (DConst::<2>, DConst::<3>));
        let d_target =
            d_dev.tensor_from_vec(vec![0., 1., 0., 0., 0., 1.], (DConst::<2>, DConst::<3>));
        let d_b = d_model.forward(d_a.trace(Gradients::leaky()));
        let d_loss = dfdx::prelude::cross_entropy_with_logits_loss(d_b, d_target);

        assert_close(&loss.data(), &d_loss.as_vec());

        let d_grads = d_loss.backward();
        assert_close(
            &get_vec(*grads.last().unwrap(), &mut cx),
            &d_grads.get(&d_a).as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.ff.2.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads.get(&d_model.ff.0 .2.weight).permute().as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.ff.0.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads.get(&d_model.ff.0 .0.weight).permute().as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_o.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_o.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_q.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_q.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_k.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_k.weight)
                .permute()
                .as_vec(),
        );
        assert_close(
            &get_vec(
                grads[model_params
                    .iter()
                    .position(|i| *i == model.attention.w_v.weight.id)
                    .unwrap()],
                &mut cx,
            ),
            &d_grads
                .get(&d_model.self_attn.w_v.weight)
                .permute()
                .as_vec(),
        );
    }

    #[test]
    fn test_add_grad_decreasing_idx_r1() {
        let mut cx = Graph::new();
        let a: GraphTensor<R1<2>> = cx.tensor();
        let a: GraphTensor<R3<1, 1, 2>> = a.expand::<_, LAxes2<0, 1>>();
        let a: GraphTensor<R3<2, 1, 1>> = a.permute::<_, LAxes3<2, 1, 0>>();
        assert_eq!(&a.shape.fake[..], &[false, true, true]); // has multiple fake dimensions
        assert_eq!(&a.shape.indexes[..], &[0, 2, 1]); // not strictly increasing

        // note: this tests the case when the rev indexes may decrease (0 <- 2) after increasing (2 <- 1)

        let loss: GraphTensor<R0> = a.sum_reduce();
        let _grads = cx.compile(Autograd::new(vec![a.id], loss), ());
    }

    #[test]
    fn test_add_grad_decreasing_idx_r2() {
        let mut cx = Graph::new();
        let a: GraphTensor<R2<2, 3>> = cx.tensor();
        let a: GraphTensor<R5<2, 1, 1, 1, 3>> = a.expand::<_, LAxes3<1, 2, 3>>();
        let a: GraphTensor<R5<3, 1, 2, 1, 1>> = a.permute::<_, LAxes5<4, 1, 0, 3, 2>>();
        assert_eq!(&a.shape.fake[..], &[false, false, true, true, true]); // has multiple fake dimensions
        assert_eq!(&a.shape.indexes[..], &[1, 2, 0, 4, 3]); // not strictly increasing

        // note: the difference in this test to test_add_grad_decreasing_idx_r1
        // is that the rev indexes may increase (2 <- 0) after it has decreased (0 <- 4) after it has increased (4 <- 3)

        let loss: GraphTensor<R0> = a.sum_reduce();
        let _grads = cx.compile(Autograd::new(vec![a.id], loss), ());
    }
}

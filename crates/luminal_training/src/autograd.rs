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

fn normalize_grad_for_input(
    mut grad: GraphTensor,
    fwd: &GraphTensor,
    graph: &mut Graph,
) -> GraphTensor {

    // If grad has fewer logical axes than fwd (common when forward used expand/broadcast),
    // insert broadcasted axes into grad at the logical positions where fwd is fake.
    // Do this before undoing permutes so indexes line up.
    if grad.shape.len() < fwd.shape.len() {
        let _need = fwd.shape.len() - grad.shape.len();
        // insert fake positions from highest -> lowest so indices remain valid
        let mut fake_positions: Vec<usize> =
            (0..fwd.shape.len()).filter(|&i| fwd.shape.fake[i]).collect();
        fake_positions.sort_unstable_by(|a, b| b.cmp(a));
        for &pos in &fake_positions {
            if grad.shape.len() >= fwd.shape.len() {
                break;
            }
            // Insert a logical axis with the forward logical size (broadcast view).
            let size = fwd.shape.dims[fwd.shape.indexes[pos]];
            grad.shape.expand_dim(pos, size);
            // Rewrap so the updated shape is used downstream.
            grad = GraphTensor::from_id(grad.id, grad.shape, graph as *mut Graph);
        }
        // Defensive: if we still didn't reach target rank (e.g., malformed fake flags), pad at end.
        while grad.shape.len() < fwd.shape.len() {
            let pos = grad.shape.len();
            let size = fwd.shape.dims[fwd.shape.indexes[pos]];
            grad.shape.expand_dim(pos, size);
            grad = GraphTensor::from_id(grad.id, grad.shape, graph as *mut Graph);
        }
    }

    // Undo permutes
    let mut new_indexes = ArrayVec::new();
    new_indexes.resize(fwd.shape.len(), 0);
    for i in 0..fwd.shape.len() {
        new_indexes[fwd.shape.indexes[i]] = grad.shape.indexes[i];
    }
    grad.shape.indexes = new_indexes;

    // Undo expands (sum-reduce on every fake dimension)
    let mut fake_dims: Vec<usize> =
        (0..fwd.shape.len()).filter(|&idx| fwd.shape.fake[idx]).collect();

    // Remove highest indices first so later indices stay valid.
    fake_dims.sort_unstable_by(|a, b| b.cmp(a));

    for idx in fake_dims {
        grad.id = graph
            .add_op(SumReduce(idx))
            .input(grad.id, 0, grad.shape)
            .finish();

        grad.shape.remove_dim(idx);
        grad.shape = grad.shape.contiguous();
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

    grad
}

fn add_grad(
    mut grad: GraphTensor,
    fwd: GraphTensor,
    graph: &mut Graph,
    grad_map: &mut FxHashMap<NodeIndex, (NodeIndex, ShapeTracker)>,
) {

    // Normalize/reshape incoming grad to match `fwd`'s semantics.
    let grad = normalize_grad_for_input(grad, &fwd, graph);

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

    // Helper utilities used by tests
    fn get_vec(grad: (NodeIndex, ShapeTracker), cx: &mut Graph) -> Vec<f32> {
        GraphTensor::from_id(grad.0, grad.1, cx).data()
    }

    // --- 1. Core Autograd Tests ---
    mod core_autograd {
        use super::*;

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
    }

    // --- 2. Broadcasting, Permute, and Fake Dim Handling ---
    mod broadcast_permute {
        use super::*;

        #[test]
        fn test_add_grad_decreasing_idx_r1() {
            let mut cx = Graph::new();
            // Create a tensor, expand to add fake dims and permute to produce
            // a non-monotonic indexes mapping. This mirrors the original intent
            // without relying on compile-time shape generics.
            let a = cx.tensor(2);
            let a = a.expand((1, 1, 2));
            let a = a.permute((2, 1, 0));

            // has multiple fake dimensions
            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);
            // indexes should not be strictly increasing
            let not_strict = a.shape.indexes.windows(2).any(|w| w[0] >= w[1]);
            assert!(not_strict);

            // reduce to scalar and ensure autograd compiles
            let loss = a.sum((0, 1, 2));
            let _grads = cx.compile(Autograd::new(vec![a.id], loss), ());
        }

        #[test]
        fn test_add_grad_decreasing_idx_r2() {
            let mut cx = Graph::new();
            // Start with a 2x3 tensor, expand to add fake dims, then permute to
            // create a non-monotonic indexes mapping.
            let a = cx.tensor((2, 3));
            let a = a.expand((2, 1, 1, 1, 3));
            let a = a.permute((4, 1, 0, 3, 2));

            // has multiple fake dimensions
            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 3);
            // indexes should not be strictly increasing
            let not_strict = a.shape.indexes.windows(2).any(|w| w[0] >= w[1]);
            assert!(not_strict);

            let loss = a.sum((0, 1, 2, 3, 4));
            let _grads = cx.compile(Autograd::new(vec![a.id], loss), ());
        }

        #[test]
        fn test_add_grad_with_values_r1() {
            let mut cx = Graph::new();
            let a = cx.tensor(3).set([1.0, 2.0, 3.0]);
            let orig_a_id = a.id;
            // Expand and permute to create fake dims with non-monotonic indexes
            let a = a.expand((1, 3, 1));
            let a = a.permute((1, 2, 0));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);
            let not_strict = a.shape.indexes.windows(2).any(|w| w[0] >= w[1]);
            assert!(not_strict);

            let loss = a.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradient of sum is all ones
            assert_exact(&get_vec(grads[0], &mut cx), &vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_add_grad_with_values_r2() {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 3)).set([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            let orig_a_id = a.id;
            // Expand and permute to create fake dims with non-monotonic indexes
            let a = a.expand((2, 1, 3, 1));
            let a = a.permute((2, 3, 0, 1));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);

            let loss = a.sum((0, 1, 2, 3));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradient of sum is all ones
            assert_exact(
                &get_vec(grads[0], &mut cx),
                &vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            );
        }

        #[test]
        fn test_add_grad_multiply_with_permute() {
            let mut cx = Graph::new();
            let a = cx.tensor(3).set([2.0, 3.0, 4.0]);
            let orig_a_id = a.id;
            // Expand and permute to create fake dims with non-monotonic indexes
            let a = a.expand((1, 1, 3));
            let a = a.permute((2, 1, 0));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);

            // Multiply by constant, then sum
            let b = a * 2.0;
            let loss = b.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradient should be 2.0 for each element (derivative of 2*x)
            assert_exact(&get_vec(grads[0], &mut cx), &vec![2.0, 2.0, 2.0]);
        }

        #[test]
        fn test_add_grad_complex_permute() {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 2)).set([[1.0, 2.0], [3.0, 4.0]]);
            let orig_a_id = a.id;
            // Create a complex permutation with multiple fake dims
            let a = a.expand((1, 2, 1, 2, 1));
            let a = a.permute((3, 4, 1, 0, 2));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 3);

            let loss = a.sum((0, 1, 2, 3, 4));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradient of sum is all ones
            assert_exact(&get_vec(grads[0], &mut cx), &vec![1.0, 1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_add_grad_first_last_fake() {
            let mut cx = Graph::new();
            let a = cx.tensor(3).set([1.0, 2.0, 3.0]);
            let orig_a_id = a.id;
            // Add fake dims at the beginning and end, then permute
            let a = a.expand((1, 3, 1));
            let a = a.permute((2, 0, 1));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert_eq!(fake_count, 2);

            let loss = a.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            assert_exact(&get_vec(grads[0], &mut cx), &vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_add_grad_reverse_permute() {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 3, 4));
            let orig_a_id = a.id;
            a.set(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
            ]);
            // Expand and reverse permute
            let a = a.expand((2, 1, 3, 1, 4));
            let a = a.permute((4, 3, 2, 1, 0));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);
            // Check that indexes are non-monotonic (not strictly increasing)
            let not_increasing = a.shape.indexes.windows(2).any(|w| w[0] >= w[1]);
            assert!(not_increasing);

            let loss = a.sum((0, 1, 2, 3, 4));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Should have 2*3*4 = 24 elements, all with gradient 1.0
            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 24);
            assert!(grad_vec.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_add_grad_with_addition() {
            let mut cx = Graph::new();
            let a = cx.tensor(3).set([1.0, 2.0, 3.0]);
            let b = cx.tensor(3).set([4.0, 5.0, 6.0]);
            let orig_a_id = a.id;
            let orig_b_id = b.id;

            // Expand and permute a to create fake dims with non-monotonic indexes
            let a = a.expand((1, 3, 1));
            let a = a.permute((1, 2, 0));

            // Also transform b
            let b = b.expand((1, 3, 1));
            let b = b.permute((1, 2, 0));

            let c = a + b;
            let loss = c.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(vec![orig_a_id, orig_b_id], loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradients for both should be all ones
            assert_exact(&get_vec(grads[0], &mut cx), &vec![1.0, 1.0, 1.0]);
            assert_exact(&get_vec(grads[1], &mut cx), &vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_add_grad_high_rank() {
            let mut cx = Graph::new();
            let a = cx.tensor((2, 2, 2, 2));
            let orig_a_id = a.id;
            a.set(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ]);

            // Add fake dims and permute
            let a = a.expand((2, 1, 2, 2, 1, 2));
            let a = a.permute((5, 4, 2, 3, 1, 0));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 2);

            let loss = a.sum((0, 1, 2, 3, 4, 5));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 16);
            assert!(grad_vec.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_add_grad_partial_fake() {
            let mut cx = Graph::new();
            let a = cx.tensor((3, 4)).set(vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ]);
            let orig_a_id = a.id;

            // Expand only in the middle, creating just one fake dim
            let a = a.expand((3, 1, 4));
            let a = a.permute((2, 0, 1));

            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert_eq!(fake_count, 1);

            let loss = a.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 12);
            assert!(grad_vec.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_add_grad_with_elementwise_ops() {
            let mut cx = Graph::new();
            let a = cx.tensor(4).set([1.0, 2.0, 3.0, 4.0]);
            let orig_a_id = a.id;

            // Expand and permute to create fake dims with non-monotonic indexes
            let a = a.expand((1, 1, 4));
            let a = a.permute((2, 1, 0));

            // Apply element-wise operations
            let b = a.sin();
            let loss = b.sum((0, 1, 2));

            let grads = cx.compile(Autograd::new(orig_a_id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Gradient should be cos(a) for each element
            let expected: Vec<f32> = vec![1.0_f32, 2.0, 3.0, 4.0]
                .into_iter()
                .map(|x| x.cos())
                .collect();
            assert_close(&get_vec(grads[0], &mut cx), &expected);
        }

        #[test]
        fn test_undo_permute_behavior() {
            // Test that the undo-permute step correctly aligns gradient coordinates with forward tensor
            let mut cx = Graph::new();

            // Create a tensor and transform it like in the failing test
            let a = cx.tensor((2, 3)).set([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
            let orig_a_id = a.id;

            // Apply complex transformations that create non-monotonic indexes
            let a = a.expand((2, 1, 1, 1, 3));
            let a = a.permute((4, 1, 0, 3, 2));

            // Verify the shape has the expected properties
            assert_eq!(a.shape.len(), 5);
            let fake_count = a.shape.fake.iter().filter(|&&b| b).count();
            assert!(fake_count >= 3);

            // Sum to scalar - this should work without out-of-bounds errors
            let loss = a.sum((0, 1, 2, 3, 4));
            let grads = cx.compile(Autograd::new(vec![orig_a_id], loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Verify the gradient is correct (all ones)
            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 6); // 2 * 3
            assert!(grad_vec.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_add_grad_requires_real_contiguous() {
            let mut cx = Graph::new();

            // Make a 2×3 tensor, broadcast & permute so it’s decidedly non-contiguous
            let a = cx.tensor((2, 3)).set([[1., 2., 3.], [4., 5., 6.]]);
            let orig = a.id;
            let a = a.expand((1, 2, 3)).permute((2, 1, 0));

            // Loss is a *second* sum-reduce: first autograd will insert its own,
            // then this explicit one forces another op to read the result.
            // If the fake “contiguous” flag is wrong, the second reduction sees
            // bad strides and the numeric answer is off.
            let loss = a.sum((0, 1, 2)).sum(());
            let g = cx.compile(Autograd::new(orig, loss), ());
            cx.keep_tensors(&g);
            cx.execute();

            // Expected gradient is still all ones (two sums in series)
            assert_exact(&GraphTensor::from_id(g[0].0, g[0].1, &mut cx).data(),
                        &vec![1., 1., 1., 1., 1., 1.]);
        }
        #[test]
        fn test_add_grad_preserves_unfaked_dim() {
            let mut cx = Graph::new();

            // b is broadcast → fake for forward, but we turn it real with *a
            let b = cx.tensor(1).set([2.]);          // shape [1] fake dim
            let a = cx.tensor(3).set([1., 2., 3.]);  // shape [3]
            let c = b.expand((3,)) * a;              // broadcast multiply
            let loss = c.sum(0);                     // scalar

            let grads = cx.compile(Autograd::new(b.id, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // dloss/db = sum(a) = 6
            assert_exact(&GraphTensor::from_id(grads[0].0, grads[0].1, &mut cx).data(),
                        &vec![6.0]);
        }

        #[test]
        fn test_add_grad_max_rank_broadcast() {
            let mut cx = Graph::new();
            // Test with maximum supported rank (6 dimensions) with many fake dims
            // ShapeTracker uses ArrayVec with capacity 6, so this is the limit
            let a = cx.tensor((2, 1, 3, 1, 1, 4));   // 6 dims, many fake
            // Set some values - we need 2*3*4 = 24 values
            a.set(vec![1.0; 24]);
            let orig = a.id;
            let loss = a.sum((0, 1, 2, 3, 4, 5));

            let grads = cx.compile(Autograd::new(orig, loss), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Verify gradient is correct (all ones)
            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 2 * 3 * 4); // 24 elements
            assert!(grad_vec.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        }

        #[test]
        fn test_add_grad_perm_weight_noncontig() {
            let mut cx = Graph::new();
            // Weight is 2×2 but permuted so strides are swapped
            // This tests that autograd correctly handles non-contiguous tensors
            let w = cx.tensor((2,2)).set([[1.,2.],[3.,4.]]).permute((1,0));
            let x = cx.tensor(2).set([10., 5.]);
            let out = x.matmul(w).sum(0);

            // This should compile and execute without out-of-bounds errors
            let grads = cx.compile(Autograd::new(w.id, out), ());
            cx.keep_tensors(&grads);
            cx.execute();

            // Verify we got a gradient with the correct shape (2x2 = 4 elements)
            let grad_vec = get_vec(grads[0], &mut cx);
            assert_eq!(grad_vec.len(), 4);
            // Verify gradient values are reasonable (not NaN or Inf)
            assert!(grad_vec.iter().all(|&v| v.is_finite()));
        }
    }

        // ---------------------------
    // Unit tests for normalize_grad_for_input
    // ---------------------------
    mod normalize_tests {
        use super::*;

        // Helper to get materialized data from a GraphTensor
        fn data_of(t: &GraphTensor, cx: &mut Graph) -> Vec<f32> {
            GraphTensor::from_id(t.id, t.shape, cx as *mut Graph).data()
        }

        /// White-box unit test replicating `test_add_grad_decreasing.rs`.
        /// It verifies that `normalize_grad_for_input` correctly handles a `fwd` tensor
        /// with multiple fake dimensions and non-monotonic `indexes`.
        #[test]
        fn test_normalize_handles_non_monotonic_indexes_and_multi_fake_dims() {
            let mut cx = Graph::new();
            // 1. Construct `fwd` tensor state with non-monotonic indexes and multiple fake dims.
            // Use a real tensor (constant reshaped) instead of an uninitialized placeholder
            // so execution doesn't attempt to load an unset value for this node.
            let fwd_base = cx.constant(0.0).reshape((2, 3));
            let fwd_expanded = fwd_base.expand((2, 1, 3, 1));
            // Sanity-check the expanded state before permuting
            assert_eq!(fwd_expanded.shape.shape_usize(), vec![2, 1, 3, 1]);
            assert_eq!(fwd_expanded.shape.indexes.as_slice(), &[0, 2, 1, 3]);
            assert_eq!(fwd_expanded.shape.fake.as_slice(), &[false, false, true, true]);
            let fwd = fwd_expanded.permute((2, 3, 0, 1));

            // Sanity-check the forward tensor's shape properties.
            assert_eq!(fwd.shape.shape_usize(), vec![3, 1, 2, 1]);
            assert_eq!(fwd.shape.indexes.as_slice(), &[1, 3, 0, 2]);
            assert_eq!(fwd.shape.fake.as_slice(), &[false, false, true, true]); 

            // 2. Simulate upstream gradient of broadcasted `1.0`.
            let grad: GraphTensor = cx.constant(1.0).expand((3, 1, 2, 1));

            // 3. Call the function under test.
            let out = super::normalize_grad_for_input(grad, &fwd, &mut cx);

            // 4. White-box verification:
            //    - The output should have had two SumReduce ops applied.
            //    - We can verify this by checking the final shape, which should be [2, 3].
            assert_eq!(out.shape.shape_usize(), vec![2, 3]);

            // 5. Execute graph and verify final data.
            //    The gradient of 1.0, reduced over two fake dims of size 1, should
            //    result in a [2, 3] tensor of ones.
            cx.keep_tensors(vec![(out.id, out.shape)]);
            cx.execute();
            let expected_data = vec![1.0; 6];
            assert_exact(&data_of(&out, &mut cx), &expected_data);
        }

        /// Unit test to verify `normalize_grad_for_input` does nothing when no fake
        /// dimensions or permutes are present.
        #[test]
        fn test_normalize_is_noop_for_simple_shapes() {
            let mut cx = Graph::new();
            // 1. `fwd` tensor is a simple, contiguous tensor.
            let fwd = cx.tensor(3).set([1.0, 2.0, 3.0]);

            // 2. Upstream grad is ones with the same shape.
            let grad = cx.tensor(3).set([1.0, 1.0, 1.0]);
            let original_grad_id = grad.id;

            // 3. Call the function under test.
            let out = super::normalize_grad_for_input(grad, &fwd, &mut cx);

            // 4. White-box verification:
            //    - No new ops should be added. The output ID should be the same as the input.
            assert_eq!(out.id, original_grad_id, "No new nodes should be created");
            //    - The output shape should be identical to the input shape.
            assert_eq!(out.shape.shape_usize(), vec![3]);

            // 5. Verify data.
            cx.keep_tensors(vec![(out.id, out.shape)]);
            cx.execute();
            assert_exact(&data_of(&out, &mut cx), &vec![1.0, 1.0, 1.0]);
        }
    }
}
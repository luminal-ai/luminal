use luminal::prelude::*;

/// A fixed number of selected experts for every token.
///
/// The final axis of `expert_ids` and `weights` is the route-slot axis. All
/// preceding axes identify tokens. Routing policy is intentionally external:
/// callers decide how scores are produced, which experts are selected, and
/// whether the selected weights are normalized.
#[derive(Clone, Copy)]
pub struct TopKRoutes {
    expert_ids: GraphTensor,
    weights: GraphTensor,
}

impl TopKRoutes {
    /// Construct routes from already-selected expert IDs and route weights.
    pub fn new(expert_ids: GraphTensor, weights: GraphTensor) -> Self {
        assert_eq!(
            expert_ids.dtype,
            DType::Int,
            "TopKRoutes expert IDs must be Int"
        );
        assert!(
            expert_ids.rank() > 0,
            "TopKRoutes requires a final route-slot axis"
        );
        assert_same_graph(expert_ids, weights, "TopKRoutes");
        assert_same_shape(
            &expert_ids.dims(),
            &weights.dims(),
            "TopKRoutes expert IDs and weights",
        );
        Self {
            expert_ids,
            weights,
        }
    }

    /// Select route weights from the last axis of a score tensor.
    ///
    /// This performs no activation or normalization. For example, a model can
    /// pass softmax probabilities, sigmoid scores, or raw learned weights.
    pub fn from_scores(scores: GraphTensor, expert_ids: GraphTensor) -> Self {
        assert_same_graph(scores, expert_ids, "TopKRoutes::from_scores");
        let score_dims = scores.dims();
        let route_dims = expert_ids.dims();
        assert!(
            !score_dims.is_empty(),
            "scores and expert IDs require an expert/route axis"
        );
        assert_eq!(
            score_dims.len(),
            route_dims.len(),
            "scores and expert IDs must have the same rank"
        );
        assert_same_shape(
            &score_dims[..score_dims.len() - 1],
            &route_dims[..route_dims.len() - 1],
            "scores and expert IDs token axes",
        );

        let token_rank = route_dims.len() - 1;
        let mut coords = Vec::with_capacity(score_dims.len());
        for axis in 0..token_rank {
            coords.push(scores.graph().iota(route_dims.clone(), |c| c[axis]));
        }
        coords.push(expert_ids);
        Self::new(expert_ids, scores.gather(&coords))
    }

    pub fn expert_ids(&self) -> GraphTensor {
        self.expert_ids
    }

    pub fn weights(&self) -> GraphTensor {
        self.weights
    }

    /// The final axis containing the selected expert slots.
    pub fn route_axis(&self) -> usize {
        self.expert_ids.rank() - 1
    }

    /// Replace the route weights without changing the selected experts.
    pub fn with_weights(self, weights: GraphTensor) -> Self {
        Self::new(self.expert_ids, weights)
    }

    /// Normalize each token's selected weights to sum to one.
    pub fn normalize(self) -> Self {
        let axis = self.route_axis();
        let slots = self.expert_ids.dims()[axis];
        let denominator = self.weights.sum(axis).expand_dim(axis, slots);
        self.with_weights(self.weights / denominator)
    }

    /// Broadcast token inputs across the route-slot axis.
    ///
    /// `input` begins with the token axes and may have any trailing payload
    /// shape. For example, `[batch, sequence, hidden]` becomes
    /// `[batch, sequence, K, hidden]`.
    pub fn dispatch(&self, input: GraphTensor) -> GraphTensor {
        assert_same_graph(self.expert_ids, input, "TopKRoutes::dispatch");
        let route_dims = self.expert_ids.dims();
        let token_rank = self.route_axis();
        let input_dims = input.dims();
        assert!(
            input_dims.len() >= token_rank,
            "input does not contain all route token axes"
        );
        assert_same_shape(
            &route_dims[..token_rank],
            &input_dims[..token_rank],
            "route and input token axes",
        );
        input.expand_dim(token_rank, route_dims[token_rank])
    }

    /// Select an arbitrary tensor from an expert parameter bank.
    ///
    /// The parameter bank's first axis is the expert axis. `[E, ...]` becomes
    /// `[..., K, ...]`, prefixed by the routes' token and slot axes.
    pub fn select(&self, expert_tensor: GraphTensor) -> GraphTensor {
        select_expert_tensor(self.expert_ids, expert_tensor)
    }

    /// Apply route weights and sum over the route-slot axis.
    pub fn combine(&self, routed_output: GraphTensor) -> GraphTensor {
        assert_same_graph(self.expert_ids, routed_output, "TopKRoutes::combine");
        let route_dims = self.expert_ids.dims();
        let output_dims = routed_output.dims();
        assert!(
            output_dims.len() >= route_dims.len(),
            "routed output does not contain all route axes"
        );
        assert_same_shape(
            &route_dims,
            &output_dims[..route_dims.len()],
            "routes and routed output prefix",
        );
        let weights = self
            .weights
            .cast(routed_output.dtype)
            .expand_rhs(&output_dims[route_dims.len()..]);
        (routed_output * weights).sum(self.route_axis())
    }

    /// Convert structured top-k routes into the general flat route table.
    pub fn into_routes(self) -> Routes {
        let route_dims = self.expert_ids.dims();
        let token_rank = self.route_axis();
        let token_dims = &route_dims[..token_rank];
        let token_count = token_dims
            .iter()
            .copied()
            .fold(IntExpr::from(1), |acc, dim| acc * dim)
            .simplify();
        let token_ids = self.expert_ids.graph().iota(route_dims.clone(), |c| {
            let mut id = IntExpr::from(0);
            let mut stride = IntExpr::from(1);
            for axis in (0..token_rank).rev() {
                id += c[axis] * stride;
                stride = (stride * token_dims[axis]).simplify();
            }
            id
        });
        let slot_ids = self
            .expert_ids
            .graph()
            .iota(route_dims.clone(), |c| c[token_rank]);
        Routes::new(
            token_ids.flatten(),
            self.expert_ids.flatten(),
            slot_ids.flatten(),
            self.weights.flatten(),
            token_count,
            route_dims[token_rank],
        )
    }
}

/// A general sparse token-to-expert routing table.
///
/// Each element describes one route. `(token_id, slot_id)` pairs must be
/// unique, and `slot_id` must be in `0..max_routes_per_token`. This uniqueness
/// lets [`Routes::combine`] use ordinary assignment scatter followed by a sum;
/// it does not require scatter-add semantics.
#[derive(Clone, Copy)]
pub struct Routes {
    token_ids: GraphTensor,
    expert_ids: GraphTensor,
    slot_ids: GraphTensor,
    weights: GraphTensor,
    token_count: IntExpr,
    max_routes_per_token: IntExpr,
}

impl Routes {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        token_ids: GraphTensor,
        expert_ids: GraphTensor,
        slot_ids: GraphTensor,
        weights: GraphTensor,
        token_count: impl Into<IntExpr>,
        max_routes_per_token: impl Into<IntExpr>,
    ) -> Self {
        assert_eq!(token_ids.dtype, DType::Int, "route token IDs must be Int");
        assert_eq!(expert_ids.dtype, DType::Int, "route expert IDs must be Int");
        assert_eq!(slot_ids.dtype, DType::Int, "route slot IDs must be Int");
        assert_eq!(token_ids.rank(), 1, "route tensors must be rank one");
        for tensor in [expert_ids, slot_ids, weights] {
            assert_same_graph(token_ids, tensor, "Routes");
            assert_same_shape(&token_ids.dims(), &tensor.dims(), "route table columns");
        }
        Self {
            token_ids,
            expert_ids,
            slot_ids,
            weights,
            token_count: token_count.into(),
            max_routes_per_token: max_routes_per_token.into(),
        }
    }

    pub fn token_ids(&self) -> GraphTensor {
        self.token_ids
    }

    pub fn expert_ids(&self) -> GraphTensor {
        self.expert_ids
    }

    pub fn slot_ids(&self) -> GraphTensor {
        self.slot_ids
    }

    pub fn weights(&self) -> GraphTensor {
        self.weights
    }

    pub fn token_count(&self) -> IntExpr {
        self.token_count
    }

    pub fn max_routes_per_token(&self) -> IntExpr {
        self.max_routes_per_token
    }

    /// Replace the route weights without changing the routing table.
    pub fn with_weights(self, weights: GraphTensor) -> Self {
        Self::new(
            self.token_ids,
            self.expert_ids,
            self.slot_ids,
            weights,
            self.token_count,
            self.max_routes_per_token,
        )
    }

    /// Gather token inputs into route order.
    ///
    /// The first input axis is the flattened token axis. `[T, ...]` becomes
    /// `[R, ...]`.
    pub fn dispatch(&self, input: GraphTensor) -> GraphTensor {
        assert_same_graph(self.token_ids, input, "Routes::dispatch");
        let input_dims = input.dims();
        assert!(
            !input_dims.is_empty(),
            "dispatched input must have a token axis"
        );
        assert_same_shape(
            &[self.token_count],
            &input_dims[..1],
            "route token count and input token axis",
        );

        let payload_dims = &input_dims[1..];
        let out_dims = route_output_shape(self.token_ids, payload_dims);
        let mut coords = Vec::with_capacity(input_dims.len());
        coords.push(self.token_ids.expand_rhs(payload_dims));
        for axis in 0..payload_dims.len() {
            coords.push(input.graph().iota(out_dims.clone(), |c| c[axis + 1]));
        }
        input.gather(&coords)
    }

    /// Select an arbitrary tensor from an expert parameter bank.
    ///
    /// The parameter bank's first axis is the expert axis. `[E, ...]` becomes
    /// `[R, ...]`.
    pub fn select(&self, expert_tensor: GraphTensor) -> GraphTensor {
        select_expert_tensor(self.expert_ids, expert_tensor)
    }

    /// Apply route weights, scatter into unique token/slot positions, and sum
    /// the slot axis.
    ///
    /// `[R, ...]` becomes `[T, ...]`. The temporary semantic shape is
    /// `[T, max_routes_per_token, ...]`.
    pub fn combine(&self, routed_output: GraphTensor) -> GraphTensor {
        assert_same_graph(self.token_ids, routed_output, "Routes::combine");
        let output_dims = routed_output.dims();
        assert!(
            !output_dims.is_empty(),
            "routed output must have a route axis"
        );
        assert_same_shape(
            &self.token_ids.dims(),
            &output_dims[..1],
            "route table and routed output route axes",
        );

        let payload_dims = &output_dims[1..];
        let weights = self
            .weights
            .cast(routed_output.dtype)
            .expand_rhs(payload_dims);
        let weighted = routed_output * weights;

        let mut destination_dims = Vec::with_capacity(output_dims.len() + 1);
        destination_dims.push(self.token_count);
        destination_dims.push(self.max_routes_per_token);
        destination_dims.extend_from_slice(payload_dims);
        let destination = routed_output
            .graph()
            .constant(0)
            .cast(routed_output.dtype)
            .expand_rhs(destination_dims);

        let mut coords = Vec::with_capacity(output_dims.len() + 1);
        coords.push(self.token_ids.expand_rhs(payload_dims));
        coords.push(self.slot_ids.expand_rhs(payload_dims));
        for axis in 0..payload_dims.len() {
            coords.push(
                routed_output
                    .graph()
                    .iota(output_dims.clone(), |c| c[axis + 1]),
            );
        }
        destination.scatter(&coords, weighted).sum(1)
    }
}

fn select_expert_tensor(expert_ids: GraphTensor, expert_tensor: GraphTensor) -> GraphTensor {
    assert_same_graph(expert_ids, expert_tensor, "expert selection");
    assert_eq!(
        expert_ids.dtype,
        DType::Int,
        "selected expert IDs must be Int"
    );
    let expert_dims = expert_tensor.dims();
    assert!(
        !expert_dims.is_empty(),
        "expert parameter bank must have an expert axis"
    );
    let route_dims = expert_ids.dims();
    let parameter_dims = &expert_dims[1..];
    let out_dims = route_output_shape(expert_ids, parameter_dims);
    let mut coords = Vec::with_capacity(expert_dims.len());
    coords.push(expert_ids.expand_rhs(parameter_dims));
    for axis in 0..parameter_dims.len() {
        coords.push(
            expert_tensor
                .graph()
                .iota(out_dims.clone(), |c| c[route_dims.len() + axis]),
        );
    }
    expert_tensor.gather(&coords)
}

fn route_output_shape(routes: GraphTensor, payload_dims: &[IntExpr]) -> Vec<IntExpr> {
    let mut dims = routes.dims();
    dims.extend_from_slice(payload_dims);
    dims
}

fn assert_same_graph(lhs: GraphTensor, rhs: GraphTensor, context: &str) {
    assert!(
        lhs.graph_ref == rhs.graph_ref,
        "{context} tensors must belong to the same graph"
    );
}

fn assert_same_shape(lhs: &[IntExpr], rhs: &[IntExpr], context: &str) {
    assert!(
        lhs.len() == rhs.len()
            && lhs
                .iter()
                .zip(rhs)
                .all(|(left, right)| left == right || left.egglog_equal(*right)),
        "{context} shapes differ: {lhs:?} vs {rhs:?}"
    );
}

#[cfg(test)]
mod tests {
    use super::{Routes, TopKRoutes};
    use luminal::prelude::*;

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-4 * expected.abs().max(1.0),
                "element {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn top_k_routes_select_dispatch_normalize_and_combine() {
        const TOKENS: usize = 2;
        const EXPERTS: usize = 3;
        const K: usize = 2;
        const INPUT: usize = 2;
        const OUTPUT: usize = 2;

        let mut cx = Graph::new();
        let scores = cx.tensor((TOKENS, EXPERTS), DType::F32);
        let expert_ids = cx.tensor((TOKENS, K), DType::Int);
        let input = cx.tensor((TOKENS, INPUT), DType::F32);
        let expert_weights = cx.tensor((EXPERTS, INPUT, OUTPUT), DType::F32);

        let routes = TopKRoutes::from_scores(scores, expert_ids).normalize();
        let dispatched = routes.dispatch(input);
        let selected = routes.select(expert_weights);
        let routed = dispatched.unsqueeze(2).matmul(selected).squeeze(2);
        let output = routes.combine(routed).output();

        let score_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0];
        let id_values = vec![2, 0, 1, 2];
        let input_values = vec![1.0, 2.0, -1.0, 3.0];
        let expert_values = vec![
            1.0, 0.0, 0.0, 1.0, // expert 0
            2.0, 0.0, 0.0, 2.0, // expert 1
            1.0, 1.0, 1.0, -1.0, // expert 2
        ];
        let expected = vec![2.5, -0.25, -4.0 / 3.0, 13.0 / 3.0];

        let runtime = luminal_reference::harness::run_reference(
            &cx,
            &[
                (scores.id, score_values.into()),
                (expert_ids.id, id_values.into()),
                (input.id, input_values.into()),
                (expert_weights.id, expert_values.into()),
            ],
        );
        assert_close(runtime.get_f32(output.id).expect("output"), &expected);
    }

    #[test]
    fn top_k_routes_preserve_all_token_axes() {
        let mut cx = Graph::new();
        let expert_ids = cx.tensor((2, 3, 2), DType::Int);
        let weights = cx.tensor((2, 3, 2), DType::F32);
        let routes = TopKRoutes::new(expert_ids, weights);

        let dispatched = routes.dispatch(cx.tensor((2, 3, 4), DType::F32));
        let selected = routes.select(cx.tensor((5, 4, 6), DType::F32));
        let combined = routes.combine(cx.tensor((2, 3, 2, 7), DType::F32));

        let concrete = |tensor: GraphTensor| {
            tensor
                .dims()
                .iter()
                .map(|dim| dim.to_usize().expect("static test dimension"))
                .collect::<Vec<_>>()
        };
        assert_eq!(concrete(dispatched), vec![2, 3, 2, 4]);
        assert_eq!(concrete(selected), vec![2, 3, 2, 4, 6]);
        assert_eq!(concrete(combined), vec![2, 3, 7]);
    }

    #[test]
    fn general_routes_scatter_into_slots_then_sum() {
        const TOKENS: usize = 3;
        const EXPERTS: usize = 2;
        const ROUTES: usize = 5;
        const SLOTS: usize = 2;
        const WIDTH: usize = 2;

        let mut cx = Graph::new();
        let token_ids = cx.tensor(ROUTES, DType::Int);
        let expert_ids = cx.tensor(ROUTES, DType::Int);
        let slot_ids = cx.tensor(ROUTES, DType::Int);
        let weights = cx.tensor(ROUTES, DType::F32);
        let input = cx.tensor((TOKENS, WIDTH), DType::F32);
        let expert_weights = cx.tensor((EXPERTS, WIDTH, WIDTH), DType::F32);

        let routes = Routes::new(token_ids, expert_ids, slot_ids, weights, TOKENS, SLOTS);
        let dispatched = routes.dispatch(input);
        let selected = routes.select(expert_weights);
        let routed = dispatched.unsqueeze(1).matmul(selected).squeeze(1);
        let output = routes.combine(routed).output();

        let token_values = vec![2, 0, 1, 2, 0];
        let expert_values = vec![1, 0, 1, 0, 1];
        let slot_values = vec![1, 0, 0, 0, 1];
        let weight_values = vec![0.4, 0.25, 1.0, 0.6, 0.75];
        let input_values = vec![1.0, 2.0, -1.0, 3.0, 2.0, -2.0];
        let matrix_values = vec![
            1.0, 0.0, 0.0, 1.0, // expert 0: identity
            2.0, 0.0, 0.0, 2.0, // expert 1: 2 * identity
        ];
        let expected = vec![1.75, 3.5, -2.0, 6.0, 2.8, -2.8];

        let runtime = luminal_reference::harness::run_reference(
            &cx,
            &[
                (token_ids.id, token_values.into()),
                (expert_ids.id, expert_values.into()),
                (slot_ids.id, slot_values.into()),
                (weights.id, weight_values.into()),
                (input.id, input_values.into()),
                (expert_weights.id, matrix_values.into()),
            ],
        );
        assert_close(runtime.get_f32(output.id).expect("output"), &expected);
    }

    #[test]
    fn top_k_routes_convert_to_general_routes() {
        const TOKENS: usize = 2;
        const K: usize = 2;
        const WIDTH: usize = 2;

        let mut cx = Graph::new();
        let expert_ids = cx.tensor((TOKENS, K), DType::Int);
        let weights = cx.tensor((TOKENS, K), DType::F32);
        let routed = cx.tensor((TOKENS, K, WIDTH), DType::F32);
        let top_k = TopKRoutes::new(expert_ids, weights);
        let structured = top_k.combine(routed).output();
        let general = top_k
            .into_routes()
            .combine(routed.merge_dims(0, 1))
            .output();

        let runtime = luminal_reference::harness::run_reference(
            &cx,
            &[
                (expert_ids.id, vec![0, 1, 1, 0].into()),
                (weights.id, vec![0.25, 0.75, 0.6, 0.4].into()),
                (
                    routed.id,
                    vec![1.0, 2.0, 3.0, 4.0, -1.0, 2.0, 5.0, 1.0].into(),
                ),
            ],
        );
        assert_close(
            runtime.get_f32(general.id).expect("general output"),
            runtime.get_f32(structured.id).expect("structured output"),
        );
    }
}

use luminal::prelude::*;

/// A layer of E experts and a router
pub struct MoE {
    pub expert_weights: GraphTensor, // [E, in, out]
    pub router: GraphTensor,         // [in, E]
    pub k: usize,
}

impl MoE {
    pub fn forward(&self, activations: GraphTensor) -> GraphTensor {
        let n = activations.dims().len();
        let e_dim = *self.router.dims().last().unwrap();
        let (_, in_size, out_size) = self.expert_weights.dims3();
        let io = in_size * out_size;
        let k_expr = Expression::from(self.k);

        // 1. Routing probabilities: [batch.., E]
        let routing_weights = activations.matmul(self.router).softmax(n - 1);

        // 2. Top-k expert indices: [batch.., k] (Int)
        let top_k_indices = routing_weights.topk_indexes(self.k, n - 1);

        // 3. Gather top-k routing values: [batch.., k]
        //    flat_idx = batch_row * E + expert_idx
        //    iota(z / k * E) gives batch_row * E at each position in [batch.., k]
        let row_offsets = activations
            .graph()
            .iota(Expression::from('z') / k_expr * e_dim, top_k_indices.dims());
        let routing_flat_idx =
            (row_offsets.cast(DType::F32) + top_k_indices.cast(DType::F32)).cast(DType::Int);
        let top_k_values = routing_weights.gather(routing_flat_idx); // [batch.., k]

        // 4. Gather expert weight matrices: [batch.., k, in, out]
        //    flat_idx[.., ki, i, o] = expert_idx[.., ki] * in*out + i * out + o
        let base = (top_k_indices * io).cast(DType::F32); // [batch.., k]
        let within = activations
            .graph()
            .iota(Expression::from('z'), (in_size, out_size))
            .cast(DType::F32); // [in, out] values 0..in*out-1

        // Expand base to [batch.., k, in, out]
        let n_base = base.dims().len();
        let exp_base = base
            .expand_dim(n_base, in_size)
            .expand_dim(n_base + 1, out_size);

        // Expand within to [batch.., k, in, out]
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }

        let expert_flat_idx = (exp_base + exp_within).cast(DType::Int);
        let gathered = self.expert_weights.gather(expert_flat_idx); // [batch.., k, in, out]

        // 5. Batched matmul: [batch.., k, 1, in] @ [batch.., k, in, out] → [batch.., k, out]
        let expanded_act = activations
            .expand_dim(n - 1, self.k) // [batch.., k, in]
            .unsqueeze(n); // [batch.., k, 1, in]
        let expert_out = expanded_act.matmul(gathered).squeeze(n); // [batch.., k, out]

        // 6. Weighted sum over experts: [batch.., k, out] * [batch.., k, 1] → sum(k) → [batch.., out]
        let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [batch.., k, 1]
        weights_exp.shape.expand(expert_out.dims());
        (expert_out * weights_exp).sum(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::MoE;
    use luminal::prelude::*;
    use rand::{Rng, rng};

    fn random_vec(n: usize) -> Vec<f32> {
        let mut r = rng();
        (0..n).map(|_| r.random_range(-0.5..0.5)).collect()
    }

    fn assert_close(a: &[f32], b: &[f32]) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (x - y).abs();
            if diff > 1e-3 {
                panic!(
                    "{x} is not close to {y} at index {i}, diff={diff}\n  actual:   {a:?}\n  expected: {b:?}"
                );
            }
        }
    }

    /// Reference MoE computation for a single input vector.
    /// input: [in_dim], router: [in_dim, n_experts] (row-major),
    /// expert_weights: [n_experts, in_dim, out_dim] (row-major)
    fn moe_reference_1d(
        input: &[f32],
        router: &[f32],
        expert_weights: &[f32],
        n_experts: usize,
        in_dim: usize,
        out_dim: usize,
        k: usize,
    ) -> Vec<f32> {
        // 1. Router logits: input @ router → [n_experts]
        let mut logits = vec![0.0f32; n_experts];
        for e in 0..n_experts {
            for i in 0..in_dim {
                logits[e] += input[i] * router[i * n_experts + e];
            }
        }

        // 2. Softmax
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|x| (x - max_l).exp()).collect();
        let sum_e: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|x| x / sum_e).collect();

        // 3. Top-k indices (descending by probability)
        let mut indices: Vec<usize> = (0..n_experts).collect();
        indices.sort_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());
        let top_k_idx = &indices[..k];
        let top_k_w: Vec<f32> = top_k_idx.iter().map(|&i| probs[i]).collect();

        // 4. Weighted sum of expert outputs (no renormalization, matching code intent)
        let mut output = vec![0.0f32; out_dim];
        for (ki, &eidx) in top_k_idx.iter().enumerate() {
            for o in 0..out_dim {
                let mut val = 0.0f32;
                for i in 0..in_dim {
                    val += input[i] * expert_weights[eidx * in_dim * out_dim + i * out_dim + o];
                }
                output[o] += top_k_w[ki] * val;
            }
        }
        output
    }

    /// Reference MoE for batched input [batch, in_dim]
    #[allow(clippy::too_many_arguments)]
    fn moe_reference_batch(
        input: &[f32],
        router: &[f32],
        expert_weights: &[f32],
        n_experts: usize,
        in_dim: usize,
        out_dim: usize,
        k: usize,
        batch: usize,
    ) -> Vec<f32> {
        let mut output = Vec::with_capacity(batch * out_dim);
        for b in 0..batch {
            let inp = &input[b * in_dim..(b + 1) * in_dim];
            let out = moe_reference_1d(inp, router, expert_weights, n_experts, in_dim, out_dim, k);
            output.extend_from_slice(&out);
        }
        output
    }

    // ── Test: 1D input, k=1, strongly-routed to expert 0 ────────────────
    #[test]
    fn test_moe_1d_k1() {
        let n_experts = 2;
        let in_dim = 3;
        let out_dim = 2;
        let k = 1;

        let mut cx = Graph::new();
        let input = cx.tensor(in_dim);
        let expert_w = cx.tensor((n_experts, in_dim, out_dim));
        let router_w = cx.tensor((in_dim, n_experts));

        let moe = MoE {
            expert_weights: expert_w,
            router: router_w,
            k,
        };
        let output = moe.forward(input).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        let input_data = vec![1.0, 2.0, 3.0];
        // Router strongly favors expert 0
        let router_data = vec![
            10.0, -10.0, // feature 0
            10.0, -10.0, // feature 1
            10.0, -10.0, // feature 2
        ];
        // Expert 0: simple linear, Expert 1: different
        let expert_data = vec![
            // Expert 0: [3x2]
            1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // Expert 1: [3x2]
            2.0, 0.0, 0.0, 2.0, 2.0, 2.0,
        ];

        rt.set_data(input.id, input_data.clone());
        rt.set_data(router_w.id, router_data.clone());
        rt.set_data(expert_w.id, expert_data.clone());
        rt.execute(&cx.dyn_map);

        let expected = moe_reference_1d(
            &input_data,
            &router_data,
            &expert_data,
            n_experts,
            in_dim,
            out_dim,
            k,
        );
        // With strong routing to expert 0: output ≈ [1,2,3]@[[1,0],[0,1],[1,1]] = [4, 5]
        assert_close(rt.get_f32(output.id), &expected);
    }

    // ── Test: 1D input, k=E (all experts selected) ─────────────────────
    #[test]
    fn test_moe_1d_k_equals_e() {
        let n_experts = 3;
        let in_dim = 2;
        let out_dim = 2;
        let k = 3; // select all experts

        let mut cx = Graph::new();
        let input = cx.tensor(in_dim);
        let expert_w = cx.tensor((n_experts, in_dim, out_dim));
        let router_w = cx.tensor((in_dim, n_experts));

        let moe = MoE {
            expert_weights: expert_w,
            router: router_w,
            k,
        };
        let output = moe.forward(input).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        let input_data = vec![1.0, 1.0];
        // Nearly-equal routing to all experts (slight differences to avoid argsort ties)
        let router_data = vec![0.01, 0.02, 0.03, 0.01, 0.02, 0.03];
        // Each expert: identity-scaled by index+1
        let expert_data = vec![
            // Expert 0: identity
            1.0, 0.0, 0.0, 1.0, // Expert 1: 2x
            2.0, 0.0, 0.0, 2.0, // Expert 2: 3x
            3.0, 0.0, 0.0, 3.0,
        ];

        rt.set_data(input.id, input_data.clone());
        rt.set_data(router_w.id, router_data.clone());
        rt.set_data(expert_w.id, expert_data.clone());
        rt.execute(&cx.dyn_map);

        let expected = moe_reference_1d(
            &input_data,
            &router_data,
            &expert_data,
            n_experts,
            in_dim,
            out_dim,
            k,
        );
        // Equal routing: each expert weight = 1/3
        // output = 1/3 * [1,1] + 1/3 * [2,2] + 1/3 * [3,3] = [2, 2]
        assert_close(rt.get_f32(output.id), &expected);
    }

    // ── Test: 2D batched input ──────────────────────────────────────────
    #[test]
    fn test_moe_batched() {
        let n_experts = 2;
        let in_dim = 3;
        let out_dim = 2;
        let k = 1;
        let batch = 2;

        let mut cx = Graph::new();
        let input = cx.tensor((batch, in_dim));
        let expert_w = cx.tensor((n_experts, in_dim, out_dim));
        let router_w = cx.tensor((in_dim, n_experts));

        let moe = MoE {
            expert_weights: expert_w,
            router: router_w,
            k,
        };
        let output = moe.forward(input).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        let input_data = vec![
            1.0, 0.0, 0.0, // batch 0: routes to expert via feature 0
            0.0, 1.0, 0.0, // batch 1: routes to expert via feature 1
        ];
        // Router: feature 0 → expert 0, feature 1 → expert 1
        let router_data = vec![
            10.0, -10.0, // feature 0 → expert 0
            -10.0, 10.0, // feature 1 → expert 1
            0.0, 0.0, // feature 2 → neutral
        ];
        let expert_data = vec![
            // Expert 0: [3x2]
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // Expert 1: [3x2]
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];

        rt.set_data(input.id, input_data.clone());
        rt.set_data(router_w.id, router_data.clone());
        rt.set_data(expert_w.id, expert_data.clone());
        rt.execute(&cx.dyn_map);

        let expected = moe_reference_batch(
            &input_data,
            &router_data,
            &expert_data,
            n_experts,
            in_dim,
            out_dim,
            k,
            batch,
        );
        assert_close(rt.get_f32(output.id), &expected);
    }

    // ── Test: random inputs with k=2 ────────────────────────────────────
    #[test]
    fn test_moe_random_k2() {
        let n_experts = 4;
        let in_dim = 8;
        let out_dim = 4;
        let k = 2;

        let mut cx = Graph::new();
        let input = cx.tensor(in_dim);
        let expert_w = cx.tensor((n_experts, in_dim, out_dim));
        let router_w = cx.tensor((in_dim, n_experts));

        let moe = MoE {
            expert_weights: expert_w,
            router: router_w,
            k,
        };
        let output = moe.forward(input).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        let input_data = random_vec(in_dim);
        let router_data = random_vec(in_dim * n_experts);
        let expert_data = random_vec(n_experts * in_dim * out_dim);

        rt.set_data(input.id, input_data.clone());
        rt.set_data(router_w.id, router_data.clone());
        rt.set_data(expert_w.id, expert_data.clone());
        rt.execute(&cx.dyn_map);

        let expected = moe_reference_1d(
            &input_data,
            &router_data,
            &expert_data,
            n_experts,
            in_dim,
            out_dim,
            k,
        );
        assert_close(rt.get_f32(output.id), &expected);
    }

    // ── Test: batched random inputs ─────────────────────────────────────
    #[test]
    fn test_moe_batched_random() {
        let n_experts = 3;
        let in_dim = 4;
        let out_dim = 3;
        let k = 2;
        let batch = 4;

        let mut cx = Graph::new();
        let input = cx.tensor((batch, in_dim));
        let expert_w = cx.tensor((n_experts, in_dim, out_dim));
        let router_w = cx.tensor((in_dim, n_experts));

        let moe = MoE {
            expert_weights: expert_w,
            router: router_w,
            k,
        };
        let output = moe.forward(input).output();

        cx.build_search_space::<ReferenceRuntime>(CompileOptions::default());
        let mut rt = cx.search(
            ReferenceRuntime::default(),
            CompileOptions::default().search_graph_limit(1),
        );

        let input_data = random_vec(batch * in_dim);
        let router_data = random_vec(in_dim * n_experts);
        let expert_data = random_vec(n_experts * in_dim * out_dim);

        rt.set_data(input.id, input_data.clone());
        rt.set_data(router_w.id, router_data.clone());
        rt.set_data(expert_w.id, expert_data.clone());
        rt.execute(&cx.dyn_map);

        let expected = moe_reference_batch(
            &input_data,
            &router_data,
            &expert_data,
            n_experts,
            in_dim,
            out_dim,
            k,
            batch,
        );
        assert_close(rt.get_f32(output.id), &expected);
    }

    /// Dump the egglog HLIR for a QwenMoE-style GLU-MoE pattern.
    /// This helps identify the exact pattern for the GLUMoE backend HostOp.
    #[test]
    fn dump_glu_moe_egglog() {
        use luminal::dtype::DType;
        use luminal::egglog_utils::hlir_to_egglog;

        let n_experts = 4;
        let hidden = 8;
        let intermediate = 4;
        let top_k: usize = 2;

        let mut cx = Graph::new();

        // Input tensors
        let x = cx.tensor(('s', hidden));
        let router = cx.tensor((n_experts, hidden));
        let gate_up_weights = cx
            .tensor((n_experts, intermediate * 2, hidden))
            .as_dtype(DType::Bf16);
        let down_weights = cx
            .tensor((n_experts, hidden, intermediate))
            .as_dtype(DType::Bf16);

        let n = x.dims().len(); // 2
        let e_dim = *router.dims().first().unwrap(); // E
        let k_expr = luminal::shape::Expression::from(top_k);

        // 1. Router: softmax(x @ router^T) → [s, E]
        let routing_weights = x.matmul(router.t()).softmax(n - 1);

        // 2. TopK expert selection → [s, k] (Int)
        let top_k_indices = routing_weights.topk_indexes(top_k, n - 1);

        // 3. Gather top-k routing values → [s, k]
        let row_offsets = cx.iota(
            luminal::shape::Expression::from('z') / k_expr * e_dim,
            top_k_indices.dims(),
        );
        let routing_flat_idx =
            (row_offsets.cast(DType::F32) + top_k_indices.cast(DType::F32)).cast(DType::Int);
        let top_k_values = routing_weights.gather(routing_flat_idx);

        // 4. Gather gate_up expert weights → [s, k, intermediate*2, H]
        let gate_up_gathered =
            gather_experts_test(x, top_k_indices, gate_up_weights).cast(DType::F32);
        let x_exp = x.expand_dim(n - 1, top_k).unsqueeze(n); // [s, k, 1, H]
        let gate_up_out = x_exp.matmul(gate_up_gathered.transpose(2, 3)).squeeze(n); // [s, k, intermediate*2]

        // 5. SwiGLU: silu(gate) * up → [s, k, intermediate]
        let gate = gate_up_out.slice((.., .., ..intermediate));
        let up = gate_up_out.slice((.., .., intermediate..));
        let hidden_act = gate.silu() * up;

        // 6. Gather down expert weights → [s, k, H, intermediate]
        let down_gathered = gather_experts_test(x, top_k_indices, down_weights).cast(DType::F32);
        let hidden_exp = hidden_act.unsqueeze(2); // [s, k, 1, intermediate]
        let down_out = hidden_exp.matmul(down_gathered.transpose(2, 3)).squeeze(2); // [s, k, H]

        // 7. Weighted sum over k experts → [s, H]
        let mut weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [s, k, 1]
        weights_exp.shape.expand(down_out.dims());
        let _output = (down_out * weights_exp).sum(n - 1).output();

        // Dump the HLIR to egglog
        let (program, root) = hlir_to_egglog(&cx);
        println!("=== GLU-MoE HLIR Egglog Dump ===");
        println!("Root: {root}");
        println!("{program}");
    }

    /// Helper: gather expert weight matrices using topk indices.
    fn gather_experts_test(
        graph_source: GraphTensor,
        top_k_indices: GraphTensor,
        weights: GraphTensor,
    ) -> GraphTensor {
        let (_, d1, d2) = weights.dims3();
        let io = d1 * d2;
        let base = (top_k_indices * io).cast(DType::F32);
        let within = graph_source
            .graph()
            .iota(luminal::shape::Expression::from('z'), (d1, d2))
            .cast(DType::F32);
        let n_base = base.dims().len();
        let exp_base = base.expand_dim(n_base, d1).expand_dim(n_base + 1, d2);
        let mut exp_within = within;
        for (i, dim) in base.dims().iter().enumerate() {
            exp_within = exp_within.expand_dim(i, *dim);
        }
        let expert_flat_idx = (exp_base + exp_within).cast(DType::Int);
        weights.gather(expert_flat_idx)
    }
}

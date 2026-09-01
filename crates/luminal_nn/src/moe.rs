use crate::Linear;
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
        // 1. Routing probabilities: [batch.., E]
        let routing_weights = activations.matmul(self.router).softmax(n - 1);

        // 2. Top-k expert indices: [batch.., k] (Int)
        let top_k_indices = routing_weights.topk_indexes(self.k, n - 1);

        // 3. Gather top-k routing values: [batch.., k]
        //    flat_idx = batch_row * E + expert_idx
        //    iota(z / k * E) gives batch_row * E at each position in [batch.., k]
        // Batch-row offset · E as a coordinate function: leading batch
        // axes carry their row-major strides scaled by E; the trailing k
        // axis contributes 0 (P1, 2026-08-07 — the old flat form recorded
        // a TruncDiv chain here).
        let idx_dims = top_k_indices.dims();
        let row_offsets = activations.graph().iota(idx_dims.clone(), |c| {
            let n = idx_dims.len();
            let mut stride = e_dim;
            let mut acc = IntExpr::from(0);
            for i in (0..n - 1).rev() {
                acc += c[i] * stride;
                stride = (stride * idx_dims[i]).simplify();
            }
            acc
        });
        // Int-native flat-index assembly (2026-08-11): both operands are
        // already Int; the old f32 round trip ended in a refused cast.
        let routing_flat_idx = row_offsets + top_k_indices;
        let top_k_values = routing_weights.gather1d(routing_flat_idx); // [batch.., k]

        // 4. Gather expert weight matrices: [batch.., k, in, out]
        //    flat_idx[.., ki, i, o] = expert_idx[.., ki] * in*out + i * out + o
        let base = top_k_indices * io; // [batch.., k] (Int)
        let within = activations
            .graph()
            .iota((in_size, out_size), |c| c[0] * out_size + c[1]); // [in, out] (Int)

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

        let expert_flat_idx = exp_base + exp_within;
        let gathered = self.expert_weights.gather1d(expert_flat_idx); // [batch.., k, in, out]

        // 5. Batched matmul: [batch.., k, 1, in] @ [batch.., k, in, out] → [batch.., k, out]
        let expanded_act = activations
            .expand_dim(n - 1, self.k) // [batch.., k, in]
            .unsqueeze(n); // [batch.., k, 1, in]
        let expert_out = expanded_act.matmul(gathered).squeeze(n); // [batch.., k, out]

        // 6. Weighted sum over experts: [batch.., k, out] * [batch.., k, 1] → sum(k) → [batch.., out]
        let weights_exp = top_k_values.unsqueeze(top_k_values.dims().len()); // [batch.., k, 1]
        let weights_exp = weights_exp.expand(expert_out.dims());
        (expert_out * weights_exp).sum(n - 1)
    }
}

#[cfg(test)]
mod tests {
    use super::MoE;
    use luminal::prelude::*;

    fn assert_close(ours: &[f32], expected: &[f32]) {
        assert_eq!(ours.len(), expected.len(), "length mismatch");
        for (index, (a, b)) in ours.iter().zip(expected).enumerate() {
            assert!(
                (a - b).abs() <= 1e-4 * b.abs().max(1.0),
                "element {index}: ours {a} vs expected {b}"
            );
        }
    }

    /// Full MoE forward (router softmax → topk → routing-value and
    /// expert-weight gathers → batched matmul → weighted sum) against a
    /// scalar reference, k=1 over 2 experts. Runs the plain extraction
    /// path — the chain rides stable_argsort's rank scatter, both flat
    /// gather sugars, and Int↔F32 index arithmetic.
    #[test]
    fn moe_forward_matches_scalar_reference() {
        const E: usize = 2;
        const IN: usize = 2;
        const OUT: usize = 2;
        const BATCH: usize = 3;

        let mut cx = Graph::new();
        let model = MoE {
            expert_weights: cx.named_tensor("Experts", (E, IN, OUT)),
            router: cx.named_tensor("Router", (IN, E)),
            k: 1,
        };
        let x = cx.tensor((BATCH, IN));
        let out = model.forward(x).output();

        let x_vals = vec![1.0f32, 0.5, -1.0, 2.0, 0.25, -0.75];
        // Router picks expert 0 for positive-x0-heavy rows, expert 1 otherwise
        // (logits differ per row; no ties).
        let router_vals = vec![2.0f32, -1.0, -0.5, 1.5];
        let expert_vals: Vec<f32> = (0..E * IN * OUT).map(|v| v as f32 * 0.3 - 1.0).collect();

        // Scalar reference.
        let mut expected = vec![0.0f32; BATCH * OUT];
        for b in 0..BATCH {
            let xr = &x_vals[b * IN..(b + 1) * IN];
            let mut logits = [0.0f32; E];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..IN).map(|i| xr[i] * router_vals[i * E + e]).sum();
            }
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let best = (0..E)
                .max_by(|a, b| logits[*a].partial_cmp(&logits[*b]).unwrap())
                .unwrap();
            let weight = exps[best] / denom;
            let w = &expert_vals[best * IN * OUT..(best + 1) * IN * OUT];
            for o in 0..OUT {
                expected[b * OUT + o] =
                    (0..IN).map(|i| xr[i] * w[i * OUT + o]).sum::<f32>() * weight;
            }
        }

        let rt = luminal::test_support::run_reference(
            &cx,
            &[
                (x.id, x_vals.into()),
                (model.router.id, router_vals.into()),
                (model.expert_weights.id, expert_vals.into()),
            ],
        );
        assert_close(rt.get_f32(out.id).expect("output"), &expected);
    }
}

/// The full-fidelity top-k mixture (Qwen3-MoE form, ruling 2026-08-12):
/// scores = softmax over ALL experts FIRST, then top-k selection, then
/// the selected probabilities RENORMALIZE to sum 1 (norm_topk_prob).
/// Expert weights are host-pre-stacked — gate_up [E, 2·I, H] (each
/// expert's gate rows then up rows) and down [E, H, I] — and fetched
/// per selected expert by COORDINATE-form gather (the primary; flat
/// indices at expert-tensor scale would also be provable but the
/// coordinate spelling never builds them). The in-graph gate/up split
/// slices a COMPUTE output (no concat downstream — not the divergence
/// road).
pub struct MoETopK {
    /// (E, hidden) router — F32, learned; HF orientation.
    pub router: Linear,
    /// (E, 2·intermediate, hidden) stacked gate;up weights.
    pub gate_up: GraphTensor,
    /// (E, hidden, intermediate) stacked down weights.
    pub down: GraphTensor,
    pub experts: usize,
    pub top_k: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

impl MoETopK {
    pub fn new(
        hidden: usize,
        intermediate: usize,
        experts: usize,
        top_k: usize,
        ns: &Ns,
        cx: &mut Graph,
    ) -> Self {
        Self {
            router: Linear::new_permuted(hidden, experts, false, &ns.child("gate"), cx),
            gate_up: cx.named_tensor(
                ns.leaf("gate_up_weights"),
                (experts, 2 * intermediate, hidden),
            ),
            down: cx.named_tensor(ns.leaf("down_weights"), (experts, hidden, intermediate)),
            experts,
            top_k,
            hidden,
            intermediate,
        }
    }

    /// Build the (E, 2I, H)/(E, H, I) stacks IN-GRAPH from per-expert
    /// (gate, up, down) tensors — checkpoint anatomy enters unfused and
    /// the graph performs the fusion (ruling 2026-08-12: no
    /// value-transforming preparation steps). Dims derive from the
    /// parts; gate/up are (I, H), down is (H, I).
    pub fn from_per_expert(
        router: Linear,
        parts: &[(GraphTensor, GraphTensor, GraphTensor)],
        top_k: usize,
    ) -> Self {
        let experts = parts.len();
        let (gate0, _, _) = parts[0];
        let intermediate = gate0.dims()[0].to_usize().expect("static intermediate");
        let hidden = gate0.dims()[1].to_usize().expect("static hidden");
        let mut gate_up_stack: Option<GraphTensor> = None;
        let mut down_stack: Option<GraphTensor> = None;
        for (gate, up, down) in parts {
            let fused = gate.concat_along(*up, 0).expand_dim(0, 1); // (1, 2I, H)
            let down_row = down.expand_dim(0, 1); // (1, H, I)
            gate_up_stack = Some(match gate_up_stack {
                Some(acc) => acc.concat_along(fused, 0),
                None => fused,
            });
            down_stack = Some(match down_stack {
                Some(acc) => acc.concat_along(down_row, 0),
                None => down_row,
            });
        }
        Self {
            router,
            gate_up: gate_up_stack.expect("at least one expert"),
            down: down_stack.expect("at least one expert"),
            experts,
            top_k,
            hidden,
            intermediate,
        }
    }

    /// x (s, hidden) → (s, hidden).
    pub fn forward(&self, x: GraphTensor) -> GraphTensor {
        let s = x.dims()[0];
        let (k, h, i2) = (self.top_k, self.hidden, 2 * self.intermediate);
        let cx = x.graph();

        // 1. Qwen3 scoring order: softmax over ALL experts, THEN top-k.
        let probs = self.router.forward(x).softmax(1); // (s, E)
        let idx = probs.topk_indexes(k, 1); // (s, k) Int

        // 2. Selected probabilities, renormalized to sum 1.
        let row_of = cx.iota((s, k), |c| c[0]);
        let picked = probs.gather(&[row_of, idx]); // (s, k)
        let denom = picked.sum(1).expand_dim(1, k);
        let weights = picked / denom;

        // 3. Fetch the selected experts' gate_up matrices: (s, k, 2I, H).
        let e4 = idx.expand_dim(2, i2).expand_dim(3, h);
        let r4 = cx.iota((s, k, i2, h), |c| c[2]);
        let c4 = cx.iota((s, k, i2, h), |c| c[3]);
        let gate_up = self.gate_up.gather(&[e4, r4, c4]);

        // 4. Per-expert projection: (s,k,1,H) @ (s,k,H,2I) → (s,k,2I).
        let x_e = x.expand_dim(1, k).expand_dim(2, 1);
        let projected = x_e.matmul(gate_up.permute((0, 1, 3, 2))).squeeze(2);

        // 5. SwiGLU on the fused halves (slices of a compute output).
        let gate = projected.slice_along(..self.intermediate, 2);
        let up = projected.slice_along(self.intermediate.., 2);
        let hidden_states = gate.silu() * up; // (s, k, I)

        // 6. Down projection: (s,k,1,I) @ (s,k,I,H) → (s,k,H).
        let e_down = idx.expand_dim(2, h).expand_dim(3, self.intermediate);
        let r_down = cx.iota((s, k, h, self.intermediate), |c| c[2]);
        let c_down = cx.iota((s, k, h, self.intermediate), |c| c[3]);
        let down = self.down.gather(&[e_down, r_down, c_down]); // (s,k,H,I)
        let out_e = hidden_states
            .expand_dim(2, 1)
            .matmul(down.permute((0, 1, 3, 2)))
            .squeeze(2); // (s, k, H)

        // 7. Weighted sum over the k experts.
        (out_e * weights.expand_dim(2, h)).sum(1)
    }
}

#[cfg(test)]
mod topk_tests {
    use super::MoETopK;
    use luminal::prelude::*;
    use scalar_refs::*;

    /// The full Qwen3-MoE chain against a scalar reference: softmax over
    /// ALL experts first, top-k by stable ranking, renormalized picked
    /// probabilities, per-expert fused gate;up projection, SwiGLU on the
    /// sliced halves, down projection, weighted sum.
    #[test]
    fn moe_topk_matches_scalar_reference() {
        const S: usize = 2;
        const H: usize = 4;
        const I: usize = 3;
        const E: usize = 4;
        const K: usize = 2;

        let mut cx = Graph::new();
        let moe = MoETopK::new(H, I, E, K, &Ns::root().child("mlp"), &mut cx);
        let x = cx.tensor((S, H));
        let out = moe.forward(x).output();

        let x_vals = weights(S * H, 3);
        let router_vals = weights(E * H, 5); // (E, H) HF orientation
        let gate_up_vals = weights(E * 2 * I * H, 7);
        let down_vals = weights(E * H * I, 9);

        // Scalar reference.
        let mut expected = vec![0f32; S * H];
        for s_i in 0..S {
            let xr = &x_vals[s_i * H..(s_i + 1) * H];
            // Router logits (x @ router.t()) then softmax over E.
            let mut logits = [0f32; E];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..H).map(|j| xr[j] * router_vals[e * H + j]).sum();
            }
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / denom).collect();
            // Top-k, largest first, stable (ties by lower index).
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap().then(a.cmp(b)));
            let picked: Vec<usize> = order[..K].to_vec();
            let picked_sum: f32 = picked.iter().map(|e| probs[*e]).sum();
            for expert in &picked {
                let weight = probs[*expert] / picked_sum;
                // Fused gate;up: (2I, H) rows.
                let w = &gate_up_vals[expert * 2 * I * H..(expert + 1) * 2 * I * H];
                let mut projected = [0f32; 2 * I];
                for (r, slot) in projected.iter_mut().enumerate() {
                    *slot = (0..H).map(|j| xr[j] * w[r * H + j]).sum();
                }
                let hidden: Vec<f32> = (0..I)
                    .map(|r| {
                        let g = projected[r];
                        let silu = g / (1.0 + (-g).exp());
                        silu * projected[I + r]
                    })
                    .collect();
                let d = &down_vals[expert * H * I..(expert + 1) * H * I];
                for r in 0..H {
                    let v: f32 = (0..I).map(|j| hidden[j] * d[r * I + j]).sum();
                    expected[s_i * H + r] += weight * v;
                }
            }
        }

        let rt = luminal::test_support::run_reference(
            &cx,
            &[
                (x.id, x_vals.into()),
                (moe.router.weight.id, router_vals.into()),
                (moe.gate_up.id, gate_up_vals.into()),
                (moe.down.id, down_vals.into()),
            ],
        );
        assert_close(rt.get_f32(out.id).expect("moe out"), &expected);
    }

    /// THE IN-GRAPH STACKING SPELLING (ruling 2026-08-12: no
    /// value-transforming preparation steps): per-expert gate/up/down
    /// enter as separate named tensors — HF checkpoint anatomy — and
    /// the graph itself builds (E, 2I, H)/(E, H, I) by concat, feeding
    /// the same runtime-indexed gather. This is a DIVERGENCE PROBE as
    /// much as a fidelity test: gather-downstream-of-concat is
    /// structurally adjacent to the slice-of-concat rejoin family, so
    /// first runs happen under the RSS watchdog.
    #[test]
    fn moe_topk_in_graph_stacking_matches_scalar_reference() {
        const S: usize = 2;
        const H: usize = 4;
        const I: usize = 3;
        const E: usize = 4;
        const K: usize = 2;

        let mut cx = Graph::new();
        let per_expert: Vec<(GraphTensor, GraphTensor, GraphTensor)> = (0..E)
            .map(|e| {
                (
                    cx.named_tensor(format!("experts.{e}.gate_proj.weight"), (I, H)),
                    cx.named_tensor(format!("experts.{e}.up_proj.weight"), (I, H)),
                    cx.named_tensor(format!("experts.{e}.down_proj.weight"), (H, I)),
                )
            })
            .collect();
        let router = crate::Linear::new_permuted(H, E, false, &Ns::root().child("gate"), &mut cx);
        let moe = MoETopK::from_per_expert(router, &per_expert, K);
        let x = cx.tensor((S, H));
        let out = moe.forward(x).output();

        let x_vals = weights(S * H, 3);
        let router_vals = weights(E * H, 5);
        let gate_up_vals = weights(E * 2 * I * H, 7);
        let down_vals = weights(E * H * I, 9);

        // Identical scalar reference as the host-stacked test.
        let mut expected = vec![0f32; S * H];
        for s_i in 0..S {
            let xr = &x_vals[s_i * H..(s_i + 1) * H];
            let mut logits = [0f32; E];
            for (e, logit) in logits.iter_mut().enumerate() {
                *logit = (0..H).map(|j| xr[j] * router_vals[e * H + j]).sum();
            }
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
            let denom: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / denom).collect();
            let mut order: Vec<usize> = (0..E).collect();
            order.sort_by(|a, b| probs[*b].partial_cmp(&probs[*a]).unwrap().then(a.cmp(b)));
            let picked: Vec<usize> = order[..K].to_vec();
            let picked_sum: f32 = picked.iter().map(|e| probs[*e]).sum();
            for expert in &picked {
                let weight = probs[*expert] / picked_sum;
                let w = &gate_up_vals[expert * 2 * I * H..(expert + 1) * 2 * I * H];
                let mut projected = [0f32; 2 * I];
                for (r, slot) in projected.iter_mut().enumerate() {
                    *slot = (0..H).map(|j| xr[j] * w[r * H + j]).sum();
                }
                let hidden: Vec<f32> = (0..I)
                    .map(|r| {
                        let g = projected[r];
                        let silu = g / (1.0 + (-g).exp());
                        silu * projected[I + r]
                    })
                    .collect();
                let d = &down_vals[expert * H * I..(expert + 1) * H * I];
                for r in 0..H {
                    let v: f32 = (0..I).map(|j| hidden[j] * d[r * I + j]).sum();
                    expected[s_i * H + r] += weight * v;
                }
            }
        }

        // Stage the SLICES of the same value streams per expert.
        let mut pairs: Vec<(
            petgraph::graph::NodeIndex,
            luminal::buffer_tensor_ir::TypedBuffer,
        )> = vec![
            (x.id, x_vals.into()),
            (moe.router.weight.id, router_vals.into()),
        ];
        for (e, (gate, up, down)) in per_expert.iter().enumerate() {
            let fused = &gate_up_vals[e * 2 * I * H..(e + 1) * 2 * I * H];
            pairs.push((gate.id, fused[..I * H].to_vec().into()));
            pairs.push((up.id, fused[I * H..].to_vec().into()));
            pairs.push((
                down.id,
                down_vals[e * H * I..(e + 1) * H * I].to_vec().into(),
            ));
        }
        let rt = luminal::test_support::run_reference(&cx, &pairs);
        assert_close(rt.get_f32(out.id).expect("moe out"), &expected);
    }
}

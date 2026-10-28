//! Scaled-dot-product attention lowerings (port batch 6).
//!
//! All `scaled_dot_product_attention` ATen variants (unified, efficient,
//! flash, flash_for_cpu, cudnn, overrideable) share one body that lowers to
//! `softmax((Q@K^T)*scale + mask) @ V`. Args are resolved by name so one
//! body serves every variant. Every tensor slot a variant's schema returns
//! is bound (see [`SdpaSlot`]): AOTAutograd's forward graph returns the
//! saved-for-backward slots too, so a slot left unbound fails any
//! grad-mode compile.
//!
//! Q/K/V arrive at the op's compute dtype (F32 for bf16/f16, the
//! translator's opmath pattern). Keep exponential weights and both
//! contractions at that dtype; round only the original operator output.
//! Normalization follows the value contraction;
//! `is_causal` is a top-left iota mask; bool masks are keep-masks and
//! fully-masked query rows output zeros; grouped K/V heads are
//! repeat-interleaved (GQA).

use anyhow::{Context, Result};
use luminal::prelude::*;

use super::Translator;
use super::util::{self, same_dim};
use crate::pt2_schema::{Node, NodeInput};

/// What one return slot of an SDPA variant's schema carries
/// (`torch.ops.aten.<variant>.default._schema`).
#[derive(Clone, Copy, Debug)]
enum SdpaSlot {
    /// `output`: the attention result.
    Output,
    /// `logsumexp`: per query row, `ln sum_j exp(s_ij)` over the scaled,
    /// masked scores the softmax normalises. `pad` is the value a kernel
    /// stores past the last query row when its declared row extent is
    /// rounded up.
    LogSumExp { pad: Option<f32> },
    /// Dropout RNG state (`philox_seed`, `philox_offset`, `rng_state`,
    /// `unused`). Only the matching backward kernel reads it, to replay the
    /// dropout mask, and only when `dropout_p > 0`, which this lowering
    /// refuses. At `dropout_p == 0` no value of it is ever read, so zeros of
    /// the declared shape and dtype stand in. Only the CUDA variants return
    /// it; no eager value was compared.
    RngState,
    /// `cum_seq_q` / `cum_seq_k`: cumulative sequence offsets of a nested
    /// (jagged) batch. Dense inputs return an undefined tensor (`as_none`).
    CumSeq,
    /// `max_q` / `max_k`: SymInt sequence extents, not tensors. The export
    /// records their values in `sym_int_values`, where shape arguments
    /// resolve them, so there is nothing to bind.
    SeqExtent,
    /// `debug_attn_mask`: the kernel's dropout-encoded probabilities when
    /// `return_debug_mask` is set, else an empty (or undefined) tensor.
    DebugMask,
}

/// Each SDPA variant's return slots in schema order, keyed by the
/// dispatch target.
fn sdpa_slots(target: &str) -> Option<&'static [(&'static str, SdpaSlot)]> {
    use SdpaSlot::*;
    const LSE: SdpaSlot = LogSumExp { pad: None };
    Some(match target {
        "scaled_dot_product_attention.default" => &[("output", Output)],
        "_scaled_dot_product_flash_attention_for_cpu.default" => {
            &[("output", Output), ("logsumexp", LSE)]
        }
        // Memory-efficient attention aligns the logsumexp row axis to 32
        // (`ceil(L / 32) * 32`; L itself on ROCm; 0 without
        // `compute_log_sumexp`) and its forward kernel stores +inf in the
        // padding rows (mem_eff_attention/kernel_forward.h: "we pad
        // logsumexp with `inf`"). CUDA-only: not compared with an eager run.
        "_scaled_dot_product_efficient_attention.default" => &[
            ("output", Output),
            (
                "log_sumexp",
                LogSumExp {
                    pad: Some(f32::INFINITY),
                },
            ),
            ("philox_seed", RngState),
            ("philox_offset", RngState),
        ],
        "_scaled_dot_product_flash_attention.default" => &[
            ("output", Output),
            ("logsumexp", LSE),
            ("cum_seq_q", CumSeq),
            ("cum_seq_k", CumSeq),
            ("max_q", SeqExtent),
            ("max_k", SeqExtent),
            ("rng_state", RngState),
            ("unused", RngState),
            ("debug_attn_mask", DebugMask),
        ],
        "_scaled_dot_product_cudnn_attention.default"
        | "_scaled_dot_product_fused_attention_overrideable.default" => &[
            ("output", Output),
            ("logsumexp", LSE),
            ("cum_seq_q", CumSeq),
            ("cum_seq_k", CumSeq),
            ("max_q", SeqExtent),
            ("max_k", SeqExtent),
            ("philox_seed", RngState),
            ("philox_offset", RngState),
            ("debug_attn_mask", DebugMask),
        ],
        _ => return None,
    })
}

/// The tensor name each schema slot is bound to, in slot order (`None`
/// for an undefined tensor or a SymInt). A tuple return serializes one
/// argument per slot; a single `as_tensors` list cannot hold the `None`
/// and SymInt slots, so its positions are slot positions only for a
/// schema without them.
fn sdpa_slot_names(node: &Node, slots: &[(&str, SdpaSlot)]) -> Result<Vec<Option<String>>> {
    let names: Vec<Option<String>> = match node.outputs.as_slice() {
        [only] if only.as_tensors.is_some() => {
            anyhow::ensure!(
                slots
                    .iter()
                    .all(|(_, slot)| !matches!(slot, SdpaSlot::CumSeq | SdpaSlot::SeqExtent)),
                "SDPA {}: a tensor-list output cannot be aligned to a schema with \
                 None/SymInt slots",
                node.target
            );
            only.as_tensors
                .iter()
                .flatten()
                .map(|value| Some(value.name.clone()))
                .collect()
        }
        outputs => outputs
            .iter()
            .map(|output| {
                anyhow::ensure!(
                    output.as_tensors.is_none(),
                    "SDPA {}: a tensor-list output in a tuple return",
                    node.target
                );
                Ok(output.as_tensor.as_ref().map(|value| value.name.clone()))
            })
            .collect::<Result<_>>()?,
    };
    anyhow::ensure!(
        names.len() <= slots.len(),
        "SDPA {}: {} outputs, but the schema returns {}",
        node.target,
        names.len(),
        slots.len()
    );
    Ok(names
        .into_iter()
        .map(|name| name.filter(|name| !name.is_empty()))
        .collect())
}

/// Insert the unit axes a declared layout adds to `t` (cudnn's trailing
/// `[.., L, 1]` logsumexp; the overrideable op's batch of 1 for 3-D
/// inputs). Unit axes hold no data, so the values and their order are
/// unchanged; `None` when `declared` is not `t`'s dims plus unit axes.
fn insert_unit_axes(mut t: GraphTensor, declared: &[IntExpr]) -> Option<GraphTensor> {
    let dims = t.dims();
    let mut matched = 0;
    for (axis, extent) in declared.iter().enumerate() {
        if matched < dims.len() && same_dim(*extent, dims[matched]) {
            matched += 1;
        } else if extent.to_usize() == Some(1) {
            t = t.expand_dim(axis, 1usize);
        } else {
            return None;
        }
    }
    (matched == dims.len()).then_some(t)
}

/// The per-query-row statistics of the attention softmax, shared by the
/// output and the logsumexp slot.
#[derive(Clone, Copy)]
struct RowStats {
    /// `max_j s_ij` over the scaled, masked scores.
    max: GraphTensor,
    /// `sum_j exp(s_ij - max)`, before fully-masked rows are zeroed.
    denominator: GraphTensor,
    /// Bool keep-mask only: 1.0 where the row keeps at least one key.
    any_keep: Option<GraphTensor>,
}

/// Broadcast-add an additive offset (causal mask, attention bias) onto the
/// scores. Handles dtype promotion and rank extension: a `(S_q, S_k)` offset
/// broadcasts across any batch/head prefix dims of `scores`.
fn add_offset(scores: GraphTensor, offset: GraphTensor) -> GraphTensor {
    let (s, o) = util::ensure_same_dtype(scores, offset);
    let (s, o) = util::broadcast_binary(s, o);
    s + o
}

fn named_input<'a>(node: &'a Node, name: &str) -> Option<&'a NodeInput> {
    node.inputs.iter().find(|input| input.name == name)
}

impl Translator<'_> {
    /// Translate all `scaled_dot_product_attention` ATen variants (unified,
    /// efficient, flash, flash_for_cpu, cudnn, overrideable) into
    /// `softmax((Q@K^T)*scale + causal_mask + attn_bias) @ V`. Args resolve
    /// by name so one body serves every variant; every tensor slot of the
    /// variant's schema is bound (`bind_sdpa_outputs`).
    pub(super) fn translate_sdpa(&mut self, node: &Node) -> Result<()> {
        let query = self.operand(&node.inputs[0])?;
        let mut key = self.operand(&node.inputs[1])?;
        let mut value = self.operand(&node.inputs[2])?;

        let q_ndim = query.rank();
        anyhow::ensure!(
            q_ndim >= 2,
            "SDPA: query must have at least 2 dims (got {q_ndim})"
        );

        // Q/K share head_dim and K/V share seq by op contract, but dynamo
        // gives each placeholder its own SymInt — unify so matmul
        // dim-equality checks hold. The recorder has no mutable ShapeTracker,
        // so re-view the tensor when the dims differ but have equal extents;
        // when they already compare equal this is a no-op.
        if key.rank() == q_ndim && value.rank() == q_ndim {
            let q_head_dim = query.dims()[q_ndim - 1];
            let k_head_dim = key.dims()[q_ndim - 1];
            if !same_dim(q_head_dim, k_head_dim)
                && let (Some(q), Some(k)) = (q_head_dim.to_usize(), k_head_dim.to_usize())
            {
                anyhow::ensure!(q == k, "SDPA: query/key head_dim mismatch ({q} vs {k})");
                let mut target = key.dims();
                target[q_ndim - 1] = q_head_dim;
                key = util::reshape_tensor(key, &target);
            }
            let k_seq = key.dims()[q_ndim - 2];
            let v_seq = value.dims()[q_ndim - 2];
            if !same_dim(k_seq, v_seq)
                && let (Some(k), Some(v)) = (k_seq.to_usize(), v_seq.to_usize())
            {
                anyhow::ensure!(k == v, "SDPA: key/value seq mismatch ({k} vs {v})");
                let mut target = value.dims();
                target[q_ndim - 2] = k_seq;
                value = util::reshape_tensor(value, &target);
            }
        }

        // attn_bias (Efficient/Cudnn/Unified) or attn_mask (FlashForCpu/Unified),
        // read as recorded: a Bool mask is a predicate, not an operand of
        // the score arithmetic.
        let mut additive: Option<GraphTensor> = None;
        for name in ["attn_bias", "attn_mask"] {
            if let Some(input) = named_input(node, name)
                && let Some(tensor) = self.optional_raw_operand(input)?
            {
                additive = Some(tensor);
                break;
            }
        }
        let is_causal = self.named_bool_arg(node, "is_causal").unwrap_or(false);

        let dropout_p = self.named_float_arg(node, "dropout_p").unwrap_or(0.0) as f32;
        anyhow::ensure!(
            dropout_p == 0.0,
            "SDPA: dropout_p={dropout_p} unsupported (inference only)"
        );

        // Default scale 1/sqrt(head_dim) needs a concrete head_dim;
        // symbolic-shape HF graphs always pass `scale` explicitly.
        let scale = match self.named_float_arg(node, "scale") {
            Some(v) => v,
            None => {
                let head_dim = query.dims().last().and_then(|d| d.to_usize()).context(
                    "SDPA: query head_dim must be concrete to derive the default \
                         scale (pass `scale` explicitly for symbolic head dims)",
                )?;
                1.0_f64 / (head_dim as f64).sqrt()
            }
        };

        // GQA: repeat-interleave K/V heads up to Q's count, gated on
        // `enable_gqa` — only the unified op carries the flag, and the
        // pipeline emits only the unified op (sdpa decomps are stripped).
        // Flagless graphs with mismatched heads fail the ensure below.
        let enable_gqa = self.named_bool_arg(node, "enable_gqa").unwrap_or(false);
        if enable_gqa && q_ndim >= 3 && !same_dim(query.dims()[q_ndim - 3], key.dims()[q_ndim - 3])
        {
            let h_axis = q_ndim - 3;
            let h_q = query.dims()[h_axis]
                .to_usize()
                .context("SDPA GQA: query head count must be concrete")?;
            let h_kv = key.dims()[h_axis]
                .to_usize()
                .context("SDPA GQA: kv head count must be concrete")?;
            anyhow::ensure!(
                h_kv > 0 && h_q % h_kv == 0,
                "SDPA GQA: query heads ({h_q}) must be a positive multiple of kv heads ({h_kv})"
            );
            let group = h_q / h_kv;
            key = key
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
            value = value
                .expand_dim(h_axis + 1, group)
                .merge_dims(h_axis, h_axis + 1);
        } else if q_ndim >= 3 {
            anyhow::ensure!(
                same_dim(query.dims()[q_ndim - 3], key.dims()[q_ndim - 3]),
                "SDPA: query/key head counts differ ({:?} vs {:?}) without enable_gqa",
                query.dims()[q_ndim - 3],
                key.dims()[q_ndim - 3]
            );
        }

        // scores = (Q @ K^T) * scale.
        // Q/K arrive at the compute dtype, so the scores, the scale, the
        // masks and the softmax are all F32 for half-precision inputs.
        let (q_for_mm, k_for_mm) =
            util::ensure_same_dtype(query, key.transpose(q_ndim - 2, q_ndim - 1));
        let scores = q_for_mm.matmul(k_for_mm);
        let scale_const = self.constant_like(scores, scale);
        let mut scores = scores * scale_const;

        if is_causal {
            let s_q = scores.dims()[q_ndim - 2];
            let s_k = scores.dims()[q_ndim - 1];
            let row = self.cx.arange(s_q).cast(DType::F32).expand_dim(1, s_k);
            let col = self.cx.arange(s_k).cast(DType::F32).expand_dim(0, s_q);
            // 1.0 strictly above the diagonal (j > i = masked); -1e9 ~= -inf.
            let masked = col.gt(row).cast(DType::F32);
            let neg_inf = self.constant_like(masked, -1e9);
            scores = add_offset(scores, masked * neg_inf);
        }

        // Bool masks: track per-row any-keep to zero fully-masked rows
        // after softmax.
        let mut row_any_keep: Option<GraphTensor> = None;
        let mut any_keep: Option<GraphTensor> = None;
        if let Some(mask) = additive {
            let offset = if mask.dtype == DType::Bool {
                let keep = mask.cast(DType::F32);
                let key_axis = keep.rank() - 1;
                let rows = keep.max(vec![key_axis]);
                any_keep = Some(rows);
                row_any_keep = Some(rows.expand_to_shape_on_axes(keep.dims(), vec![key_axis]));
                let one = self.cx.constant_f32(1.0).expand_rhs(keep.dims());
                (one - keep) * self.constant_like(keep, -1e9)
            } else {
                self.widen(mask)
            };
            scores = add_offset(scores, offset);
        }

        // Keep the stable softmax weights, denominator and value contraction
        // in the compute dtype. Some flash kernels round the exponentials
        // to the input dtype internally; that loses precision within this
        // single ATen operator and compounds across attention blocks.
        // bind_value restores the declared dtype at the operator boundary.
        let axis = q_ndim - 1;
        let row_max = scores.max(vec![axis]);
        let maximum = row_max.expand_to_shape_on_axes(scores.dims(), vec![axis]);
        let mut weights = (scores - maximum).exp();
        let row_sum = weights.sum(vec![axis]);
        if let Some(indicator) = row_any_keep {
            let (a, i) = util::ensure_same_dtype(weights, indicator);
            let (a, i) = util::broadcast_binary(a, i);
            weights = a * i;
        }
        let numerator = weights.matmul(value);
        let denominator = row_sum.expand_to_shape_on_axes(numerator.dims(), vec![axis]);
        let (numerator, denominator) = util::ensure_same_dtype(numerator, denominator);
        let out = numerator / denominator;

        let stats = RowStats {
            max: row_max,
            denominator: row_sum,
            any_keep,
        };
        self.bind_sdpa_outputs(node, out, stats)
    }

    /// Bind every tensor slot the variant's schema returns, under its
    /// export name, at the shape and dtype the export declares for it
    /// (`bind_value` rounds each to its own declared dtype). A slot that
    /// cannot be produced faithfully is refused by name, never left
    /// unbound and never guessed.
    fn bind_sdpa_outputs(&mut self, node: &Node, out: GraphTensor, stats: RowStats) -> Result<()> {
        let target = node
            .target
            .strip_prefix("torch.ops.aten.")
            .unwrap_or(&node.target);
        let slots = sdpa_slots(target)
            .with_context(|| format!("SDPA: no return schema for {}", node.target))?;
        let names = sdpa_slot_names(node, slots)?;
        anyhow::ensure!(
            names.first().is_some_and(Option::is_some),
            "SDPA: no output tensor name found on node {}",
            node.target
        );
        for (name, &(slot_name, slot)) in names.into_iter().zip(slots) {
            let Some(name) = name else {
                continue;
            };
            let refuse = |why: &str| {
                anyhow::anyhow!(
                    "SDPA {}: output slot `{slot_name}` ({name}) {why}",
                    node.target
                )
            };
            let meta = self.tensor_meta(&name)?.clone();
            let shape = self.tensor_meta_to_shape(&meta)?;
            let dtype = super::dtype_of(meta.dtype)?;
            let value = match slot {
                SdpaSlot::Output => out,
                SdpaSlot::LogSumExp { pad } => self
                    .sdpa_logsumexp(stats, &shape, dtype, pad)
                    .ok_or_else(|| {
                        refuse(&format!(
                            "is declared {shape:?}, which is not the per-row logsumexp \
                             layout {:?}",
                            stats.max.dims()
                        ))
                    })?,
                SdpaSlot::RngState => self.full_tensor(shape, dtype, 0.0),
                SdpaSlot::DebugMask if shape.iter().any(|d| d.to_usize() == Some(0)) => {
                    self.full_tensor(shape, dtype, 0.0)
                }
                SdpaSlot::DebugMask => {
                    return Err(refuse(
                        "(return_debug_mask=True) holds the kernel's dropout-encoded \
                         probabilities, which are not produced",
                    ));
                }
                SdpaSlot::CumSeq => {
                    return Err(refuse(
                        "is a tensor (a nested batch); only dense inputs are supported",
                    ));
                }
                SdpaSlot::SeqExtent => {
                    return Err(refuse(
                        "is a SymInt in the schema but was serialized as a tensor",
                    ));
                }
            };
            self.bind_value(name, value);
        }
        Ok(())
    }

    /// The logsumexp slot: `max + ln(sum exp(s - max))`, built from the
    /// output's own row max and denominator, so it honours scale,
    /// `is_causal` and `attn_mask` exactly as the output does, in the
    /// compute dtype. A bool-masked row that keeps no key outputs zeros and
    /// its logsumexp is 0, as torch's CPU flash kernel returns for a row
    /// with no admissible key (output 0, logsumexp 0). The rows are then
    /// laid out as declared: padded to a rounded-up row extent with `pad`,
    /// and given the declared unit axes. `None` when the declared shape is
    /// not that layout.
    fn sdpa_logsumexp(
        &mut self,
        stats: RowStats,
        declared: &[IntExpr],
        dtype: DType,
        pad: Option<f32>,
    ) -> Option<GraphTensor> {
        if pad.is_some() && declared.last()?.to_usize() == Some(0) {
            // compute_log_sumexp=False: the slot is declared empty.
            return Some(self.full_tensor(declared.to_vec(), dtype, 0.0));
        }
        let mut lse = stats.max + stats.denominator.log();
        if let Some(any_keep) = stats.any_keep {
            let zero = self.constant_like(lse, 0.0);
            lse = self.select(any_keep, lse, zero);
        }
        if let Some(fill) = pad {
            let axis = lse.rank() - 1;
            let rows = lse.dims()[axis];
            let extent = *declared.last()?;
            if !same_dim(rows, extent) {
                if let (Some(rows), Some(extent)) = (rows.to_usize(), extent.to_usize())
                    && extent < rows
                {
                    return None;
                }
                lse = lse.pad_along(0usize, (extent - rows).simplify(), axis, fill);
            }
        }
        insert_unit_axes(lse, declared)
    }
}

#[cfg(test)]
mod output_slot_tests {
    use luminal::prelude::*;

    use super::super::test_support::*;
    use crate::pt2_schema::{ScalarName, TensorRef};
    use crate::translate::translate;

    const LONG: u32 = 5;
    const UINT64: u32 = 35;

    /// One serialized return slot, as `torch.export.save` writes it.
    enum Out {
        Tensor(&'static str),
        None,
        SymInt(&'static str),
    }

    fn output_ref(out: &Out) -> TensorRef {
        match out {
            Out::Tensor(name) => tensor_ref(name),
            Out::None => TensorRef {
                as_tensor: None,
                as_tensors: None,
                as_sym_int: None,
                as_sym_float: None,
                as_sym_bool: None,
            },
            Out::SymInt(name) => TensorRef {
                as_tensor: None,
                as_tensors: None,
                as_sym_int: Some(ScalarName {
                    as_name: name.to_string(),
                }),
                as_sym_float: None,
                as_sym_bool: None,
            },
        }
    }

    /// Translate one SDPA variant over q [2,4,5,8], k/v [2,4,6,8] with the
    /// given serialized outputs and tensor metadata; every tensor slot is a
    /// graph output.
    fn translate_variant(
        variant: &str,
        extra_inputs: Vec<crate::pt2_schema::NodeInput>,
        outputs: &[Out],
        slot_meta: &[(&str, u32, &[i64])],
    ) -> anyhow::Result<crate::translate::Translation> {
        let mut inputs = vec![
            tensor_input("query", "q"),
            tensor_input("key", "k"),
            tensor_input("value", "v"),
        ];
        inputs.extend(extra_inputs);
        let mut sdpa = node(&format!("torch.ops.aten.{variant}"), inputs, &[]);
        sdpa.outputs = outputs.iter().map(output_ref).collect();
        let mut tensors: Vec<(&str, u32, &[i64])> = vec![
            ("q", FLOAT, &[2, 4, 5, 8]),
            ("k", FLOAT, &[2, 4, 6, 8]),
            ("v", FLOAT, &[2, 4, 6, 8]),
        ];
        tensors.extend_from_slice(slot_meta);
        let graph_outputs: Vec<&str> = slot_meta.iter().map(|(name, _, _)| *name).collect();
        translate(&parsed_nodes(
            vec![sdpa],
            &tensors,
            &["q", "k", "v"],
            &graph_outputs,
        ))
    }

    /// Each graph output's bound value carries its declared shape and dtype.
    fn assert_bound_as_declared(
        t: &crate::translate::Translation,
        slot_meta: &[(&str, u32, &[i64])],
    ) {
        assert_eq!(t.outputs.len(), slot_meta.len());
        for (output, (name, code, shape)) in t.outputs.iter().zip(slot_meta) {
            let value = &t.graph.logical.petgraph()[output.tensor];
            let dims: Vec<Option<usize>> = value.dims.iter().map(|d| d.to_usize()).collect();
            let declared: Vec<Option<usize>> = shape.iter().map(|d| Some(*d as usize)).collect();
            assert_eq!(dims, declared, "slot {name}: bound dims");
            let dtype = match *code {
                FLOAT => DType::F32,
                LONG => DType::I64,
                UINT64 => DType::U64,
                other => panic!("unexpected dtype code {other}"),
            };
            assert_eq!(value.dtype, dtype, "slot {name}: bound dtype");
        }
    }

    #[test]
    fn flash_for_cpu_binds_output_and_logsumexp() {
        let meta: &[(&str, u32, &[i64])] = &[
            ("getitem", FLOAT, &[2, 4, 5, 8]),
            ("getitem_1", FLOAT, &[2, 4, 5]),
        ];
        let t = translate_variant(
            "_scaled_dot_product_flash_attention_for_cpu.default",
            vec![],
            &[Out::Tensor("getitem"), Out::Tensor("getitem_1")],
            meta,
        )
        .unwrap();
        assert_bound_as_declared(&t, meta);
        let lse = &t.graph.logical.petgraph()[t.outputs[1].tensor];
        assert!(
            !matches!(lse.op, LogicalOp::Constant(_)),
            "logsumexp is computed, not a placeholder"
        );
    }

    #[test]
    fn efficient_pads_logsumexp_rows_and_binds_rng_state() {
        let meta: &[(&str, u32, &[i64])] = &[
            ("getitem", FLOAT, &[2, 4, 5, 8]),
            ("getitem_1", FLOAT, &[2, 4, 32]),
            ("getitem_2", LONG, &[]),
            ("getitem_3", LONG, &[]),
        ];
        let t = translate_variant(
            "_scaled_dot_product_efficient_attention.default",
            vec![none("attn_bias"), scalar_bool("compute_log_sumexp", true)],
            &[
                Out::Tensor("getitem"),
                Out::Tensor("getitem_1"),
                Out::Tensor("getitem_2"),
                Out::Tensor("getitem_3"),
            ],
            meta,
        )
        .unwrap();
        assert_bound_as_declared(&t, meta);

        // compute_log_sumexp=False declares an empty row axis.
        let meta: &[(&str, u32, &[i64])] = &[
            ("getitem", FLOAT, &[2, 4, 5, 8]),
            ("getitem_1", FLOAT, &[2, 4, 0]),
        ];
        let t = translate_variant(
            "_scaled_dot_product_efficient_attention.default",
            vec![none("attn_bias"), scalar_bool("compute_log_sumexp", false)],
            &[Out::Tensor("getitem"), Out::Tensor("getitem_1")],
            meta,
        )
        .unwrap();
        assert_bound_as_declared(&t, meta);
    }

    #[test]
    fn flash_cudnn_and_overrideable_skip_none_and_symint_slots() {
        let flash: &[(&str, u32, &[i64])] = &[
            ("getitem", FLOAT, &[2, 4, 5, 8]),
            ("getitem_1", FLOAT, &[2, 4, 5]),
            ("getitem_6", UINT64, &[2]),
            ("getitem_7", UINT64, &[]),
            ("getitem_8", FLOAT, &[0]),
        ];
        let t = translate_variant(
            "_scaled_dot_product_flash_attention.default",
            vec![],
            &[
                Out::Tensor("getitem"),
                Out::Tensor("getitem_1"),
                Out::None,
                Out::None,
                Out::SymInt("max_q"),
                Out::SymInt("max_k"),
                Out::Tensor("getitem_6"),
                Out::Tensor("getitem_7"),
                Out::Tensor("getitem_8"),
            ],
            flash,
        )
        .unwrap();
        assert_bound_as_declared(&t, flash);

        for (variant, lse_shape) in [
            (
                "_scaled_dot_product_cudnn_attention.default",
                &[2i64, 4, 5, 1][..],
            ),
            (
                "_scaled_dot_product_fused_attention_overrideable.default",
                &[2, 4, 5][..],
            ),
        ] {
            let meta: &[(&str, u32, &[i64])] = &[
                ("getitem", FLOAT, &[2, 4, 5, 8]),
                ("getitem_1", FLOAT, lse_shape),
                ("getitem_6", LONG, &[]),
                ("getitem_7", LONG, &[]),
            ];
            let t = translate_variant(
                variant,
                vec![],
                &[
                    Out::Tensor("getitem"),
                    Out::Tensor("getitem_1"),
                    Out::None,
                    Out::None,
                    Out::SymInt("max_q"),
                    Out::SymInt("max_k"),
                    Out::Tensor("getitem_6"),
                    Out::Tensor("getitem_7"),
                    Out::None,
                ],
                meta,
            )
            .unwrap();
            assert_bound_as_declared(&t, meta);
        }
    }

    #[test]
    fn unfaithful_slots_are_refused_by_name() {
        // return_debug_mask=True declares the dropout-encoded probabilities.
        let err = translate_variant(
            "_scaled_dot_product_flash_attention.default",
            vec![scalar_bool("return_debug_mask", true)],
            &[
                Out::Tensor("getitem"),
                Out::Tensor("getitem_1"),
                Out::None,
                Out::None,
                Out::SymInt("max_q"),
                Out::SymInt("max_k"),
                Out::Tensor("getitem_6"),
                Out::Tensor("getitem_7"),
                Out::Tensor("getitem_8"),
            ],
            &[
                ("getitem", FLOAT, &[2, 4, 5, 8]),
                ("getitem_1", FLOAT, &[2, 4, 5]),
                ("getitem_6", UINT64, &[2]),
                ("getitem_7", UINT64, &[]),
                ("getitem_8", FLOAT, &[2, 4, 5, 128]),
            ],
        )
        .err()
        .expect("a populated debug mask is refused");
        assert!(
            format!("{err:#}").contains("`debug_attn_mask`"),
            "the refusal names the slot: {err:#}"
        );

        // A logsumexp declared in a layout other than the per-row one.
        let err = translate_variant(
            "_scaled_dot_product_flash_attention_for_cpu.default",
            vec![],
            &[Out::Tensor("getitem"), Out::Tensor("getitem_1")],
            &[
                ("getitem", FLOAT, &[2, 4, 5, 8]),
                ("getitem_1", FLOAT, &[2, 5, 4]),
            ],
        )
        .err()
        .expect("a mismatched logsumexp layout is refused");
        assert!(
            format!("{err:#}").contains("`logsumexp`"),
            "the refusal names the slot: {err:#}"
        );
    }
}

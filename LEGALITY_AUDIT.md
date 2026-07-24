# Legality-by-Construction Audit

**Contract.** Every rewrite rule must only add e-nodes that are unconditionally legal
and semantically equivalent to their e-class, for every extractable combination.
Legality established by post-extraction machinery (candidate filters, fusion-region
validation, alias validation, NaN output checks, arity assertions, choice repair)
violates the contract. Goal: make all post-extraction **semantic** checks removable;
**resource** checks (device limits, workspace sizing) may remain but must be
fail-stop — they must never silently select a semantically different candidate.

**Categories.**
- **V1** conditional legality: RHS legal only under conditions verified later.
- **V2** floating-point non-equivalence: intermediate precision/rounding changes.
  Accepted line (per the `delete-low-precision-sum` precedent and
  `tests/qwen_bf16_repro.rs`): reassociation within the same accumulator
  precision is OK; changing intermediate precision or substituting approximate
  implementations is a violation.
- **V3** extraction incoherence: unions whose per-class child choices can
  combine into an inconsistent extraction.

**Scope.** All rewrite rules in `src/` (expression algebra, HLIR, matmul
flattening) and `crates/luminal_cuda_lite/src/` (kernel + host). Cleared-rule
lists from each sweep are preserved in section 7 for coverage accountability.

---

## 1. The validation layer being masked (what we want to delete)

### Semantic checks (removal targets)
| # | Check | Site | Masks |
|---|---|---|---|
| S1 | `validate_fusion_regions` ("unproved relation is rejected as an unsupported layout") | `fusion/region_codegen.rs:587-969`, invoked from **`resource.rs:743/759`** — semantic validation smuggled into the resource planner as `ResourceViolation::InvalidFusionRegion` | all speculative fusion grow/merge rules (§2.1) |
| S2 | `validate_mutating_aliases` + `CyclicLlir` toposort rejection | `resource.rs:641/733` | ScatterNoCopy family, rope fused scatters, FS-absorption cycles (§2.2) |
| S3 | `has_nan_outputs` candidate rejection | `runtime.rs:3415`; `graph.rs` search ×2 | residual net for every silent V1/V2 below |
| S4 | Kernel input-arity assertions | `to_host.rs:2817-2830` | IList tail-merge incoherence (§2.6) |
| S5 | FlashInfer "matched a gather without recoverable compact gather_idx" panic | `flashinfer/mod.rs:429` | flashinfer island rules' unproven gather-index structure (§2.3-B2) |
| S6 | Choice-coherence machinery: `opkind_metadata_consistent`, `repair_choice_cycles`, `validate_choice_set`, marker-choice bias, `inline_static_loop_inputs` | `egglog_utils/mod.rs`, `graph.rs` | marker transparency unions + length-changing metadata rewrites (§2.5, §2.6) |
| S7 | Extraction-time staticness panics (`.expect("… is static")`) | moe_gemv.rs, rope.rs, rms_norm.rs, swiglu.rs, GLUMoE mode | rules lacking staticness premises |
| S8 | NVRTC-failure / "candidate compile panicked" as rejection | search catch sites | any rule emitting uncompilable spellings |

### Resource / capability checks (keep, fail-stop only)
`plan_static_llir_resources` + `validate_resource_plan` (intermediate bytes,
kernel-source bytes, param ABI, device limits); retained/aggregate bucket plans;
per-token `validate_compiled_bucket_resources`; cuBLASLt heuristic "No suitable
algorithm"; FlashInfer workspace/K-V-length sizing and explicit-indptr content
validation; GLUMoE buffer-size bails. **Condition:** rejection must be loud
(finalist rejection logging now guarantees this in search) and must never fall
back to a semantically different graph.
One repair mechanism must flip polarity: `clamp_ld_for_order`
(`cublaslt/mod.rs:888`) silently **rewrites** an invalid ld into a different
memory layout — convert to a hard assertion (all emitted rules are ld-valid by
construction today, so the clamp is provably dead).

---

## 2. Violations

Ranked tiers: **T1** silent wrong-answer with *no* masking check; **T2** masked
by post-extraction validation (the legality-by-validation debt); **T3** latent
(unsound in an unexercised domain).

### 2.1 Fusion (kernel/fusion/) — T2, the #373 epicenter
PR #373 deleted 11 cleanup rules from `markers.rs` and added ~1,200 lines of
`validate_fusion_regions`; the speculation moved from "cleaned in-graph" to
"filtered post-extraction".
- **grow-FE-B-lhs/rhs-{Add,Mul}** (markers.rs:217-244) — V1: stamps FE dtype on
  the absorbed operand with no `(dtype ?b)` fact, no supported-dtype guard.
- **grow-U-FS / grow-B-FS-{Add,Mul} / grow-Cast-FS** (markers.rs:202-277) — V1:
  FS-absorption can create self-referential e-classes; own comment: "static
  candidate validation rejects only selections that actually form an LLIR cycle".
- **grow-FE-U / grow-FE-Cast / merge-FE-FE-{Add,Mul}** (markers.rs:165-295) — V3:
  FE/FS/elem metadata (shape/stride ELIST children) extract independently per
  node; incoherent spellings rejected only by S1's rank/layout proofs.
- **cuda-elem-singleton-{Sin,Sqrt,Exp2,Log2,Recip,Add,Mul}**
  (elementwise.rs:77-310) — V1+V3: no dtype-class guard (Bool/packed rejected
  only by S1); `singleton-Mul` stamps `(dtype ?a)` without checking `?b`. Own
  admission at elementwise.rs:326-329: "A selected candidate with inconsistent
  ranks is rejected by fusion-region validation."
- **Remedy pattern:** dtype facts for every operand + a `fusion_dtype_supported`
  relation; canonical single-child metadata terms (or canonical-spelling
  witness) so only one extractable spelling exists; an in-egraph
  non-reachability witness for absorption (or restructure absorption as a
  directed rewrite on a region IR rather than a union).

### 2.2 In-place aliasing (T2)
- **scatter→scatter-no-copy + consumed-buffer-resolve** (other_ops.rs:300-340)
  — V1 textbook; own comments: "Alias validity is checked per extracted LLIR
  candidate, not per eclass." Masked by S2.
- **rope fused scatters** (rope.rs:546-583, 1292-1334) — V1 inherited: mutating
  alias kernels, safety only S2-checked; plus staticness panics (S7) without
  premises.
- **Remedy:** in-egraph exclusivity witness (`exclusive-last-read ?dest ?op`
  derived over the consumer set, or a linear/consume node type constructible
  only with proof all other readers precede the mutation), consumed by both
  rules as an LHS fact.

### 2.3 FlashInfer (host/flashinfer/) — T1, most severe silent class
- **B1 "FlashInfer batch decode attention"** (flashinfer_attention.egg:103-242):
  (a) `fi_scaled_offset_mask` accepts any `Add(Mul(x, ≈1e10), y)`; the mask is
  then *discarded* ("proof-only") and runtime attends the full `[0,c]` prefix —
  padding/bidirectional masks silently become full-prefix attention, **no check
  can catch it**. (b) no dtype premise (F64 dies at a post-extraction assert).
  (c) no s∈[1,1] guard unlike sibling rules.
- **B2 gather-index structure unproven** (all six island rules): `?k_idx` is a
  free e-class; `try_find_compact_gather_idx` post-extraction either panics
  (S5) or — worse — accepts the first `Input` under any `Mul` without verifying
  the multiplier equals `kvdim`: an index like `idx·(kvdim+8)+off` reads wrong
  cache rows **silently**.
- **B3 free operand-layout variables** (Q/K/V/GQA/out stride lists unpinned in
  all island rules): permuted-but-shape-compatible views union with an op that
  computes something else; `(MDiv ?kvdim ?hdim)` never asserts divisibility.
- **B4 triu/direct-causal mask facts** (:656-769): the LessThan stride lists
  distinguishing causal from **anti-causal** are free variables; `?q_pos` is
  bound but never consumed (decode assumes q_pos = c−1). The "window term fact"
  (:771-815) pins its strides exactly — the correct idiom, unused by its
  siblings.
- **Remedy:** structural mask proof mirroring the triu chain with pinned
  strides; dtype relation `{F32,F16,Bf16}`; s-interval guard; move the
  compact-index arithmetic into the premise and emit `?compact_idx` as a direct
  op input (kills S5 and the silent factor mismatch); pin GQA/matmul stride
  spellings; factorization premise `?kvdim = ?hdim · ?kv_heads`.

### 2.4 MoE (host/moe/, kernel/moe_gemv.rs) — T1/T2 mix
- **C1 GLUMoE fusion** (glumoe_rewrite.egg all 3 final rules + markers) — V1:
  gate/up split completely unpinned (a halves-swapped model silently computes
  the wrong function — **no check catches it**); `?io` never tied to the expert
  extent (masked by execute()'s bail-cascade); matmul stride lists free;
  expert-bounds semantics diverge from HLIR Gather. — V2: execute() casts
  activations F32→BF16 before the gate_up matmul, a precision reduction absent
  from the matched graph; `expf` vs `Exp2(x·1.442695)`; Gemma coefficient
  matched by tolerance window but hardcoded in the kernel.
- **C2 KernelMoEGemv** (moe_gemv.rs:87-138) — V1: pins dtypes/shapes/operand
  strides (learned the hard way), but `?io` is never constrained to `O·D` and
  gather-index strides are free — a different flat-index arithmetic **runs and
  returns wrong numbers**. Out-of-range expert yields 0.0 here, a bail in
  GLUMoE, and Gather semantics in HLIR — three behaviors in one e-class.
- **Remedy:** premise `?io = (MMul ?o ?d)` (+ canonical-spelling seeding for
  folded products), pinned within-expert iota form, pinned gate/up slice iotas
  (offset/extent/strides), staticness premises for extracted-as-usize fields,
  and either F32 activations in the GLUMoE host op or an explicit documented
  BF16-activation contract.

### 2.5 Loop-marker transparency (src/hlir.rs) — T2, root of S6
- **LoopInputStatic inline** (hlir.rs:703-708) and **LoopInput→LoopInputStatic**
  (hlir.rs:695-701) — V3: `(union ?e ?x)` merges a wrapper with its own source,
  creating self-referential e-classes; extraction can materialize marker chains
  and choice cycles. This single pair is why `repair_choice_cycles`, the
  marker-choice bias, and the unroll-time `inline_static_loop_inputs` pass
  exist.
- **Remedy:** never union an identity wrapper with its child. Express
  transparency as a relation (`sees_through ?e ?x`) consumed by fusion
  matchers, or subsume/delete the marker spelling at union time so it is never
  extractable.

### 2.6 Metadata coherence (core + to_host) — T2
- **KNOWN BUG (found by the search-reliability harness, 2026-07-24): the
  single-bucket-combo stitched load path never uploads KernelConstant
  buffers** ("missing cached buffer" at first execute). No shipped model
  exercises one combo (all have >= 2 's' buckets); the harness works
  around it with two 's' buckets. Fix when the load path is next open.

- **FIXED (2026-07-23): `dtype` fact corruption via loop-marker unions.** The
  rolling prepass inferred marker dtypes with a heuristic that had no case
  for `Iota`/`Constant`/`LessThan` (F32 fallback), so Int index expressions
  got F32-typed `LoopInput` markers; the "LoopInputStatic inline" rule then
  unioned the marker class into the Int source class, and `dtype`'s
  `:merge new` silently let the last write win. Singleton fusion lowering
  read the flip-flopping fact when stamping `FusionStart`, producing the
  nondeterministic `fusion-region-reject` storms (26/292/659 across
  identical gemma4_moe runs) that randomly culled decode families (TPOT 12
  vs 107 ms). Fixed by (a) stamping each `FusionStart` with its own input's
  dtype in the singleton rules and (b) making `infer_node_dtype_cached`
  mirror `dtype_prop` exactly (fixed dtypes, first-input propagation,
  Gather data-input case). Acceptance: two consecutive runs with zero
  fusion-region rejects.
- **`dtype` with `:merge new`** (egglog_utils/mod.rs:155): the above is one
  instance of a systemic hole — any future cross-dtype union silently
  corrupts the fact instead of failing saturation. **Remedy:** conflict-
  detecting merge (assert-equal) so illegal unions crash loudly in tests;
  this is the construction-enforcement backstop for the whole dtype system.
- **`len`/`nth_from_end`/`n_elements` with `merge: new`** (base.rs:1234-1304):
  last-write-wins silently swallows contradictory facts when length-changing
  backend rewrites leave an ELIST class with different-length spellings — the
  reason `opkind_metadata_consistent` exists. **Remedy:** assert-equal merge;
  length-changing rewrites must build fresh kind terms, never union ELists.
- **KernelGather `?__tail` workaround** (hlir.rs:810-840): documents that op
  input-list e-classes can gain foreign tails via unrelated unions — what S4's
  arity asserts guard. **Remedy:** length-indexed list constructors (ground,
  non-unionable input lists).

### 2.7 Numerics (V2) — T1 (nothing masks ulp-level divergence)
- **cuda-elem-rsqrt-from-sqrt-recip** (elementwise.rs:92-105): emits `rsqrtf`
  (approximate) for `Recip(Sqrt(x))` and drops the intermediate rounding.
- **direct-exp-region / direct-sigmoid-*** (elementwise.rs:107-180): constant
  tolerance windows (1.44–1.45 etc.) union a *family* of distinct programs with
  one `expf`/sigmoid body; the matched constant is discarded.
- **cuBLASLt GELU epilogue** (epilogue_rewrite.egg:375-579): same window defect
  + tanh-approx epilogue vs the HLIR sigmoid spelling. RELU epilogues differ on
  NaN/−0.0 edge cases (document).
- **cuBLASLt alpha/beta fusions with 16-bit D** (scale/beta rewrites): fused
  wide-epilogue = one rounding; unfused = round16 then 16-bit op — precision
  change for F16/BF16 D (F32 D is exact). Column-bias rules share this when D
  is 16-bit.
- **generic-matmul-cuda-mul-sum** (generic_matmul.rs:175-200): unions the
  bf16/f16 product-materializing reduction with the F32-accumulating backend
  contract, then *repairs* the class with the `delete-low-precision-*` cleanup
  rules — a destructive, scheduling-fragile mask by the file's own admission.
  **Remedy:** guard the union to F32(+F64), introduce low-precision contractions
  only via a first-class accumulator-contract marker emitted by
  `GraphTensor::matmul` (the comment already proposes it); the deletes then
  become unnecessary.
- **rope π/2 window** (rope.rs:943-945) — matched constant discarded, kernel
  hardcodes `1.570796f`. **KernelGemvF8 scale reassociation** (gemv.rs:599) —
  same-precision reassociation; minor, at the accepted line.
- **kernel-cast-sum-{F16,Bf16} Kahan accumulation** (hlir.rs:582-603) —
  *more* accurate than the plain F32 Sum in the same class; still an effective
  accumulator-precision change. Decide: amend the contract to admit compensated
  summation explicitly, or emit plain accumulation.
- **±1e10 additive mask vs FlashInfer hard mask**: differs only for
  fully-masked rows / extreme scores; with 2.3 fixed, document as an accepted
  approximation.
- **Remedy pattern for windows:** constants round-trip exactly since c6b6a038 —
  match exactly and thread the matched constant into the kernel body.

### 2.8 Core integer algebra — T3 (latent; wrong under truncating `/`/`%` for negative operands)
`Expression` division truncates (`shape/expression.rs:235-236`). Counterexamples
exist for: **add-div** (base.rs:1011), **div-mul-num-plus-rem** (:671),
**mod-mul-num-plus-rem** (:762), **mod-mod-larger** (:776) and
**mod-mod-smaller** (:788) — the last two have apparently operand-swapped
guards making them unsound on negatives *and* vacuous on positives. Benign
today only because shape/index expressions are non-negative — an unstated
axiom. **Remedy:** interval witnesses (`lower(x) ≥ 0`, `lower(b) ≥ 1`) using
the existing machinery; fix or delete the mod-mod pair.
Also: **num-neg1-to-float** (base.rs:727) injects an `MFloat` spelling into
every integer −1 class while `build_expression` has no `MFloat` arm (extractor
panic on tie-break) — delete the rule, keep the safe direction.
**replace-var-miss/iter-miss** (base.rs:1214-1231): non-monotonic `!=`-on-eclass
guard can union unreplaced terms once interval facts merge a var with a
constant — make miss-detection syntactic on names. **replace distribution**
rules assume atomic `from` (unstated invariant) — match `(MVar _)`/`(MIter)`
structurally.

### 2.9 cuBLASLt guard hygiene — T2 (fail-stop, moderate)
All 17 base layout rules use `(!= ?m (MNum 0))`-style e-class disequalities:
non-monotone, and dynamic dims pass the guard with legality deferred to the
runtime zero-dim error. Beta fusions leave `?c_dtype` unconstrained (one-line
fix: `(= ?c_dtype ?d_dtype)`). **Remedy:** interval guards
`(> (lower ?m) 0)` — the pattern the new single-row-dyn rule already uses.
The new free-stride batched rule's interval guards (`ld ≥ rows` via
lower/upper) were assessed **construction-legal**: monotone merges, absent
facts ⇒ rule doesn't fire.

### 2.10 Remaining stride/staticness pins — T1-lite
rope bf16 concat out-strides; rms_norm/swiglu root out-strides and interior
stride atoms; staticness premises for every field `extract()` requires static
(S7 list). All currently guarded by nothing but S3.

---

## 3. Check-removal dependency map

| Check | Removable after |
|---|---|
| S1 `validate_fusion_regions` → debug_assert | §2.1 fusion dtype witnesses + canonical metadata + cycle witness |
| S2 alias/cycle validation → debug_assert | §2.2 exclusivity witness (ScatterNoCopy, rope fused); §2.1 absorption restructure |
| S3 NaN check → **delete** | all of §2.3, §2.4, §2.7, §2.10 (it is the residual net for every silent class) |
| S4 arity asserts → debug_assert | §2.6 ground input lists |
| S5 FlashInfer gather panic → delete | §2.3-B2 premise restructure |
| S6 choice-repair machinery → delete | §2.5 marker redesign + §2.6 assert-equal merges/fresh kinds |
| S7 staticness panics → premise-guaranteed | staticness atoms per rule (§2.4, §2.2, §2.10) |
| S8 NVRTC-failure rejection → hard error | consequence of the above: a failing compile becomes a compiler bug, not a candidate property |

## 4. The construction-legal idiom (already in-tree, extend everywhere)

1. **Witness relations proving layout before the union** — conv2d suite
   (`prove … conv2d … semantics`), `generic_matmul_exact_{2d,3d}`,
   `kernel_embed_row_major`.
2. **Canonical-spelling seeding** so folded stride products still match
   witnesses — `seed canonical 2d/3d matmul stride spellings`.
3. **Interval guards for value-domain conditions** — `single-row dyn`,
   free-stride `ld ≥ rows`; monotone `lower`/`upper` merges make these sound.
4. **Exact constant matching** with the constant threaded into codegen —
   available since constants round-trip exactly (c6b6a038).
5. **Provenance pinning via shared structure** — FlashInfer rolled-cache-pair
   requiring the same `?loop_id` and `?scatter_idx` for K and V.

## 5. Ranked fix shortlist (silent-wrong-answer first)

1. FlashInfer `fi_scaled_offset_mask` (2.3-B1a) — any mask becomes full-prefix.
2. FlashInfer causal-mask stride orientation + unconsumed `q_pos` (2.3-B4).
3. GLUMoE gate/up split + F32→BF16 activation quantization (2.4-C1).
4. KernelMoEGemv `?io = O·D` premise (2.4-C2).
5. FlashInfer gather-index premise restructure (2.3-B2) — kills S5 + silent case.
6. Numerics windows: rsqrt/exp/sigmoid/GELU/π-2; 16-bit alpha-beta fusions (2.7).
7. generic-matmul low-precision union → contract marker; retire the deletes (2.7).
8. Marker transparency redesign (2.5) — retires the whole S6 family.
9. Fusion witnesses + metadata canonicalization (2.1, 2.6) — retires S1/S4.
10. Alias exclusivity witness (2.2) — retires S2.
11. Core integer-algebra interval guards + `num-neg1-to-float` deletion (2.8).
12. cuBLASLt guard hygiene + `clamp_ld_for_order` → assert (2.9, §1-A3).

## 5b. Contract rulings (project decisions, 2026-07-22)

1. **Compensated (Kahan) summation is inside the F32-accumulator contract** —
   `kernel-cast-sum-{F16,Bf16}` is cleared; no change required.
2. **Epilogue/mask substitutions must reproduce the HLIR exactly.** The
   cuBLASLt GELU epilogue fusion (tanh-approx) and RELU NaN/−0.0 edge cases,
   and the FlashInfer hard-mask vs ±1e10 additive-mask delta, are violations
   to fix (make exact or restrict to provably-identical cases), not
   approximations to document.
3. **The matched dtype is binding.** If the frontend/HLIR spelling computes in
   F32, the op must stay F32 (GLUMoE's F32→BF16 activation cast is a bug: the
   host op must consume F32). If the frontend says BF16, the op stays BF16.
4. **Integer-algebra rules get non-negativity witnesses** via the interval
   analysis machinery — guard, don't delete.

## 6. Notes on scope of "resource" checks
Resource rejection is compatible with legality-by-construction *only* as a
fail-stop: with all semantic legality in-graph, a candidate that exceeds a
resource cap may be rejected and a **different legal candidate** selected —
that changes performance, never semantics. The one standing hazard class was
silent repair (`clamp_ld_for_order`) and silent fallback (pre-logging finalist
substitution); the first should become an assert, the second is now logged.

## 7. Coverage: cleared rules
- **Core:** full base.rs algebra/interval/RowMajor/replace suites except the
  eight flagged; both matmul-flattening `.egg` files; `binary_op_unroll_rules`;
  `identical_inputs`; all dtype-propagation rules (modulo `merge: new` note).
- **Kernel:** conv2d witness+union suite (model citizens); generic_matmul
  seeds + witnesses; `kernel gemv m1 {dt} {static,dyn}` + KernelGemvF8 anchor;
  hlir.rs 1:1 lowerings, constant-cast folds, row-major witnesses, embed suite;
  matmul2d/to_host/cuda_graph (no rules).
- **Host:** RmCm/CmRm/CmCm layout rules; RmRm base+batched incl. single-row,
  single-row-dyn, free-strides; row-order suite; output-witness facts; fp8
  suite; epilogue column-bias (modulo 16-bit D note); FlashInfer cache-pair
  facts, window-term fact, s/dtype guards on prefill islands; CuBlasLt extract
  projections.

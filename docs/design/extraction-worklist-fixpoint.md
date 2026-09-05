# Design: a worklist fixpoint for genome extraction (LUM-798)

Status: design proposal, 2026-09-05. Not implemented. Docs-only PR against
`logical-ssa-project`; all line anchors are to trunk `c0ee79cd` unless
stated otherwise. Companion tickets: LUM-798 ("Extraction / Search
Algorithm Performance"), LUM-799 ("Genetic Search Interaction with
cycles"). Instrumentation this design's verification leans on: PR #509
(`feat/search-timing-breakdown`, open at the time of writing).

Two kinds of statement appear below and are marked: **verified** means read
in the code at the cited line or reproduced by a run whose command and
output are in Appendix A; **inferred** means a consequence I argue for but
did not execute.

---

## 0. Summary

`Extractor::relax_to_fixpoint` (`src/extraction.rs:1199-1349`) turns a
genome into a plan per e-class by sweeping every class in the discovered
universe, in root-first BFS order, until a sweep changes nothing. Because
the sweep order is the reverse of the dataflow direction, one sweep moves
plan information exactly one level up the dataflow: the pass count equals
the depth of the plan DAG plus one, and the loop's cost is
`(depth + 1) x (number of candidates)`. Verified on every mini and both
runtimes: `passes = max_first_pass + 1` on all 522 probe lines, and depth
is about 0.45 x universe on these models, so the loop is quadratic in
model size. On the A100, qwen3-4B pays ~2,300 passes x ~5,000 classes
(~11.6 M class visits, ~10 s) per genome, 64 times per search.

The replacement is a leaf-driven worklist with per-candidate
pending-child counters (the same counter shape the SCC sampler already
uses, `src/search_support.rs:448-510`): a class is evaluated when one of
its candidates becomes eligible or one of its children gets cheaper, and
never otherwise. It computes the same fixpoint (Section 2 gives the
argument and Section 7 the test that checks it), visits each class about
once in genome mode (measured: zero re-plans, zero multi-candidate classes
under a genome on every mini), and is linear in the size of the candidate
graph. Expected effect on qwen3-4B: the relaxation bucket drops from ~10 s
to ~10 ms per genome; what remains per genome is discovery and assembly,
which this document also covers (Section 3.5) because at scale they are
the next wall.

Recommendation in one line: land option (a) (plain worklist with reverse
dependencies and pending counters) behind a differential harness that runs
the old loop beside it; reject SCC condensation (b) as machinery the
counters make unnecessary; defer cross-genome incremental re-extraction
(c) as a <=1.3x lever on a phase that (a) already makes ~1,000x faster.

---

## 1. Problem statement, with measurements

### 1.1 What the current loop does

Verified, `src/extraction.rs:1199-1349`:

1. **Discovery** (`:1200-1242`): a BFS from the output roots. For each
   class it calls `candidates_for_class` (`:1135-1197`) and pushes every
   child class of every candidate. It records `universe: Vec<ClassId>` in
   BFS order and `candidate_lists: Vec<Vec<Candidate>>` parallel to it.
2. **Relaxation** (`:1244-1325`): `loop { passes += 1; for (class,
   candidates) in universe.iter().zip(&candidate_lists) { ... } if
   !changed { break } }`. For each class it builds the input-terminal plan
   if the class is a terminal (`:1266-1276`), then for each candidate whose
   children ALL have a `Some` plan in `memo` it builds a `Plan` whose cost
   is the candidate's own bytes-moved plus the sum of the children's plan
   costs (`:1277-1309`), keeps the best by `is_better` (`:1623-1635`), and
   writes it to `memo` only if strictly better than the current entry
   (`:1311-1320`). The loop is bounded by an assertion at 100,000 passes
   (`:1249-1252`) and narrates itself every 5 s (`:1253-1263`, the
   `[extract] SLOW RELAX` line).
3. **Settle and blockage record** (`:1326-1348`): every universe class
   without a plan gets an explicit `None`; unplanned classes with no
   candidates go to `no_candidates`, the others to `blocked` with the list
   of their candidates' unplanned children.

### 1.2 Why the pass count is what it is

The universe is in BFS order from the roots, so a class is (almost
always) visited before its children within a pass. A class can only plan
after its children plan, so pass 1 plans only the leaves (input terminals,
`BufferTensorNil`, zero-input ops), pass 2 their consumers, and so on. The
number of passes is the longest chain in the plan DAG plus one final
no-change pass.

Verified by instrumentation (Appendix A.2): the probe recorded, per
class, the pass on which it was first planned. On every one of the 522
lines (8 genomes x 4 minis on the reference runtime, 8 x 3 minis plus 192
2D-matmul genomes on CUDA-lite with the cuBLASLt marker on, 264 genomes
across the `scc_sampler` boards, `rehomed`, `r10_debug`),
`passes == max_first_pass + 1`.

What propagates per pass is therefore **plan existence and cost, one
dataflow level at a time**. Cycle structure does not add passes in genome
mode: the probe counted zero re-plans (a memo entry overwritten by a
better plan) on every genome-mode line, because under a genome each class
has at most one candidate (`multi = 0` on every genome-mode line; see
2.3). Re-plans do occur in the plain, genome-less fixture extractor
(`r10_debug` plain lines: `replans = 4` with 23-25 multi-candidate
classes) — that is the true min-cost relaxation, and it converged in the
same 8 passes as depth predicted.

### 1.3 Asymptotic form

Per genome the loop costs

    T_relax = passes x sum_over_classes(|candidates|) x t_visit
            = (depth + 1) x |C| x t_visit

with `|C|` the total candidate count (~= universe size in genome mode).
Depth scales with the model: mini gemma3 has 452 classes and depth 198
(0.44); qwen3-4B on the A100 had 5,058 classes and ~2,300 passes (0.45).
So `T_relax ~ 0.45 x |U| x |C| x t_visit`, quadratic in model size.

Measured `t_visit`: ~2.1 us per class visit on the host in the debug
profile (gemma3: 199 x 452 visits in 190 ms); ~0.85 us on the A100 host in
release (11.8 M visits in 10.0 s, from the SLOW RELAX line "10.0s: pass
2341, 4928 planned / 5058 classes"). Each visit that finds an eligible
candidate also allocates a full `Plan` (four `Vec` clones and a
`Box<dyn LayoutIrOp>` clone, `:1293-1306`) only to compare it and usually
discard it — a constant factor the new design removes as well.

### 1.4 Host measurements (this design's own reproduction)

Command: reference-runtime mini smoke tests (`harness_search_options`: 2
generations x 4 genomes, seed 0) and CUDA-lite election rows (same budget,
cuBLASLt marker registered), debug profile for `luminal` itself
(dependencies are `opt-level = 3` per `Cargo.toml:98-99`), Apple-silicon
host, with a temporary counter at the end of `relax_to_fixpoint` (Appendix
A.1; not committed). Per-genome figures are the warm steady state; the
first genome's discovery is the genome-independent `op_cache` fill.

| runtime / mini | universe | candidates | passes | relax / genome | discovery / genome (first) |
|---|---|---|---|---|---|
| ref conv | 66 | 62 | 39 | 5.4 ms | 0.37 ms (23 ms) |
| ref llama3 | 146 | 133 | 69-70 | 21.8 ms | 0.9 ms (100 ms) |
| ref whisper | 194 | 180 | 94 | 40 ms | 1.2 ms (93 ms) |
| ref qwen3 | 181 | 166 | 78-79 | 31 ms | 1.0 ms (182 ms) |
| ref gemma3 | 451-452 | 411-412 | 197-199 | 190 ms | 2.9 ms (288 ms) |
| CL whisper (marker) | 201-225 | 187-211 | 86-97 | 41-53 ms | 2.4 ms (5.3 s; genomes 2-4: 1.2-2.7 s) |
| CL qwen3 (marker) | 176-183 | 161-168 | 73-77 | 29-42 ms | 1.6 ms (4.0 s; genomes 2-4: 0.5-1.1 s) |
| CL gemma3 (marker) | 442-466 | 402-426 | 193-202 | 192-223 ms | 4.6 ms (9.3 s; genomes 2-4: ~1 s) |

Two sizes on one family: mini qwen3 (181 classes, 79 passes, 31 ms) vs
qwen3-4B on the A100 (5,058 classes, ~2,300 passes, ~10 s relax): 28x the
classes, 29x the passes, ~320x the time — the quadratic form.

### 1.5 A100 numbers (2026-09-04, examples with cuBLASLt on by default)

From the coordinator's brief, not reproduced here: qwen3-4B extraction
951 s over ~64 genomes (14.9 s/genome, of which the SLOW RELAX line places
~10 s in the pass loop; the remaining ~4-5 s per genome is discovery plus
assembly, split unknown until PR #509's buckets report); gemma3-4B 926 s;
whisper 161 s; yolo_v11n 3,881 s before the render memo (PR #439). PR
#509's first host sample on the conv example: extract 539 ms = discovery
392 ms, relax 52 ms over 305 passes, assemble 94 ms — at small depth
discovery dominates; at 4B depth the relaxation dominates. Both are
addressed (Sections 3.1 and 3.5).

### 1.6 Order-reversal experiment (the mechanism, confirmed)

The probe ran the identical pass loop a second time over the same
universe in REVERSE BFS order and compared the two memos entry by entry
(cost, source enode, output slot, children, label). Results, all lines:
`mismatches = 0` (the fixpoint is order-independent — Section 2.4's
argument, checked), and `passes_rev` 10-28 against `passes` 69-202 on the
minis (6-8x fewer): reverse BFS is closer to dataflow order but is not a
topological order, so it still iterates. A leaf-driven worklist is the
topological order, obtained without computing one.

---

## 2. Exact semantics to preserve

### 2.1 Vocabulary

- **e-class / e-node**: `egraph_serialize::ClassId` / `NodeId`. Class ids
  are process-random (ruling 2026-09-02); nothing below may depend on
  their spelling.
- **LayoutTensor class**: a value class the plan produces or reads.
- **op class, OpSpec, ProducerRef**: `collect_op_specs` (`:3717-3787`)
  reads every `LayoutTensorOpLit` enode; an op class carries one `OpSpec`
  per distinct (inputs, outputs) list pair; `producer_index[output
  class]` lists `ProducerRef { op_class, spec_index, output_index }`.
- **Input terminal**: a class seeded from a bound input
  (`collect_input_terminals`, `:3920-3963`); planned from the boundary at
  cost 0, offered no candidates (`:1135-1157`), and holding no producer
  row (`:940`).
- **Candidate** (`:1924-1948`): one way to produce a class — a structural
  spelling (`BufferOutputLit`, `BufferTensorCons/Nil/Lit`,
  `:1351-1395`) or an implementation enode with one spec
  (`candidate_for_layout_op`, `:1484-1550`). Its `children` are the
  spec's input classes with operand port names.
- **Genome / choice** (`:525-541`): `choices: HashMap<ClassId,
  ProducerChoice { enode, output_index }>`. Under a genome a class in the
  producer index offers exactly the candidates whose `(source_enode,
  selected_output_index)` equal its choice (`:1158-1170`); a produced
  class with no row offers nothing (`:1171`, the totality contract);
  classes outside the producer index (structural spellings) offer all
  their spellings (`:1172-1185`).
- **Universe** `U`: the BFS closure from the roots over the children of
  ALL candidates of each discovered class (eligible or not).
- **Plan** (`:115-132`) and **memo** `HashMap<ClassId, Option<Plan>>`
  (`:71`).
- **Tuple order**: `is_better(a, b)` (`:1623-1635`) compares
  `(heuristic_cost, plan_label, stable_key)` lexicographically with a
  strict `<`. `plan_label` (`:3634-3644`) is the kind or op label;
  `stable_key` (`:1637-1649`) is the depth-3 render of the source enode,
  memoized per enode for the session.
- **Eligible**: a candidate all of whose children have `Some` plans.
- **Planned**: `memo[c] = Some(plan)`.
- **Dead-end**: an unplanned class with no candidates (`no_candidates`).
- **Blocked**: an unplanned class with candidates; its record is the list
  of its candidates' unplanned children (`blocked`).
- **Choice-cycle**: a strongly connected component of size >= 2 (or a
  self-loop) in the `blocked` graph (`failure_breakdown`, `:279-418`).

### 2.2 The fixpoint the current code computes

Define, over `U`, the operator `F` on memo states:

    F(memo)[c] = Input plan (cost 0)                        if c is an input terminal
               = min over eligible candidates k of c of
                   (own_cost(k) + sum_{d in children(k)} memo[d].cost,
                    label(k), key(k))                         otherwise
               = None                                        if no candidate is eligible

`own_cost` is `candidate_heuristic_cost` (`:1679-1706`), a function of the
candidate only; `label` and `key` are functions of the candidate only.
The sum saturates (`:1290`). The order on states is pointwise: `None` is
the top (worst) element, and `Some` plans compare by the tuple.

The pass loop is a Gauss-Seidel iteration of `F` from the all-`None`
state: within a pass, later classes see earlier classes' fresh writes.
It stops when a full pass writes nothing.

**Invariant I3 (the fixpoint).** For every class in `U`, the final tuple
of `memo[c]` is the greatest fixpoint of `F` below top — equivalently,
the result of Kleene iteration of `F` from the all-`None` state. Cycles
never self-enable: a class is planned only through a chain of enablements
that bottoms out at leaves. This is the property that makes a choice
cycle a refusal and not a plan.

### 2.3 Plan identity, and the one corner where history matters

The tuple at each class is order-independent (2.4). The PLAN stored is
the candidate achieving it. If two eligible candidates of one class have
equal tuples but different children, the current code keeps whichever
was written first: `best` is recomputed from scratch each pass, taking
the first minimal candidate in list order (`:1306-1308`), but the write is
gated on strict improvement over the current entry (`:1311-1320`), so an
equal-tuple plan written in an earlier pass is never replaced.

When can two candidates of one class tie exactly with different
children? They must share `stable_key` (same enode, or two enodes
rendering identically to depth 3) and label (same op) and total cost. Same
enode with different children means two `OpSpec`s of one op class at the
same output slot — the "several distinct input lists at that slot" case
`choice_input_classes` documents (`:1409-1415`).

Verified by the probe: under a genome, `multi = 0` on every line — no
class had more than one candidate on any mini, either runtime, marker on
or off, nor on the 264 sampler-board genomes. So in genome mode the
corner cannot occur on today's estate. In plain mode the probe counted
equal-tuple/different-children ties at the fixpoint: `ties = 0` on every
plain line (rehomed add_mul_fused with 2 multi-candidate classes;
r10_debug plain fixtures with 23-25 multi-candidate classes and 200-221
candidates). The corner exists in the code, not in the measured graphs.

Where it would occur, the current outcome is already process-random:
candidate list order follows `producer_index[class]`'s `Vec` order,
filled by `collect_op_specs` iterating `class_nodes`, a `std::HashMap`
(`:3717-3728`) — so which equal-tuple candidate is "first" changes run to
run. Nothing can pin it; no test does.

**Invariant I4 (plan identity).** `memo[c]` is the FIRST candidate in
the class's candidate-list order that achieves the fixpoint tuple. This
equals the current behaviour whenever the achieving candidate is unique
(always, in every measurement), and canonicalizes the process-random
corner. The differential harness (7.1) counts occurrences of the corner
so the claim is checked on every fixture and mini, not assumed.

### 2.4 Why a worklist computes the same fixpoint

`F` is monotone in the pointwise order: if every child's tuple gets
better or stays (lower cost, or `None` -> `Some`), then for each
candidate the eligibility set grows and the summed cost falls or stays,
so the minimum falls or stays. `F` at class `c` depends only on the
COSTS of `c`'s children (labels and keys are the candidate's own). The
current loop and any worklist that re-evaluates a class whenever one of
its children is first planned or gets cheaper are both fair chaotic
iterations of `F` from top (every class whose `F`-value could have
changed is eventually re-evaluated; a class none of whose children
changed cannot change under `F`, so skipping it loses nothing). Chaotic
iterations of a monotone operator from top on a finite-height lattice all
converge to the same greatest fixpoint (Cousot & Cousot 1977; Kildall
1973 is the worklist form for dataflow). Finite height: costs are `u64`
and strictly decrease on every re-write, labels and keys are finite sets.

Checked empirically by the reversal experiment (1.6): two very different
fair orders, 522 (e-graph, genome) pairs, zero memo mismatches.

### 2.5 What "identical" means for the caller

For every (serialized e-graph, genome, matcher set, allow list) the new
extractor must produce:

- **I1** the same universe `U` (memo key set), because the settle step
  writes `None` for every universe class (`:1326-1330`) and
  `blocked`/`no_candidates` are computed over it.
- **I2** the same candidate list per class, in the same order (same
  `candidates_for_class`; Phase 2's chosen-only construction must
  preserve the surviving subsequence and its order).
- **I3, I4** the same memo (tuples and plans), per 2.2-2.3.
- **I5** the same `blocked` map (same key set; each value the same `Vec`
  in the same order — candidates order x children order, `:1342-1347`)
  and the same `no_candidates` `Vec` (universe order, `:1338-1339`).
- **I6** the same refusal: `extract()`'s `bail!` text (`:1066-1128`) is a
  function of the memo at the roots and the output spine, and
  `failure_breakdown()` (`:279-418`) of `blocked`/`no_candidates`, so I3
  and I5 imply identical refusal classification (choice-cycle vs
  dead-end), summary text, and `RefusalBreakdown` accounting.
- **I7** the same `ExtractedGraph`: `build_extracted_graph` (`:3069-3086`)
  and `IrBuilder` (`:3327-3559`) read only the memo and the genome, and
  are untouched; hence identical node insertion order, edges, provenance,
  per-op `heuristic_cost` (path-counting semantics included), identical
  `plan_fingerprint` (`:849-893`), and identical `heuristic_cost_of`
  (`crates/luminal_cuda_lite/src/heuristic.rs:45-55`).
- **I8** the genome-independent caches (`op_cache`, `stable_key_cache`,
  `bounds_index`, `dtype_index`, `tensor_bytes_cache`, the `RenderCtx`
  memo) end with the same contents; only the order of first fills changes
  (they are memos, never skips — ruling 2026-09-01).

**What the algorithm may NOT change:** the `Genome` and `ProducerChoice`
types and their semantics (choice per class -> (enode, slot)); the sampler
(`sample_genome`, `mutate_genome`, `flip_closes_cycle`,
`bufferize_cycle_tripwire`, `SamplingSpace`) and therefore RNG consumption
— the extractor consumes no randomness and this design adds none; the
producer index and the viability filter (`apply_viability_filter`,
`:1000-1056`) — a different, session-level fixpoint that this design does
not touch (it is quadratic too, but paid once per session; a follow-up);
the election helper (`crates/luminal_cuda_lite/src/ops/cublaslt/election.rs:82-380`),
which builds genomes from the producer index and its own demand walk and
never calls the fixpoint; the consumers' API (`ExtractionSession::{new_with_matcher_set,
producer_index, sampling_space, failure_breakdown, blockage_anatomy,
extract_with_genome}` and the free functions at `:196-206, :512-523,
:550-563, :565-572`); the render memo and lazy text (PR #439).

### 2.6 The two cycle species, under the current semantics

- **Copy <-> Copy weld** (LUM-799 first bullet): classes `x`, `y` of one
  logical value in two layouts, each offering `Copy(other)`. A genome
  electing both copies: neither candidate is ever eligible, both stay
  `None`, `blocked[x] = [y]`, `blocked[y] = [x]`, `failure_breakdown`
  reports one choice-cycle. Electing a progressing producer for `y`: `y`
  plans when its inputs do, then `x`'s copy becomes eligible. The sampler
  never emits the first shape except through its documented fallback.
- **Double-transpose collapse** `x == apply(apply(x, T), T)` (second
  bullet): two DIFFERENT logical values re-describing each other through
  zero-byte views. Same mechanics; the view candidates cost 0 own bytes.
  In genome mode identical to the weld. In PLAIN mode, when both classes
  also have kernel routes, the view-through candidate ties the kernel
  route on cost and the label decides; if `IndexMapApplyView...` sorts
  below the kernel's label the fixpoint plan is `x = view(y)`, `y =
  view(x)` — a cyclic extracted graph only bufferize's toposort catches.
  This is pre-existing (the input-terminal instance was fixed 2026-09-02
  by making terminals candidate-less), identical under both algorithms
  (same fixpoint), out of scope here, and noted for LUM-799 (Section 8,
  R8).

---

## 3. The design

### 3.1 Option (a): plain worklist with reverse dependencies and pending counters

Keep discovery as it is (Phase 1) or as Phase 2 improves it. After
discovery, build for the universe:

- `position: HashMap<ClassId, u32>` — class -> universe index (Phase 3
  makes everything below index by `u32`; Phase 1 may keep `ClassId` keys).
- `pending[c][k]: u32` — for candidate `k` of class `c`, the number of
  DISTINCT child classes not yet planned. Initialized to the number of
  distinct children.
- `dependents[d]: Vec<(c, k)>` — for each class `d`, every (class,
  candidate) pair that lists `d` among its distinct children. One entry
  per distinct occurrence, so the decrements match the initial counts.
- `queue: VecDeque<u32>` + `in_queue: Vec<bool>` — FIFO, each class at
  most once in the queue at a time.

Seeding: input terminals in `U` are written to `memo` immediately with
the cost-0 Input plan and their planned-event is fired; every class with
a candidate whose `pending` is already 0 (no children: `BufferTensorNil`,
zero-input ops) is enqueued.

Evaluate(`c`): compute `best` exactly as the pass body does today
(`:1266-1309`) but only over candidates with `pending[c][k] == 0`; compare
with the current entry by `is_better`; on a write:

- if the class was `None` before (first plan): for every `(c', k)` in
  `dependents[c]`: `pending[c'][k] -= 1`; if it reaches 0, enqueue `c'`.
- if the class re-planned with a LOWER cost: for every `(c', k)` in
  `dependents[c]` with `pending[c'][k] == 0`, enqueue `c'`. A re-plan at
  equal cost (label/key improvement) changes no parent's tuple, because
  `F` reads children's costs only; nothing is enqueued.

Drain the queue. Then run the settle and blockage steps verbatim
(`:1326-1348`), over the same `(universe, candidate_lists)` pairs in the
same order, so I5 holds by construction.

**Complexity.** Building the counters and dependents is
`O(sum_k |children(k)|)`. In genome mode every class is evaluated once per
eligible-candidate event plus once per child cost decrease; measured zero
re-plans, so visits ~= `|U|` and the work is `O(|C| + sum_k |children(k)|)`
— linear where the current loop is `depth x |C|`. Plain mode adds one
visit per actual cost improvement (Bellman-Ford-like; bounded by
`(height + 1) x |U|`, see 5.4). Memory: two `Vec`s of `u32` per candidate
plus the dependents lists — a few hundred KB at 5,000 classes.

**Interaction with the SCC sampler and the tripwires.** None on the
sampler: it runs before extraction over the genome-independent candidate
graph. The pending counters are the extractor-side dual of the counters
`sample_genome_reporting` keeps (`src/search_support.rs:448-510`): the
sampler keeps chosen intra-component edges acyclic so that the
extractor's counters can reach zero everywhere; where they do not, the
blockage record and `failure_breakdown` are computed exactly as today, so
the "choice cycle on an acyclic chosen-edge graph" ensure in both search
loops (`crates/luminal_cuda_lite/src/search.rs:452-467`,
`crates/luminal_reference/src/search.rs:314`) and
`bufferize_cycle_tripwire` see identical inputs.

**What it buys on the measured cases.** The relax bucket becomes ~`|U|`
class evaluations: qwen3-4B ~5,000 evaluations instead of ~11.6 M class
visits — ~2,000x fewer visits per genome, and each evaluation is cheaper
(no `Plan` built for a candidate that is not written). Estimated relax
time per genome on the A100: ~10 ms (inferred from 0.85 us/visit and one
`Plan` allocation per class), against ~10 s today; over 64 genomes, ~640 s
of qwen3's 951 s.

### 3.2 Option (b): SCC condensation plus per-SCC iteration

Per genome: build the genome-restricted candidate graph over `U`, run
Tarjan (`petgraph::algo::tarjan_scc`, already a dependency), process
components in reverse topological order, and iterate to convergence
inside each component only.

Complexity: `O(|U| + edges)` for Tarjan plus the same evaluation work as
(a). What it buys: nothing in genome mode — with pending counters the
worklist already IS a topological order of the eligible sub-DAG (Kahn's
algorithm with the eligibility rule as the in-degree), and acyclic regions
settle in one visit without any SCC computation; classes inside an SCC
that never becomes enabled are never visited at all. In plain mode, (b)
bounds re-evaluation to within a component, but (a)'s worklist already
re-evaluates only actual dependents of an actual cost decrease, and
re-plans are rare (4 in the worst measured plain fixture). Cost: a Tarjan
per genome, ~100 more lines, a second graph representation to keep
consistent with the candidate lists, and no measurement that asks for it.
The session already owns the SCCs of the UNRESTRICTED candidate graph
(`SamplingSpace`, `:650-806`) for the sampler; they are genome-independent
and serve diagnostics, but a per-genome restriction is a different graph.
**Rejected**; the visit-count tripwire (5.4) is the guard that would
reopen the question.

### 3.3 Option (c): incremental re-extraction across genomes

A child genome differs from its parent in `mutations` classes (2 in the
harness; `CompileOptions::mutations` in general). Keep the parent's memo
and universe; for each changed class `c`, invalidate `c` and every class
whose plan transitively depends on `c` (the upward cone through
`dependents`), reset them to `None`, rebuild their candidate lists (the
choice changed) and the counters they participate in, and re-run the
worklist seeded from the cone's boundary.

Correctness requires the reset: the greatest-fixpoint semantics forbid a
stale plan surviving without support, and a class in a cycle that was
enabled only through a route the mutation removed would otherwise stay
planned — a wrong plan, not a slower one. The universe also changes (the
new candidate reaches new children; old ones may become unreachable and
must leave `U` for I1/I5), so discovery must be re-run at least over the
cone.

What it buys: the cone of a random class in a chain-like plan DAG of
depth `D` averages `D/2`; with depth ~= 0.45 |U| and 2 mutations, roughly
half to three quarters of `U` is re-evaluated anyway. Inferred saving:
<= ~1.3-2x on the relax bucket, which (a) already reduces ~1,000x — i.e.
milliseconds. It buys nothing on discovery unless discovery is also made
incremental, and nothing on assembly. Cost: the session must retain the
previous genome and memo, `extract_with_genome`'s "clear everything"
contract (`:495-501`) becomes a diff, and the differential harness must
also test incremental == from-scratch for every mutation. **Deferred**,
not rejected: revisit only if, after Phases 1-2 land and PR #509 reports
A100 buckets, per-genome extraction still dominates the search and the
cost is in the relax bucket. The cheaper lever for discovery and assembly
is genome-independent hoisting (3.5), which has no staleness hazard.

### 3.4 Recommendation

Option (a), staged: Phase 1 the worklist behind the differential harness;
Phase 2 the discovery fix; Phase 3 (dense interning) only if the A100
buckets after Phase 1 still show relax above a few percent of extraction.
Justification: (a) alone removes the quadratic term; it changes one
function body and adds no new representation the sampler or the
diagnostics must agree with; its correctness argument is the standard
one (2.4) and is checked mechanically (7.1); (b) and (c) add machinery
whose benefit no measurement demands.

### 3.5 Discovery and assembly: what remains per genome, and what to do

Verified: under a genome, `candidates_for_class` (`:1158-1170`) calls
`producer_candidates_for_output(class)` (`:1452-1482`), which builds a
`Candidate` for EVERY (spec, enode) pair of every producer op class of
the class — each costing an `op_cache` lookup, a `Box<dyn LayoutIrOp>`
clone, a `metadata()` `Vec`, one `String` per operand port
(`op_children`, `:1552-1561`) and two `Vec<ClassId>` clones — and then
`retain`s the one matching the choice (`:1166-1170`). The comment above
it ("candidates are built for the chosen enode only, never
built-then-discarded per spelling", `:1132-1134`) describes the intended
behaviour, not the code. On minis with few spellings per class this is
1-5 ms per genome; on graphs with many spellings per site (the cuBLASLt
marker mints several per matmul) it is where PR #509's conv sample put
392 of 539 ms.

Phase 2 (Section 6) builds only the chosen candidates: resolve the
choice's op class as `egraph.nodes[choice.enode].eclass` (O(1)), take the
`producer_index[class]` entries with that op class and the chosen output
index, and call `candidate_for_layout_op` for the chosen enode only — the
filter `choice_input_classes` already applies (`:1427-1445`), applied to
construction. I2 holds because the surviving subsequence and its order
are unchanged (the same producer entries in the same order, one enode).
A session-level `Rc<[Candidate]>` cache keyed by `(class, enode,
output_index)` — never cleared, like `op_cache` — then makes per-genome
discovery a BFS of hash lookups. The first-genome `op_cache` fill (5-9 s
on CL minis) also changes shape: only elected enodes are parsed per
genome, so the parse cost spreads over genomes and its total is bounded by
today's.

Assembly (`build_extracted_graph`) is out of this design's scope but is
the likely next wall at 4B scale: `layout_tensor_info` (`:3088-3153`) is
called per value per genome from `ensure_value` (`:3352-3468`) and is
genome-independent except for the waste-slot relabel (`:3396-3406`); a
session-level memo returning a clone is the obvious lever. Recorded as
open question R7, to be sized from PR #509's `assemble` bucket.

---

## 4. Refactor map

Anchors: `src/extraction.rs` at `c0ee79cd`.

| Site | Today | Becomes |
|---|---|---|
| `Extractor` struct `:50-105` | `memo: HashMap<ClassId, Option<Plan>>` `:71`, `blocked` `:76`, `no_candidates` `:77` | Unchanged in Phases 1-2. Phase 3 adds a `universe: Option<Universe>` scratch (dense indices) and may turn `memo` into `Vec<Option<Plan>>` behind `plan()` `:3062-3067`. |
| `ExtractionSession::extract_with_genome` `:495-501` | clears memo/blocked/no_candidates, calls `extract()` | Unchanged signature. Phase 0 adds `extract_with_genome_differential` beside it and an env-gated call to it from here (`LUMINAL_EXTRACT_DIFFERENTIAL=1`; same pattern as `SEARCH_LOG` via `log_channel_enabled`, `src/search_support.rs:48-57`). Deleted again in Phase 4. |
| `Extractor::extract` `:1058-1133` | calls `relax_to_fixpoint(&roots)` `:1064` | Unchanged. |
| `candidates_for_class` `:1135-1197` | genome arm `:1158-1170` builds all spellings then `retain`s | Phase 2: genome arm calls new `candidates_for_choice(class, choice)`; the `Some(None)` `:1171` and `None` `:1172-1185` arms unchanged; fix the comment `:1132-1134`. |
| `relax_to_fixpoint` `:1199-1349` | discovery `:1200-1242`, pass loop `:1244-1325`, settle `:1326-1330`, blockage `:1331-1348` | Phase 1: discovery extracted to `discover(&self, roots) -> Universe` (verbatim body); pass loop REPLACED by `worklist_fixpoint(&mut self, &Universe) -> RelaxStats`; settle and blockage kept verbatim (over `universe.classes`/`universe.candidates`). The old loop survives Phases 1-3 as `relax_passes_reference(&mut self, &Universe)` (the differential's oracle) and is deleted in Phase 4. |
| `SLOW RELAX` narration `:1253-1263` | every 5 s: pass, planned/classes | `SLOW WORKLIST`: elapsed, visits, writes, queue length, planned/classes. |
| convergence assert `:1249-1252` | `passes <= 100_000` | `visits <= |U| x (|U| + 1)` with the same "did not converge" wording (5.4). |
| `producer_candidates_for_output` `:1452-1482` | all (spec, enode) pairs | Unchanged (plain path). Phase 2 adds `producer_candidates_for_choice(class, choice)`. |
| `candidate_for_layout_op` `:1484-1550`, `is_better` `:1623-1635`, `stable_key` `:1637-1649`, `candidate_heuristic_cost` `:1679-1706`, `Candidate` `:1924-1948` | | Unchanged; called from the worklist exactly as from the pass body. |
| `build_extracted_graph` `:3069-3086`, `IrBuilder` `:3327-3559`, `plan_fingerprint` `:849-893` | | Untouched. |
| `apply_viability_filter` `:1000-1056` | its own repeated-pass fixpoint (session-level) | Untouched by this design (follow-up: same counter technique applies; paid once per session, not per genome). |
| `src/search_support.rs` `SearchTimings` `:307-350`; PR #509's `ExtractTimings { discovery_nanos, relax_nanos, relax_passes, assemble_nanos }` | `relax_passes` counts passes | `relax_passes` -> `relax_visits` (class evaluations) plus `relax_writes` (memo writes). If #509 merges first this is a two-line follow-up in its summary line; if this lands first, #509 rebases onto the new names (R3). |
| Consumers: `crates/luminal_cuda_lite/src/search.rs:353,433,446,484`; `crates/luminal_reference/src/search.rs:211,285,298,336`; `crates/luminal_cuda_lite/src/finalists.rs:233-239`; `crates/luminal_cuda_lite/src/ops/cublaslt/election.rs:90`; `crates/luminal_reference/src/harness.rs:39-66,115-122`; `tests/test_runtime/src/lib.rs:36-37,212` | call the session/free functions | **No edits.** The API surface is unchanged; the differential lever is inside `extract_with_genome`. |
| Untouched by construction | `Genome`, `ProducerChoice`, `SamplingSpace` `:650-806`, `edges_have_cycle` `:808-847`, the sampler (`src/search_support.rs:394-696`), the render memo (`RenderCtx`/`ClassRenderer` `:1975-3059`), `failure_breakdown`/`blockage_anatomy` `:279-493` | | |

New test file: `tests/test_runtime/tests/extraction_differential.rs`
(Phase 0). New helper in core test support:
`luminal::test_support::extracted_graphs_identical(&ExtractedGraph,
&ExtractedGraph) -> Result<(), String>` (7.1).

---

## 5. Specs

### 5.1 `discover(&self, roots: &[ClassId]) -> Universe`

```rust
struct Universe {
    classes: Vec<ClassId>,            // BFS order from the roots (as today)
    position: HashMap<ClassId, u32>,  // inverse of `classes`
    candidates: Vec<Vec<Candidate>>,  // parallel to `classes`, from candidates_for_class
    total_candidates: usize,
}
```

Pre: `roots` non-empty (the caller returns `Ok(None)` otherwise, `:1060`).
Post: `classes` is exactly the sequence today's loop pushes at `:1231`, in
the same order; `candidates[i] == candidates_for_class(&classes[i])`;
every child class of every candidate is in `position` (BFS closure). The
`SLOW DISCOVERY` narration (`:1221-1230`) is kept.

### 5.2 `worklist_fixpoint(&mut self, u: &Universe) -> RelaxStats`

```rust
struct RelaxStats { visits: u64, writes: u64, queue_peak: usize }
```

Pre: `self.memo`, `self.blocked`, `self.no_candidates` are empty (cleared
by `extract_with_genome` `:497-499`, or fresh on the free-function paths).
Post: for every `c` in `u.classes`, `self.memo[c]` satisfies I3 and I4;
classes not planned are NOT yet in the memo (the settle step that follows
writes their `None`, as today `:1326-1330`); `blocked` and `no_candidates`
are untouched by this function (the blockage step fills them).

Algorithm (normative):

```
pending[c][k]   = |distinct children of candidate k of class c|
dependents[d]   = [(c, k) for each c, k with d in distinct children(c, k)]   // built in (c, k) order
in_queue        = [false; |U|]; queue = empty FIFO
visits = writes = 0

for c in U (universe order):
    if c is an input terminal:
        memo[c] = Input plan (cost 0, PlanKind::Input(info))        // as :1266-1276
        writes += 1
        planned(c)
    else if some k has pending[c][k] == 0:
        push(c)

while let Some(c) = queue.pop_front():
    in_queue[c] = false; visits += 1
    best = None
    for k in candidates(c) in list order, with pending[c][k] == 0:
        plan = Plan { heuristic_cost: own(k) saturating+ sum(memo[child].cost), ... }  // as :1277-1306
        if is_better(plan, best) { best = plan }                      // strict: first minimal wins (I4)
    current = memo.get(c)
    match (best, current):
        (Some(new), None)                      => write(c, new); planned(c)
        (Some(new), Some(old)) if is_better(new, old) =>
            cheaper = new.cost < old.cost; write(c, new)
            if cheaper { for (c2, k2) in dependents[c] where pending[c2][k2] == 0 { push(c2) } }
        _ => ()

planned(c): for (c2, k2) in dependents[c]: pending[c2][k2] -= 1; if pending[c2][k2] == 0 { push(c2) }
push(c):    if !in_queue[c] { in_queue[c] = true; queue.push_back(c) }
```

Notes that make it identical to today's loop: `best` is recomputed over
the eligible candidates in list order with the same strict `is_better`,
so the first minimal candidate wins exactly as at `:1306-1308`; the write
is gated on strict improvement over the current entry exactly as at
`:1311-1320`; the input-terminal plan is the same literal (`:1266-1276`);
`source_eclass` defaults to the class as at `:1295-1298`. Input terminals
never have candidates (`:1135-1157`), so they are written once at seeding
and never revisited.

### 5.3 Settle and blockage (unchanged text, new inputs)

Run `:1326-1348` verbatim over `u.classes.iter().zip(&u.candidates)`.
Post: I1 (`memo` has every universe class), I5 (`blocked` values in
candidate x children order; `no_candidates` in universe order).

### 5.4 Termination and the tripwire

Every write to `memo[c]` strictly decreases `c`'s tuple in the
lexicographic order on `(u64, label, key)`, whose label and key
components range over finite sets, so each class is written finitely
often; each write pushes at most `|dependents[c]|` classes; each visit
does `O(sum_k |children(k)|)` work; the queue is therefore emptied after
finitely many visits. Bound for the tripwire (inferred, Bellman-Ford
style): with a FIFO queue, after `r` "rounds" every class whose fixpoint
plan tree has height <= `r` is final, each class is queued at most once
per round, and height <= `|U|`, so `visits <= |U| x (|U| + 1)`. Assert
that with the message "extraction fixpoint did not converge after
{visits} class visits over {} classes" — the analogue of today's
100,000-pass assertion (`:1249-1252`), which the 2026-08-07 minimal-repo
ruling kept as an algorithm invariant. In genome mode visits ~= `|U|`.

### 5.5 Refusals

Unchanged code paths, unchanged inputs (I5, I6): `extract()`'s spine walk
`:1066-1128` names the failing output positions from `memo`;
`failure_breakdown` classifies `blocked` SCCs and `no_candidates`. A
choice-cycle is a set of classes whose counters never reach zero because
each waits on another; a dead-end is a class with an empty candidate
list. Both are recorded by the settle/blockage step, never by the
worklist itself.

### 5.6 The cycle species

Copy <-> Copy weld, genome elects both copies: `pending[x][copy] = 1`
(child `y`), `pending[y][copy] = 1` (child `x`); neither `planned()`
fires; both stay `None`; blockage records `blocked[x] = [y]`, `blocked[y]
= [x]`; `failure_breakdown` finds the 2-SCC; identical to today. Genome
elects a progressing producer for `y`: `y` is pushed when its inputs are
planned, written once, `planned(y)` drops `pending[x][copy]` to 0, `x` is
written once. Two visits.

Double-transpose collapse in genome mode: identical to the weld with
`IndexMapApplyView` candidates (own cost 0). In plain mode both views and
both kernel routes are candidates; the first class to plan does so via its
kernel route, the other's view becomes eligible and is compared against
its own kernel route by `is_better` (cost tie possible, label decides) —
exactly today's fixpoint, including the pre-existing cyclic-plan hazard
described in 2.6, which this design neither fixes nor worsens.

### 5.7 Phase 2: `candidates_for_choice(&self, class, choice) -> Vec<Candidate>`

Pre: `class` is in `producer_index`; `choice` is the genome's row.
Post: equals today's `:1158-1170` result — the structural candidate for
the chosen enode if it is a structural spelling (`candidate_for_node`),
followed by, for each `ProducerRef` of `class` in index order whose
`op_class == egraph.nodes[choice.enode].eclass` and `output_index ==
choice.output_index`, the `candidate_for_layout_op` of the chosen enode
with that spec (skipping subsumed/`[...]`/unmatched enodes as
`candidate_for_layout_op` does). Nothing else is constructed. The
optional session cache `choice_cache: RefCell<HashMap<(ClassId, NodeId,
usize), Rc<[Candidate]>>>` memoizes the result; never cleared by
`extract_with_genome` (genome-independent by construction: the key IS
the choice).

### 5.8 Phase 0: the differential

```rust
pub struct ExtractionDifferential {
    pub universe: usize,
    pub reference_passes: usize,       // old loop
    pub visits: u64, pub writes: u64,  // new loop
    pub memo_mismatches: Vec<ClassId>, // classes whose plans differ (2.3 fields)
    pub blockage_equal: bool,
    pub tie_corners: usize,            // classes with >=2 eligible equal-tuple candidates with different children (2.3)
    pub graph_equal: Result<(), String>,
    pub fingerprint_equal: bool,
    pub refusal_equal: bool,           // Display of Err, and failure_breakdown() triple
}
impl ExtractionSession<'_> {
    pub fn extract_with_genome_differential(&mut self, genome: &Genome)
        -> (Result<Option<ExtractedGraph>>, ExtractionDifferential);
}
```

Runs the reference loop on a cleared session, snapshots memo/blocked/
no_candidates/result/fingerprint/breakdown; clears; runs the worklist;
compares; leaves the WORKLIST result in the session. Plans compare on
`heuristic_cost, source_eclass, source_enode, selected_output_index,
input_list, output_list, children (port, class) sequence, metadata (name,
class) sequence, plan_label`. Graphs compare structurally (7.1). An
`Err` result compares by `format!("{err:#}")`.

---

## 6. Implementation plan

Each phase compiles green on its own, lands alone, and carries its gate.
Gates follow the 2026-09-03 MINIMAL-gate ruling (build + the named tests
+ fmt + lib-only clippy blocking; the full ~10-minute gate detached).
Line counts are estimates.

**Phase 0 — the differential harness first** (~+300 lines, no behaviour
change). Add `extracted_graphs_identical` to `src/test_support.rs`; add
`ExtractionDifferential` and `extract_with_genome_differential` to
`src/extraction.rs`, initially comparing the pass loop against ITSELF
(second run on a cleared memo) so the harness is exercised end to end;
add `tests/test_runtime/tests/extraction_differential.rs` sweeping the 9
`tests/test_runtime/fixtures/*.egg`, the 30 `src/egglog_core/test_scripts/*.egg`
that assemble against the test runtime's registry, the 15
`GOLDEN_SCRIPTS` (`src/test_support.rs:1308-1324`) in plain mode, and the
`scc_sampler` boards' e-graphs, each with 16 seeds of
`sample_genome_with_seed` and 16 of `mutate_genome_with_seed` plus plain
mode. Include one deliberately perturbed comparison (drop an edge) to
prove the helper can fail. Gate: the new test green; `cargo test -p
luminal --lib`; fmt; clippy.

**Phase 1 — the worklist** (~+220 / -0 lines; the old loop stays as
`relax_passes_reference`). Implement 5.1-5.4; `relax_to_fixpoint`
dispatches to the worklist; the differential compares old vs new; the env
lever runs it inside `extract_with_genome`. Rename `relax_passes` ->
`relax_visits` + `relax_writes` in whichever of #509/this lands second.
Gate: `extraction_differential` green with `memo_mismatches` empty and
`tie_corners == 0` reported (a nonzero count is a finding, not a failure,
see R1); `cargo test -p luminal --lib`; `cargo test -p test_runtime
--test scc_sampler --test rehomed --test r10_debug --test mutation --test
planner --test views`; `cargo test -p luminal_cuda_lite --test
cublaslt_election --test finalists_lattice --test dim_buckets --test
scc_sampler_marker --test ladder_refusals`; `cargo test -p
luminal_reference --test mini_model_smoke --test corpus`; `cargo test -p
luminal_cuda_lite --test example_smoke`; fmt; clippy. Report the host
table of 1.4 with `visits` in place of `passes`.

**Phase 2 — chosen-only discovery** (~+90 / -12 lines). Implement 5.7;
fix the comment at `:1132-1134`; optional `choice_cache`. Gate: as Phase 1
plus PR #509's `extract_discovery_nanos` on the CL election rows before/
after (host), expecting the genomes-2..4 discovery (0.5-2.7 s on the
minis, table 1.4) to fall to the per-genome steady state.

**Phase 3 — dense interning** (~+150 / -60 lines; optional). Index
`pending`, `dependents`, memo lookups inside the worklist by `u32`;
compute tuples without materializing a `Plan` for candidates that do not
win. Land only if the A100 buckets after Phases 1-2 still show
`extract_relax_nanos` above ~5% of `extract_nanos`. Gate: as Phase 1.

**Phase 4 — the A100 run, then deletion** (~-180 lines). Run the
examples (whisper, qwen3, gemma3, yolo_v11n) with
`LUMINAL_EXTRACT_DIFFERENTIAL=1` on the A100 box per the device
verification recipe (push -> pull; release examples build). Any mismatch
panics with the report and stops the phase. Then delete
`relax_passes_reference`, the env lever, and
`extract_with_genome_differential` (keep `extracted_graphs_identical` and
the differential TEST only if it can compare against a pinned structural
digest — otherwise delete it too; recommendation: delete, per the
minimal-repo ruling — the identity argument is then carried by 2.4 and
the pins). Gate: the standard suites plus the performance report (7.3).

Order rationale: the harness exists before any algorithm changes (Phase
0), so Phase 1's first run is a real differential; Phase 2 is separable
because I2 is stated in terms of the surviving subsequence; Phase 3 is
measurement-gated; Phase 4 removes the scaffolding only after the largest
graphs have been compared.

---

## 7. Verification

### 7.1 Differential test (Phase 0/1)

`extracted_graphs_identical(a, b)`: same `outputs` (as node-index
sequences), same node count, and for each index `i` the same node kind
with: `LayoutOp` — `op.label()`, `provenance` (`op_eclass`,
`source_enode`, `selected_output_index` as strings), `inputs` (port,
value) sequence, `outputs` (eclass, label, dims, element_bits,
dtype_enum) sequence, `heuristic_cost`; `BufferInput` — `value.eclass`,
`value.label`, `buffer.{tensor_eclass, id_eclass, access, freed_by,
lit}`; `BufferOutput` — `eclass`, `label`, `slots` (index, value, buffer
ids). Edges: the sorted multiset of `(source, target, value, port)`.
Plus `plan_fingerprint(a) == plan_fingerprint(b)`. Node-index equality is
legitimate because both graphs are produced by the same deterministic
`build_extracted_graph` from identical memos in one process, where class
ids are stable; this is a within-process differential, not a
cross-process pin. Tooltips (`LazyText`) are built from the same `Plan`
fields; the test forces and compares `OpNode.tooltip` on fixtures only.

Refusal identity: compare `format!("{err:#}")` and the
`failure_breakdown()` triple.

Scope: fixtures and scripts listed in Phase 0; every mini through both
runtimes' `ExtractionSession` with the harness budget and additionally
32 sampled + 32 mutated seeds; plain mode for all fixtures. Hand-built
cyclic genomes (from the `scc_sampler` boards' components: elect the
intra-component candidate for every member) to exercise the refusal
path.

Old implementation retention: as `relax_passes_reference` only, deleted
in Phase 4 after the A100 differential (Section 6).

### 7.2 Election pins bit-identical

`cargo test -p luminal_cuda_lite --test cublaslt_election --test
finalists_lattice --test dim_buckets` unchanged, plus `tests/test_runtime/
tests/r10_debug.rs` and the `rehomed` golden. These are implied by I7 but
run anyway: `election_row_*` print the marker election counts
(`marker_elected=12 computes=140` for whisper, `6/126` qwen3, `12/332`
gemma3 in this design's runs) and must print the same rows.

### 7.3 Performance targets

Instrumentation: PR #509's `ExtractTimings` (`discovery_nanos`,
`relax_nanos`, `relax_passes` -> `relax_visits`/`relax_writes`,
`assemble_nanos`) summed per search and printed by the search summary.

Hard targets (per genome, genome mode):

- `relax_visits <= |U| + relax_writes` and `relax_writes == planned
  classes` (one write per planned class; zero re-plans) on every mini;
  report any excess as a finding.
- `extract_relax_nanos <= 2% of extract_nanos` on whisper/qwen3/gemma3
  examples on the A100.

Soft targets (A100, examples, 64 genomes, against 2026-09-04): qwen3
extraction 951 s -> <= 300 s after Phase 1 (removing ~640 s of relax; the
remaining ~4-5 s/genome of discovery + assembly is the unknown the #509
buckets will split), <= 60 s after Phase 2 if discovery is the bulk of the
remainder; gemma3 926 s similarly; whisper 161 s -> <= 30 s. yolo_v11n:
report only. These are predictions from 1.3-1.5, to be replaced by the
measured buckets; a miss is information about assembly, not about the
worklist, as long as the hard targets hold.

Host: table 1.4 re-run in the same profile; relax column expected to fall
below the discovery column on every row.

### 7.4 Property tests worth adding

- **Order independence** (extends the reversal experiment): for a random
  permutation of the worklist's initial seed order (or a LIFO queue),
  the memo equals the FIFO memo. Cheap, catches any accidental order
  dependence introduced later.
- **Relabel commutation**: for a random relabeling `rho` of class ids
  (the `proto/id-relabel` harness), `new(rho(egraph), rho(genome)) ==
  rho(new(egraph, genome))` up to the known pre-existing volatility
  (`stable_key` spells depth-0 class ids; `producer_index` order follows
  `HashMap` iteration). Verified fact from the relabel harness (memory,
  2026-09-02): the CURRENT extractor's outcomes are NOT relabel-invariant,
  and that item is parked by ruling. So the property to add is relative
  — old and new agree under every relabeling — which the differential
  already gives when run on a relabeled e-graph; absolute invariance is
  not this design's claim.
- **Incremental == from-scratch**: only if option (c) is ever built.
- **Universe equality**: `discover` returns the same class set and order
  for old and new (I1) — folded into the differential.

### 7.5 What would falsify the design

1. A memo mismatch on any pair with `tie_corners == 0`: the identity
   argument (2.4) is wrong somewhere — stop, do not land.
2. A mismatch only where `tie_corners > 0`: the corner is real on some
   graph; land only with Austin's ruling on I4 (R1).
3. `relax_visits / |U|` well above 1 in genome mode: the "one visit per
   class" claim fails (multi-spec re-plans exist at scale); correctness
   is unaffected, the estimate in 3.1 is.
4. A100 `extract_relax_nanos` not falling by >= 100x after Phase 1: the
   pass loop was not where the time went (e.g. `is_better`/`stable_key`
   or `candidate_heuristic_cost` dominate); the buckets say which.
5. After Phases 1-2, per-genome extraction still > 1 s on qwen3-4B:
   assembly dominates; that is R7's design, not this one.

---

## 8. Risks and open questions for Austin

**R1. Canonical tie-break (I4).** Where two candidates of a class tie on
`(cost, label, key)` with different children, today's winner is whichever
was planned first, which depends on `HashMap` iteration order and is
therefore process-random; the worklist takes the first in candidate-list
order. Measured: zero such classes on every fixture and mini in both
modes. Recommendation: adopt I4, have the differential count the corner,
add no pin. If a count ever comes back nonzero, the doctrine answer
("never depend on spelling", "tests invariant to id/order") already says
neither outcome may be pinned.

**R2. Delete the old loop after the A100 differential.** Recommendation:
yes, in Phase 4, together with the env lever and the differential entry
point (minimal-repo ruling). Keep the structural-equality helper only if
another test wants it.

**R3. `relax_passes` in PR #509.** Recommendation: rename to
`relax_visits` and add `relax_writes`; whichever PR lands second carries
the two-line change.

**R4. Env lever vs cargo feature for the transition.** `SEARCH_LOG`
sets the precedent for an env-read debug lever; a cargo feature on the
core crate would ripple through three runtime crates. Recommendation: the
env lever, deleted in Phase 4.

**R5. `op_cache` fill order changes under Phase 2** (only elected enodes
are parsed per genome). Content-identical, order-irrelevant; the first
genome gets cheaper and later genomes may pay small parse costs. No
action; noted so a discovery-timing regression on genome 2 is not
misread.

**R6. Phase 3 (dense interning) at all?** Recommendation: only if the
post-Phase-1 A100 buckets show relax above ~5% of extraction. Otherwise
it is complexity for a bucket that no longer matters.

**R7. Assembly at 4B scale.** Unmeasured split today; the likely lever is
a session-level `layout_tensor_info` memo (genome-independent except the
waste-slot relabel). Recommendation: a separate ticket sized from #509's
`assemble` bucket after Phase 1 lands; not folded into LUM-798.

**R8. Plain-mode zero-cost view cycles** can still yield a cyclic
extracted graph (2.6), identical under both algorithms. Out of scope;
recommend a note on LUM-799. The genome path is protected by the sampler
and the input-terminal leaf rule.

**R9. The viability filter's own repeated-pass fixpoint**
(`apply_viability_filter`, `:1000-1056`) is quadratic in the same way but
runs once per session (inside `analysis_nanos`). Recommendation: leave
it; convert with the same counter technique only if #509's `analysis`
bucket says so.

**R10. `blocked` value ORDER (I5).** The differential compares `Vec`
order, which is stricter than `failure_breakdown` needs (it builds a
graph). Keeping the settle/blockage code verbatim makes this free; if
Phase 3 changes the iteration to dense indices it must still iterate in
universe order.

---

## Appendix A. Reproduction

### A.1 Pass-count counter (temporary, worktree only, not committed)

Worktree `.claude/worktrees/design-worklist` on `c0ee79cd`, `vendor/
egglog-checkout` copied from the primary, shared warm target
`CARGO_TARGET_DIR=.../rejoin-p1/target`. One `eprintln!` at the end of
`relax_to_fixpoint` printing `passes`, `universe.len()`,
`total_candidates`, discovery and relax elapsed. Commands:

    cargo test -p luminal_reference --test mini_model_smoke mini_<x>_runs -- --exact --nocapture
    cargo test -p luminal_cuda_lite --test cublaslt_election election_row_<x> -- --exact --nocapture

Raw lines are in the session scratchpad (`measure.log`); the table in 1.4
is their per-genome steady state.

### A.2 Order-reversal and tie probe (temporary, worktree only)

The pass loop was moved into `probe_passes(&mut self, universe,
candidate_lists, reverse: bool) -> (passes, replans, first_pass)`; under
`EXTRACT_PROBE=1` the loop ran forward, snapshotted the memo, ran again
in reverse over a cleared memo, compared every class (cost, source enode,
output slot, children, label), restored the forward memo, and counted
classes with >= 2 eligible equal-tuple candidates whose children differ.
Printed per extraction: `genome passes passes_rev mismatches ties replans
max_first_pass multi universe candidates discovery relax relax_rev`.

Aggregate over the run (522 lines):

| section | genome | lines | mismatch lines | tie lines | re-plan lines (max) | multi>0 lines | max passes | max passes_rev |
|---|---|---|---|---|---|---|---|---|
| ref llama3 | yes | 8 | 0 | 0 | 0 | 0 | 70 | 11 |
| ref whisper | yes | 8 | 0 | 0 | 0 | 0 | 94 | 14 |
| ref qwen3 | yes | 8 | 0 | 0 | 0 | 0 | 79 | 13 |
| ref gemma3 | yes | 8 | 0 | 0 | 0 | 0 | 199 | 28 |
| CL whisper (marker) | yes | 8 | 0 | 0 | 0 | 0 | 97 | 15 |
| CL qwen3 (marker) | yes | 8 | 0 | 0 | 0 | 0 | 77 | 12 |
| CL gemma3 (marker) | yes | 8 | 0 | 0 | 0 | 0 | 202 | 26 |
| CL 2D matmul marker | yes | 192 | 0 | 0 | 0 | 0 | 11 | 2 |
| test_runtime scc_sampler | yes | 264 | 0 | 0 | 0 | 0 | 13 | 3 |
| test_runtime rehomed | mixed | 6 | 0 | 0 | 0 | 1 (plain: 2 classes) | 6 | 3 |
| test_runtime r10_debug | mixed | 6 | 0 | 0 | 1 (4 re-plans, plain) | 2 (plain: 23, 25 classes) | 9 | 4 |

`passes == max_first_pass + 1` on every line.

## Appendix B. References

- Cousot, P. and Cousot, R. (1977). Automatic synthesis of optimal
  invariant assertions: mathematical foundations — chaotic iterations of
  monotone operators converge to the same fixpoint regardless of order.
- Kildall, G. (1973). A unified approach to global program optimization —
  the worklist formulation of dataflow fixpoints.
- Kahn, A. B. (1962). Topological sorting of large networks — the
  in-degree-counter scheduling the pending counters generalize.
- Tarjan, R. (1972). Depth-first search and linear graph algorithms —
  the SCC algorithm `failure_breakdown` and `SamplingSpace` already use
  via petgraph.
- Memory notes in this repository's project memory: `extraction-blowup-fixes`
  (the 2026-08-07 fixpoint's origin and the 2026-09-01 render-memo wall),
  `scc-sampler-design` and `cycle-anatomy-and-sampling` (the cycle
  species), `class-id-stability` and `permutation-invariant-graph-tests`
  (what may not be pinned).

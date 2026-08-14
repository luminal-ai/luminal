# Luminal Core Spec

This document describes the core design, semantics, and contracts of the Luminal compiler.

## Compile Flow

```
Frontend Program (PyTorch Core Aten IR, GraphTensor API, StableHLO (future), etc.)
    -> HLIR Graph
    -> Loop-rolled HLIR Graph
    -> Egglog Saturation (including backend-specific rewrites)
    -> EGraph
    -> Extraction Search (Looped)
        -> Genetic Algorithm chosen graph extraction
        -> Looped LLIR Graph
        -> Backend Profiling
    -> Best-performing unrolled LLIR Graph
    -> Runtime
```

## Global Contracts

- Every selectable LLIR graph for a given HLIR graph must be semantically equivalent for all valid inputs, dtypes, shapes, strides, and dynamic dims.
- `ReferenceRuntime` is the full semantic reference backend, ran on CPU. All HLIR semantics are supported. Performance is not a priority, correctness is.
- Frontend-exported (HLIR) programs are specifications of intended semantics. Do not change them to hide compiler, search, runtime, or backend bugs.

## Frontend

Frontends have two roles:

- Provide a convenient interface to construct HLIR graphs.
- Interface with the backend through the Runtime API.

An example frontend is luminal_python (PyTorch), which uses the torch.compile machinery to export a Core Aten IR graph, which is then decomposed to an HLIR graph. It also gives the user a way to drive the runtime through the Runtime API.

Inputs and outputs specified from the frontend must have physical layouts, all other HLIR does not have physical layouts.

## HLIR (High Level Intermediate Representation)

An HLIR graph is a complete bit-level logical specification of a tensor program.

HLIR is a directed acyclic graph made of strictly the following ops:

- Input: An input tensor provided by the user, with a given dtype, physical layout, and string label (for debugging).
- Output: An output tensor returned to the user, with a given physical layout.
- Constant: A constant value of a given dtype, specified at compile time.
- Cast: An elementwise type conversion, using casting semantics specified in the ReferenceBackend
- Iota: A value expression and a size expression. Value is evaluated size # of times to create a flat 1D tensor.
- Exp2: A base 2 elementwise exponential.
- Log2: A base 2 elementwise logarithmic.
- Sin: An elementwise sin.
- Recip: An elementwise reciprocal.
- Sqrt: An elementwise square-root.
- Add: An elementwise addition.
- Mul: An elementwise multiply.
- Mod: An elementwise modulo.
- LessThan: An elementwise less-than. Always outputs a boolean tensor.
- Gather: Given a data tensor and an index tensor, produces an output tensor with the same shape as the index tensor, made up of values contained in the data tensor at an index specified by each element of the index tensor.
- Scatter: Given a data tensor, an index tensor, and a destination tensor, produces an output tensor as the destination tensor with indexes specified by elements in the index tensor overwriten by cooresponding elements of the data tensor.
- SumReduce: Sum reduction over one dimension of a tensor.
- MaxReduce: Max reduction over one dimension of a tensor.

Input and Output ops are special ops in that they are both HLIR and LLIR ops. They persist through the full lowering process.

Dynamic dimensions are provided as variables to any expressions present in the HLIR graph. Through this, complex shapes can be derived. For instance, a LLM may define activation shape as `[batch, previous_sequence_length + current_sequence_length, 4096]`, where batch and sequence lengths are not known at compile time.

## LLIR (Low-Level Intermediate Representation)

LLIR (Low-Level Intermediate Representation) ops implement specified backend behavior. They need not be independant, but any extractable LLIR graph must be comprehensable by the backend's load_llir function, convertable into a valid executable.

An example of an LLIR operation might be a CUDA kernel, or a threadblock operation.

## Egglog

Egglog defines a nondestructive rewriting system, which when ran to saturation builds an e-graph.

Backends are responsible for providing rewrites to lower HLIR into backend-specific LLIR.

Luminal comes with a number of built-in egglog functions and analyses that can be used by backend-provided rules.

Egglog should be the only phase where rewriting and pattern matching happens. Every extractable LLIR graph should faithfully represent the final execution program. No post-passes should be used on the LLIR graph. Every extractable LLIR graph must be semantically valid, equivalent to the logical HLIR program, and executable.

- Core emits the HLIR program as the initial e-graph starting state.
- Egglog pulls LLIR ops and rewrites from the registered backend.
- `EgglogOp::sort` declares the egglog op shape. `rewrites` declares semantic
  equivalences. `cleanup` controls removal of intermediate forms.
- Rewrites must be valid as equivalences over the matched shape, stride, dtype,
  and dynamic-dim domain.
- Backend-specific fused or specialized ops must be introduced by egglog rules
  and selected by extraction. Do not add Rust-side LLIR post-passes that search
  for patterns, fuse kernels, or choose backend ops after extraction.
- HLIR ops are deleted as a final `cleanup` ruleset phase so they are not present during extraction.

## Search

Search samples choices from the saturated e-graph using a genetic algorithm, extracts LLIR candidates, and profiles them with the runtime. Every extractable LLIR graph must be valid. Validity / output checking is _not_ part of the search process, as outputs are assumed to be correct.

- A profiled candidate counts against the search limit only after extraction and
  backend filtering succeed.
- Candidate timeout covers compile plus run viability. Execution timeout covers
  a profiling trial only.
- Dynamic-dim buckets create separate search spaces and selected LLIR graphs.
  Backends that load buckets must dispatch according to runtime `dyn_map`.
- Profiling uses representative dynamic values, optionally overridden by
  `CompileOptions`. The final compiled graph must remain valid for the full
  declared dynamic domain, not only the representative point.
- Search may rank performance only after semantic equivalence is established.
  It must not select around a known incorrect candidate.

## Dynamism

Runtime dynamism is expressed through dynamic variables which can be used in expressions. Since tensor dimensions are represented as expressions, tensor dimension expressons containing dynamic variables vary at runtime.

Each execution of a backend uses a cooresponding `dyn_map`, which is a mapping from dynamic variables to concrete values. For example: `{'c': 5, 'b': 8, 'p': 3048}`.

Buckets are specifiable for each runtime variable, and at compile time seperate compiles will be done for each possible bucket combination. At runtime these compiled executables are dynamically dispatched depending on the current concrete values for each dynamic variable.

## Backend

A backend is a specific execution environment. Through implementing the Runtime trait, a backend provides:

- A set of LLIR ops.
- A set of rewrites that rewrite from HLIR patterns to LLIR patterns, LLIR patterns to other LLIR patterns, or any mix of the two.
- `.set_data(NodeIndex, T)` and `.get_data(NodeIndex) -> T` for setting and retriving inputs and outputs respectively.
- `initialize()` to set up the backend, and load any fixed state.
- `load_llir(llir_graph)` to initialize an LLIR graph in the backend. This is where an LLIR graph is converted to an executable artifact.
- `execute()` to execute the already-loaded executable artifact.
- `profile(llir_graph)` to return a fitness metric minimized by search for a given LLIR graph.

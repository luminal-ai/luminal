# Associative Fold HLIR Idea

Status: deferred design idea; not approved for implementation.

## Motivation

Luminal currently represents summation and maximum reduction as separate HLIR
operations. Adding a third operation for product reduction would repeat their
common shape, stride, extraction, and symbolic-dimension machinery. The current
frontend product implementation is not a suitable substitute: it computes
`exp(sum(log(x)))`, which does not preserve multiplication semantics for
negative values, signed zero, integer dtypes, overflow and underflow, or the
frontend-only representation of complex values.

Reductions and scans are also closely related. A reduction returns only the
final accumulator, while a scan returns the intermediate accumulator for every
position. A single structural abstraction could express both while allowing
backends to keep specialized kernels.

## Proposed abstraction

Introduce one `Fold` or `AssociativeFold` HLIR operation parameterized by a
closed reducer and an output mode:

```rust
struct Fold {
    reducer: Reducer,
    axis: usize,
    output: FoldOutput,
    direction: FoldDirection,
    init: FoldInit,
    input_shape: ShapeTracker,
}

enum Reducer {
    Add,
    Multiply,
    Maximum,
    Minimum,
    And,
    Or,
}

enum FoldOutput {
    Final,
    Inclusive,
    Exclusive,
}

enum FoldDirection {
    Forward,
    Reverse,
}

enum FoldInit {
    Identity,
    FirstElement,
    Explicit,
}
```

`Final` removes the folded axis and implements a reduction. `Inclusive` and
`Exclusive` preserve the axis and implement scans. An explicit initial value
could be represented as another input if and when an operation requires it.

## Frontend mappings

| Frontend operation | Fold representation |
| --- | --- |
| `sum` | `Fold(Add, Final)` |
| `prod` | `Fold(Multiply, Final)` |
| `max` | `Fold(Maximum, Final)` |
| `min` | `Fold(Minimum, Final)` |
| `all` | `Fold(And, Final)` |
| `any` | `Fold(Or, Final)` |
| `cumsum` | `Fold(Add, Inclusive)` |
| `cumprod` | `Fold(Multiply, Inclusive)` |
| cumulative maximum values | `Fold(Maximum, Inclusive)` |
| cumulative minimum values | `Fold(Minimum, Inclusive)` |

`mean` should remain a sum fold followed by division by the symbolic element
count. It does not need a distinct reducer.

## Initialization and empty inputs

Initialization must be typed and reducer-aware rather than represented by an
`f32` sentinel:

- Add uses zero.
- Multiply uses one.
- And uses true.
- Or uses false.
- Maximum and minimum normally initialize from the first valid element.

Using the first element avoids incorrect sentinels such as `f32::MIN` for an
F64 maximum. It also gives the operation a place to define empty-reduction and
empty-scan behavior explicitly. An explicit initial value can provide an
identity where the frontend API supports one.

## Optimization model

The reducer is deliberately a closed enum rather than an arbitrary computation
region. This makes its algebraic and dtype behavior visible to egglog and to
backend selection. The HLIR can remain generic while backend rewrites produce
specialized operations, for example:

```text
Fold(Add, Final)          -> KernelSumReduce
Fold(Maximum, Final)      -> KernelMaxReduce
Fold(Multiply, Final)     -> KernelProductReduce
Fold(Multiply, Inclusive) -> KernelProductScan
```

Sum-specific matmul and cast-sum fusion rules would continue to match only the
`Add` reducer. A single HLIR operation therefore does not require a single
generic backend kernel.

The operation must define ordering and IEEE behavior precisely. Floating-point
reducers are not associative at the bit level, and maximum/minimum must specify
NaN propagation, signed-zero treatment, and tie behavior rather than inheriting
those details accidentally from a host-language comparison.

## Rich accumulator state

A scalar reducer covers ordinary value reductions and scans, but some operations
need multiple loop-carried values:

- `cummax` and `cummin` need `(value, index)` to reproduce latest-occurrence,
  NaN, and signed-zero semantics.
- Complex cumulative product needs `(real, imaginary)` because complex is not
  an HLIR dtype.
- Numerically stable variance can use `(count, mean, m2)`.

A later version of `Fold` could accept tuple state and a constrained reducer
region. That would make the operation much more general, but it also requires
multi-output/state representation, stronger legality checks, more complex
autodiff, and more difficult backend matching. It should not be part of the
initial closed-reducer design.

## Why not an arbitrary fold now?

An arbitrary `body(accumulator, element) -> accumulator` could express all of
the cases above, but the compiler would no longer know whether it may reorder,
parallelize, vectorize, or tree-reduce the body. Proving those properties inside
egglog and generating efficient kernels for arbitrary regions would be a much
larger project. The closed reducer set captures the immediate duplication and
correctness problems without committing HLIR to higher-order regions.

## Possible implementation sequence

If this idea is revisited:

1. Introduce `Reducer` and `Fold(Final)` and migrate the existing sum and maximum
   reductions without changing behavior.
2. Add typed multiplication, minimum, and boolean reducers to the reference
   backend.
3. Replace the log-based frontend product implementation.
4. Migrate backend-specific reduction rewrites and kernels while keeping their
   specializations.
5. Add inclusive and exclusive scan output modes with symbolic-axis tests.
6. Evaluate tuple accumulator state separately for indices, complex product,
   and statistical reductions.

## Open questions

- Should the operation be named `Fold`, `AssociativeFold`, or `Aggregate`?
- Which floating-point reassociations are legal, and which output differences
  are part of the backend contract?
- Should maximum/minimum expose latest-index semantics directly, or should
  value-and-index folds be a separate operation?
- How should an explicit initial value be represented and typed?
- Should scans be part of the first implementation or added only after generic
  reductions have replaced the current operations?

This document records the idea only. No HLIR migration or implementation is
currently planned.

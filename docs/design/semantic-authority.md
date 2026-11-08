# Model semantic authority

Status: repository invariant. This document is the durable project memory for
compiler, frontend, model, runtime, and backend work.

## Single source of truth

The authored model definition is the single source of functional truth:

- For PyTorch, it is the PyTorch program as captured with its dtype, aliasing,
  mutation, autograd, DTensor placement, sharding, and collective semantics.
- For Rust, it is the recorded `GraphTensor` program and any explicit
  distribution metadata.

The specification includes operations and their dependency ordering, tensor shapes and
dynamic domains, dtypes, explicit casts, accumulator and output dtypes,
explicit rounding boundaries, constants, exceptional-value behavior, parameter sharing,
mutation and aliasing, randomness, sharding placements, collectives, and
collective order. Checkpoint storage metadata, device capabilities, compiler
defaults, and performance goals do not override it.

Reduction association and evaluation order are unspecified unless the model
explicitly authors an ordered or sequential operation. Implementations may use
a different tree, reassociation, or contraction within a logical reduction, but
that freedom does not change the authored operand, accumulator, or output dtype
and does not cross an explicit cast or rounding boundary.

## Compiler permission

The compiler may change the physical implementation only. Egglog can introduce
and select a kernel, library call, fusion, layout, schedule, buffer reuse, or
communication implementation only when it is semantically equivalent to the
authored model over the entire valid input and dynamic-shape domain.

The compiler must not silently:

- insert or remove casts, mixed precision, quantization, or dequantization;
- change operand, accumulator, intermediate, or output dtypes;
- move an explicit rounding boundary or replace exact authored arithmetic with an
  approximation;
- reshard, replicate, or change placements;
- add, remove, reorder, or replace model collectives in a way that changes the
  authored distribution contract; or
- edit model code, weights, prompts, constants, or outputs to make compilation
  pass, improve speed, or fit memory.

An explicit frontend transform may change a model only before compilation and
only when the user requested that transform; the transformed program then
becomes the new authored specification. It is not a hidden compiler
optimization.

## Failure behavior

If no exact implementation fits the backend, declared dynamic domain, memory
budget, or supported distribution mechanisms, compilation must refuse with a
diagnostic. Numerical closeness, a tolerance-based test, common vendor behavior,
or a faster implementation cannot authorize different semantics.

All selectable e-graph candidates are therefore claims of equivalence. When
two candidates disagree, the rewrite or admission proof is wrong. Remove the
invalid equivalence and add a regression at the layer where it entered; never
teach search to avoid that candidate by luck.

## Model corrections

Model code can be corrected only with direct evidence that it disagrees with
its authoritative source. Such a correction is a model change, must preserve
the source framework's documented behavior—including dtype and sharding—and
must carry a model-level regression. A compiler or memory failure is not
evidence that the model definition is wrong.

# Affine normal form before expansion

*2026-10-26. Preamble section "AFFINE NORMAL FORM, BEFORE EXPANSION", `main_ruleset`.*

## The problem

Composed view offsets are integer expressions over coordinates. A view of a
view substitutes one index map into another's offset, so a slice of a reshape
of a concatenation arrives as terms such as

```
(c1 * 128) * 32            a coordinate scaled by the strides of two reshapes
Max((c1 + 64) + -64, 0)    a slice start under a pad's clamp
(Min(c1, 63) + c0) * 128   a literal factor outside a sum
```

The equivalence-discovery rules for integers — associativity, both
distributivity directions — were written to find that two such spellings are
the same function, and they are fenced (certificate guards, the G2 constant
refusals) so that they never regroup through the zero class. Inside the
fences they still *enumerate*: every regrouping of a term is a new e-node, and
distribution and association mint new sub-term classes (`l*k`, `r*k`, `a*b`),
each of which must then be proven closed, bounded and nonzero before anything
composes with it, and each of which is itself eligible for the same rules.

On a joined stream (a `cat`, spelled as clamped views and masks) with a tail
slice and a reshape around a reduction in between — FLUX's single-stream
block: join, q/k RMSNorm over heads, split — that enumeration does not
converge. The Trainium backend's reproducer (`spike/micro.py wm_no_residual`
in the Kontext workspace),

```
cat([a, b], 1) -> Linear -> unflatten(-1, (4, 32)) -> mean(x², -1, keepdim)
               -> q * m -> flatten(2, 3) -> [:, 64:]
```

grows from 0.06 GB after five rounds of the main stratum to over 24 GB in the
sixth. Per-round match counts: `IntMul closed when both operands are closed`
– / – / 1106 / 2074, `bounds of IntMul, four-corner` 49 / 147 / 723 / 904,
`int-mul-associate` 0 / 4 / 374 / 478, `int-distribute-factor` – / 4 / 79 /
127 (rounds 2 to 5). The e-graph dumps show the growth in `Mul(Mul(c, k), k)`,
`Mul(k, Mul(c, k))`, `Add(Mul(c, k), Mul(c, k))` and `Mul(k, k)` spellings:
nested literal scalings, distributed literal factors, and the literal
products those leave behind.

## The change

Three folds in the preamble (`main_ruleset`, section "AFFINE NORMAL FORM,
BEFORE EXPANSION") rewrite the non-canonical affine spellings into the
canonical one and **subsume** the spelling they matched, both operand
orientations (commutativity presents the swapped twin; the like-term
combiner established the pattern). The affine fragment — literals,
coordinates, dimensions, `IntAdd`, `IntMul` by a literal — has a canonical
spelling: a sum of monomials, each a non-literal scaled by one literal, plus
one literal.

| spelling | canonical | rule |
|---|---|---|
| `(x * a) * b` | `x * (a·b)` | nested literal scaling folds |
| `(x + y) * k` | `x*k + y*k` | a literal factor distributes |
| `(x + a) + b` | `x + (a+b)` | nested literal summands fold |

Every rule is a ring identity, so the union is sound; subsuming the matched
node removes a representative from matching and extraction, never an
equality. Once a term is canonical the expansion rules have nothing
non-canonical to regroup, and the like-term combiner finishes the sum.

Guards on the folds, each one load-bearing:

- Checked i64 arithmetic: literals beyond 2^31 (a slice end an exporter
  spells as `i64::MAX`) are left unfolded.
- No identity literal (1 under `IntMul`, 0 under `IntAdd`) and no constant
  (point-bound) inner term. A class holds its identity spelling for an
  iteration before the identity rule subsumes it — `c` is also `c * 1` — so
  without this a fold binds `?inner` to the operand's own class through that
  spelling and subsumes the node it matched (`c * 8` as `(c * 1) * 8`),
  leaving a live term with no live spelling: a backend's recorded chain
  pattern then matches nothing (the Trainium group-norm election). A constant
  inner is a preserved factorization (`8 = 2 * 4`) the constant-folding
  discipline keeps live on purpose.
- Distribution asks the nonzero certificate of both summands, as the
  expansion rules do; the zero class is never certified, so `x + 0` cannot
  bind the sum to `x`'s own class.

Only literal factors and summands move. A product of two unknowns, a
symbolic dimension as a factor, and every `Min`/`Max`/`div`/`rem` term stay
where they are: nothing here is a polynomial normal form for them, and
expanding them is what produced the combinatorics. A backend that cannot
read such a term by stride reads it element by element through the offset
function (the Trainium materializer does).

## What subsumption costs, and who pays it

A structural matcher that matched the recorded spelling of such a term has
to match the canonical one. Two did:

- **Core's bit-to-element layout discovery** (3c) matched a bit offset as
  `(IntMul elements (IntLit bits))`; once the folds scale `elements`'
  monomials by the width, that spelling is subsumed. The discovery now also
  reads the canonical form through `bits-quotient`, a quotient by the width
  demanded per bit layout and built bottom-up (a divisible literal, a
  monomial with a divisible literal factor, a sum of two quotients); anything
  else stalls and the layout keeps its bit form.
- **A backend's chain rules** are patterns recorded from a frontend graph,
  with symbolic pattern dimensions where the model has literals, so a
  recorded `(c * K) * 32` is spelled `c * 128` by the model. The Trainium
  generalizer now emits the canonical shape with the folded literal as a
  class the rule computes.

Two alternatives were measured and rejected. Unions without subsumption,
with the expansion rules refusing literal outer operands, converge, and keep
every spelling live, but the substitution walk then visits every live
spelling: 14.9 s instead of 8.9 s on one FLUX double block and ten minutes
instead of eighteen seconds on a two-block stack. Demand-gating the literal
exact-division route by divisor left a div/rem normalization non-convergent.

## What it does to the reproducer and the blocks

Unbounded saturation of `wm_no_residual`: 0.48 s, 39 LLIR nodes. FLUX's
single block in the per-block form (join at entry, split at exit, the form
that ran away): 2.2 s. The double block with per-stream attention: 8.9 s.
Double block, join, single block, split: 17.6 s (previously divergent).
Core's library tests and the backend's election fixtures (group norm and
the vocabulary head among them) pass.

## What it does not change

The associativity and distributivity rules and their fences stay as they
are: they are still the discovery mechanism for the non-literal cases, and
their fixtures still hold. Constant folding still does not subsume (the
preserved-factorization exception), so `Mul(k, k)` spellings of literal
classes remain; the nested-scaling fold produces the folded literal
directly, so those spellings are no longer the fuel for a second scaling.

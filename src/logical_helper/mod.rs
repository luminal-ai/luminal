//! Helper spellings of common `LogicalIndexMapApply` patterns.
//!
//! A helper is a `LogicalTensor` constructor that names one view pattern
//! and carries every field needed to rebuild it. Each instance has two
//! rules: recognize (the apply's class gains the helper) and expand (the
//! helper's class gains the apply), so rewrite rules can match and mint
//! the pattern by name. Helpers own no shape or dtype rules: the union
//! with the apply carries both. They live beside the core logical-op
//! vocabulary, not inside it, and register through their own list.

use crate::logical_op::LogicalOp;

mod broadcast_axis;
mod matrix_transpose;

pub use broadcast_axis::LogicalHelperBroadcastAxis;
pub use matrix_transpose::LogicalHelperMatrixTranspose;

/// THE registration list for logical helpers, kept apart from
/// [`crate::logical_op::built_in_logical_ops`] so the core vocabulary
/// stays small. Consulted after the ops wherever a constructor is looked
/// up by name.
pub fn built_in_logical_helpers() -> &'static [Box<dyn LogicalOp + Send + Sync>] {
    static HELPERS: std::sync::OnceLock<Vec<Box<dyn LogicalOp + Send + Sync>>> =
        std::sync::OnceLock::new();
    HELPERS.get_or_init(|| {
        vec![
            Box::new(LogicalHelperMatrixTranspose),
            Box::new(LogicalHelperBroadcastAxis),
        ]
    })
}

/// Registry lookup by egglog constructor name.
pub fn logical_helper_for(constructor: &str) -> Option<&'static (dyn LogicalOp + Send + Sync)> {
    built_in_logical_helpers()
        .iter()
        .find(|helper| helper.egglog_constructor() == constructor)
        .map(|helper| helper.as_ref())
}

#[cfg(test)]
mod tests {
    use crate::egglog_snippet::{assembled_program_for, new_egraph};

    const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run) (run prop)) (run subst-walk)))";

    /// Run `script` under the core program (no runtime matchers) — its
    /// `check`s are the assertions.
    fn run(script: &str) {
        let program = format!("{}\n\n{script}", assembled_program_for(&[]));
        new_egraph()
            .parse_and_run_program(None, &program)
            .unwrap_or_else(|err| panic!("helper script: {err}"));
    }

    #[test]
    fn matrix_transpose_recognizes_expands_and_involutes() {
        run(&format!(
            r#"
(let pshape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil)))))
(let oshape (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 5) (IntExprNil)))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") pshape (F32)))
; the recorder's spelling of x.permute((1, 0))
(let xt (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar oshape 0) (IntExprCons (CoordVar oshape 1) (IntExprNil))) pshape)
  oshape))
; a helper minted directly, as a rewrite would
(let ht (LogicalHelperMatrixTranspose x))
(let htt (LogicalHelperMatrixTranspose (LogicalHelperMatrixTranspose x)))
; a shrinking read through the same entries is NOT a transpose
(let sshape (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 5) (IntExprNil)))))
(let xs (LogicalIndexMapApply x
  (IndexMapLit (IntExprCons (CoordVar sshape 0) (IntExprCons (CoordVar sshape 1) (IntExprNil))) pshape)
  sshape))
; an extent-1 axis: the transpose still saturates without a shape conflict
(let p14 (ShapeLit (IntExprCons (IntLit 1) (IntExprCons (IntLit 4) (IntExprNil)))))
(let y (LogicalTensorInputLit (LogicalIdLit "y") p14 (F32)))
(let yt (LogicalHelperMatrixTranspose y))
{SCHEDULE}
(check (= xt (LogicalHelperMatrixTranspose x)))
(check (= ht xt))
(check (= (shape-of ht) oshape))
(check (= (dtype-of ht) (F32)))
(check (= htt x))
(fail (check (= xs (LogicalHelperMatrixTranspose x))))
(check (= (shape-of yt) (ShapeLit (IntExprCons (IntLit 4) (IntExprCons (IntLit 1) (IntExprNil))))))
"#
        ));
    }

    #[test]
    fn broadcast_axis_recognizes_and_expands_every_instance() {
        run(&format!(
            r#"
(let pshape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil)))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") pshape (F32)))
(let n (IntLit 7))
; the matmul A leg: a[m,k] read through (c2 c0) into P = [m, n, k]
(let p3 (ShapeLit (IntExprCons (IntLit 5) (IntExprCons n (IntExprCons (IntLit 3) (IntExprNil))))))
(let a_b (LogicalIndexMapApply a
  (IndexMapLit (IntExprCons (CoordVar p3 2) (IntExprCons (CoordVar p3 0) (IntExprNil))) pshape)
  p3))
; helpers minted directly at every position of a rank-2 parent
(let h0 (LogicalHelperBroadcastAxis a 0 n))
(let h1 (LogicalHelperBroadcastAxis a 1 n))
(let h2 (LogicalHelperBroadcastAxis a 2 n))
; a rank-1 parent, both positions
(let vshape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let v (LogicalTensorInputLit (LogicalIdLit "v") vshape (F32)))
(let v_out (ShapeLit (IntExprCons n (IntExprCons (IntLit 4) (IntExprNil)))))
(let v_b (LogicalIndexMapApply v (IndexMapLit (IntExprCons (CoordVar v_out 0) (IntExprNil)) vshape) v_out))
(let g0 (LogicalHelperBroadcastAxis v 0 n))
; the same entries over a shrunk out shape is NOT an insertion (shape tie)
(let bshape (ShapeLit (IntExprCons (IntLit 5) (IntExprCons n (IntExprCons (IntLit 2) (IntExprNil))))))
(let bad (LogicalIndexMapApply a
  (IndexMapLit (IntExprCons (CoordVar bshape 2) (IntExprCons (CoordVar bshape 0) (IntExprNil))) pshape)
  bshape))
{SCHEDULE}
(check (= a_b (LogicalHelperBroadcastAxis a 1 n)))
(check (= h1 a_b))
(check (= (shape-of h1) p3))
(check (= (dtype-of h1) (F32)))
(check (= (shape-of h0) (ShapeLit (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprCons n (IntExprNil)))))))
(check (= (shape-of h2) (ShapeLit (IntExprCons n (IntExprCons (IntLit 5) (IntExprCons (IntLit 3) (IntExprNil)))))))
(check (= v_b (LogicalHelperBroadcastAxis v 1 n)))
(check (= (shape-of g0) (ShapeLit (IntExprCons (IntLit 4) (IntExprCons n (IntExprNil))))))
(fail (check (= bad (LogicalHelperBroadcastAxis a 0 n))))
(fail (check (= bad (LogicalHelperBroadcastAxis a 1 n))))
(fail (check (= bad (LogicalHelperBroadcastAxis a 2 n))))
"#
        ));
    }
}

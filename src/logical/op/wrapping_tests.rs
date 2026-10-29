//! The wrapping family's value bounds and its bridge to the exact ops: a
//! wrapping result lies in its dtype's range, and where the operands'
//! bounds prove the exact result in width the wrapping op IS the exact op.

use crate::egglog_snippet::{assembled_program_for, new_egraph};

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

fn run(script: &str) {
    let program = format!("{}\n\n{script}", assembled_program_for(&[]));
    new_egraph()
        .parse_and_run_program(None, &program)
        .unwrap_or_else(|err| panic!("wrapping script: {err}"));
}

/// Two Int inputs of shape [4], with value bounds `[lo, hi]` attested on
/// each (`None` keeps only the storage bounds every Int input carries).
fn inputs(dtype: &str, bounds: Option<(i64, i64)>) -> String {
    let attest = bounds
        .map(|(lo, hi)| {
            format!(
                "(set (value-lower-bound-of a) (bigint {lo})) (set (value-upper-bound-of a) (bigint {hi}))
                 (set (value-lower-bound-of b) (bigint {lo})) (set (value-upper-bound-of b) (bigint {hi}))"
            )
        })
        .unwrap_or_default();
    format!(
        r#"
(let shape (ShapeLit (IntExprCons (IntLit 4) (IntExprNil))))
(let a (LogicalTensorInputLit (LogicalIdLit "a") shape ({dtype})))
(let b (LogicalTensorInputLit (LogicalIdLit "b") shape ({dtype})))
{attest}
(let wsum (LogicalWrappingAdd a b))
(let wprod (LogicalWrappingMul a b))
(let wtotal (LogicalWrappingReduceSum a 0))
(let wcumsum (LogicalWrappingScanSum a 0))
(let wcumprod (LogicalWrappingScanProd a 0))
"#
    )
}

#[test]
fn proven_wrapping_ops_are_their_exact_twins() {
    run(&format!(
        "{}{SCHEDULE}
(check (= wsum (LogicalAdd a b)))
(check (= wprod (LogicalMul a b)))
(check (= wtotal (LogicalReduceSum a 0)))
(check (= (value-lower-bound-of wsum) (bigint -200)))
(check (= (value-upper-bound-of wprod) (bigint 10000)))
",
        inputs("Int", Some((-100, 100)))
    ));
}

#[test]
fn unproven_wrapping_ops_stay_wrapping_and_lie_in_their_dtype_range() {
    run(&format!(
        "{}{SCHEDULE}
(fail (check (= wsum (LogicalAdd a b))))
(fail (check (= wprod (LogicalMul a b))))
(fail (check (= wtotal (LogicalReduceSum a 0))))
(check (= (value-lower-bound-of wsum) (bigint -2147483648)))
(check (= (value-upper-bound-of wsum) (bigint 2147483647)))
(check (= (value-lower-bound-of wcumsum) (bigint -2147483648)))
(check (= (value-upper-bound-of wcumprod) (bigint 2147483647)))
",
        inputs("Int", None)
    ));
    run(&format!(
        "{}{SCHEDULE}
(fail (check (= wsum (LogicalAdd a b))))
(check (= (value-upper-bound-of wtotal) (bigint 9223372036854775807)))
(check (= (value-lower-bound-of wcumprod) (bigint -9223372036854775808)))
",
        inputs("Int64", None)
    ));
}

#[test]
fn uint64_arithmetic_already_wraps() {
    run(&format!(
        "{}{SCHEDULE}
(check (= wsum (LogicalAdd a b)))
(check (= wprod (LogicalMul a b)))
(check (= wtotal (LogicalReduceSum a 0)))
",
        inputs("UInt64", None)
    ));
}

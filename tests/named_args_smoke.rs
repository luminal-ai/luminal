//! Smoke test: named-argument syntax (via egglog-experimental) parses and runs
//! through luminal's `new_egraph()`. Covers named declarations, out-of-order
//! named calls, leading-positional + `...`, and `...` fresh-var binding in a
//! query. This is the feature the hand-written matmul-flatten rules rely on.

use luminal::egglog_utils::new_egraph;

#[test]
fn named_args_and_ellipsis_work_in_luminal_egraph() {
    let mut egraph = new_egraph();
    let program = r#"
        ; Declare an op-like constructor with named fields.
        (datatype Vehicle
          (MyCar :color i64 :numwheel i64 :doors i64))

        ; Build with named args, out of declaration order.
        (let c1 (MyCar :numwheel 4 :doors 2 :color 7))
        ; Build positionally — must still work against a named schema.
        (let c2 (MyCar 7 4 2))
        (run 0)
        (check (= c1 c2))

        ; Query with a named arg + `...`: bind :color, let the rest be fresh.
        (relation SawColor (i64))
        (rule ((MyCar :color c ...)) ((SawColor c)))

        ; Query with a leading positional arg + `...`.
        (relation SawFirst (i64))
        (rule ((MyCar first ...)) ((SawFirst first)))

        (run 2)
        (check (SawColor 7))
        (check (SawFirst 7))
    "#;

    let commands = egraph
        .parser
        .get_program_from_string(None, program)
        .expect("named-argument program should parse");
    egraph
        .run_program(commands)
        .expect("named-argument program should run and all checks pass");
}

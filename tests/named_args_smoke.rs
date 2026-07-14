//! Smoke test: named-argument syntax (via egglog-experimental) parses and runs
//! through luminal's `new_egraph()`. Covers named declarations, out-of-order
//! named calls, leading-positional + `...`, and `...` fresh-var binding in a
//! query. This is the feature the hand-written matmul-flatten rules rely on.

use egglog::prelude::*; // egglog!/expr! quasiquotes
use luminal::egglog_utils::new_egraph;

/// End-to-end: the forked-egglog `egglog!`/`expr!` quasiquotes, driven with
/// luminal's experimental parser, expand named args + `...` (no text strings).
#[test]
fn quote_macros_with_named_args_in_luminal() {
    let mut egraph = new_egraph();

    // Declare a named constructor and build/run a program via `egglog!`.
    let prog = egglog!(
        egraph.parser,
        (datatype Vehicle (MyCar :color i64 :numwheel i64))
        (let c1 (MyCar :numwheel 4 :color 7))
        (let c2 (MyCar 7 4))
        (run 0)
        (check (= c1 c2))
    )
    .unwrap();
    egraph.run_program(prog).unwrap();

    // A rule matching `:color` and `...`-ing the rest, built with the macro.
    let rule = egglog!(
        egraph.parser,
        (relation SawColor (i64))
        (rule ((MyCar :color c ...)) ((SawColor c)))
        (run 1)
        (check (SawColor 7))
    )
    .unwrap();
    egraph.run_program(rule).unwrap();

    // `expr!` builds a single term against the experimental parser too.
    let e = expr!(egraph.parser, (MyCar :numwheel 4 :color 7)).unwrap();
    assert_eq!(format!("{e}"), "(MyCar 7 4)"); // named -> positional
}

/// Step-2 mechanism: a rule generator only knows the constructor and field
/// names at runtime, so it splices them with `#kind` (head) and `:#field`
/// (keyword) into an `egglog!` scaffold. The schema-aware parser then expands
/// the named args + `...` exactly as if they'd been written literally.
#[test]
fn quote_splices_runtime_kind_and_field() {
    let mut egraph = new_egraph();

    // Declare a named constructor + seed a value. Parsing the datatype on
    // `egraph.parser` registers MyCar's named-call macro (parse-time effect).
    let setup = egglog!(
        egraph.parser,
        (datatype Vehicle (MyCar :color i64 :numwheel i64))
        (relation SawColor (i64))
        (let c (MyCar :color 7 :numwheel 4))
    )
    .unwrap();
    egraph.run_program(setup).unwrap();

    // Runtime names — exactly what a per-op generator has in hand.
    let kind = "MyCar";
    let field = "color";
    let rule = egglog!(
        egraph.parser,
        (rule ((= ?e (#kind :#field ?c ...))) ((SawColor ?c)))
        (run 1)
        (check (SawColor 7))
    )
    .unwrap();
    egraph.run_program(rule).unwrap();
}

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

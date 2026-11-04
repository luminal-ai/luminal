//! Views of views through the composition closure, from torch's own
//! spelling: a permute undone by the permute back is the operand in the
//! e-graph, and a transpose reshaped (an arithmetic composite the e-graph
//! leaves closed) executes correctly on the reference runtime.
use super::test_support::*;
use luminal::egglog_snippet::{assembled_program_for, new_egraph};
use luminal_reference::{ReferenceRuntime, TypedBuffer, harness_search_options};
use rustc_hash::FxHashMap;

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

#[test]
fn a_permute_undone_by_the_permute_back_is_the_operand() {
    let t = translate_nodes(
        vec![
            node(
                "torch.ops.aten.permute.default",
                vec![tensor_input("self", "x"), ints("dims", &[1, 2, 0])],
                &["y"],
            ),
            node(
                "torch.ops.aten.permute.default",
                vec![tensor_input("self", "y"), ints("dims", &[2, 0, 1])],
                &["z"],
            ),
        ],
        &[
            ("x", FLOAT, &[2, 3, 4]),
            ("y", FLOAT, &[3, 4, 2]),
            ("z", FLOAT, &[2, 3, 4]),
        ],
        &["x"],
        &["z"],
    );
    let logical = &t.graph.logical;
    let x = logical.let_name(t.inputs[0].tensor);
    let z = logical.let_name(t.outputs[0].tensor);
    assert_ne!(x, z, "the translator records the two permutes as views");
    let program = format!(
        "{}\n{}\n{SCHEDULE}\n(check (= {z} {x}))\n",
        assembled_program_for(&[]),
        logical.render_all().unwrap()
    );
    new_egraph()
        .parse_and_run_program(None, &program)
        .expect("the permute round trip composes to the identity and is its operand");
}

#[test]
fn a_transpose_reshaped_executes_through_its_composite() {
    let t = translate_nodes(
        vec![
            node(
                "torch.ops.aten.transpose.int",
                vec![
                    tensor_input("self", "x"),
                    scalar_int("dim0", 0),
                    scalar_int("dim1", 1),
                ],
                &["y"],
            ),
            node(
                "torch.ops.aten.view.default",
                vec![tensor_input("self", "y"), ints("size", &[3, 8])],
                &["z"],
            ),
        ],
        &[
            ("x", FLOAT, &[2, 3, 4]),
            ("y", FLOAT, &[3, 2, 4]),
            ("z", FLOAT, &[3, 8]),
        ],
        &["x"],
        &["z"],
    );
    let x: Vec<f32> = (0..24).map(|i| i as f32).collect();
    let mut runtime = ReferenceRuntime::load(&t.graph).expect("load");
    let mut data: FxHashMap<_, TypedBuffer> = FxHashMap::default();
    data.insert(t.inputs[0].tensor, TypedBuffer::F32(x.clone()));
    runtime
        .search(
            &Default::default(),
            &Default::default(),
            &data,
            &harness_search_options(),
        )
        .expect("the transposed reshape plans");
    runtime.set_data(t.inputs[0].tensor, TypedBuffer::F32(x.clone()));
    runtime.execute().expect("the transposed reshape executes");
    let z = runtime.get_f32(t.outputs[0].tensor).expect("z");
    // z[j][i*4 + k] = y[j][i][k] = x[i][j][k]
    let expected: Vec<f32> = (0..3)
        .flat_map(|j| (0..2).flat_map(move |i| (0..4).map(move |k| x_at(i, j, k))))
        .collect();
    assert_eq!(z, &expected);
}

fn x_at(i: usize, j: usize, k: usize) -> f32 {
    (i * 12 + j * 4 + k) as f32
}

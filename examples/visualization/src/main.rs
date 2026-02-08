use std::fs;

use luminal::{
    self,
    egglog_utils::{egglog_to_llir, hlir_to_egglog, random_initial_choice},
    op::IntoEgglogOp,
    prelude::{
        egglog::var,
        egglog_ast::{RustSpan, Span},
        *,
    },
    visualization::{ToDot, ToHtml},
};
use luminal_cuda::runtime::CudaRuntime;

fn main() {
    // Create a new graph
    let mut cx = Graph::new();

    // Create input tensor using constant values

    let (m, n, k) = (4096, 14336, 9192);

    let a = cx.tensor((m, k));
    let b = cx.tensor((k, n));

    let _c = a.matmul(b);

    println!("Visualizing HLIR");
    fs::write("HLIR.dot", cx.graph.to_dot().unwrap()).unwrap();

    let (program, root) = hlir_to_egglog(&cx);

    type Ops = (<CudaRuntime as Runtime>::Ops, luminal::hlir::HLIROps);
    let ops = <Ops as IntoEgglogOp>::into_vec();

    // run e-graph saturation
    println!("Building and Saturating E-Graph");
    let mut egglog_obj = egglog::EGraph::default();
    let code = luminal::egglog_utils::full_egglog(&program, &ops, true);
    egglog_obj.parse_and_run_program(None, &code).unwrap();

    // EGraph Optimization Complete
    println!("Visualizing EGraph");
    // save the egraph visualizations
    fs::write("egraph.html", egglog_obj.to_html().unwrap()).unwrap();
    fs::write("egraph.dot", egglog_obj.to_dot().unwrap()).unwrap();

    let (sort, value) = egglog_obj.eval_expr(&var!(root)).unwrap();
    let s_egraph = SerializedEGraph::new(&egglog_obj, vec![(sort, value)]);
    let example_llir_graph = egglog_to_llir(
        &s_egraph,
        random_initial_choice(&s_egraph, &mut rand::rng()),
        &ops,
        &[],
        &mut FxHashMap::default(),
        &mut FxHashMap::default(),
        None,
    );

    println!("Visualizing LLIR Graph");
    fs::write("LLIR.dot", example_llir_graph.to_dot().unwrap()).unwrap();
}

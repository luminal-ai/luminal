//! Diagnostic (simplify-revert verification, 2026-08-27): print the
//! recorded model text for a symbolic repeat/merge/pad fixture so
//! folded-vs-raw dim/expr spellings can be inspected directly. Not part
//! of any gate.

use luminal::prelude::*;

fn main() {
    let mut cx = Graph::default();
    cx.set_dim('s', 4);
    cx.set_dim('t', 2);
    // rank-2 symbolic tensor: ('s', 't')
    let x = cx.named_tensor("x", ('s', 't'), DType::F32);
    // repeat: tiled extents s*2, t*3
    let r = x.repeat((2, 3));
    // merge the repeated dims: merged extent (s*2)*(t*3)
    let m = r.merge_dims(0, 1);
    let _ = m.output();
    // pad along axis 0 of a fresh symbolic tensor: out dim 1 + s + 2
    let y = cx.named_tensor("y", ('s',), DType::F32);
    let _ = y.pad_along(1, 2, 0, 0.0).output();
    match cx.logical.model_text() {
        Ok(model) => println!("{model}"),
        Err(e) => println!("POISONED: {e}"),
    }
}

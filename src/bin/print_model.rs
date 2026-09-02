//! Diagnostic (R-C verification, 2026-08-26): print the recorded model
//! text for one iota-heavy fixture so raw-vs-simplified spellings can be
//! inspected directly. Not part of any gate.

use luminal::prelude::*;

fn main() {
    let mut cx = Graph::new();
    let x = cx.tensor((2, 4, 8), DType::F32);
    let idx = cx.tensor((2, 4, 3), DType::Int);
    let _ = x.gather_elements(idx, 2).output();
    match cx.logical.model_text() {
        Ok(model) => println!("{model}"),
        Err(e) => println!("POISONED: {e}"),
    }
}

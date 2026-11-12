//! Small matrix multiplication on the reference runtime.

use anyhow::Result;
use luminal::prelude::*;
use luminal_reference::ReferenceRuntime;

fn main() -> Result<()> {
    let mut cx = Graph::new();
    let a = cx.tensor(vec![3, 1], DType::F32);
    let b = cx.tensor(vec![1, 4], DType::F32);
    let c = a.matmul(b);

    // Compile the graph. `runtime.search` instead autotunes it against sample inputs.
    let mut runtime = ReferenceRuntime::load(&cx)?;
    runtime.compile(&Default::default(), &Default::default())?;
    runtime.set_data(a.id, vec![1.0f32, 2.0, 3.0]);
    runtime.set_data(b.id, vec![1.0f32, 2.0, 3.0, 3.0]);
    runtime.execute()?;

    println!("Result: {:?}", runtime.get_f32(c.id)?);
    Ok(())
}

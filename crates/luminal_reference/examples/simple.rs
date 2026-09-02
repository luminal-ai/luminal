//! Small matrix multiplication on the reference runtime.

use anyhow::Result;
use luminal::prelude::*;
use luminal_reference::ReferenceRuntime;

fn main() -> Result<()> {
    let mut cx = Graph::new();
    let a = cx.tensor((3, 1), DType::F32);
    let b = cx.tensor((1, 4), DType::F32);
    let c = a.matmul(b).output();

    let pairs = vec![
        (a.id, vec![1.0f32, 2.0, 3.0].into()),
        (b.id, vec![1.0f32, 2.0, 3.0, 3.0].into()),
    ];
    let data = pairs.iter().cloned().collect();

    let mut runtime = ReferenceRuntime::load(&cx)?;
    runtime.search(&data, &luminal::test_support::harness_search_options())?;
    for (id, value) in pairs {
        runtime.set_data(id, value);
    }
    runtime.execute()?;

    println!("Result: {:?}", runtime.get_f32(c.id)?);
    Ok(())
}

use luminal::prelude::*;
fn main() {
    let mut cx = Graph::new();
    let x = cx.tensor((8, 16), DType::F32);
    let w = cx.tensor((16, 12), DType::F32);
    let _ = x.matmul(w);
    println!("{}", cx.logical.render_all().unwrap());
}

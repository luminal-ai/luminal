use luminal::prelude::*;
fn main() {
    let mut cx = Graph::new();
    let x = cx.tensor((2usize, 4usize), DType::F32);
    let w = cx.tensor((3usize, 4usize), DType::F32);
    let _ = x.matmul(w.permute((1, 0)));
    println!("{}", cx.logical.render_all().unwrap());
}

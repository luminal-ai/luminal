//! Shared plan layouts and dense interpretation of returned backing storage.

use anyhow::Result;
pub use luminal::layouts::DecodedLayout;

pub type MetalPlan = luminal::bufferize::BufferIrGraph<DecodedLayout>;

pub fn eval_term(expr: &luminal::layouts::IntExprTerm, coords: &[usize]) -> Result<i64> {
    expr.eval_at(coords)
}

pub fn element_index(layout: &DecodedLayout, coords: &[usize]) -> Result<usize> {
    layout.element_index(coords)
}

pub fn dense_f32(backing: &[f32], layout: &DecodedLayout) -> Result<Vec<f32>> {
    let dims = layout
        .literal_extents()
        .ok_or_else(|| anyhow::anyhow!("dense read: symbolic extents"))?;
    let numel: usize = dims.iter().product();
    if layout.has::<luminal::layouts::RightMajorContiguousElementLayout>() {
        anyhow::ensure!(
            numel <= backing.len(),
            "dense output exceeds backing storage"
        );
        return Ok(backing[..numel].to_vec());
    }
    let rank = dims.len();
    let mut coords = vec![0usize; rank];
    let mut out = Vec::with_capacity(numel);
    for _ in 0..numel {
        let flat = element_index(layout, &coords)?;
        anyhow::ensure!(
            flat < backing.len(),
            "element index {flat} exceeds the backing buffer ({} elements)",
            backing.len()
        );
        out.push(backing[flat]);
        for axis in (0..rank).rev() {
            coords[axis] += 1;
            if coords[axis] < dims[axis] {
                break;
            }
            coords[axis] = 0;
        }
    }
    Ok(out)
}

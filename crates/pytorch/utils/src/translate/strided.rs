//! Storage-addressed ATen operations expressed as coordinate Gather/Scatter.
//!
//! PT2 logical values are not flat storage: a transpose or a slice still
//! addresses its original backing. Keep that provenance until a copy creates
//! fresh storage. A boundary that exposes only a subview must first expose its
//! backing storage; never invent values for holes outside a logical input.
use super::{Translator, util::same_dim};
use crate::pt2_schema::Node;
use anyhow::{Context, Result, ensure};
use luminal::prelude::*;

#[derive(Clone, Copy)]
pub(super) struct Backing {
    flat: GraphTensor,
    origin: IntExpr,
}

impl Translator<'_> {
    fn storage_geometry(&self, name: &str) -> Result<(Vec<IntExpr>, IntExpr)> {
        if let Some(geometry) = self.storage_geometries.get(name) {
            return Ok(geometry.clone());
        }
        let meta = self.tensor_meta(name)?;
        let dims = self.tensor_meta_to_shape(meta)?;
        let strides = if let Some(strides) = &meta.strides {
            ensure!(
                strides.len() == dims.len(),
                "{name}: stride rank differs from shape rank"
            );
            strides
                .iter()
                .map(|s| self.dim_size_to_expr(s))
                .collect::<Result<Vec<_>>>()?
        } else {
            (0..dims.len())
                .map(|i| super::dim_arith::product_of_dims(dims[i + 1..].iter().copied()))
                .collect()
        };
        let offset = meta
            .storage_offset
            .as_ref()
            .map(|s| self.dim_size_to_expr(s))
            .transpose()?
            .unwrap_or(0.into());
        // Correct offsets of aliases following a scatter. Some PyTorch Fake
        // kernels reset scatter's output offset although native ATen retains
        // it; preserve the schema's geometry rather than that stale metadata.
        let parent = self.alias_parent(name);
        let offset = if let Some(parent) = parent {
            let (_, actual) = self.storage_geometry(&parent)?;
            let raw = self
                .tensor_meta(&parent)?
                .storage_offset
                .as_ref()
                .map(|s| self.dim_size_to_expr(s))
                .transpose()?
                .unwrap_or(0.into());
            offset - raw + actual
        } else {
            offset
        };
        Ok((strides, offset))
    }

    fn alias_parent(&self, name: &str) -> Option<String> {
        self.parsed
            .program
            .graph_module
            .graph
            .nodes
            .iter()
            .find(|n| n.outputs.iter().any(|o| o.value_name() == Some(name)))
            .filter(|n| {
                matches!(
                    n.target
                        .strip_prefix("torch.ops.aten.")
                        .unwrap_or(&n.target),
                    "view.default"
                        | "_unsafe_view.default"
                        | "_reshape_alias.default"
                        | "transpose.int"
                        | "t.default"
                        | "permute.default"
                        | "slice.Tensor"
                        | "select.int"
                        | "narrow.default"
                        | "squeeze.default"
                        | "squeeze.dim"
                        | "squeeze.dims"
                        | "unsqueeze.default"
                        | "expand.default"
                        | "detach.default"
                        | "alias.default"
                        | "as_strided.default"
                        | "diagonal.default"
                        | "unfold.default"
                )
            })
            .and_then(|n| n.inputs.first())
            .and_then(|i| i.arg.as_tensor_name())
            .map(str::to_owned)
    }

    fn backing(&mut self, name: &str) -> Result<Backing> {
        if let Some(backing) = self.storage_values.get(name) {
            return Ok(*backing);
        }
        // Only schema-guaranteed aliases share backing. In particular reshape
        // and contiguous may copy; export functionalizes them to explicit views
        // and clones before storage-addressed operations reach this translator.
        let parent = self.alias_parent(name);
        let backing = if let Some(parent) = parent {
            self.backing(&parent)?
        } else {
            let value = *self
                .values
                .get(name)
                .with_context(|| format!("unknown storage value {name}"))?;
            let dims = value.dims();
            let (strides, origin) = self.storage_geometry(name)?;
            let numel = super::dim_arith::product_of_dims(dims.iter().copied());
            // A dense root can be converted to storage order by inverse physical
            // coordinates. Dimensions of size one impose no stride constraint.
            let mut remaining: Vec<_> = (0..dims.len())
                .filter(|i| dims[*i].to_usize() != Some(1))
                .collect();
            let mut expected = IntExpr::from(1);
            if numel.to_usize() != Some(0) {
                while !remaining.is_empty() {
                    let next = remaining
                        .iter()
                        .position(|i| same_dim(strides[*i], expected));
                    let Some(next) = next else {
                        anyhow::bail!(
                            "{name}: storage-addressed operation needs the full backing storage of a non-dense input; expose its backing at the input boundary"
                        );
                    };
                    let axis = remaining.remove(next);
                    expected *= dims[axis];
                }
            }
            let row_major = (0..dims.len()).all(|i| {
                dims[i].to_usize() == Some(1)
                    || same_dim(
                        strides[i],
                        super::dim_arith::product_of_dims(dims[i + 1..].iter().copied()),
                    )
            });
            let flat = if row_major || numel.to_usize() == Some(0) || dims.is_empty() {
                value.flatten()
            } else {
                let coordinates: Vec<_> = (0..dims.len())
                    .map(|i| {
                        if dims[i].to_usize() == Some(1) {
                            self.cx.iota(vec![numel], |_| 0.into())
                        } else {
                            self.cx.iota(vec![numel], |c| (c[0] / strides[i]) % dims[i])
                        }
                    })
                    .collect();
                value.gather(&coordinates)
            };
            Backing { flat, origin }
        };
        self.storage_values.insert(name.to_owned(), backing);
        Ok(backing)
    }

    fn strided_arguments(
        &self,
        node: &Node,
        size_index: usize,
        default_offset: IntExpr,
    ) -> Result<(Vec<IntExpr>, Vec<IntExpr>, IntExpr)> {
        let sizes = self.get_int_exprs_arg(node, size_index)?;
        let strides = self.get_int_exprs_arg(node, size_index + 1)?;
        ensure!(
            sizes.len() == strides.len(),
            "as_strided size and stride must have equal length"
        );
        ensure!(
            sizes
                .iter()
                .chain(&strides)
                .all(|v| v.as_num().is_none_or(|n| n >= 0)),
            "as_strided requires nonnegative sizes and strides"
        );
        let offset = match node.inputs.get(size_index + 2) {
            None => default_offset,
            Some(i) if matches!(&i.arg, crate::pt2_schema::Argument::Other(v) if v.is_null() || v.get("as_none").and_then(|v| v.as_bool()) == Some(true)) => {
                default_offset
            }
            Some(i) => self
                .resolve_arg_as_expression(&i.arg)
                .context("unresolved as_strided storage_offset")?,
        };
        ensure!(
            offset.as_num().is_none_or(|n| n >= 0),
            "as_strided requires nonnegative storage_offset"
        );
        Ok((sizes, strides, offset))
    }

    fn storage_indices(
        &mut self,
        backing: Backing,
        sizes: &[IntExpr],
        strides: &[IntExpr],
        offset: IntExpr,
    ) -> Result<GraphTensor> {
        let offset = offset - backing.origin;
        if !sizes.iter().any(|s| s.to_usize() == Some(0)) {
            let last = sizes
                .iter()
                .zip(strides)
                .fold(offset, |a, (n, s)| a + (*n - 1) * *s);
            if let (Some(first), Some(last), Some(span)) = (
                offset.simplify().as_num(),
                last.simplify().as_num(),
                backing.flat.dims()[0].simplify().as_num(),
            ) {
                ensure!(
                    first >= 0 && last < span,
                    "as_strided addresses storage outside the represented backing [{first}, {last}] vs {span} elements; expose the full backing at the input boundary"
                );
            }
        }
        Ok(self.cx.iota(sizes.to_vec(), |c| {
            c.iter().zip(strides).fold(offset, |a, (c, s)| a + *c * *s)
        }))
    }

    pub(super) fn translate_as_strided(&mut self, node: &Node) -> Result<GraphTensor> {
        let name = node.inputs[0]
            .arg
            .as_tensor_name()
            .context("as_strided input must be a tensor")?;
        let (_, offset) = self.storage_geometry(name)?;
        let backing = self.backing(name)?;
        let (sizes, strides, offset) = self.strided_arguments(node, 1, offset)?;
        let indices = self.storage_indices(backing, &sizes, &strides, offset)?;
        let value = backing.flat.gather(&[indices]);
        if node.target.ends_with(".as_strided.default") {
            for output in Self::tensor_output_names(node) {
                self.storage_geometries
                    .insert(output.clone(), (strides.clone(), offset));
                self.storage_values.insert(output, backing);
            }
        }
        Ok(value)
    }

    pub(super) fn translate_as_strided_scatter(&mut self, node: &Node) -> Result<GraphTensor> {
        let name = node.inputs[0]
            .arg
            .as_tensor_name()
            .context("as_strided_scatter input must be a tensor")?;
        let value = self.operand(&node.inputs[0])?;
        let source = self.operand(&node.inputs[1])?;
        let (self_strides, self_offset) = self.storage_geometry(name)?;
        let backing = self.backing(name)?;
        let (sizes, strides, offset) = self.strided_arguments(node, 2, self_offset)?;
        ensure!(
            source.dims().len() == sizes.len()
                && source
                    .dims()
                    .iter()
                    .zip(&sizes)
                    .all(|(a, b)| same_dim(*a, *b)),
            "as_strided_scatter source must have the requested size"
        );
        let indices = self.storage_indices(backing, &sizes, &strides, offset)?;
        let source = super::convert(source, value.dtype);
        let flat = if sizes.iter().any(|s| s.to_usize() == Some(0)) {
            backing.flat
        } else {
            backing.flat.scatter(&[indices], source)
        };
        let updated = Backing { flat, ..backing };
        let indices = self.storage_indices(updated, &value.dims(), &self_strides, self_offset)?;
        let out = flat.gather(&[indices]);
        // The fresh result retains the input's storage geometry, including
        // writes outside its visible view that a later as_strided may read.
        for output in Self::tensor_output_names(node) {
            self.storage_geometries
                .insert(output.clone(), (self_strides.clone(), self_offset));
            self.storage_values.insert(output, updated);
        }
        Ok(out)
    }
}

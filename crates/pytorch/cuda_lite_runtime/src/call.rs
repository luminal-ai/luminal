//! What the CUDA runtime reads off one call's tensors, and the two things it
//! refuses about them.
//!
//! The runtime binds a caller's tensor in place: it is given an address and
//! a byte count and reads the elements through the layout the program was
//! compiled for. A tensor's type, dtype, device, shape and strides are
//! checked by Dynamo's guards before a compiled graph is called, so nothing
//! here checks them again. Two facts Dynamo does not guard are checked
//! here: a tensor's storage offset, and two tensors sharing storage while
//! one is written back into.
//!
//! Everything in this module is a pure function of the code object's
//! tables, what the slot readers read off a tensor, and the call's
//! dimensions.

use anyhow::{Result, anyhow, bail};
use luminal::prelude::{DynMap, IntExpr, Symbol};
use luminal_torch_slots::TensorInfo;

use crate::code::{Input, Strides};

/// The dimension an extent IS, when it is exactly one symbol. A compound
/// extent (`2*s0`) is a statement about dimensions and is never inverted
/// into one.
fn bare(extent: &IntExpr) -> Option<Symbol> {
    let symbols = extent.to_symbols();
    (symbols.len() == 1 && *extent == IntExpr::from(symbols[0])).then(|| symbols[0])
}

/// The names of the dimensions `expr` still needs once `dims` is applied.
fn unstated(expr: &IntExpr, dims: &DynMap) -> String {
    let mut names: Vec<String> = expr
        .to_symbols()
        .into_iter()
        .filter(|symbol| !dims.contains_key(symbol))
        .map(|symbol| symbol.to_string())
        .collect();
    names.sort();
    names.dedup();
    names.join(", ")
}

/// The value one call gives each dynamic dimension.
///
/// An integer argument that is the dimension states it, and so does every
/// axis of a bound tensor whose extent is that dimension alone. `integers`
/// are `(the dimension an argument defines, its value)`. The first statement
/// of a dimension is the one read: that every statement of it agrees is what
/// Dynamo's shape guards check.
pub fn state_dimensions(integers: &[(Symbol, usize)], tensors: &[(&Input, &TensorInfo)]) -> DynMap {
    let mut values = DynMap::default();
    for (symbol, value) in integers {
        values.entry(*symbol).or_insert(*value);
    }
    for (input, info) in tensors {
        for (extent, size) in input.shape.iter().zip(&info.shape) {
            if let Some(symbol) = bare(extent) {
                // A size is never negative; the conversion cannot fail.
                values.entry(symbol).or_insert(*size as usize);
            }
        }
    }
    values
}

/// Each extent at the call's dimensions.
pub fn resolve(name: &str, shape: &[IntExpr], dims: &DynMap) -> Result<Vec<i64>> {
    shape
        .iter()
        .map(|extent| {
            extent
                .exec(dims)
                .or_else(|| extent.to_usize())
                .map(|size| size as i64)
                .ok_or_else(|| {
                    anyhow!("{name}: extent {extent} names a dimension no call has stated")
                })
        })
        .collect()
}

/// The element strides a row is returned at, at this shape and these
/// dimensions: what an output is allocated with and what a view is made at.
///
/// A stride that still names a dimension once the call's dimensions are
/// applied is refused by name.
pub fn strides_at(name: &str, strides: &Strides, shape: &[i64], dims: &DynMap) -> Result<Vec<i64>> {
    match strides {
        Strides::Elements(strides) => {
            if strides.len() != shape.len() {
                bail!(
                    "{name}: compiled with {} element strides, at rank {}",
                    strides.len(),
                    shape.len()
                );
            }
            strides
                .iter()
                .enumerate()
                .map(|(axis, stride)| {
                    let value = stride.exec(dims).ok_or_else(|| {
                        anyhow!(
                            "{name}: element stride {stride} on axis {axis} still names {} once \
                             this call's dimensions are substituted; no argument of this call \
                             gives that dimension an extent",
                            unstated(stride, dims)
                        )
                    })?;
                    Ok(value as i64)
                })
                .collect()
        }
        // torch's own contiguous strides: the running product multiplies by
        // `max(size, 1)`, so a zero extent leaves the strides outside it as
        // a full tensor's.
        Strides::RowMajor => {
            let mut strides = vec![1i64; shape.len()];
            for axis in (0..shape.len().saturating_sub(1)).rev() {
                strides[axis] = strides[axis + 1] * shape[axis + 1].max(1);
            }
            Ok(strides)
        }
        Strides::ColumnMajor => {
            let mut strides = vec![1i64; shape.len()];
            for axis in 1..shape.len() {
                strides[axis] = strides[axis - 1] * shape[axis - 1].max(1);
            }
            Ok(strides)
        }
    }
}

/// Refuse a tensor that does not start at its storage's base. A binding
/// names a buffer's base, and two bindings on one buffer share that base, so
/// a view at an offset has no binding. Dynamo guards a tensor's shape and
/// strides, not where in its storage it starts.
pub fn require_base(name: &str, info: &TensorInfo) -> Result<()> {
    if info.storage_offset != 0 {
        bail!(
            "{name}: storage offset {} is not bound today; a binding names a buffer's base, and \
             two bindings on one buffer share that base",
            info.storage_offset
        );
    }
    Ok(())
}

/// Bytes the bound buffer spans, reachable from the tensor's first element:
/// the last element the strides reach, plus one. A bound tensor has storage
/// offset zero, so its address is the base of that span.
pub fn span_bytes(info: &TensorInfo) -> usize {
    if info.shape.contains(&0) {
        return 0;
    }
    let last: i64 = info
        .shape
        .iter()
        .zip(&info.strides)
        .map(|(size, stride)| (size - 1) * stride)
        .sum();
    (1 + last) as usize * info.dtype.element_size()
}

/// Two bound tensors whose device storage overlaps are refused when either
/// one is written through.
///
/// Read-only aliasing (tied weights, a tensor passed twice) is two buffers
/// carrying one address, a fact about the caller's memory and harmless. A
/// writeback is not: the program knows one buffer per binding, so a write
/// through one of an overlapping pair would land in the other's reads with
/// no edge ordering them.
///
/// Checked on every call: Dynamo guards tensor identity, not storage
/// overlap, so a program compiled on distinct tensors can be called as
/// `fn(x, x[:])`.
pub fn refuse_overlapping_writebacks(tensors: &[(&Input, &TensorInfo)]) -> Result<()> {
    if !tensors.iter().any(|(input, _)| input.written) {
        return Ok(());
    }
    let spans: Vec<(&Input, usize, usize)> = tensors
        .iter()
        .map(|(input, info)| (*input, info.data, info.data + span_bytes(info)))
        .collect();
    for (index, (input, start, stop)) in spans.iter().enumerate() {
        for (other, other_start, other_stop) in &spans[index + 1..] {
            if start >= other_stop || other_start >= stop {
                continue;
            }
            let Some(target) = [input, other].into_iter().find(|row| row.written) else {
                continue;
            };
            bail!(
                "'{}' and '{}' share device storage and '{}' is written back into; one buffer \
                 per binding cannot order a write against the other's reads",
                input.name,
                other.name,
                target.name
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use luminal_torch_slots::{Device, DeviceType, TorchDtype};

    fn input(name: &str, shape: Vec<IntExpr>, written: bool) -> Input {
        Input {
            name: name.to_string(),
            buffer: 0,
            shape,
            written,
            constant: None,
        }
    }

    fn literal(values: &[i64]) -> Vec<IntExpr> {
        values.iter().map(|value| IntExpr::from(*value)).collect()
    }

    fn tensor(shape: &[i64], strides: &[i64]) -> TensorInfo {
        TensorInfo {
            data: 0x1000,
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            dtype: TorchDtype::Float32,
            device: Device {
                kind: DeviceType::Cuda,
                index: 0,
            },
            storage_offset: 0,
        }
    }

    /// A dimension is read off the first argument that states it: an integer
    /// argument that is the dimension, else a tensor axis compiled as it.
    /// A compound extent states nothing.
    #[test]
    fn a_dimension_is_read_from_the_first_argument_that_states_it() {
        let s0 = Symbol::new("s0");
        let s1 = Symbol::new("s1");
        let x = input(
            "x",
            vec![IntExpr::from(s0), IntExpr::from(s1) * IntExpr::from(2i64)],
            false,
        );
        let value = tensor(&[3, 8], &[8, 1]);
        let stated = state_dimensions(&[], &[(&x, &value)]);
        assert_eq!(stated.get(&s0), Some(&3));
        assert_eq!(
            stated.get(&s1),
            None,
            "2*s1 is a statement about s1, not s1"
        );
        let stated = state_dimensions(&[(s0, 7)], &[(&x, &value)]);
        assert_eq!(
            stated.get(&s0),
            Some(&7),
            "the integer argument comes first"
        );
    }

    /// Dynamo guards shapes, not storage offsets, so a program compiled on a
    /// tensor at its storage's base can be called with one that is not.
    #[test]
    fn a_storage_offset_is_refused() {
        let mut value = tensor(&[4, 4], &[4, 1]);
        require_base("x", &value).unwrap();
        value.storage_offset = 8;
        let message = require_base("x", &value).unwrap_err().to_string();
        assert!(message.contains("x: storage offset 8"), "{message}");
    }

    /// An output is allocated at the strides it was compiled for, at the
    /// call's dimensions. A stride that names a dimension no argument states
    /// is refused by name.
    #[test]
    fn strides_are_evaluated_at_the_calls_dimensions() {
        let s0 = Symbol::new("s0");
        let strides = Strides::Elements(vec![IntExpr::from(s0), IntExpr::from(1i64)]);
        let stated: DynMap = [(s0, 3)].into_iter().collect();
        assert_eq!(
            strides_at("out", &strides, &[4, 3], &stated).unwrap(),
            [3, 1]
        );
        let message = strides_at("out", &strides, &[4, 3], &DynMap::default())
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("s0") && message.contains("element stride"),
            "{message}"
        );
        let message = strides_at("out", &strides, &[4], &stated)
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("2 element strides, at rank 1"),
            "{message}"
        );
    }

    /// The contiguous layouts state torch's own strides, a zero extent
    /// included.
    #[test]
    fn the_contiguous_layouts_state_torchs_own_strides() {
        let none = DynMap::default();
        let at = |strides: &Strides, shape: &[i64]| strides_at("out", strides, shape, &none);
        assert_eq!(at(&Strides::RowMajor, &[2, 3, 4]).unwrap(), [12, 4, 1]);
        assert_eq!(at(&Strides::RowMajor, &[2, 0, 3]).unwrap(), [3, 3, 1]);
        assert_eq!(at(&Strides::ColumnMajor, &[2, 3, 4]).unwrap(), [1, 2, 6]);
    }

    /// The span is what the strides reach from the first element, not the
    /// element count: a transposed view reaches further than it holds.
    #[test]
    fn the_span_is_what_the_strides_reach() {
        assert_eq!(span_bytes(&tensor(&[3, 4], &[4, 1])), 12 * 4);
        assert_eq!(span_bytes(&tensor(&[2, 4], &[1, 3])), (1 + 1 + 9) * 4);
        assert_eq!(span_bytes(&tensor(&[0, 4], &[4, 1])), 0);
        assert_eq!(span_bytes(&tensor(&[], &[])), 4);
    }

    /// Overlap is refused only when one of the pair is written back into.
    #[test]
    fn overlapping_storage_is_refused_only_under_a_writeback() {
        let first = tensor(&[4], &[1]);
        let mut second = tensor(&[4], &[1]);
        second.data = first.data + 8; // two elements in

        let (a, b) = (
            input("a", literal(&[4]), false),
            input("b", literal(&[4]), false),
        );
        refuse_overlapping_writebacks(&[(&a, &first), (&b, &second)]).unwrap();

        let written = input("b", literal(&[4]), true);
        let message = refuse_overlapping_writebacks(&[(&a, &first), (&written, &second)])
            .unwrap_err()
            .to_string();
        assert!(
            message.contains("'a' and 'b'") && message.contains("'b' is written back"),
            "{message}"
        );
        // Disjoint storage is never refused.
        second.data = first.data + 16;
        refuse_overlapping_writebacks(&[(&a, &first), (&written, &second)]).unwrap();
    }
}

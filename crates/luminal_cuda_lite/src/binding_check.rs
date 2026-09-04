//! CONTRACT-1 (M4 Phase 2 companion, enforced at Phase 5): distinct
//! BufferIds must be backed by DISJOINT device pointer ranges. The
//! composed-access machinery redirects folded-view readers into their
//! parent's storage by BufferId identity; two distinct BufferIds
//! silently sharing bytes would let a WAR-ordered writer clobber a
//! folded reader's parent without any Anti edge ever having been
//! minted. Today the device executor allocates one fresh `CudaSlice`
//! per BufferId, so non-overlap holds by construction — this module is
//! the contract's enforcement face for when RAW CALLER POINTERS arrive
//! at the binding surface: the executor calls [`assert_disjoint`] on
//! every (buffer, base, len) it binds and refuses loudly on violation,
//! never mistranslating.
//!
//! Deliberately device-free (no cudarc types): pointer ranges are plain
//! integers so the checker itself is unit-testable on any host.

use anyhow::{Result, bail};

/// One bound storage range: a buffer's display name, its base address,
/// and its extent in bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundRange {
    pub buffer: String,
    pub base: u64,
    pub bytes: u64,
}

/// Refuse unless every pair of ranges (across DISTINCT BufferIds — the
/// caller passes one entry per buffer) is disjoint. Zero-length ranges
/// occupy no bytes and cannot overlap anything. O(n log n): sort by
/// base, check each neighbor.
pub fn assert_disjoint(ranges: &[BoundRange]) -> Result<()> {
    let mut sorted: Vec<&BoundRange> = ranges.iter().filter(|range| range.bytes > 0).collect();
    sorted.sort_by_key(|range| range.base);
    for pair in sorted.windows(2) {
        let (lo, hi) = (pair[0], pair[1]);
        let lo_end = lo
            .base
            .checked_add(lo.bytes)
            .unwrap_or_else(|| panic!("buffer {} range overflows the address space", lo.buffer));
        if lo_end > hi.base {
            bail!(
                "CONTRACT-1 violation: buffers {} [{:#x}, {:#x}) and {} \
                 [{:#x}, {:#x}) overlap — distinct BufferIds must be \
                 backed by disjoint storage (folded-view reads and WAR \
                 ordering are both keyed on BufferId identity)",
                lo.buffer,
                lo.base,
                lo_end,
                hi.buffer,
                hi.base,
                hi.base + hi.bytes,
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{BoundRange, assert_disjoint};

    fn range(buffer: &str, base: u64, bytes: u64) -> BoundRange {
        BoundRange {
            buffer: buffer.to_string(),
            base,
            bytes,
        }
    }

    #[test]
    fn disjoint_ranges_pass() {
        assert_disjoint(&[
            range("a", 0x1000, 64),
            range("b", 0x1040, 64),
            range("c", 0x2000, 8),
        ])
        .expect("disjoint ranges are legal");
    }

    #[test]
    fn overlapping_ranges_refuse_loudly_naming_both() {
        let err = assert_disjoint(&[
            range("weights", 0x1000, 64),
            range("activations", 0x1020, 64),
        ])
        .expect_err("overlap must refuse");
        let msg = err.to_string();
        assert!(msg.contains("CONTRACT-1"), "{msg}");
        assert!(
            msg.contains("weights") && msg.contains("activations"),
            "{msg}"
        );
    }

    #[test]
    fn identical_bases_refuse() {
        assert_disjoint(&[range("a", 0x1000, 4), range("b", 0x1000, 4)])
            .expect_err("aliased bases must refuse");
    }

    #[test]
    fn touching_ranges_are_disjoint() {
        // [0x1000, 0x1040) then [0x1040, ...): end-exclusive, no shared byte.
        assert_disjoint(&[range("a", 0x1000, 64), range("b", 0x1040, 64)])
            .expect("end-exclusive adjacency shares no byte");
    }

    #[test]
    fn zero_length_ranges_cannot_overlap() {
        assert_disjoint(&[range("empty", 0x1000, 0), range("a", 0x1000, 64)])
            .expect("a zero-byte range occupies nothing");
    }
}

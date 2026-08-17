//! Post-extraction mask-event accounting.
//!
//! Luminal's contract is legality by construction: every rewrite may only add
//! e-nodes that are unconditionally legal and equivalent, so the checks that
//! reject or repair candidates *after* extraction must never fire. Each such
//! check records an event here. Under the contract every counter reads zero;
//! nonzero counts are the legality burn-down list.
//!
//! Counters are always on (relaxed atomics — nanoseconds on cold rejection
//! paths). Per-event details are printed when `LUMINAL_MASK_LOG=1`. A summary
//! of nonzero counters is printed at the end of every search.

use std::sync::atomic::{AtomicU64, Ordering};

pub struct MaskEvent {
    pub name: &'static str,
    count: AtomicU64,
}

impl MaskEvent {
    pub const fn new(name: &'static str) -> Self {
        Self {
            name,
            count: AtomicU64::new(0),
        }
    }

    pub fn record(&self) {
        self.count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record and, under `LUMINAL_MASK_LOG=1`, print the event detail.
    pub fn record_with(&self, detail: impl FnOnce() -> String) {
        self.record();
        if mask_log_enabled() {
            eprintln!("[mask:{}] {}", self.name, detail());
        }
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    fn reset(&self) {
        self.count.store(0, Ordering::Relaxed);
    }
}

fn mask_log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("LUMINAL_MASK_LOG").is_some())
}

/// Semantic rejections — must become zero for the corresponding check to be
/// deletable.
pub static ALIAS_HAZARD_REJECT: MaskEvent = MaskEvent::new("alias-hazard-reject");
pub static CYCLIC_LLIR_REJECT: MaskEvent = MaskEvent::new("cyclic-llir-reject");
/// Any panic caught while building/filtering/profiling a candidate (arity
/// asserts, staticness expects, FlashInfer gather recovery, NVRTC failures).
/// The detail carries the panic payload for attribution.
pub static CANDIDATE_PANIC: MaskEvent = MaskEvent::new("candidate-panic");
/// Genome-coherence repairs — evidence of extraction-incoherent unions.
pub static CHOICE_CYCLE_REPAIR: MaskEvent = MaskEvent::new("choice-cycle-repair");
pub static OPKIND_INCONSISTENT: MaskEvent = MaskEvent::new("opkind-inconsistent-variant");
pub static MARKER_CHOICE_RESTRICTED: MaskEvent = MaskEvent::new("marker-choice-restricted");
pub static INVARIANT_MARKER_INLINED: MaskEvent = MaskEvent::new("invariant-marker-inlined");
/// Non-panic finalization rejections (detail disambiguates resource vs
/// semantic reasons).
pub static FINALIST_REJECT: MaskEvent = MaskEvent::new("finalist-reject");
pub static AGGREGATE_REJECT: MaskEvent = MaskEvent::new("aggregate-reject");
/// Resource-cap rejections. Allowed to fire under the contract (fail-stop),
/// tracked for completeness.
pub static RESOURCE_REJECT: MaskEvent = MaskEvent::new("resource-reject");

fn all() -> [&'static MaskEvent; 10] {
    [
        &ALIAS_HAZARD_REJECT,
        &CYCLIC_LLIR_REJECT,
        &CANDIDATE_PANIC,
        &CHOICE_CYCLE_REPAIR,
        &OPKIND_INCONSISTENT,
        &MARKER_CHOICE_RESTRICTED,
        &INVARIANT_MARKER_INLINED,
        &FINALIST_REJECT,
        &AGGREGATE_REJECT,
        &RESOURCE_REJECT,
    ]
}

/// Nonzero counters as `(name, count)` pairs.
pub fn nonzero_counts() -> Vec<(&'static str, u64)> {
    all()
        .into_iter()
        .filter(|event| event.count() > 0)
        .map(|event| (event.name, event.count()))
        .collect()
}

/// Human-readable summary of nonzero counters, if any.
pub fn report() -> Option<String> {
    let counts = nonzero_counts();
    if counts.is_empty() {
        return None;
    }
    let mut out = String::from("mask events (legality-by-construction burn-down):");
    for (name, count) in counts {
        out.push_str(&format!("\n  {name:<32} {count}"));
    }
    Some(out)
}

/// Reset all counters (tests and multi-model harnesses).
pub fn reset() {
    for event in all() {
        event.reset();
    }
}

/// Best-effort string form of a caught panic payload for attribution.
pub fn panic_payload(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

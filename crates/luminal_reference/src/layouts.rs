//! The reference runtime's PLAN-LAYOUT vocabulary.
//!
//! There is no reference-flavored layout type and no reference-flavored
//! decoder any more (ruling D9, 2026-09-03): core owns the decoder and
//! publishes the struct it produces, and the runtimes import it
//! directly. What survives here is the crate-local name for this
//! runtime's bufferized plan.
//!
//! Core's bufferizer stays generic over an opaque layout type it only
//! clones and transports; THIS runtime instantiates it with core's
//! [`luminal::layouts::DecodedLayout`] — the shared mirror vocabulary
//! plus the value's `dtype-of` fact. The dtype is what makes plans
//! self-contained for `load_plan` callers: staging, allocation, and
//! readback all read the carried layout instead of a table an external
//! caller never had.

/// The reference runtime's bufferized plan.
pub type ReferencePlan = luminal::bufferize::BufferIrGraph<luminal::layouts::DecodedLayout>;

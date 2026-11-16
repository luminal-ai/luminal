//! What every PyTorch backend's code object states the same way.
//!
//! A code object is one compiled graph as plain data: everything a call
//! needs and nothing the compiler needed. Part of it is the same for every
//! backend, because it describes torch's side of a call and not the
//! backend's: what kind of value each argument slot holds and what a call
//! does with it, where the value of each output leaf comes from, the
//! conditions torch asserted, and the value each dimension starts at. That
//! part is [`Boundary`], and it lives here once.
//!
//! What a backend reads and writes, and what it runs, is its own: its
//! inputs, its rows and its program stay in its runtime crate, beside a
//! `Boundary`.
//!
//! - [`boundary`]: the tables, and what must hold for them to index one
//!   another.
//! - [`bytes`]: reading a code object from its byte form, version first.
//! - [`call`] (with the `python` feature): the steps of a call that read
//!   only the tables: the arguments, slot by slot, and the output leaves.
//! - [`serialized`] (with the `python` feature): one call at a time into
//!   what a compiled graph holds, from whichever thread.

pub mod boundary;
pub mod bytes;
#[cfg(feature = "python")]
pub mod call;
#[cfg(feature = "python")]
pub mod serialized;

pub use boundary::{Boundary, Check, Leaf, Literal, SlotAction, Source};
pub use bytes::hex;
#[cfg(feature = "python")]
pub use call::{Arguments, TensorArgument, to_py, type_name};
#[cfg(feature = "python")]
pub use serialized::Serialized;

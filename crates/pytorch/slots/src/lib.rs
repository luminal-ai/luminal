//! The boundary every PyTorch backend shares with torch.
//!
//! torch calls a compiled graph with one argument per placeholder, in order.
//! Each of those positions is a *slot*. This crate holds the two things all
//! backends agree on about slots, and nothing else:
//!
//! * [`SlotKind`]: what kind of value a slot holds. It is plain data and
//!   serializes, so a backend can store one kind per slot in its saved
//!   program.
//! * With the `python` feature: functions that read a value out of a slot
//!   ([`read_tensor`], [`read_int`], [`read_float`], [`read_bool`]) and that
//!   create tensors torch owns ([`empty_strided`], [`view_of`]).
//!
//! What a backend then *does* with a slot's value (bind a device pointer,
//! copy bytes, bind a dimension) is the backend's own business and lives in
//! the backend.
//!
//! # How tensors are reached
//!
//! The tensor functions go through torch's stable C entry points (the
//! `aoti_torch_*` shims and, from torch 2.14, `torch_tensor_from_pyobject`).
//! They are found by name, at first use, in the libtorch that the running
//! torch already loaded. Nothing here is compiled against torch's headers,
//! so one build of a backend works with every torch that exports them.

mod kind;

pub use kind::SlotKind;

#[cfg(feature = "python")]
pub mod debug;
#[cfg(feature = "python")]
mod scalar;
#[cfg(feature = "python")]
mod tensor;
#[cfg(feature = "python")]
mod torch_api;

#[cfg(feature = "python")]
pub use scalar::{read_bool, read_float, read_int};
#[cfg(feature = "python")]
pub use tensor::{
    Device, DeviceType, TensorInfo, TorchDtype, current_cuda_stream, empty_strided, read_tensor,
    view_of,
};

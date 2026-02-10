#![allow(unused)]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream};
use luminal::prelude::*;
use tracing_perfetto_sdk_schema::{
    self as schema, TrackEvent, debug_annotation::NameField, trace_packet, track_event,
};
use uuid::Uuid;

pub mod cuda_graph;
pub mod hlir;
pub mod other_ops;

pub use cuda_graph::*;

pub type Ops = (hlir::Ops, other_ops::Ops);

/// Build a mapping from interned string IDs to their string values for a given sequence.
fn build_interned_strings(trace: &schema::Trace) -> std::collections::HashMap<(u32, u64), String> {
    use tracing_perfetto_sdk_schema::trace_packet;
    let mut interned: std::collections::HashMap<(u32, u64), String> =
        std::collections::HashMap::new();
    for packet in &trace.packet {
        let seq_id = match &packet.optional_trusted_packet_sequence_id {
            Some(trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(seq)) => {
                *seq
            }
            _ => 0,
        };
        // interned_data is a field on TracePacket, not a Data variant
        if let Some(data) = &packet.interned_data {
            for entry in &data.debug_annotation_names {
                if let Some(name) = &entry.name {
                    interned.insert((seq_id, entry.iid()), name.clone());
                }
            }
        }
    }
    interned
}

/// Check if a debug annotation has key "id" and the given UUID value.
fn annotation_matches_id(
    a: &schema::DebugAnnotation,
    id: &Uuid,
    interned: &std::collections::HashMap<(u32, u64), String>,
    seq_id: u32,
) -> bool {
    let key_matches = match &a.name_field {
        Some(NameField::Name(k)) => k == "id",
        Some(NameField::NameIid(iid)) => interned
            .get(&(seq_id, *iid))
            .map(|s| s == "id")
            .unwrap_or(false),
        None => false,
    };
    if !key_matches {
        return false;
    }
    match &a.value {
        Some(tracing_perfetto_sdk_schema::debug_annotation::Value::StringValue(v)) => {
            *v == format!("{id}")
        }
        _ => false,
    }
}

/// Record CUDA graph kernel timings as nested slices in perfetto trace
pub fn record_cuda_graph_timings(
    trace: &schema::Trace,
    cuda_graph_timings: &[(CudaGraphTiming, Uuid)],
) -> Vec<schema::TracePacket> {
    use tracing_perfetto_sdk_schema::{trace_packet, track_descriptor};

    // Build interned string lookup table
    let interned = build_interned_strings(trace);

    let mut packets = Vec::new();
    for (graph_timing, id) in cuda_graph_timings {
        let parent_info = trace.packet.iter().find_map(|p| {
            let seq_id = match &p.optional_trusted_packet_sequence_id {
                Some(trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                    seq,
                )) => *seq,
                _ => 0,
            };
            match &p.data {
                Some(trace_packet::Data::TrackEvent(TrackEvent {
                    r#type: ty,
                    track_uuid,
                    debug_annotations,
                    ..
                })) if *ty == Some(track_event::Type::SliceBegin as i32)
                    && debug_annotations
                        .iter()
                        .any(|a| annotation_matches_id(a, id, &interned, seq_id)) =>
                {
                    Some((p.timestamp?, p.timestamp_clock_id?, (*track_uuid)?, seq_id))
                }
                _ => None,
            }
        });
        let Some((span_start_time, clock_id, track_uuid, sequence_id)) = parent_info else {
            continue;
        };
        // Use span_start_time + setup_duration + launch_latency as the base for kernel timings.
        // - setup_duration_ns: time spent on host between span entry and launch call
        // - launch_latency_ns: GPU-side time from launch to first kernel execution
        // This ensures kernel spans are accurately positioned within the cuda_graph span.
        let base_time =
            span_start_time + graph_timing.setup_duration_ns + graph_timing.launch_latency_ns;
        for kernel_timing in &graph_timing.kernel_timings {
            packets.push(schema::TracePacket {
                timestamp: Some(base_time + kernel_timing.start_ns),
                timestamp_clock_id: Some(clock_id),
                optional_trusted_packet_sequence_id: Some(
                    trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                        sequence_id,
                    ),
                ),
                data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                    track_uuid: Some(track_uuid),
                    r#type: Some(track_event::Type::SliceBegin as i32),
                    name_field: Some(track_event::NameField::Name(
                        kernel_timing.kernel_name.to_owned(),
                    )),
                    ..Default::default()
                })),
                ..Default::default()
            });
            packets.push(schema::TracePacket {
                timestamp: Some(base_time + kernel_timing.end_ns),
                timestamp_clock_id: Some(clock_id),
                optional_trusted_packet_sequence_id: Some(
                    trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                        sequence_id,
                    ),
                ),
                data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                    track_uuid: Some(track_uuid),
                    r#type: Some(track_event::Type::SliceEnd as i32),
                    name_field: Some(track_event::NameField::Name(
                        kernel_timing.kernel_name.to_owned(),
                    )),
                    ..Default::default()
                })),
                ..Default::default()
            });
        }
    }

    packets
}

pub trait KernelOp: std::fmt::Debug + as_any::AsAny {
    #[allow(clippy::type_complexity)]
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
        compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    );

    /// Returns the output buffer size in elements.
    fn output_size(&self) -> Expression;

    /// Returns the number of bytes this kernel will load from global memory.
    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    /// Returns the number of bytes this kernel will store to global memory.
    fn bytes_stored(&self) -> Expression {
        0.into()
    }

    /// Returns the number of floating point operations this kernel performs.
    fn flops(&self) -> Expression {
        0.into()
    }

    /// Returns the name of this kernel for profiling display.
    fn kernel_name(&self) -> &'static str {
        "Unknown"
    }

    /// Allocate internal buffers this kernel needs. Called once during graph building.
    /// Default: no internal buffers.
    fn allocate_internal_buffers(
        &self,
        _stream: &Arc<CudaStream>,
        _dyn_map: &FxHashMap<char, usize>,
    ) -> Vec<CudaSlice<u8>> {
        vec![]
    }

    /// Returns the set of dynamic dimensions that affect internal buffer sizes.
    /// When any of these dimensions change, internal buffers should be reallocated.
    /// Default: empty set (no dimensions affect internal buffers).
    fn internal_buffer_dyn_dims(&self) -> FxHashSet<char> {
        FxHashSet::default()
    }

    /// Build kernel parameters. Returns the u64 values to pass to the kernel.
    /// Default: [output_ptr, input_ptrs..., dyn_dims_ptr (if non-zero)]
    fn build_params(
        &self,
        _stream: &Arc<CudaStream>,
        output_ptr: u64,
        input_ptrs: &[u64],
        _internal_bufs: &[CudaSlice<u8>],
        dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        let mut params = vec![output_ptr];
        params.extend_from_slice(input_ptrs);
        if dyn_dims_ptr != 0 {
            params.push(dyn_dims_ptr);
        }
        params
    }

    /// Called before each kernel execution. Update internal state if needed.
    /// `all_buffer_ptrs` contains pointers for all buffers this kernel might use.
    /// `constants` are device constants returned by compile() that may need updating.
    fn pre_execute(
        &self,
        _stream: &Arc<CudaStream>,
        _internal_bufs: &mut [CudaSlice<u8>],
        _constants: &mut FxHashMap<char, CudaSlice<u8>>,
        _all_buffer_ptrs: &FxHashMap<NodeIndex, u64>,
        _dyn_map: &FxHashMap<char, usize>,
    ) {
    }

    /// Returns indices of internal buffers containing timing data, if any.
    /// Returns (timings_idx, start_times_idx, sm_count).
    fn timing_buffer_indices(&self) -> Option<(usize, usize, usize)> {
        None
    }
}

luminal::impl_into_ops!(KernelOp);

// Kernel to host op compilation
mod to_host;
pub use to_host::{CudaGraphOp, kernel_to_host};

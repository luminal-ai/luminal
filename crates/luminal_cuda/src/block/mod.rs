#![allow(clippy::mutable_key_type)]
pub mod cstruct;
mod ops;
mod to_kernel;

use itertools::Itertools;
pub use ops::*;
pub use to_kernel::block_to_kernel;

use cudarc::{
    driver::{
        CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DeviceRepr, ValidAsZeroBits,
    },
    nvrtc::{CompileOptions, compile_ptx_with_opts},
};
use luminal::{
    graph::LLIRGraph,
    hlir::Input,
    prelude::{
        FxHashMap, FxHashSet, NodeIndex,
        petgraph::{Direction, algo::toposort, visit::EdgeRef},
    },
    shape::{Expression, flatten_z_strides},
};
use std::{
    collections::HashMap,
    fmt::Debug,
    hash::{DefaultHasher, Hash, Hasher},
    iter::once,
    sync::Arc,
};
use tracing::{Level, span};
use tracing_perfetto_sdk_schema::{
    self as schema, TrackEvent, debug_annotation::NameField, trace_packet, track_descriptor,
    track_event,
};

use crate::block::cstruct::CStruct;

pub const N_TIMING_SLOTS: usize = 1000;

#[allow(unused_variables)]
pub trait BlockOp: Debug + as_any::AsAny {
    fn op_name(&self) -> &'static str;
    fn launch_range(&self) -> Vec<Expression> {
        unimplemented!()
    }
    /// Returns the output buffer size in elements.
    fn output_size(&self) -> Expression {
        unimplemented!()
    }
    fn producer_barriers_seperate(&self) -> Vec<bool>;
    fn consumer_barriers_seperate(&self) -> Vec<Vec<bool>>;
    /// C function body
    fn cuda_function(&self) -> String {
        "".to_string()
    }

    /// Device-global variable declarations (e.g., "__device__ int my_global;")
    fn device_globals(&self) -> String {
        "".to_string()
    }

    /// Returns the number of bytes this op will load from global memory.
    fn bytes_loaded(&self) -> Expression {
        0.into()
    }

    /// Returns the number of bytes this op will store to global memory.
    fn bytes_stored(&self) -> Expression {
        0.into()
    }

    /// Returns the number of floating point operations this op performs.
    fn flops(&self) -> Expression {
        0.into()
    }
    /// Build C-struct paylod
    fn build_payload<'a>(&self, stream: &Arc<CudaStream>, payload: CStruct<'a>) -> CStruct<'a> {
        unimplemented!()
    }
    fn prologue_a(&self) -> String {
        "".to_string()
    }
    fn prologue_a_flops(&self) -> Expression {
        0.into()
    }
    fn prologue_a_bytes_loaded(&self) -> Expression {
        0.into()
    }
    fn prologue_b(&self) -> String {
        "".to_string()
    }
    fn prologue_b_flops(&self) -> Expression {
        0.into()
    }
    fn prologue_b_bytes_loaded(&self) -> Expression {
        0.into()
    }
    fn prologue_c(&self) -> String {
        "".to_string()
    }
    fn prologue_c_flops(&self) -> Expression {
        0.into()
    }
    fn prologue_c_bytes_loaded(&self) -> Expression {
        0.into()
    }
}

luminal::impl_into_ops!(BlockOp);

#[tracing::instrument(skip_all)]
fn compute_barrier_strides(
    mut prod_range: Vec<Expression>,
    mut prod_shared: Vec<bool>,
    mut cons_range: Vec<Vec<Expression>>,
    mut cons_shared: Vec<Vec<bool>>,
) -> (Vec<Expression>, Vec<Vec<Expression>>) {
    // returns (producer strides, consumer strides)
    fn transpose<T>(v: Vec<Vec<T>>) -> Vec<Vec<T>> {
        if v.is_empty() {
            return vec![];
        }
        let len = v[0].len();
        let mut iters: Vec<_> = v.into_iter().map(|n| n.into_iter()).collect();
        (0..len)
            .map(|_| {
                iters
                    .iter_mut()
                    .map(|n| n.next().unwrap())
                    .collect::<Vec<T>>()
            })
            .collect()
    }
    let max_range_len = prod_range
        .len()
        .max(cons_range.iter().map(|i| i.len()).max().unwrap_or_default());
    let prod_range_len = prod_range.len();
    let cons_range_lens = cons_range.iter().map(|c| c.len()).collect_vec();
    prod_range.append(&mut vec![1.into(); max_range_len - prod_range.len()]);
    prod_shared.append(&mut vec![true; max_range_len - prod_shared.len()]);
    for v in &mut cons_range {
        v.append(&mut vec![1.into(); max_range_len - v.len()]);
    }
    for v in &mut cons_shared {
        v.append(&mut vec![false; max_range_len - v.len()]);
    }
    let cons_range_t = transpose(cons_range);
    let cons_shared_t = transpose(cons_shared);
    assert_eq!(cons_shared_t.len(), prod_range.len());
    let r = prod_range
        .iter()
        .zip(&prod_shared)
        .zip(&cons_range_t)
        .zip(cons_shared_t)
        .rev()
        .scan(Expression::from(1), |acc, (((pr, ps), cr), cs)| {
            let prev = *acc;
            if *ps && cs.iter().all(|i| *i) {
                if cr.iter().all(|cr| *pr == *cr) {
                    *acc *= *pr;
                    Some((Expression::from('z') * prev, vec![prev * 'z'; cr.len()]))
                } else if let Some(Some(factor)) = cr.iter().try_fold(None, |acc, cr| {
                    // Multiple producers per consumer
                    if !(*pr % *cr).to_usize().map(|i| i == 0).unwrap_or_default() {
                        return None;
                    }
                    if let Some(prev) = acc
                        && prev != (*pr / *cr)
                    {
                        return None;
                    }
                    Some(Some(*pr / *cr))
                }) {
                    *acc *= *pr / factor;
                    assert!(factor.to_usize().map(|i| i > 0).unwrap_or(true));
                    Some((
                        Expression::from('z') / factor * prev,
                        vec![prev * 'z'; cr.len()],
                    ))
                } else if let Some(Some(factor)) = cr.iter().try_fold(None, |acc, cr| {
                    // Multiple consumers per producer
                    if !(*cr % *pr).to_usize().map(|i| i == 0).unwrap_or_default() {
                        return None;
                    }
                    if let Some(prev) = acc
                        && prev != (*cr / *pr)
                    {
                        return None;
                    }
                    Some(Some(*cr / *pr))
                }) {
                    assert!(factor.to_usize().map(|i| i > 0).unwrap_or(true));
                    *acc *= cr[0] / factor;
                    Some((
                        prev * 'z',
                        vec![Expression::from('z') / factor * prev; cr.len()],
                    ))
                } else {
                    Some((0.into(), vec![0.into(); cr.len()]))
                }
            } else {
                Some((0.into(), vec![0.into(); cr.len()]))
            }
        })
        .collect_vec();
    let (mut p, c): (Vec<Expression>, Vec<Vec<Expression>>) = r.into_iter().rev().unzip();
    let mut c = transpose(c);
    // Re-trim down to original range lengths
    p = p[..prod_range_len].to_vec();
    for (c, r) in c.iter_mut().zip(cons_range_lens) {
        *c = c[..r].to_vec();
    }
    (p, c)
}

#[tracing::instrument(skip_all)]
#[allow(clippy::type_complexity)]
fn get_barrier_strides(
    graph: &LLIRGraph,
    block_ops: &FxHashSet<NodeIndex>,
) -> (
    FxHashMap<NodeIndex, Vec<Expression>>,
    FxHashMap<(NodeIndex, usize), Vec<Expression>>,
    FxHashMap<NodeIndex, Expression>,
    Expression,
) {
    // Resolve dependencies
    let mut producer_barrier_strides = FxHashMap::default();
    let mut consumer_barrier_strides = FxHashMap::default();
    for node in block_ops {
        if !graph
            .neighbors_directed(*node, Direction::Outgoing)
            .any(|n| block_ops.contains(&n))
        {
            producer_barrier_strides.insert(
                *node,
                vec![
                    0.into();
                    graph[*node]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .launch_range()
                        .len()
                ],
            ); // TODO: is this right?
            continue;
        }
        let consumers = graph
            .edges_directed(*node, Direction::Outgoing)
            .sorted_by_key(|e| e.id())
            .map(|e| {
                let n_input = graph
                    .edges_directed(e.target(), Direction::Incoming)
                    .sorted_by_key(|e| e.id())
                    .position(|ie| ie.id() == e.id())
                    .unwrap();
                (e.target(), n_input)
            })
            .filter(|(n, _)| block_ops.contains(n))
            .collect_vec();
        let prod_op = graph[*node].to_dialect::<dyn BlockOp>().unwrap();
        let prod_range = prod_op.launch_range();
        let prod_shared = prod_op.producer_barriers_seperate();
        let cons_range: Vec<Vec<Expression>> = consumers
            .iter()
            .map(|(n, _)| {
                graph[*n]
                    .to_dialect::<dyn BlockOp>()
                    .unwrap()
                    .launch_range()
            })
            .collect();
        let (producer_strides, consumer_strides) = compute_barrier_strides(
            prod_range.clone(),
            prod_shared,
            cons_range.clone(),
            consumers
                .iter()
                .map(|(n, i)| {
                    graph[*n]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .consumer_barriers_seperate()
                        .remove(*i)
                })
                .collect(),
        );

        producer_barrier_strides.insert(*node, producer_strides);
        assert_eq!(consumers.len(), consumer_strides.len());
        for ((cons, inp), strides) in consumers.into_iter().zip(consumer_strides) {
            consumer_barrier_strides.insert((cons, inp), strides);
        }
    }
    let mut n_barriers = Expression::from(1); // Starts at 1 to account for GMEM producers
    let mut producer_barrier_bases = FxHashMap::default();
    for op in block_ops {
        producer_barrier_bases.insert(*op, n_barriers);
        n_barriers = (n_barriers
            + producer_barrier_strides[op]
                .iter()
                .zip(
                    graph[*op]
                        .to_dialect::<dyn BlockOp>()
                        .unwrap()
                        .launch_range(),
                )
                .map(|(stride, range)| stride.substitute('z', range))
                .sum::<Expression>()
            + 1)
        .simplify();
    }
    (
        producer_barrier_strides,
        consumer_barrier_strides,
        producer_barrier_bases,
        n_barriers,
    )
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SMEvent {
    pub start: u64,
    pub stop: u64,
    pub event: i32,
}
unsafe impl DeviceRepr for SMEvent {}
unsafe impl ValidAsZeroBits for SMEvent {}

#[derive(Clone, Debug)]
pub(crate) struct TaskQueue {
    data: Vec<u8>,
    task_stride: usize,
    payload_align: usize,
    num_tasks: usize,
}

impl TaskQueue {
    pub fn new(payload_size: usize, payload_align: usize) -> Self {
        // Task layout (must match C struct with alignment):
        // - 11 ints (44 bytes at offset 0)
        // - 6 source indices (24 bytes at offset 44)
        // - 1 out index (4 bytes at offset 68)
        // = 72 bytes base, then padding for payload alignment, then payload
        let int_section = size_of::<i32>() * 11; // 44 bytes
        let index_section = size_of::<i32>() * 7; // 28 bytes (6 source + 1 out)
        let base_size = int_section + index_section; // 72 bytes

        // Add padding before payload if needed for alignment
        let payload_offset = if payload_align > 1 {
            (base_size + payload_align - 1) & !(payload_align - 1)
        } else {
            base_size
        };
        let total = payload_offset + payload_size;

        // Final alignment is max of 4 (for int fields) and payload_align
        let struct_align = 4.max(payload_align);
        let task_stride = (total + struct_align - 1) & !(struct_align - 1);
        Self {
            data: Vec::new(),
            task_stride,
            payload_align,
            num_tasks: 0,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn push_task(
        &mut self,
        op: i32,
        range: i32,
        remaining: i32,
        in_dep_a_stride: i32,
        in_dep_a_base: i32,
        in_dep_b_stride: i32,
        in_dep_b_base: i32,
        in_dep_c_stride: i32,
        in_dep_c_base: i32,
        out_dep_stride: i32,
        out_dep_base: i32,
        source_indices: [i32; 6],
        out_index: i32,
        payload: &[u8],
        expressions: &FxHashMap<Expression, i32>,
    ) {
        let mut bytes = CStruct::new(Some(expressions))
            .int("op", op)
            .int("range", range)
            .int("remaining", remaining)
            .int("in_dep_a_stride", in_dep_a_stride)
            .int("in_dep_a_base", in_dep_a_base)
            .int("in_dep_b_stride", in_dep_b_stride)
            .int("in_dep_b_base", in_dep_b_base)
            .int("in_dep_c_stride", in_dep_c_stride)
            .int("in_dep_c_base", in_dep_c_base)
            .int("out_dep_stride", out_dep_stride)
            .int("out_dep_base", out_dep_base)
            .int_arr("source_indices", &source_indices)
            .int("out_index", out_index)
            .bytes(self.payload_align, "payload", payload) // Add payload with proper alignment
            .finish_struct();

        // Pad to task_stride
        bytes.resize(self.task_stride, 0);

        self.data.extend_from_slice(&bytes);
        self.num_tasks += 1;
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    pub fn len(&self) -> usize {
        self.num_tasks
    }
}

struct ManualTrackBuilder {
    packets: Vec<schema::TracePacket>,
    track_uuid: u64,
    sequence_id: u32,
    state_cleared: bool,
    core_index: u32,
}

impl ManualTrackBuilder {
    pub fn new(core_index: u32, ts0: u64, clock_id: u32) -> Self {
        let track_uuid = manual_track_uuid(core_index);
        let sequence_id = manual_sequence_id(core_index);
        let track_name = format!("SM {core_index}");
        let synthetic_tid = 10_000 + core_index;
        let descriptor = schema::TracePacket {
            timestamp: Some(ts0.saturating_sub(1)),
            timestamp_clock_id: Some(clock_id),
            data: Some(trace_packet::Data::TrackDescriptor(
                schema::TrackDescriptor {
                    parent_uuid: None,
                    uuid: Some(track_uuid),
                    static_or_dynamic_name: Some(track_descriptor::StaticOrDynamicName::Name(
                        track_name.clone(),
                    )),
                    thread: Some(schema::ThreadDescriptor {
                        pid: Some(std::process::id() as i32),
                        tid: Some(synthetic_tid as i32),
                        thread_name: Some(track_name),
                        ..Default::default()
                    }),
                    ..Default::default()
                },
            )),
            ..Default::default()
        };

        let mut builder = Self {
            packets: Vec::new(),
            track_uuid,
            sequence_id,
            state_cleared: false,
            core_index,
        };
        builder.push_packet(descriptor);
        builder
    }

    pub fn push_slice(&mut self, label: &str, start: u64, end: u64, ts0: u64, clock_id: u32) {
        self.push_packet(self.slice_packet(label, ts0 + start, clock_id, true));
        self.push_packet(self.slice_packet(label, ts0 + end, clock_id, false));
    }

    pub fn slice_packet(
        &self,
        label: &str,
        timestamp_ns: u64,
        clock_id: u32,
        is_begin: bool,
    ) -> schema::TracePacket {
        let mut debug_annotations = Vec::new();
        debug_annotations.push(schema::DebugAnnotation {
            name_field: Some(schema::debug_annotation::NameField::Name("sm".into())),
            value: Some(schema::debug_annotation::Value::IntValue(
                self.core_index as i64,
            )),
            ..Default::default()
        });
        debug_annotations.push(schema::DebugAnnotation {
            name_field: Some(schema::debug_annotation::NameField::Name(
                "span.label".into(),
            )),
            value: Some(schema::debug_annotation::Value::StringValue(label.into())),
            ..Default::default()
        });

        schema::TracePacket {
            timestamp: Some(timestamp_ns),
            timestamp_clock_id: Some(clock_id),
            data: Some(trace_packet::Data::TrackEvent(schema::TrackEvent {
                track_uuid: Some(self.track_uuid),
                r#type: Some(if is_begin {
                    track_event::Type::SliceBegin as i32
                } else {
                    track_event::Type::SliceEnd as i32
                }),
                name_field: Some(track_event::NameField::Name(label.to_owned())),
                debug_annotations,
                ..Default::default()
            })),
            ..Default::default()
        }
    }

    pub fn push_packet(&mut self, mut packet: schema::TracePacket) {
        packet.optional_trusted_packet_sequence_id = Some(
            trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                self.sequence_id,
            ),
        );
        if !self.state_cleared {
            packet.sequence_flags =
                Some(trace_packet::SequenceFlags::SeqIncrementalStateCleared as i32 as u32);
            self.state_cleared = true;
        }
        self.packets.push(packet);
    }

    pub fn into_packets(self) -> Vec<schema::TracePacket> {
        self.packets
    }
}

fn manual_track_uuid(core_index: u32) -> u64 {
    hash64((1u32, 42u32, core_index))
}

fn manual_sequence_id(core_index: u32) -> u32 {
    hash32((2u32, 42u32, core_index))
}

fn hash64<T: Hash>(val: T) -> u64 {
    let mut hasher = DefaultHasher::new();
    val.hash(&mut hasher);
    hasher.finish()
}

fn hash32<T: Hash>(val: T) -> u32 {
    (hash64(val) & 0xffff_ffff) as u32
}

/// Build a mapping from interned string IDs to their string values for a given sequence.
fn build_interned_strings(trace: &schema::Trace) -> std::collections::HashMap<(u32, u64), String> {
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
    id: &uuid::Uuid,
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

/// Record block op timings from megakernels to perfetto trace packets
pub fn record_block_op_timings(
    trace: &schema::Trace,
    ops: &[Arc<Box<dyn BlockOp>>],
    timings: &[Vec<(Vec<SMEvent>, u64, uuid::Uuid)>],
) -> Vec<schema::TracePacket> {
    // Build interned string lookup table
    let interned = build_interned_strings(trace);

    let host_start_times: Vec<(u64, u32)> = timings
        .iter()
        .flatten()
        .map(|(_, _, id)| {
            trace
                .packet
                .iter()
                .find_map(|p| {
                    let seq_id = match &p.optional_trusted_packet_sequence_id {
                        Some(
                            trace_packet::OptionalTrustedPacketSequenceId::TrustedPacketSequenceId(
                                seq,
                            ),
                        ) => *seq,
                        _ => 0,
                    };
                    match &p.data {
                        Some(trace_packet::Data::TrackEvent(TrackEvent {
                            r#type: ty,
                            debug_annotations,
                            ..
                        })) if *ty == Some(track_event::Type::SliceBegin as i32)
                            && debug_annotations
                                .iter()
                                .any(|a| annotation_matches_id(a, id, &interned, seq_id)) =>
                        {
                            Some((p.timestamp?, p.timestamp_clock_id?))
                        }
                        _ => None,
                    }
                })
                .expect("Couldn't find span with correct uuid for gpu timing dump")
        })
        .collect();

    let mut packets = Vec::new();
    let n_ops = ops.len();
    for ((device_timings, device_start_time, _), (host_time, host_clock_id)) in
        timings.iter().flatten().zip(host_start_times)
    {
        for (sm, sm_timings) in device_timings.chunks(1000).enumerate() {
            let mut builder = ManualTrackBuilder::new(sm as u32, host_time, host_clock_id);
            for n_op in 0..sm_timings.len() - 1 {
                let event = sm_timings[n_op].event as usize;
                let op_label = if event == 0 {
                    "Issue".to_string()
                } else if event == 1 {
                    "Wait".to_string()
                } else if event >= 2 && event < 2 + n_ops {
                    ops[event - 2].op_name().to_string()
                } else if event >= 2 + n_ops {
                    let prologue_event = event - 2 - n_ops;
                    let op_idx = prologue_event / 3;
                    let prologue_type = prologue_event % 3;
                    if op_idx < n_ops {
                        let suffix = match prologue_type {
                            0 => "prologue A",
                            1 => "prologue B",
                            2 => "prologue C",
                            _ => "prologue ?",
                        };
                        format!("{} ({})", ops[op_idx].op_name(), suffix)
                    } else {
                        format!("Unknown({})", event)
                    }
                } else {
                    format!("Unknown({})", event)
                };
                if sm_timings[n_op + 1].start == 0 {
                    break;
                }
                builder.push_slice(
                    &op_label,
                    sm_timings[n_op].start - *device_start_time,
                    sm_timings[n_op + 1].start - *device_start_time,
                    host_time,
                    host_clock_id,
                );
            }
            packets.extend(builder.into_packets());
        }
    }
    packets
}

#[tracing::instrument(skip_all)]
#[allow(clippy::type_complexity)]
fn compile_interpreter(
    cuda_stream: &Arc<CudaStream>,
    ops: &Vec<Arc<Box<dyn BlockOp>>>,
    expressions: &FxHashSet<Expression>,
    payload_size: usize,
    n_tasks: usize,
    n_barriers: &Expression,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> (
    CudaFunction,
    Arc<CudaModule>,
    FxHashMap<Expression, i32>,
    FxHashMap<char, CudaSlice<u8>>,
) {
    let expression_map = expressions
        .iter()
        .enumerate()
        .map(|(i, e)| (*e, i as i32))
        .collect::<FxHashMap<_, _>>();

    // Compile the interpreter
    let mut kernel = include_str!("interpreter.cu").to_string();
    let n_ops = ops.len();
    kernel = kernel.replace(
        "const int N_OPS = 0;",
        &format!("const int N_OPS = {};", n_ops),
    );
    kernel = kernel.replace(
        "const int N_TIMING_SLOTS = 0;",
        &format!("const int N_TIMING_SLOTS = {N_TIMING_SLOTS};"),
    );
    kernel = kernel.replace(
        "const int N_TASKS = 0;",
        &format!("const int N_TASKS = {n_tasks};"),
    );
    // N_BARRIERS is an expression that may depend on dyn dims, render it as a macro
    kernel = kernel.replace(
        "//%n_barriers_const%",
        &format!("#define N_BARRIERS ({})", n_barriers.simplify().to_kernel()),
    );
    kernel = kernel.replace(
        "//%extra_op_codes%",
        &ops.iter()
            .enumerate()
            .map(|(i, op)| format!("{}Op = {i}", op.op_name()))
            .join(", "),
    );
    kernel = kernel.replace(
        "//%extra_op_structs%",
        &ops.iter()
            .map(|op| {
                format!(
                    "struct {}Payload {{{}}};",
                    op.op_name(),
                    op.build_payload(cuda_stream, CStruct::new(Some(&expression_map))),
                )
            })
            .join("\n"),
    );
    kernel = kernel.replace(
        "//%extra_op_functions%",
        &ops
            .iter()
            .map(|op| {
                let op_name = op.op_name();
                let op_body = op.cuda_function();
                format!(
                    "__device__ __forceinline__ void {op_name}_function({op_name}Payload payload, const float* const source_ptrs[6], float* out_ptr, const int current, int t, float* scratchpad) {{
{op_body}
}}"
                )
            })
            .join("\n"),
    );
    kernel = kernel.replace(
        "//%extra_op_payloads%",
        &format!(
            "{} char _padding[{}];",
            ops.iter()
                .map(|op| {
                    let op_name = op.op_name();
                    format!("{op_name}Payload {op_name};")
                })
                .join(" "),
            payload_size
        ),
    );
    kernel = kernel.replace("//%extra_op_calls%", &ops.iter().map(|op| {
            let op_name = op.op_name();
            format!("case OpCode::{op_name}Op: {op_name}_function(t->payload.{op_name}, source_ptrs, out_ptr, nt.current, threadIdx.x, scratchpad); break;")
        }).join("\n"));

    // Generate prologue functions (only for non-empty prologues)
    {
        let _span = span!(Level::TRACE, "render_prologue_functions").entered();
        kernel = kernel.replace(
        "//%extra_prologue_functions%",
        &ops
            .iter()
            .flat_map(|op| {
                let op_name = op.op_name();
                let prologue_a = op.prologue_a();
                let prologue_b = op.prologue_b();
                let prologue_c = op.prologue_c();
                let mut funcs = Vec::new();
                if !prologue_a.is_empty() {
                    funcs.push(format!(
                        "__device__ __forceinline__ void {op_name}_prologue_a({op_name}Payload payload, const float* const source_ptrs[6], float* out_ptr, const int current, int t, float* scratchpad) {{
{prologue_a}
}}"
                    ));
                }
                if !prologue_b.is_empty() {
                    funcs.push(format!(
                        "__device__ __forceinline__ void {op_name}_prologue_b({op_name}Payload payload, const float* const source_ptrs[6], float* out_ptr, const int current, int t, float* scratchpad) {{
{prologue_b}
}}"
                    ));
                }
                if !prologue_c.is_empty() {
                    funcs.push(format!(
                        "__device__ __forceinline__ void {op_name}_prologue_c({op_name}Payload payload, const float* const source_ptrs[6], float* out_ptr, const int current, int t, float* scratchpad) {{
{prologue_c}
}}"
                    ));
                }
                funcs
            })
            .join("\n"),
    );

        // Generate prologue A calls (only for non-empty prologues, with event recording)
        kernel = kernel.replace(
        "//%prologue_a_calls%",
        &ops.iter().enumerate().filter_map(|(i, op)| {
            let op_name = op.op_name();
            if op.prologue_a().is_empty() {
                None
            } else {
                // Event code: 2 + N_OPS + op_idx * 3 + 0
                let event_code = 2 + n_ops + i * 3;
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_a(t->payload.{op_name}, source_ptrs, out_ptr, nt.current, threadIdx.x, scratchpad); __syncthreads(); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); break;"))
            }
        }).join("\n"),
    );

        // Generate prologue B calls (only for non-empty prologues, with event recording)
        kernel = kernel.replace(
        "//%prologue_b_calls%",
        &ops.iter().enumerate().filter_map(|(i, op)| {
            let op_name = op.op_name();
            if op.prologue_b().is_empty() {
                None
            } else {
                // Event code: 2 + N_OPS + op_idx * 3 + 1
                let event_code = 2 + n_ops + i * 3 + 1;
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_b(t->payload.{op_name}, source_ptrs, out_ptr, nt.current, threadIdx.x, scratchpad); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); __syncthreads(); break;"))
            }
        }).join("\n"),
    );

        // Generate prologue C calls (only for non-empty prologues, with event recording)
        kernel = kernel.replace(
        "//%prologue_c_calls%",
        &ops.iter().enumerate().filter_map(|(i, op)| {
            let op_name = op.op_name();
            if op.prologue_c().is_empty() {
                None
            } else {
                // Event code: 2 + N_OPS + op_idx * 3 + 2
                let event_code = 2 + n_ops + i * 3 + 2;
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_c(t->payload.{op_name}, source_ptrs, out_ptr, nt.current, threadIdx.x, scratchpad); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); __syncthreads(); break;"))
            }
        }).join("\n"),
    );
    }

    let span = span!(Level::TRACE, "render_expressions").entered();
    let constants = expressions
        .iter()
        .flat_map(|e| e.dyn_vars())
        .collect::<FxHashSet<_>>();
    let constant_string = constants
        .iter()
        .map(|v| format!("__constant__ int const_{v}[1];"))
        .join("\n");
    let lambdas = expression_map
        .iter()
        .sorted_by_key(|(_, i)| **i)
        .map(|(e, i)| format!("case {i}: return {};", e.simplify().to_kernel()))
        .join("\n");
    kernel = kernel.replace("//%expr_fns%", &lambdas);

    // Collect device globals from all ops
    let device_globals = ops
        .iter()
        .map(|op| op.device_globals())
        .filter(|s| !s.is_empty())
        .join("\n");

    kernel = kernel.replace(
        "//%constants%",
        &format!("{constant_string}{device_globals}"),
    );
    drop(span);

    let arch = crate::cuda_nvrtc_arch(cuda_stream.context()).unwrap_or_else(|| "sm_75".to_string());
    // NVRTC output depends on both source and target arch.
    let kernel_cache_key = format!("{arch}\n{kernel}");
    let (module, func) = if let Some((module, kernel)) = kernel_cache.get(&kernel_cache_key) {
        (module.clone(), kernel.clone())
    } else {
        let _span = span!(Level::TRACE, "nvrtc", arch = %arch).entered();
        let ptx = compile_ptx_with_opts(
            &kernel,
            CompileOptions {
                arch: Some(arch.as_str()),
                ..Default::default()
            },
        )
        .unwrap();
        let module = cuda_stream.context().load_module(ptx).unwrap();
        let func = module.load_function("worker_kernel").unwrap();

        kernel_cache.push(kernel_cache_key, (module.clone(), func.clone()));

        kernel_cache.insert(kernel.clone(), (module.clone(), func.clone()));

        (module, func)
    };
    let constants: FxHashMap<char, CudaSlice<u8>> = constants
        .into_iter()
        .map(|d| {
            let global = module
                .get_global(&format!("const_{d}"), cuda_stream)
                .unwrap();
            (d, global)
        })
        .collect();
    (func, module, expression_map, constants)
}

/// A compiled megakernel that implements KernelOp.
/// This allows megakernels to flow through the same compilation pipeline as regular kernels.
///
/// Internal buffers are managed via the `internal_buffers()` method, which returns
/// the buffers the kernel needs allocated and passed as parameters.
///
/// Note: Device buffers are managed separately in the runtime.
/// This struct holds only compile-time information.
#[derive(Debug)]
pub struct MegakernelOp {
    /// The compiled interpreter kernel function
    pub interpreter: CudaFunction,
    /// The CUDA module containing the kernel
    pub module: Arc<CudaModule>,
    /// Device-side constants for dynamic dimensions
    pub interpreter_constants: FxHashMap<char, CudaSlice<u8>>,
    /// Number of barriers needed for synchronization
    pub n_barriers: Expression,
    /// Serialized task queue (all operations to execute)
    pub(crate) work_queue: TaskQueue,
    /// Mapping from LLIR node to buffer index
    pub node_to_buffer_index: FxHashMap<NodeIndex, i32>,
    /// Number of SMs on the device (for grid size)
    pub sm_count: i32,
}

impl crate::kernel::KernelOp for MegakernelOp {
    fn compile(
        &self,
        _stream: &Arc<CudaStream>,
        _compile_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        // Return the pre-compiled interpreter function with appropriate launch config.
        (
            self.interpreter.clone(),
            self.module.clone(),
            "megakernel".to_string(),
            (self.sm_count.into(), 1.into(), 1.into()), // grid: one block per SM
            (256.into(), 1.into(), 1.into()),           // block: 256 threads
            0.into(), // No dynamic shared memory (static scratchpad is sufficient)
            self.interpreter_constants.clone(), // Return constants for runtime to manage
        )
    }

    fn output_size(&self) -> Expression {
        // Megakernels don't have a single output size - they write to multiple buffers.
        // Return 0 as a placeholder; the actual buffer allocation is handled by the
        // individual BlockOps that make up the megakernel.
        0.into()
    }

    fn kernel_name(&self) -> &'static str {
        "Megakernel"
    }

    fn allocate_internal_buffers(
        &self,
        stream: &Arc<CudaStream>,
        dyn_map: &FxHashMap<char, usize>,
    ) -> Vec<CudaSlice<u8>> {
        let buffer_count = self.buffer_count();
        let n_barriers = self.n_barriers.exec(dyn_map).unwrap();

        vec![
            // 0: tasks - upload task queue
            stream.clone_htod(self.work_queue.as_slice()).unwrap(),
            // 1: head - reset in-kernel
            stream
                .alloc_zeros::<u8>(std::mem::size_of::<i32>())
                .unwrap(),
            // 2: ready - barrier array, reset in-kernel
            stream
                .alloc_zeros::<u8>(n_barriers * std::mem::size_of::<i32>())
                .unwrap(),
            // 3: queue_lock - reset in-kernel
            stream
                .alloc_zeros::<u8>(std::mem::size_of::<i32>())
                .unwrap(),
            // 4: timings - per-SM timing events
            stream
                .alloc_zeros::<u8>(
                    self.sm_count as usize * N_TIMING_SLOTS * std::mem::size_of::<SMEvent>(),
                )
                .unwrap(),
            // 5: start_times - per-SM start times
            stream
                .alloc_zeros::<u8>(self.sm_count as usize * std::mem::size_of::<u64>())
                .unwrap(),
            // 6: buffers - array of buffer pointers
            stream
                .alloc_zeros::<u8>(buffer_count * std::mem::size_of::<u64>())
                .unwrap(),
        ]
    }

    fn build_params(
        &self,
        stream: &Arc<CudaStream>,
        _output_ptr: u64,
        _input_ptrs: &[u64],
        internal_bufs: &[CudaSlice<u8>],
        _dyn_dims_ptr: u64,
    ) -> Vec<u64> {
        // Megakernel params: [tasks, head, ready, queue_lock, timings, start_times, buffers, dyn_dims]
        // dyn_dims is handled via constants, pass 0
        internal_bufs
            .iter()
            .map(|buf| buf.device_ptr(stream).0)
            .chain(std::iter::once(0u64)) // dyn_dims placeholder
            .collect()
    }

    fn pre_execute(
        &self,
        stream: &Arc<CudaStream>,
        internal_bufs: &mut [CudaSlice<u8>],
        _constants: &mut FxHashMap<char, CudaSlice<u8>>,
        all_buffer_ptrs: &FxHashMap<NodeIndex, u64>,
        dyn_map: &FxHashMap<char, usize>,
    ) {
        // Update dyn dims in interpreter constants by getting fresh handles from the module.
        // We do NOT use the `_constants` parameter because CudaSlice.clone() creates copies,
        // not references to the original __constant__ memory.
        for (dyn_dim, val) in dyn_map {
            let global_name = format!("const_{}", dyn_dim);
            if let Ok(mut global) = self.module.get_global(&global_name, stream) {
                let mut view = global.as_view_mut();
                let mut symbol = unsafe { view.transmute_mut::<i32>(1).unwrap() };
                stream
                    .memcpy_htod(&[*val as i32], &mut symbol)
                    .expect("Failed to update dyn dim constant");
                // IMPORTANT: Don't drop `global` - it would try to free __constant__ memory!
                // Leak it intentionally to prevent the Drop from running.
                std::mem::forget(global);
            }
        }

        // Re-upload tasks with remaining=-1 (index 0)
        // This ensures fresh task state for each execution
        let task_data = self.work_queue.as_slice();
        stream
            .memcpy_htod(task_data, &mut internal_bufs[0].as_view_mut())
            .expect("Failed to re-upload tasks");

        // Reset head to 0 (index 1)
        {
            let mut head_view = internal_bufs[1].as_view_mut();
            let mut head_typed = unsafe { head_view.transmute_mut::<i32>(1).unwrap() };
            stream
                .memcpy_htod(&[0i32], &mut head_typed)
                .expect("Failed to reset head");
        }

        // Reset barriers to 0 (index 2)
        // Use the allocated size to avoid buffer overflow
        let allocated_barrier_size = internal_bufs[2].len();
        let allocated_n_barriers = allocated_barrier_size / std::mem::size_of::<i32>();
        {
            let zeros: Vec<i32> = vec![0; allocated_n_barriers];
            let mut ready_view = internal_bufs[2].as_view_mut();
            let mut ready_typed = unsafe {
                ready_view
                    .transmute_mut::<i32>(allocated_n_barriers)
                    .unwrap()
            };
            stream
                .memcpy_htod(&zeros, &mut ready_typed)
                .expect("Failed to reset barriers");
        }

        // Reset queue_lock to 0 (index 3)
        {
            let mut lock_view = internal_bufs[3].as_view_mut();
            let mut lock_typed = unsafe { lock_view.transmute_mut::<i32>(1).unwrap() };
            stream
                .memcpy_htod(&[0i32], &mut lock_typed)
                .expect("Failed to reset queue_lock");
        }

        // Update buffer array (index 6)
        let buffer_count = self.buffer_count();
        let mut buffer_array: Vec<u64> = vec![0; buffer_count];
        for (node, &buffer_idx) in &self.node_to_buffer_index {
            if let Some(&ptr) = all_buffer_ptrs.get(node) {
                buffer_array[buffer_idx as usize] = ptr;
            }
        }

        let mut buffers_view = internal_bufs[6].as_view_mut();
        let mut buffers_typed = unsafe {
            buffers_view
                .transmute_mut::<u64>(buffer_count)
                .expect("Failed to transmute buffers")
        };
        stream
            .memcpy_htod(&buffer_array, &mut buffers_typed)
            .expect("Failed to update buffer array");

        // Ensure all uploads complete before kernel execution
        stream
            .synchronize()
            .expect("Failed to sync after pre_execute");
    }

    fn timing_buffer_indices(&self) -> Option<(usize, usize, usize)> {
        // timings at index 4, start_times at index 5, sm_count
        Some((4, 5, self.sm_count as usize))
    }

    fn internal_buffer_dyn_dims(&self) -> FxHashSet<char> {
        // The barrier buffer size depends on n_barriers, which may contain dynamic dimensions
        self.n_barriers.dyn_vars().into_iter().collect()
    }
}

impl MegakernelOp {
    /// Create a new MegakernelOp from the LLIR graph compilation result.
    pub fn new(
        llir_graph: &LLIRGraph,
        subgraph: &FxHashSet<NodeIndex>,
        cuda_stream: &Arc<CudaStream>,
        kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
    ) -> Self {
        let (
            interpreter,
            module,
            interpreter_constants,
            n_barriers,
            work_queue,
            _node_to_task_index,
            node_to_buffer_index,
        ) = make_megakernel_from_llir_graph(llir_graph, subgraph, cuda_stream, kernel_cache);

        // Get device properties
        let ctx = cuda_stream.context();
        let sm_count = ctx
            .attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )
            .expect("Failed to get SM count");

        Self {
            interpreter,
            module,
            interpreter_constants,
            n_barriers,
            work_queue,
            node_to_buffer_index,
            sm_count,
        }
    }

    /// Returns the number of buffers this megakernel uses.
    pub fn buffer_count(&self) -> usize {
        self.node_to_buffer_index
            .values()
            .map(|&i| i + 1)
            .max()
            .unwrap_or(0) as usize
    }
}

impl Drop for MegakernelOp {
    fn drop(&mut self) {
        // IMPORTANT: interpreter_constants contain CudaSlices pointing to __constant__ memory
        // in the CUDA module. We must NOT drop these slices because:
        // 1. __constant__ memory is part of the module and shouldn't be freed separately
        // 2. Dropping them would try to call cuMemFree on __constant__ addresses, which
        //    corrupts the CUDA context and causes subsequent allocations to fail
        //
        // Leak the slices intentionally to prevent their Drop from running.
        let constants = std::mem::take(&mut self.interpreter_constants);
        for (_key, slice) in constants {
            std::mem::forget(slice);
        }
    }
}

#[allow(clippy::type_complexity)]
pub(crate) fn make_megakernel_from_llir_graph(
    llir_graph: &LLIRGraph,
    subgraph: &FxHashSet<NodeIndex>,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut FxHashMap<String, (Arc<CudaModule>, CudaFunction)>,
) -> (
    CudaFunction,
    Arc<CudaModule>, // Module (needed for device globals)
    FxHashMap<char, CudaSlice<u8>>,
    Expression,
    TaskQueue,
    FxHashMap<NodeIndex, usize>,
    FxHashMap<NodeIndex, i32>, // node_to_buffer_index mapping
) {
    let block_ops = llir_graph
        .node_indices()
        .filter(|n| subgraph.contains(n))
        .filter_map(|n| llir_graph[n].to_dialect::<dyn BlockOp>())
        .map(|bo| (bo.op_name(), bo.clone()))
        .collect::<HashMap<_, _>>()
        .into_iter()
        .sorted_by_key(|(n, _)| *n)
        .map(|(_, o)| o)
        .collect_vec();
    // Render expressions
    let (
        producer_barrier_strides,
        consumer_barrier_strides,
        mut producer_barrier_bases,
        n_barriers,
    ) = crate::block::get_barrier_strides(llir_graph, subgraph);
    for node in llir_graph
        .node_indices()
        .filter(|n| llir_graph[*n].to_op::<Input>().is_some())
    {
        producer_barrier_bases.insert(node, 0.into());
    }
    #[allow(clippy::mutable_key_type)]
    let expressions = llir_graph
        .node_weights()
        .filter_map(|op| op.to_dialect::<dyn BlockOp>())
        .flat_map(|op| {
            op.build_payload(cuda_stream, CStruct::new(None))
                .recorded_expressions
                .into_iter()
                .chain(once(op.launch_range().iter().copied().product()))
        })
        .chain(producer_barrier_strides.iter().map(|(n, e)| {
            flatten_z_strides(
                &llir_graph[*n]
                    .to_dialect::<dyn BlockOp>()
                    .unwrap()
                    .launch_range(),
                e,
            )
        }))
        .chain(consumer_barrier_strides.iter().map(|((n, _), e)| {
            flatten_z_strides(
                &llir_graph[*n]
                    .to_dialect::<dyn BlockOp>()
                    .unwrap()
                    .launch_range(),
                e,
            )
        }))
        .chain(producer_barrier_bases.values().copied())
        .chain(once(0.into()))
        .chain(once(1.into()))
        .collect::<FxHashSet<_>>();
    // Build temporary expression map for calculating payload sizes
    let temp_expression_map = expressions
        .iter()
        .enumerate()
        .map(|(i, e)| (*e, i as i32))
        .collect::<FxHashMap<_, _>>();

    // Calculate actual max payload size and alignment from the ops being used
    let (max_payload_size, max_payload_align) = block_ops
        .iter()
        .map(|op| {
            op.build_payload(cuda_stream, CStruct::new(Some(&temp_expression_map)))
                .size_and_align()
        })
        .fold((0, 1), |(max_size, max_align), (size, align)| {
            (max_size.max(size), max_align.max(align))
        });

    // Count number of tasks (one per BlockOp node in subgraph)
    let n_tasks = subgraph
        .iter()
        .filter(|n| llir_graph[**n].to_dialect::<dyn BlockOp>().is_some())
        .count();

    let (interpreter, module, expressions, interpreter_constants) = compile_interpreter(
        cuda_stream,
        &block_ops,
        &expressions,
        max_payload_size,
        n_tasks,
        &n_barriers,
        kernel_cache,
    );

    // Build task queue with dynamic payload size and alignment
    let mut tasks = TaskQueue::new(max_payload_size, max_payload_align);
    let mut node_to_task_index = FxHashMap::default();
    let mut node_to_buffer_index: FxHashMap<NodeIndex, i32> = FxHashMap::default();
    let mut next_buffer_index: i32 = 0;

    // Helper to get or assign buffer index for a node
    let mut get_buffer_index = |node: NodeIndex| -> i32 {
        *node_to_buffer_index.entry(node).or_insert_with(|| {
            let idx = next_buffer_index;
            next_buffer_index += 1;
            idx
        })
    };

    for node in toposort(&llir_graph, None).unwrap() {
        if !subgraph.contains(&node) {
            continue;
        }
        let sources = llir_graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect_vec();

        // Assign buffer indices for source nodes and output node
        let source_indices: [i32; 6] = [
            get_buffer_index(sources[0]),
            sources.get(1).map(|&n| get_buffer_index(n)).unwrap_or(0),
            sources.get(2).map(|&n| get_buffer_index(n)).unwrap_or(0),
            sources.get(3).map(|&n| get_buffer_index(n)).unwrap_or(0),
            sources.get(4).map(|&n| get_buffer_index(n)).unwrap_or(0),
            sources.get(5).map(|&n| get_buffer_index(n)).unwrap_or(0),
        ];
        let out_index = get_buffer_index(node);

        let op = llir_graph[node].to_dialect::<dyn BlockOp>().unwrap();
        let op_code = block_ops
            .iter()
            .position(|o| o.op_name() == op.op_name())
            .unwrap();
        let mut payload = op
            .build_payload(cuda_stream, CStruct::new(Some(&expressions)))
            .finish_struct();
        // Pad payload to max_payload_size
        payload.resize(max_payload_size, 0);
        let range = op.launch_range();
        let in_dep_a_stride = consumer_barrier_strides
            .get(&(node, 0))
            .map(|s| flatten_z_strides(&range, s))
            .unwrap_or(0.into());
        let in_dep_b_stride = consumer_barrier_strides
            .get(&(node, 1))
            .map(|s| flatten_z_strides(&range, s))
            .unwrap_or(0.into());
        let in_dep_c_stride = consumer_barrier_strides
            .get(&(node, 2))
            .map(|s| flatten_z_strides(&range, s))
            .unwrap_or(0.into());
        let out_dep_stride = producer_barrier_strides
            .get(&node)
            .map(|s| flatten_z_strides(&range, s))
            .unwrap_or(0.into());
        node_to_task_index.insert(node, tasks.len());
        let task_range = expressions[&range.iter().copied().product()];
        let in_dep_a_stride_val = expressions[&in_dep_a_stride];
        let in_dep_a_base_val = producer_barrier_bases
            .get(&sources[0])
            .map(|e| expressions[e])
            .unwrap_or(-1);
        let in_dep_b_stride_val = expressions[&in_dep_b_stride];
        let in_dep_b_base_val = sources
            .get(1)
            .and_then(|n| producer_barrier_bases.get(n))
            .map(|e| expressions[e])
            .unwrap_or(-1);
        let in_dep_c_stride_val = expressions[&in_dep_c_stride];
        let in_dep_c_base_val = sources
            .get(2)
            .and_then(|n| producer_barrier_bases.get(n))
            .map(|e| expressions[e])
            .unwrap_or(-1);
        let out_dep_stride_val = expressions[&out_dep_stride];
        let out_dep_base_val = expressions[&producer_barrier_bases[&node]];

        tasks.push_task(
            op_code as i32,
            task_range,
            -1,
            in_dep_a_stride_val,
            in_dep_a_base_val,
            in_dep_b_stride_val,
            in_dep_b_base_val,
            in_dep_c_stride_val,
            in_dep_c_base_val,
            out_dep_stride_val,
            out_dep_base_val,
            source_indices,
            out_index,
            &payload,
            &expressions,
        );
    }
    (
        interpreter,
        module,
        interpreter_constants,
        n_barriers,
        tasks,
        node_to_task_index,
        node_to_buffer_index,
    )
}

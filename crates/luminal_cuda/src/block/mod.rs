#![allow(clippy::mutable_key_type)]
pub mod cstruct;
mod ops;
use itertools::Itertools;
use lru::LruCache;
pub use ops::*;

use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits},
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
    ptr::{null, null_mut},
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
#[derive(Copy, Clone)]
pub struct SMEvent {
    pub start: u64,
    pub stop: u64,
    pub event: i32,
}
unsafe impl DeviceRepr for SMEvent {}
unsafe impl ValidAsZeroBits for SMEvent {}

#[derive(Clone)]
pub(crate) struct TaskQueue {
    data: Vec<u8>,
    task_stride: usize,
    num_tasks: usize,
}

impl TaskQueue {
    pub fn new(payload_size: usize) -> Self {
        // Task layout (must match C struct with alignment):
        // - 11 ints (44 bytes at offset 0)
        // - 4 bytes padding for 8-byte alignment of pointers
        // - 3 pointers (24 bytes at offset 48)
        // - 1 pointer (8 bytes at offset 72)
        // = 80 bytes base + payload_size, aligned to 8 bytes
        let int_section = size_of::<i32>() * 11; // 44 bytes
        let int_section_aligned = (int_section + 7) & !7; // 48 bytes (aligned for pointers)
        let ptr_section = size_of::<*const f32>() * 3 + size_of::<*mut f32>(); // 32 bytes
        let base_size = int_section_aligned + ptr_section; // 80 bytes
        let total = base_size + payload_size;
        let task_stride = (total + 7) & !7; // Align to 8 bytes
        Self {
            data: Vec::new(),
            task_stride,
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
        source_ptrs: [*const f32; 3],
        out_ptr: *mut f32,
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
            .ptr_const_f32_arr("source_ptrs", source_ptrs.as_slice())
            .ptr_mut_f32("out_ptr", out_ptr)
            .bytes(1, "payload", payload) // Add payload with byte alignment
            .finish_struct();

        // Pad to task_stride
        bytes.resize(self.task_stride, 0);

        self.data.extend_from_slice(&bytes);
        self.num_tasks += 1;
    }

    pub fn set_out_ptr(&mut self, index: usize, ptr: *mut f32) {
        // Layout: 11 ints (44 bytes) + padding (4 bytes) + 3 ptrs (24 bytes) + out_ptr (8 bytes)
        let ints_size = size_of::<i32>() * 11; // 44
        let padding = (8 - (ints_size % 8)) % 8; // 4 bytes padding to align to 8
        let offset = index * self.task_stride + ints_size + padding + size_of::<*const f32>() * 3;
        let bytes = (ptr as usize).to_ne_bytes();
        self.data[offset..offset + size_of::<*mut f32>()].copy_from_slice(&bytes);
    }

    pub fn set_source_ptr(&mut self, index: usize, input_num: usize, ptr: *const f32) {
        // Layout: 11 ints (44 bytes) + padding (4 bytes) + source_ptrs[input_num]
        let ints_size = size_of::<i32>() * 11; // 44
        let padding = (8 - (ints_size % 8)) % 8; // 4 bytes padding to align to 8
        let offset =
            index * self.task_stride + ints_size + padding + size_of::<*const f32>() * input_num;
        let bytes = (ptr as usize).to_ne_bytes();
        self.data[offset..offset + size_of::<*const f32>()].copy_from_slice(&bytes);
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
fn compile_interpreter(
    cuda_stream: &Arc<CudaStream>,
    ops: &Vec<Arc<Box<dyn BlockOp>>>,
    expressions: &FxHashSet<Expression>,
    payload_size: usize,
    kernel_cache: &mut LruCache<String, (Arc<CudaModule>, CudaFunction)>,
) -> (
    CudaFunction,
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
                    "__device__ __forceinline__ void {op_name}_function({op_name}Payload payload, const float* const source_ptrs[3], float* out_ptr, const int current, int t, float* scratchpad) {{
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
            format!("case OpCode::{op_name}Op: {op_name}_function(t->payload.{op_name}, t->source_ptrs, t->out_ptr, nt.current, threadIdx.x, scratchpad); break;")
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
                        "__device__ __forceinline__ void {op_name}_prologue_a({op_name}Payload payload, const float* const source_ptrs[3], float* out_ptr, const int current, int t, float* scratchpad) {{
{prologue_a}
}}"
                    ));
                }
                if !prologue_b.is_empty() {
                    funcs.push(format!(
                        "__device__ __forceinline__ void {op_name}_prologue_b({op_name}Payload payload, const float* const source_ptrs[3], float* out_ptr, const int current, int t, float* scratchpad) {{
{prologue_b}
}}"
                    ));
                }
                if !prologue_c.is_empty() {
                    funcs.push(format!(
                        "__device__ __forceinline__ void {op_name}_prologue_c({op_name}Payload payload, const float* const source_ptrs[3], float* out_ptr, const int current, int t, float* scratchpad) {{
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
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_a(t->payload.{op_name}, t->source_ptrs, t->out_ptr, nt.current, threadIdx.x, scratchpad); __syncthreads(); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); break;"))
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
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_b(t->payload.{op_name}, t->source_ptrs, t->out_ptr, nt.current, threadIdx.x, scratchpad); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); __syncthreads(); break;"))
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
                Some(format!("case OpCode::{op_name}Op: if (threadIdx.x == 0) record_event(timings, &recorded_event, {event_code}); {op_name}_prologue_c(t->payload.{op_name}, t->source_ptrs, t->out_ptr, nt.current, threadIdx.x, scratchpad); if (threadIdx.x == 0) record_event(timings, &recorded_event, 1); __syncthreads(); break;"))
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

    let (module, func) = if let Some((module, kernel)) = kernel_cache.get(&kernel) {
        (module.clone(), kernel.clone())
    } else {
        let _span = span!(Level::TRACE, "nvrtc").entered();
        let ptx = compile_ptx_with_opts(
            &kernel,
            CompileOptions {
                arch: Some("sm_75"),
                ..Default::default()
            },
        )
        .unwrap();
        let module = cuda_stream.context().load_module(ptx).unwrap();
        let func = module.load_function("worker_kernel").unwrap();
        kernel_cache.push(kernel.clone(), (module.clone(), func.clone()));
        (module, func)
    };
    let constants = constants
        .into_iter()
        .map(|d| {
            (
                d,
                module
                    .get_global(&format!("const_{d}"), cuda_stream)
                    .unwrap(),
            )
        })
        .collect();
    (func, expression_map, constants)
}

#[allow(clippy::type_complexity)]
pub(crate) fn make_megakernel_from_llir_graph(
    llir_graph: &LLIRGraph,
    subgraph: &FxHashSet<NodeIndex>,
    cuda_stream: &Arc<CudaStream>,
    kernel_cache: &mut LruCache<String, (Arc<CudaModule>, CudaFunction)>,
) -> (
    CudaFunction,
    FxHashMap<char, CudaSlice<u8>>,
    Expression,
    TaskQueue,
    FxHashMap<NodeIndex, usize>,
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

    // Calculate actual max payload size from the ops being used
    let max_payload_size = block_ops
        .iter()
        .map(|op| {
            op.build_payload(cuda_stream, CStruct::new(Some(&temp_expression_map)))
                .finish_struct()
                .len()
        })
        .max()
        .unwrap_or(0);

    let (interpreter, expressions, interpreter_constants) = compile_interpreter(
        cuda_stream,
        &block_ops,
        &expressions,
        max_payload_size,
        kernel_cache,
    );

    // Build task queue with dynamic payload size
    let mut tasks = TaskQueue::new(max_payload_size);
    let mut node_to_task_index = FxHashMap::default();
    for node in toposort(&llir_graph, None).unwrap() {
        if !subgraph.contains(&node) {
            continue;
        }
        let sources = llir_graph
            .edges_directed(node, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .map(|e| e.source())
            .collect_vec();
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
            [null(); 3],
            null_mut(),
            &payload,
            &expressions,
        );
    }
    (
        interpreter,
        interpreter_constants,
        n_barriers,
        tasks,
        node_to_task_index,
    )
}

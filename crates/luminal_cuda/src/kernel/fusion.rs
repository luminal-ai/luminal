use std::sync::Arc;

use crate::{cuda_dtype, kernel::KernelOp};
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::compile_ptx,
};
use itertools::Itertools;
use luminal::{op::OpParam::*, op::*, prelude::*};

/// Trait for elementwise kernel ops that can participate in fusion.
///
/// Each fusable op describes its shape, strides, and how to compute
/// a single output element from its input values as a CUDA expression.
pub trait ElementwiseFusable {
    fn elem_out_shape(&self) -> &[Expression];
    fn elem_out_strides(&self) -> &[Expression];
    /// Returns strides for each input, in edge order.
    fn elem_input_strides(&self) -> Vec<&[Expression]>;
    fn elem_dtype(&self) -> DType;
    /// Given CUDA expressions for each input value, return the CUDA expression
    /// for this op's result. E.g. for Add: `format!("({} + {})", inputs[0], inputs[1])`
    fn elem_cuda_expr(&self, input_exprs: &[String]) -> String;
}

/// One step in the fused kernel's SSA program.
#[derive(Debug, Clone)]
pub enum FusedStep {
    /// Load from an external input parameter using the given strides.
    Load { strides: Vec<Expression> },
    /// Compute an expression from previous step results.
    /// `cuda_template` has placeholders like `{0}`, `{1}` for operand refs.
    Compute {
        cuda_template: String,
        operands: Vec<usize>,
    },
}

/// A fused elementwise kernel that combines multiple elementwise ops
/// into a single CUDA kernel.
#[derive(Debug, Clone)]
pub struct KernelFusedElementwise {
    out_shape: Vec<Expression>,
    out_strides: Vec<Expression>,
    num_inputs: usize,
    dtype: DType,
    steps: Vec<FusedStep>,
    output_step: usize,
}

impl EgglogOp for KernelFusedElementwise {
    fn term(&self) -> (String, Vec<OpParam>) {
        ("KernelFusedElem".to_string(), vec![])
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        _egraph: &'a SerializedEGraph,
        _children: &[&'a ENodeId],
        _list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        _expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        unreachable!(
            "KernelFusedElementwise is created post-extraction and should never be extracted from egglog"
        )
    }
}

impl KernelOp for KernelFusedElementwise {
    fn compile(
        &self,
        stream: &Arc<CudaStream>,
    ) -> (
        CudaFunction,
        Arc<CudaModule>,
        String,
        (Expression, Expression, Expression),
        (Expression, Expression, Expression),
        Expression,
        FxHashMap<char, CudaSlice<u8>>,
    ) {
        // Collect all dynamic variables from shapes and strides
        let mut vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(self.out_strides.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();

        for step in &self.steps {
            if let FusedStep::Load { strides } = step {
                vars.extend(strides.iter().flat_map(|e| e.dyn_vars()));
            }
        }

        let dtype = cuda_dtype(self.dtype);

        // Build parameter list: output first, then inputs
        let params = (0..self.num_inputs)
            .map(|i| format!("const {dtype} *inp_{i}"))
            .join(", ");

        // Build kernel body
        let mut body_lines = Vec::new();
        let mut load_param_idx = 0usize;
        for (step_idx, step) in self.steps.iter().enumerate() {
            match step {
                FusedStep::Load { strides } => {
                    let index_expr = flatten_mul_strides(&self.out_shape, strides).to_kernel();
                    body_lines.push(format!(
                        "        {dtype} t{step_idx} = inp_{load_param_idx}[{index_expr}];"
                    ));
                    load_param_idx += 1;
                }
                FusedStep::Compute {
                    cuda_template,
                    operands,
                } => {
                    // Substitute operand references into template
                    let mut expr = cuda_template.clone();
                    for (j, &op_idx) in operands.iter().enumerate() {
                        expr = expr.replace(&format!("{{{j}}}"), &format!("t{op_idx}"));
                    }
                    body_lines.push(format!("        {dtype} t{step_idx} = {expr};"));
                }
            }
        }

        let out_index_expr = flatten_mul_strides(&self.out_shape, &self.out_strides).to_kernel();
        body_lines.push(format!(
            "        out[{out_index_expr}] = t{};",
            self.output_step
        ));

        let kernel = format!(
            "\
{constants}
extern \"C\" {{
    __global__ void fused_elem_k({dtype} *out, {params}) {{
        int const_z = blockIdx.x * blockDim.x + threadIdx.x;
{body}
    }}
}}",
            constants = vars
                .iter()
                .map(|i| format!("__constant__ int const_{i}[1];"))
                .join("\n"),
            body = body_lines.join("\n"),
        );

        let ptx = compile_ptx(&kernel).unwrap();
        let module = stream.context().load_module(ptx).unwrap();
        let func = module.load_function("fused_elem_k").unwrap();
        let constants = vars
            .into_iter()
            .map(|d| (d, module.get_global(&format!("const_{d}"), stream).unwrap()))
            .collect();

        let out_size = self.out_shape.iter().copied().product::<Expression>();
        (
            func,
            module,
            kernel,
            (out_size.ceil_div(128), 1.into(), 1.into()),
            (out_size.min(128), 1.into(), 1.into()),
            0.into(),
            constants,
        )
    }

    fn output_size(&self) -> Expression {
        self.out_shape.iter().copied().product()
    }

    fn bytes_loaded(&self) -> Expression {
        // Each external input loads output_size elements * 4 bytes
        self.output_size() * 4 * (self.num_inputs as i32)
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        // Count compute steps as 1 FLOP each per element
        let n_compute = self
            .steps
            .iter()
            .filter(|s| matches!(s, FusedStep::Compute { .. }))
            .count();
        self.output_size() * (n_compute as i32)
    }

    fn kernel_name(&self) -> &'static str {
        "FusedElementwise"
    }
}

/// Fuse chains of elementwise CUDA kernel ops in the LLIR graph.
///
/// This operates as a post-extraction graph transformation pass.
/// Fusable ops (those implementing `ElementwiseFusable` via `as_fusable()`)
/// are grouped and replaced with a single `KernelFusedElementwise` node
/// when they form a chain with compatible shapes and strides.
pub fn fuse_elementwise(graph: &mut LLIRGraph) {
    use petgraph::visit::EdgeRef;
    use petgraph::{Direction, algo::toposort};

    let topo = match toposort(&*graph, None) {
        Ok(t) => t,
        Err(_) => return,
    };

    // Build a set of fusable nodes
    let fusable_nodes: FxHashSet<NodeIndex> = graph
        .node_indices()
        .filter(|&n| {
            graph[n]
                .to_dialect::<dyn KernelOp>()
                .and_then(|op| op.as_fusable())
                .is_some()
        })
        .collect();

    if fusable_nodes.len() < 2 {
        return;
    }

    // Identify fusable edges: producer -> consumer where both are fusable,
    // same output shape, stride compatibility, and producer has exactly 1 consumer.
    let mut fusable_edges: Vec<(NodeIndex, NodeIndex)> = Vec::new();

    for &producer in &fusable_nodes {
        // Check producer has exactly 1 outgoing edge to another node
        let outgoing: Vec<_> = graph
            .edges_directed(producer, Direction::Outgoing)
            .collect();
        if outgoing.len() != 1 {
            continue;
        }
        let consumer = outgoing[0].target();
        if !fusable_nodes.contains(&consumer) {
            continue;
        }

        let prod_op = graph[producer].to_dialect::<dyn KernelOp>().unwrap();
        let cons_op = graph[consumer].to_dialect::<dyn KernelOp>().unwrap();

        let prod_fusable = prod_op.as_fusable().unwrap();
        let cons_fusable = cons_op.as_fusable().unwrap();

        // Shape must match
        if prod_fusable.elem_out_shape() != cons_fusable.elem_out_shape() {
            continue;
        }

        // Find which input slot of consumer this edge corresponds to
        let edge_id = outgoing[0].id();
        let consumer_inputs: Vec<_> = graph
            .edges_directed(consumer, Direction::Incoming)
            .sorted_by_key(|e| e.id())
            .collect();
        let slot = consumer_inputs.iter().position(|e| e.id() == edge_id);
        let Some(slot) = slot else { continue };

        let cons_input_strides = cons_fusable.elem_input_strides();
        if slot >= cons_input_strides.len() {
            continue;
        }

        // Stride compatibility: producer's output strides must match
        // consumer's input strides at this slot
        if prod_fusable.elem_out_strides() != cons_input_strides[slot] {
            continue;
        }

        fusable_edges.push((producer, consumer));
    }

    if fusable_edges.is_empty() {
        return;
    }

    // Union-Find to group fusable nodes
    let mut parent: FxHashMap<NodeIndex, NodeIndex> = FxHashMap::default();
    let find = |parent: &mut FxHashMap<NodeIndex, NodeIndex>, mut x: NodeIndex| -> NodeIndex {
        while let Some(&p) = parent.get(&x) {
            if p == x {
                break;
            }
            x = p;
        }
        x
    };

    for &(a, b) in &fusable_edges {
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra != rb {
            parent.insert(rb, ra);
        }
        // Path compression
        parent.entry(a).or_insert(a);
        parent.entry(b).or_insert(b);
    }

    // Collect groups
    let mut groups: FxHashMap<NodeIndex, Vec<NodeIndex>> = FxHashMap::default();
    for &node in &fusable_nodes {
        if parent.contains_key(&node) {
            let root = find(&mut parent, node);
            groups.entry(root).or_default().push(node);
        }
    }

    // Filter to groups with 2+ nodes, sort each group in topo order
    let topo_pos: FxHashMap<NodeIndex, usize> =
        topo.iter().enumerate().map(|(i, &n)| (n, i)).collect();

    let mut fusion_groups: Vec<Vec<NodeIndex>> =
        groups.into_values().filter(|g| g.len() >= 2).collect();

    for group in &mut fusion_groups {
        group.sort_by_key(|n| topo_pos.get(n).copied().unwrap_or(usize::MAX));
    }

    // Process each fusion group
    for group in fusion_groups {
        let group_set: FxHashSet<NodeIndex> = group.iter().copied().collect();

        // Check that all fusable edges within the group still form a valid chain.
        // Also verify all internal edges are fusable (single consumer within group).
        let mut valid = true;
        for &node in &group {
            let outgoing: Vec<_> = graph.edges_directed(node, Direction::Outgoing).collect();
            // If node has outgoing edges to group members, it must have exactly 1 total outgoing edge
            let has_group_consumer = outgoing.iter().any(|e| group_set.contains(&e.target()));
            if has_group_consumer && outgoing.len() != 1 {
                valid = false;
                break;
            }
        }
        if !valid {
            continue;
        }

        // Find the group's output node (the one with no outgoing edge to another group member)
        let output_node = group
            .iter()
            .find(|&&n| {
                !graph
                    .edges_directed(n, Direction::Outgoing)
                    .any(|e| group_set.contains(&e.target()))
            })
            .copied();
        let Some(output_node) = output_node else {
            continue;
        };

        // Get the output node's fusable info for the final shape/strides/dtype
        let output_op = match graph[output_node].to_dialect::<dyn KernelOp>() {
            Some(op) => op.clone(),
            None => continue,
        };
        let output_fusable = match output_op.as_fusable() {
            Some(f) => f,
            None => continue,
        };

        let out_shape = output_fusable.elem_out_shape().to_vec();
        let out_strides = output_fusable.elem_out_strides().to_vec();
        let dtype = output_fusable.elem_dtype();

        // Build the SSA program by walking the group in topological order
        let mut steps: Vec<FusedStep> = Vec::new();
        let mut node_to_step: FxHashMap<NodeIndex, usize> = FxHashMap::default();
        let mut external_sources: Vec<NodeIndex> = Vec::new();

        for &node in &group {
            let node_op = match graph[node].to_dialect::<dyn KernelOp>() {
                Some(op) => op.clone(),
                None => continue,
            };
            let node_fusable = match node_op.as_fusable() {
                Some(f) => f,
                None => continue,
            };

            let input_strides_list = node_fusable.elem_input_strides();
            let incoming: Vec<_> = graph
                .edges_directed(node, Direction::Incoming)
                .sorted_by_key(|e| e.id())
                .collect();

            let mut operand_step_indices = Vec::new();

            for (input_idx, edge) in incoming.iter().enumerate() {
                let source = edge.source();
                if group_set.contains(&source) {
                    // Internal: use the step that produces this source's output
                    operand_step_indices.push(node_to_step[&source]);
                } else {
                    // External input: emit a Load step if we haven't already for this source+slot
                    // Each edge from the same external source may have different strides,
                    // so we always create a new load.
                    //
                    // But actually, for correct param ordering, each external edge
                    // maps to a unique input parameter.
                    let strides = if input_idx < input_strides_list.len() {
                        input_strides_list[input_idx].to_vec()
                    } else {
                        // fallback: shouldn't happen with correct ops
                        continue;
                    };

                    let step_idx = steps.len();
                    steps.push(FusedStep::Load { strides });
                    external_sources.push(source);
                    operand_step_indices.push(step_idx);
                }
            }

            // Build the compute expression with placeholder references
            let input_exprs: Vec<String> = (0..operand_step_indices.len())
                .map(|j| format!("{{{j}}}"))
                .collect();
            let cuda_template = node_fusable.elem_cuda_expr(&input_exprs);

            let step_idx = steps.len();
            steps.push(FusedStep::Compute {
                cuda_template,
                operands: operand_step_indices,
            });
            node_to_step.insert(node, step_idx);
        }

        let output_step = node_to_step[&output_node];

        let fused_op = KernelFusedElementwise {
            out_shape,
            out_strides,
            num_inputs: external_sources.len(),
            dtype,
            steps,
            output_step,
        };

        // Create the fused node in the graph
        let fused_node = graph.add_node(LLIROp::new::<dyn KernelOp>(
            Box::new(fused_op) as Box<dyn KernelOp>
        ));

        // Add edges from external sources to fused node (in order)
        for &source in &external_sources {
            graph.add_edge(source, fused_node, ());
        }

        // Redirect outgoing edges from the output node to the fused node
        let outgoing_edges: Vec<_> = graph
            .edges_directed(output_node, Direction::Outgoing)
            .map(|e| (e.id(), e.target()))
            .collect();
        for (edge_id, target) in outgoing_edges {
            graph.remove_edge(edge_id);
            graph.add_edge(fused_node, target, ());
        }

        // Remove all group nodes (and their edges)
        for &node in &group {
            graph.remove_node(node);
        }
    }
}

use std::sync::Arc;

use crate::{cuda_dtype, kernel::KernelOp};
use cudarc::{
    driver::{CudaFunction, CudaModule, CudaSlice, CudaStream},
    nvrtc::compile_ptx,
};
use itertools::Itertools;
use luminal::{
    graph::{extract_dtype, extract_expr_list},
    op::OpParam::*,
    op::*,
    prelude::*,
};

/// Reference to either a leaf input or a temporary (intermediate) value.
#[derive(Debug, Clone, Copy)]
enum Ref {
    Input(usize),
    Temp(usize),
}

/// A single fused computation step.
#[derive(Debug, Clone)]
struct FusedStep {
    op_code: i32,
    lhs: Ref,
    rhs: Ref,
}

/// Fused elementwise kernel op combining two binary elementwise ops into one kernel launch.
///
/// Has 3 leaf inputs (the intermediate buffer is eliminated).
/// The `steps` field encodes the two operations to perform.
#[derive(Default, Debug, Clone)]
pub struct ElementwiseFusedOp {
    out_shape: Vec<Expression>,
    out_stride: Vec<Expression>,
    input_strides: Vec<Vec<Expression>>,
    steps: Vec<FusedStep>,
    dtype: DType,
}

fn op_code_to_cuda(code: i32) -> &'static str {
    match code {
        0 => "+",
        1 => "*",
        _ => panic!("unknown elementwise op code {code}"),
    }
}

/// Known elementwise ops: (kernel term name, op code)
const ELEMENTWISE_OPS: &[(&str, i32)] = &[("KernelAdd", 0), ("KernelMul", 1)];

impl EgglogOp for ElementwiseFusedOp {
    fn term(&self) -> (String, Vec<OpParam>) {
        (
            "ElementwiseFused".to_string(),
            vec![
                EList, // 0: shape
                Input, EList, // 1,2: inp0, str0
                Input, EList, // 3,4: inp1, str1
                Input, EList, // 5,6: inp2, str2
                EList, // 7: out_strides
                Dty,   // 8: dtype
                Int, Int, Int, // 9,10,11: inner_op, outer_op, side
            ],
        )
    }

    fn rewrites(&self) -> Vec<String> {
        let mut rules = Vec::new();

        for &(outer_name, outer_code) in ELEMENTWISE_OPS {
            for &(inner_name, inner_code) in ELEMENTWISE_OPS {
                // Inner on LEFT side of outer: outer(inner(x, y), b)
                rules.push(format!(
                    "
(rule
    (
        (= ?result ({outer_name} ?shape ?a ?a_str ?b ?b_str ?out_str ?dty))
        (= ?a ({inner_name} ?m_shape ?x ?x_str ?y ?y_str ?m_out_str ?m_dty))
    )
    (
        (let ?fused (ElementwiseFused ?shape
             ?x ?x_str ?y ?y_str ?b ?b_str
             ?out_str ?dty {inner_code} {outer_code} 0))
        (union ?result ?fused)
        (set (dtype ?fused) ?dty)
    )
    :name \"fuse {outer_name}({inner_name}_left)\"
)"
                ));

                // Inner on RIGHT side of outer: outer(a, inner(x, y))
                rules.push(format!(
                    "
(rule
    (
        (= ?result ({outer_name} ?shape ?a ?a_str ?b ?b_str ?out_str ?dty))
        (= ?b ({inner_name} ?m_shape ?x ?x_str ?y ?y_str ?m_out_str ?m_dty))
    )
    (
        (let ?fused (ElementwiseFused ?shape
             ?a ?a_str ?x ?x_str ?y ?y_str
             ?out_str ?dty {inner_code} {outer_code} 1))
        (union ?result ?fused)
        (set (dtype ?fused) ?dty)
    )
    :name \"fuse {outer_name}({inner_name}_right)\"
)"
                ));
            }
        }

        rules
    }

    fn cleanup(&self) -> bool {
        false
    }

    fn extract<'a>(
        &self,
        egraph: &'a SerializedEGraph,
        children: &[&'a ENodeId],
        list_cache: &mut FxHashMap<&'a ENodeId, Vec<Expression>>,
        expr_cache: &mut FxHashMap<&'a ENodeId, Expression>,
    ) -> (LLIROp, Vec<&'a ENodeId>) {
        let out_shape = extract_expr_list(egraph, children[0], list_cache, expr_cache).unwrap();
        let str0 = extract_expr_list(egraph, children[2], list_cache, expr_cache).unwrap();
        let str1 = extract_expr_list(egraph, children[4], list_cache, expr_cache).unwrap();
        let str2 = extract_expr_list(egraph, children[6], list_cache, expr_cache).unwrap();
        let out_stride = extract_expr_list(egraph, children[7], list_cache, expr_cache).unwrap();
        let dtype = extract_dtype(egraph, children[8]);
        let inner_op = egraph.enodes[children[9]].0.parse::<i32>().unwrap();
        let outer_op = egraph.enodes[children[10]].0.parse::<i32>().unwrap();
        let side = egraph.enodes[children[11]].0.parse::<i32>().unwrap();

        let steps = if side == 0 {
            // inner is lhs of outer: step0 = inner_op(inp0, inp1), step1 = outer_op(temp0, inp2)
            vec![
                FusedStep {
                    op_code: inner_op,
                    lhs: Ref::Input(0),
                    rhs: Ref::Input(1),
                },
                FusedStep {
                    op_code: outer_op,
                    lhs: Ref::Temp(0),
                    rhs: Ref::Input(2),
                },
            ]
        } else {
            // inner is rhs of outer: step0 = inner_op(inp1, inp2), step1 = outer_op(inp0, temp0)
            vec![
                FusedStep {
                    op_code: inner_op,
                    lhs: Ref::Input(1),
                    rhs: Ref::Input(2),
                },
                FusedStep {
                    op_code: outer_op,
                    lhs: Ref::Input(0),
                    rhs: Ref::Temp(0),
                },
            ]
        };

        let op = ElementwiseFusedOp {
            out_shape,
            out_stride,
            input_strides: vec![str0, str1, str2],
            steps,
            dtype,
        };

        (
            LLIROp::new::<dyn KernelOp>(Box::new(op) as Box<dyn KernelOp>),
            vec![children[1], children[3], children[5]], // 3 leaf input enodes
        )
    }
}

impl KernelOp for ElementwiseFusedOp {
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
        let vars = self
            .out_shape
            .iter()
            .flat_map(|e| e.dyn_vars())
            .chain(
                self.input_strides
                    .iter()
                    .flat_map(|s| s.iter().flat_map(|e| e.dyn_vars())),
            )
            .chain(self.out_stride.iter().flat_map(|e| e.dyn_vars()))
            .collect::<FxHashSet<_>>();

        let dtype = cuda_dtype(self.dtype);
        let n_inputs = self.input_strides.len();

        // Build input parameter list: "const float *inp0, const float *inp1, ..."
        let input_params = (0..n_inputs)
            .map(|i| format!("const {dtype} *inp{i}"))
            .join(", ");

        // Build index expressions for each input
        let input_indices: Vec<String> = self
            .input_strides
            .iter()
            .map(|strides| {
                flatten_mul_strides(&self.out_shape, strides)
                    .to_kernel()
                    .to_string()
            })
            .collect();

        let out_index = flatten_mul_strides(&self.out_shape, &self.out_stride)
            .to_kernel()
            .to_string();

        // Build kernel body: each step produces a temp or the final output
        let mut body_lines = Vec::new();
        for (i, step) in self.steps.iter().enumerate() {
            let lhs_expr = match step.lhs {
                Ref::Input(idx) => format!("inp{idx}[{expr}]", expr = input_indices[idx]),
                Ref::Temp(idx) => format!("t{idx}"),
            };
            let rhs_expr = match step.rhs {
                Ref::Input(idx) => format!("inp{idx}[{expr}]", expr = input_indices[idx]),
                Ref::Temp(idx) => format!("t{idx}"),
            };
            let op_str = op_code_to_cuda(step.op_code);

            if i == self.steps.len() - 1 {
                // Last step writes to output
                body_lines.push(format!(
                    "        out[{out_index}] = {lhs_expr} {op_str} {rhs_expr};"
                ));
            } else {
                // Intermediate step writes to a temp
                body_lines.push(format!(
                    "        {dtype} t{i} = {lhs_expr} {op_str} {rhs_expr};"
                ));
            }
        }

        let kernel = format!(
            "
{constants}
extern \"C\" {{
    __global__ void fused_elem_k({dtype} *out, {input_params}) {{
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
        self.output_size() * 4 * self.input_strides.len() as i32
    }

    fn bytes_stored(&self) -> Expression {
        self.output_size() * 4
    }

    fn flops(&self) -> Expression {
        self.output_size() * self.steps.len() as i32
    }

    fn kernel_name(&self) -> &'static str {
        "FusedElementwise"
    }

    fn elementwise(&self) -> bool {
        true
    }
}

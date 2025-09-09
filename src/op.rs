use std::{
    any::Any,
    borrow::BorrowMut,
    fmt::Debug,
    sync::{Arc, Mutex},
};

use crate::prelude::*;

use dyn_clone::{clone_trait_object, DynClone};
use rustc_hash::FxHashMap;

/// A tensor with data. The data can be anything that implements the Data trait
#[derive(Debug, Clone)]
pub struct Tensor {
    data: Box<dyn Data>,
}

impl Tensor {
    pub fn new<T: Data>(data: T) -> Self {
        Self {
            data: Box::new(data),
        }
    }
    pub fn downcast_ref<T: Data>(&self) -> Option<&T> {
        self.data.as_any().downcast_ref()
    }
    pub fn downcast_mut<T: Data>(&mut self) -> Option<&mut T> {
        self.data.as_any_mut().downcast_mut()
    }
    pub fn is<T: Data>(&self) -> bool {
        self.data.as_any().is::<T>()
    }
}

/// Some sort of data, for instance a Vec<f32> on CPU, CudaSlice<f32> on Nvidia GPUs, or metal::Buffer for Apple GPUs
pub trait Data: Any + Debug + DynClone {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

clone_trait_object!(Data);

impl Data for Vec<f32> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Either an owned or borrowed tensor that gets consumed by ops
pub enum InputTensor<'a> {
    /// An owned tensor
    Owned(Tensor),
    /// A borrowed tensor
    Borrowed(&'a Tensor),
}

impl<'a> InputTensor<'a> {
    /// Borrow the tensor
    pub fn borrowed(&'a self) -> &'a Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t,
        }
    }

    /// Unwrap or clone the tensor, depending on if it's owned or not
    pub fn cloned(self) -> Tensor {
        match self {
            InputTensor::Owned(t) => t,
            InputTensor::Borrowed(t) => t.clone(),
        }
    }
}

/// The main operator trait.
///
/// Defines an operator that takes in a vector of input tensors and shapes and produces a vector of output tensors
pub trait Operator: Debug + as_any::AsAny {
    /// Process the input tensors and produce output tensors
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>;
    /// Implement custom functionality
    #[allow(unused)]
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        None
    }
}

impl<T: Operator> Operator for Box<T> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self, key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self, inp)
    }
}
impl<T: Operator> Operator for Arc<Mutex<T>> {
    fn custom(&mut self, key: &str, input: Box<dyn Any>) -> Option<Box<dyn Any>> {
        <T as Operator>::custom(self.lock().unwrap().borrow_mut(), key, input)
    }
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        <T as Operator>::process(self.lock().unwrap().borrow_mut(), inp)
    }
}

/// An opaque function running on CPU that takes in Vec<f32> tensors and outputs Vec<f32> tensors
#[allow(clippy::type_complexity)]
pub struct Function(
    pub String,
    pub Box<dyn Fn(Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor>>,
);

impl PartialEq for Function {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Operator for Function {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        (self.1)(inp)
    }
}

impl Debug for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A constant value placed on the graph at runtime. Can either be an expression evaluated at runtime, or a constant float
#[derive(Debug, Clone, PartialEq)]
pub enum ConstantValue {
    Expression(Expression),
    Float(f32),
}

impl From<f32> for ConstantValue {
    fn from(value: f32) -> Self {
        ConstantValue::Float(value)
    }
}
impl From<f64> for ConstantValue {
    fn from(value: f64) -> Self {
        ConstantValue::Float(value as f32)
    }
}
impl<T: Into<Expression>> From<T> for ConstantValue {
    fn from(value: T) -> Self {
        ConstantValue::Expression(value.into())
    }
}

/// Produces a single number constant from an expression or a float
#[derive(Clone, PartialEq)]
pub struct Constant(pub ConstantValue, pub *const FxHashMap<char, usize>);
impl Debug for Constant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Constant(",)?;
        match &self.0 {
            ConstantValue::Expression(e) => e.fmt(f)?,
            ConstantValue::Float(fl) => fl.fmt(f)?,
        }
        write!(f, ")")
    }
}

impl Operator for Constant {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        vec![Tensor::new(vec![match &self.0 {
            ConstantValue::Expression(e) => {
                e.exec_float(unsafe { self.1.as_ref().unwrap() }).unwrap() as f32
            }
            ConstantValue::Float(f) => *f,
        }])]
    }
}

/// Graph break for chunking search graphs
#[derive(Clone, PartialEq)]
pub struct GraphBreak;
impl Debug for GraphBreak {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GraphBreak")
    }
}

impl Operator for GraphBreak {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        inp.into_iter().map(|(t, _)| t.cloned()).collect() // inefficient, but we don't care as this won't execute on the kernel
    }
}

// Unary Op (A -> A)

/// Ensure a tensor is contiguously layed out in memory. May involve copying
#[derive(Debug, Clone, PartialEq)]
pub struct Contiguous;
impl Operator for Contiguous {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        // Copy data over to new tensor
        let inp_data = get_vec(&inp[0].0);
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Log2;
impl Operator for Log2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).log2();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Exp2;
impl Operator for Exp2 {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).exp2();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sin;
impl Operator for Sin {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sin();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recip;
impl Operator for Recip {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).recip();
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Sqrt;
impl Operator for Sqrt {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let inp_data = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(inp_data, &expr, &mut stack, i).sqrt();
        }
        vec![Tensor::new(out_data)]
    }
}

// Binary Ops (A x A -> A)

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Add;
impl Operator for Add {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) + get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mul;
impl Operator for Mul {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) * get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Mod;
impl Operator for Mod {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = get_index(lhs, &lexpr, &mut stack, i) % get_index(rhs, &rexpr, &mut stack, i);
        }
        vec![Tensor::new(out_data)]
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LessThan;
impl Operator for LessThan {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let (lhs, rhs) = (get_vec(&inp[0].0), get_vec(&inp[1].0));
        let mut out_data = vec![0.; inp[0].1.n_elements().to_usize().unwrap()];
        let lexpr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let rexpr = (inp[1].1.index_expression(), inp[1].1.valid_expression());
        let mut stack = vec![];
        for (i, out) in out_data.iter_mut().enumerate() {
            *out = (get_index(lhs, &lexpr, &mut stack, i) < get_index(rhs, &rexpr, &mut stack, i))
                as i32 as f32;
        }
        vec![Tensor::new(out_data)]
    }
}

// Reduce Ops (A -> B (different shape))

#[derive(Debug, Clone, PartialEq)]
pub struct SumReduce(pub usize);
impl Operator for SumReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![0.0; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];
        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    result[i * back_size + j] += get_index(input, &expr, &mut stack, orig_index);
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MaxReduce(pub usize);
impl Operator for MaxReduce {
    fn process(&mut self, inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let sh = inp[0].1.shape_usize();
        let front_size = sh.iter().take(self.0).product::<usize>().max(1);
        let back_size = sh.iter().skip(self.0 + 1).product::<usize>().max(1);
        let dim_size = sh[self.0];
        let mut result = vec![-f32::INFINITY; front_size * back_size];
        let input = get_vec(&inp[0].0);
        let expr = (inp[0].1.index_expression(), inp[0].1.valid_expression());
        let mut stack = vec![];

        for i in 0..front_size {
            for j in 0..back_size {
                for k in 0..dim_size {
                    let orig_index = i * dim_size * back_size + k * back_size + j;
                    let new_index = i * back_size + j;
                    result[new_index] =
                        result[new_index].max(get_index(input, &expr, &mut stack, orig_index));
                }
            }
        }
        vec![Tensor::new(result)]
    }
}

fn get_vec<'a>(tensor: &'a InputTensor<'a>) -> &'a Vec<f32> {
    tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap()
}

fn get_index(
    data: &[f32],
    (ind, val): &(Expression, Expression),
    stack: &mut Vec<i64>,
    index: usize,
) -> f32 {
    if val.exec_single_var_stack(index, stack) != 0 {
        let i = ind.exec_single_var_stack(index, stack);
        data[i]
    } else {
        0.0
    }
}

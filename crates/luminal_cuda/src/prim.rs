use crate::{compile_and_load_kernel, get_buffer_from_tensor, input_dyn_dims, CudaData, CudaFloat};

use super::{get_idx_valid_exps, render_dyn_dim_inputs};
use itertools::Itertools;
use rustc_hash::FxHashMap;

use std::{
    any::{Any, TypeId},
    marker::PhantomData,
    sync::Arc,
};

use cudarc::driver::{CudaContext, CudaFunction, LaunchConfig, PushKernelArg};

use luminal::{
    op::{Function as LFunction, *},
    prelude::{petgraph::visit::EdgeRef, *},
};

/// Copy a tensor to the GPU
#[derive(Clone)]
pub struct CudaCopyToDevice<T>(Arc<CudaContext>, PhantomData<T>);
crate::debug_type!(CudaCopyToDevice);

impl<T> CudaCopyToDevice<T> {
    pub fn new(dev: Arc<CudaContext>) -> Self {
        CudaCopyToDevice(dev, Default::default())
    }
}

impl<T: CudaFloat> Operator for CudaCopyToDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<CudaData<T>>() || inp[0].0.borrowed().is::<CudaData<u8>>() {
            // Already on device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let cpu_data = inp[0].0.borrowed().downcast_ref::<Vec<f32>>().unwrap();
        let vec = cpu_data
            .iter()
            .copied()
            .map(T::from_f32)
            .collect::<Vec<_>>();
        vec![Tensor::new(CudaData(
            self.0.default_stream().memcpy_stod(&vec).unwrap(),
        ))]
    }
}

/// Copy a tensor from the GPU
#[derive(Clone)]
pub struct CudaCopyFromDevice<T>(Arc<CudaContext>, PhantomData<T>);
crate::debug_type!(CudaCopyFromDevice);

impl<T> CudaCopyFromDevice<T> {
    pub fn new(dev: Arc<CudaContext>) -> Self {
        CudaCopyFromDevice(dev, Default::default())
    }
}

impl<T: CudaFloat> Operator for CudaCopyFromDevice<T> {
    fn process(&mut self, mut inp: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        if inp[0].0.borrowed().is::<Vec<f32>>() {
            // Already off device
            return vec![inp.pop().unwrap().0.cloned()];
        }
        let buf = self
            .0
            .default_stream()
            .memcpy_dtov(get_buffer_from_tensor::<T>(&inp[0].0))
            .unwrap();
        vec![Tensor::new(
            buf.into_iter().map(T::to_f32).collect::<Vec<_>>(),
        )]
    }
}

/// Constant value on device
#[derive(Clone)]
pub struct CudaConstant<T> {
    pub value: ConstantValue,
    device: Arc<CudaContext>,
    dyn_map: *const FxHashMap<char, usize>,
    _phantom: PhantomData<T>,
}
impl<T> core::fmt::Debug for CudaConstant<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CudaConstant({:?})", self.value)
    }
}

impl<T> CudaConstant<T> {
    pub fn new(
        device: Arc<CudaContext>,
        value: ConstantValue,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        Self {
            value,
            device,
            dyn_map,
            _phantom: Default::default(),
        }
    }
}

impl<T: CudaFloat> Operator for CudaConstant<T> {
    fn process(&mut self, _: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let stream = self.device.default_stream();
        let mut a = unsafe { stream.alloc::<T>(1).unwrap() };
        let value = match &self.value {
            ConstantValue::Expression(e) => {
                T::from_f32(e.exec(unsafe { self.dyn_map.as_ref().unwrap() }).unwrap() as f32)
            }
            ConstantValue::Float(f) => T::from_f32(*f),
        };
        stream.memcpy_htod(&vec![value], &mut a).unwrap();
        vec![Tensor::new(CudaData(a))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            if let ConstantValue::Float(f) = self.value {
                return Some(Box::new(format!("{f:?}")));
            }
        }
        None
    }
}

#[macro_export]
macro_rules! cuda_unary_op {
    ($op: expr, $op_name: ident) => {
        #[derive(Clone)]
        pub struct $op_name<T> {
            function: CudaFunction,
            device: Arc<cudarc::driver::CudaContext>,
            dyn_symbols: Vec<char>,
            dyn_map: *const FxHashMap<char, usize>,
            _phantom: PhantomData<T>,
        }

        impl<T: CudaFloat> $op_name<T> {
            pub fn new(
                shape: ShapeTracker,
                device: Arc<cudarc::driver::CudaContext>,
                dyn_map: *const FxHashMap<char, usize>,
            ) -> Self {
                let (idx_exp, valid_exp) = get_idx_valid_exps(shape);
                let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
                let type_name = T::type_name();
                let code = format!(
                    "
    #include <cuda_fp16.h>
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, int numel{rendered}) {{
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < numel && {valid_exp} != 0) {{
                out[idx] = {}(inp[{idx_exp}]);
            }}
        }}", $op
                );
                Self {
                    function: compile_and_load_kernel(code, &device),
                    device,
                    dyn_symbols,
                    dyn_map,
                    _phantom: Default::default(),
                }
            }
        }

        impl<T: CudaFloat> Operator for $op_name<T> {
            fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
            	use cudarc::driver::PushKernelArg;
                let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
                let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
                let stream = self.device.default_stream();
                let mut out = unsafe { stream.alloc::<T>(inp_size).unwrap() };
                let mut launch_args = stream.launch_builder(&self.function);
                launch_args.arg(&mut out);
                launch_args.arg(inp);
                launch_args.arg(&inp_size);
                input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
                unsafe {
                    launch_args
                        .launch(LaunchConfig::for_num_elems(inp_size as u32))
                        .unwrap();
                }

                vec![Tensor::new(CudaData(out))]
            }

            fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
                if key == "elementwise" {
                    return Some(Box::new(format!("{}(input0)", $op)));
                }

                None
            }
        }

        $crate::debug_type!($op_name);
    };
}

cuda_unary_op!("", CudaContiguous);
cuda_unary_op!("log2", CudaLog2);
cuda_unary_op!("exp2", CudaExp2);
cuda_unary_op!(if T::is_f32() { "sqrt" } else { "hsqrt" }, CudaSqrt);
cuda_unary_op!("sin", CudaSin);
cuda_unary_op!(if T::is_f32() { "__frcp_rn" } else { "hrcp" }, CudaRecip);

#[derive(Clone)]
pub struct CudaAdd<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaAdd);

impl<T: CudaFloat> CudaAdd<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!(
            "
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] =
            (({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}])
            + (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]);
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaAdd<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = unsafe { stream.alloc::<T>(inp_size).unwrap() };
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(a);
        launch_args.arg(b);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("input0 + input1".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaMul<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaMul);

impl<T: CudaFloat> CudaMul<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = (({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}]) * (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]);
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaMul<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = unsafe { stream.alloc::<T>(inp_size).unwrap() };
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(a);
        launch_args.arg(b);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("input0 * input1".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaMod<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaMod);

impl<T: CudaFloat> CudaMod<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        out[idx] = fmod((({a_valid}) == 0 ? ({type_name})0.0 : inp_a[{a_idx}]), (({b_valid}) == 0 ? ({type_name})0.0 : inp_b[{b_idx}]));
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaMod<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = unsafe { stream.alloc::<T>(inp_size).unwrap() };
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(a);
        launch_args.arg(b);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("fmod(input0, input1)".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaLessThan<T> {
    function: CudaFunction,
    device: Arc<CudaContext>,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaLessThan);

impl<T: CudaFloat> CudaLessThan<T> {
    pub fn new(
        a_shape: ShapeTracker,
        b_shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (a_idx, a_valid) = get_idx_valid_exps(a_shape);
        let (b_idx, b_valid) = get_idx_valid_exps(b_shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[a_shape, b_shape]);
        let type_name = T::type_name();
        let code = format!("
#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp_a, const {type_name} *inp_b, int numel{rendered}) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {{
        {type_name} a_t = (({a_valid}) != 0) ? inp_a[{a_idx}] : ({type_name})0.0;
        {type_name} b_t = (({b_valid}) != 0) ? inp_b[{b_idx}] : ({type_name})0.0;
        if (a_t < b_t) {{
            out[idx] = ({type_name})1.0;
        }} else {{
            out[idx] = ({type_name})0.0;
        }}
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}

impl<T: CudaFloat> Operator for CudaLessThan<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let a = get_buffer_from_tensor::<T>(&tensors[0].0);
        let b = get_buffer_from_tensor::<T>(&tensors[1].0);
        let inp_size = tensors[0].1.n_elements().to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = unsafe { stream.alloc::<T>(inp_size).unwrap() };
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(a);
        launch_args.arg(b);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }

        vec![Tensor::new(CudaData(out))]
    }

    fn custom(&mut self, key: &str, _: Box<dyn Any>) -> Option<Box<dyn Any>> {
        if key == "elementwise" {
            return Some(Box::new("(float)(input0 < input1 ? 1.0 : 0.0)".to_string()));
        }
        None
    }
}

#[derive(Clone)]
pub struct CudaSumReduce<T> {
    function: CudaFunction,
    pub device: Arc<CudaContext>,
    pub dim: usize,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaSumReduce);

impl<T: CudaFloat> CudaSumReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let type_name = T::type_name();
        let code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = 0.0;
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = reduce_value + (float)inp[{idx}];
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            dim,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}
impl<T> Operator for CudaSumReduce<T>
where
    T: CudaFloat,
    CudaData<T>: Data,
{
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.dim);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
        let front_size: usize = tensors[0]
            .1
            .dims()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .dims()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.dims()[self.dim].to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(inp_size).unwrap();
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(inp);
        launch_args.arg(&front_size);
        launch_args.arg(&back_size);
        launch_args.arg(&dim_size);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }
        vec![Tensor::new(CudaData(out))]
    }
}

#[derive(Clone)]
pub struct CudaMaxReduce<T> {
    function: CudaFunction,
    pub device: Arc<CudaContext>,
    pub dim: usize,
    _phantom: PhantomData<T>,
    dyn_symbols: Vec<char>,
    dyn_map: *const FxHashMap<char, usize>,
}
crate::debug_type!(CudaMaxReduce);

impl<T: CudaFloat> CudaMaxReduce<T> {
    pub fn new(
        dim: usize,
        shape: ShapeTracker,
        device: Arc<CudaContext>,
        dyn_map: *const FxHashMap<char, usize>,
    ) -> Self {
        let (idx, valid) = get_idx_valid_exps(shape);
        let (dyn_symbols, rendered) = render_dyn_dim_inputs(&[shape]);
        let type_name = T::type_name();
        let code = format!("#include \"cuda_fp16.h\"
extern \"C\" __global__ void kernel({type_name} *out, const {type_name} *inp, const int front_size, const int back_size, const int dim_size, int numel{rendered}) {{
    int i_ = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_ < numel) {{
        int a_ = i_ / back_size;
        int b_ = i_ % back_size;
        float reduce_value = -__int_as_float(0x7f800000);
        for (int c_ = 0; c_ < dim_size; c_++) {{
            int idx = a_ * dim_size * back_size + c_ * back_size + b_;
            if (({valid}) != 0) {{
                reduce_value = max(reduce_value, (float)inp[{idx}]);
            }}
        }}
        out[i_] = ({type_name})reduce_value;
    }}
}}");
        Self {
            function: compile_and_load_kernel(code, &device),
            device,
            dim,
            _phantom: Default::default(),
            dyn_symbols,
            dyn_map,
        }
    }
}
impl<T: CudaFloat> Operator for CudaMaxReduce<T> {
    fn process(&mut self, tensors: Vec<(InputTensor, ShapeTracker)>) -> Vec<Tensor> {
        let mut shape = tensors[0].1;
        shape.remove_dim(self.dim);
        let inp_size = shape.n_elements().to_usize().unwrap();
        let inp = get_buffer_from_tensor::<T>(&tensors[0].0);
        let front_size: usize = tensors[0]
            .1
            .dims()
            .iter()
            .take(self.dim)
            .map(|i| i.to_usize().unwrap())
            .product();
        let back_size: usize = tensors[0]
            .1
            .dims()
            .iter()
            .skip(self.dim + 1)
            .map(|i| i.to_usize().unwrap())
            .product();
        let dim_size = tensors[0].1.dims()[self.dim].to_usize().unwrap();
        let stream = self.device.default_stream();
        let mut out = stream.alloc_zeros::<T>(inp_size).unwrap();
        let mut launch_args = stream.launch_builder(&self.function);
        launch_args.arg(&mut out);
        launch_args.arg(inp);
        launch_args.arg(&front_size);
        launch_args.arg(&back_size);
        launch_args.arg(&dim_size);
        launch_args.arg(&inp_size);
        input_dyn_dims(&mut launch_args, &self.dyn_symbols, self.dyn_map);
        unsafe {
            launch_args
                .launch(LaunchConfig::for_num_elems(inp_size as u32))
                .unwrap();
        }
        vec![Tensor::new(CudaData(out))]
    }
}

/// Convert all primitive ops to cuda primitive ops, and insert copy to and from device ops
#[derive(Debug, Default)]
pub struct PrimitiveCompiler<T>(PhantomData<T>);

impl<T: CudaFloat> Compiler for PrimitiveCompiler<T> {
    type Output = ();
    fn compile<To: ToIdsMut>(&self, graph: &mut Graph, mut ids: To) {
        let dev = CudaContext::new(0).unwrap();
        // Go through the graph and insert copy ops
        // Copy function output to device and input from device
        for function_node in graph
            .node_indices()
            .filter(|n| {
                graph.node_weight(*n).unwrap().as_any().is::<Function>()
                    && graph.edges(*n).count() != 0
            })
            .collect::<Vec<_>>()
        {
            // Create copy node
            let copy_node = graph
                .add_op(CudaCopyToDevice::<T>::new(dev.clone()))
                .input(function_node, 0, ShapeTracker::new(()))
                .finish();

            // Switch outgoing edges from input to copy_node
            for (edge_id, weight, dest) in graph
                .edges_directed(function_node, petgraph::Direction::Outgoing)
                .map(|e| (e.id(), *e.weight(), e.target()))
                .filter(|(_, _, trg)| *trg != copy_node)
                .collect::<Vec<_>>()
            {
                graph.add_edge(copy_node, dest, weight);
                graph.remove_edge(edge_id);
            }

            if graph.no_delete.remove(&function_node) {
                graph.no_delete.insert(copy_node);
            }
            if let Some(v) = graph.to_retrieve.get(&function_node) {
                graph.to_retrieve.insert(copy_node, *v);
            }

            // Insert copy from device for function inputs
            for (source, edge, edge_weight) in graph
                .edges_directed(function_node, petgraph::Direction::Incoming)
                .map(|e| (e.source(), e.id(), *e.weight()))
                .collect::<Vec<_>>()
            {
                let copy_from_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(source, 0, ShapeTracker::new(()))
                    .finish();
                graph.add_edge(copy_from_node, function_node, edge_weight);
                graph.remove_edge(edge);
            }
        }

        // Copy to_retrieve from device
        for (output_node, (_, output_shape)) in graph
            .to_retrieve
            .iter()
            .map(|(a, b)| (*a, *b))
            // Filter to non-functions
            .filter(|(n, _)| !graph.node_weight(*n).unwrap().as_any().is::<LFunction>())
            .collect::<Vec<_>>()
        {
            if graph
                .node_weight(output_node)
                .unwrap()
                .as_any()
                .is::<CudaCopyToDevice<T>>()
            {
                // This output is already a copy to, instead of adding a copy from, let's remap back to the source
                let src = graph
                    .neighbors_directed(output_node, petgraph::Direction::Incoming)
                    .next()
                    .unwrap();
                graph.no_delete.remove(&output_node);
                graph.no_delete.insert(src);
                let w = graph.to_retrieve.remove(&output_node).unwrap();
                graph.to_retrieve.insert(src, w);
            } else {
                // Create copy node
                let copy_node = graph
                    .add_op(CudaCopyFromDevice::<T>::new(dev.clone()))
                    .input(output_node, 0, output_shape)
                    .finish();

                remap(output_node, copy_node, &mut ids, graph);
            }
        }

        fn is<T: Any>(type_id: TypeId) -> bool {
            type_id == TypeId::of::<T>()
        }

        // Swap primitive ops
        for id in graph.node_indices().collect::<Vec<_>>() {
            let shapes = graph
                .edges_directed(id, petgraph::Direction::Incoming)
                .filter_map(|i| i.weight().as_data())
                .sorted_by_key(|e| e.0)
                .map(|e| e.2)
                .collect::<Vec<_>>();
            let op = graph.node_weight(id).unwrap().as_any().type_id();
            let op_ref = graph.graph.node_weight_mut(id).unwrap();
            if is::<Log2>(op) {
                *op_ref = Box::new(CudaLog2::<T>::new(shapes[0], dev.clone(), &graph.dyn_map));
            } else if is::<Exp2>(op) {
                *op_ref = Box::new(CudaExp2::<T>::new(shapes[0], dev.clone(), &graph.dyn_map));
            } else if is::<Sin>(op) {
                *op_ref = Box::new(CudaSin::<T>::new(shapes[0], dev.clone(), &graph.dyn_map));
            } else if let Some(c) = op_ref.as_any().downcast_ref::<Constant>() {
                *op_ref = Box::new(CudaConstant::<T>::new(
                    dev.clone(),
                    c.0.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Recip>(op) {
                *op_ref = Box::new(CudaRecip::<T>::new(shapes[0], dev.clone(), &graph.dyn_map));
            } else if is::<Sqrt>(op) {
                *op_ref = Box::new(CudaSqrt::<T>::new(shapes[0], dev.clone(), &graph.dyn_map));
            } else if is::<Add>(op) {
                *op_ref = Box::new(CudaAdd::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mul>(op) {
                *op_ref = Box::new(CudaMul::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Mod>(op) {
                *op_ref = Box::new(CudaMod::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<LessThan>(op) {
                *op_ref = Box::new(CudaLessThan::<T>::new(
                    shapes[0],
                    shapes[1],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if is::<Contiguous>(op) {
                *op_ref = Box::new(CudaContiguous::<T>::new(
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(SumReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaSumReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            } else if let Some(MaxReduce(dim)) = op_ref.as_any().downcast_ref() {
                *op_ref = Box::new(CudaMaxReduce::<T>::new(
                    *dim,
                    shapes[0],
                    dev.clone(),
                    &graph.dyn_map,
                ));
            }
        }
    }
}

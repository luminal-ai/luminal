use luminal::{graph::Graph, op::Operator};

mod fp16;
mod fp32;
mod int_8;

#[macro_export]
macro_rules! single_unary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty, $size: expr) => {
        paste::paste! {
            #[test]
            fn [<$name _ $size>]() {
                let mut rng = StdRng::seed_from_u64(1);
                let data = random_vec_rng($size, &mut rng);
                let mut cx = Graph::new();
                let a = cx.tensor($size).set(data.clone());
                let f: fn(GraphTensor) -> GraphTensor = $luminal_func;
                let mut b = f(a).retrieve();
                cx.compile(CudaCompiler::<$type>::default(), &mut b);
                cx.execute();

                let d_dev = Cpu::default();
                let d_a = d_dev
                    .tensor_from_vec(data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let f: fn(
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                ) -> dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape> = $dfdx_func;
                let d_b = f(d_a);

                assert_close(&b.data(), &d_b.to_dtype::<f32>().as_vec());
            }
        }
    };
}

#[macro_export]
macro_rules! unary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 3);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 50);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 783);
        $crate::single_unary_test!($luminal_func, $dfdx_func, $name, $type, 4096);
    };
}

#[macro_export]
macro_rules! single_binary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty, $size: expr) => {
        paste::paste! {
            #[test]
            fn [<$name _ $size>]() {
                let mut rng = StdRng::seed_from_u64(2);
                let a_data = random_vec_rng($size, &mut rng);
                let b_data = random_vec_rng($size, &mut rng);
                let mut cx = Graph::new();
                let a = cx.tensor($size).set(a_data.clone());
                let b = cx.tensor($size).set(b_data.clone());
                let f: fn(GraphTensor, GraphTensor) -> GraphTensor =
                    $luminal_func;
                let mut c = f(a, b).retrieve();
                cx.compile(CudaCompiler::<$type>::default(), &mut c);
                cx.execute();

                let d_dev = Cpu::default();
                let d_a = d_dev
                    .tensor_from_vec(a_data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let d_b = d_dev
                    .tensor_from_vec(b_data, (dfdx::prelude::Const::<$size>,))
                    .to_dtype::<$type>();
                let f: fn(
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                    dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape>,
                ) -> dfdx::prelude::Tensor<Rank1<$size>, $type, Cpu, NoneTape> = $dfdx_func;
                let d_c = f(d_a, d_b);

                assert_close(&c.data(), &d_c.to_dtype::<f32>().as_vec());
            }
        }
    };
}

#[macro_export]
macro_rules! binary_test {
    ($luminal_func: expr , $dfdx_func: expr , $name: ident, $type: ty) => {
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 3);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 50);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 783);
        $crate::single_binary_test!($luminal_func, $dfdx_func, $name, $type, 4096);
    };
}

pub fn assert_op_in_graph<T: Operator + 'static>(graph: &Graph) {
    assert!(
        graph.node_indices().any(|i| graph.check_node_type::<T>(i)),
        "Node not found in the graph!"
    );
}

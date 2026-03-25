use std::fmt::Debug;

use luminal::{
    egglog_utils::api::{Rule, SortDef},
    hlir::unary_sort,
    op::EgglogOp,
};

pub type Ops = (Exp, Sigmoid);

#[derive(Debug, Default)]
pub struct Exp;
impl EgglogOp for Exp {
    fn sort(&self) -> SortDef {
        unary_sort("Exp")
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw(
            "(rule
            (
                (= ?exp_const (Op (Constant 1.442695) (INil)))
                (= ?mul (Op (Mul ?shape ?x_stride ?const_stride ?intermediate_stride) (ICons ?x (ICons ?exp_const (INil)))))
                (= ?exp2 (Op (Exp2 ?shape ?intermediate_stride ?out_stride) (ICons ?mul (INil))))
                (= ?dt (dtype ?x))
            )
            (
                (let ?exp (Op (Exp ?shape ?x_stride ?out_stride) (ICons ?x (INil))))
                (union ?exp2 ?exp)
                (set (dtype ?exp) ?dt)
            )
        )",
        )]
    }
}

#[derive(Default, Debug, Clone)]
pub struct Sigmoid;
impl EgglogOp for Sigmoid {
    fn sort(&self) -> SortDef {
        unary_sort("Sigmoid")
    }

    fn cleanup(&self) -> bool {
        true
    }

    fn rewrites(&self) -> Vec<Rule> {
        vec![Rule::raw("(rule
            (
                (= ?neg1 (Op (Constant -1.0) (INil)))
                (= ?neg_input (Op (Mul ?input_range ?input_stride ?const_stride ?intermediate_stride) (ICons ?input (ICons ?neg1 (INil)))))
                (= ?exp (Op (Exp ?input_range ?intermediate_stride ?exp_stride) (ICons ?neg_input (INil))))
                (= ?one (Op (Constant 1.0) (INil)))
                (= ?plus_one (Op (Add ?input_range ?exp_stride ?const_stride ?plus_one_stride) (ICons ?exp (ICons ?one (INil)))))
                (= ?sig_out (Op (Recip ?input_range ?plus_one_stride ?out_stride) (ICons ?plus_one (INil))))
                (= ?dt (dtype ?input))
            )
            (
                (let ?sig (Op (Sigmoid ?input_range ?input_stride ?out_stride) (ICons ?input (INil))))
                (union ?sig_out ?sig)
                (set (dtype ?sig) ?dt)
            )
            :name \"sigmoid\"
        )")]
    }
}

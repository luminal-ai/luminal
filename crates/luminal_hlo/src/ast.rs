use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct Operation {
    pub result_name: String,
    pub name: String,
    pub operands: Vec<String>,
    pub attributes: AttrMap,
    pub result_type_src: String,
}

#[derive(Clone, Debug)]
pub enum Attr {
    Int(i64),
    Float(f64),
    Id(String),
    IntVec(Vec<usize>),
    DimNumbers {
        input: Vec<String>,
        kernel: Vec<String>,
        output: Vec<String>,
    },
    PadPairs(Vec<(usize, usize)>),
}

pub type AttrMap = HashMap<String, Attr>;

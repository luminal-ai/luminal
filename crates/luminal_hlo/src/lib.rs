mod ast;
mod lexer;
mod lower;
mod parser;

use std::collections::HashMap;

use luminal::prelude::*;

use crate::{
    lexer::Lexer,
    lower::lower_op,
    parser::{gather_op_blocks, parse_func_args_line, Parser},
};

pub fn import_hlo(path: &str) -> (Box<Graph>, HashMap<String, GraphTensor>) {
    let contents = std::fs::read_to_string(path).expect("Failed to read file.");

    let mut cx = Box::new(Graph::new());
    let mut env: HashMap<String, GraphTensor> = HashMap::new();

    for line in contents.lines().map(str::trim) {
        if line.starts_with("func.func") {
            parse_func_args_line(line, &mut cx, &mut env);
            break;
        }
    }

    let mut ops = Vec::new();
    for raw in gather_op_blocks(&contents) {
        let trimmed = raw.trim_start();
        let toks = Lexer::new(&raw).tokenize();
        let mut p = Parser::new(&raw, toks);
        if trimmed.starts_with('%') {
            ops.push(p.parse_operation().expect("parse op failed"));
        } else if trimmed.starts_with("return") {
            ops.push(p.parse_return().expect("parse return failed"));
        }
    }

    for op in ops.into_iter() {
        if let Err(e) = lower_op(&op, &mut cx, &mut env) {
            panic!("Lowering error for op {:?}: {}", op.name, e);
        }
    }

    (cx, env)
}

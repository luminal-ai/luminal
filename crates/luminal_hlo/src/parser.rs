use crate::ast::{Attr, AttrMap, Operation};
use crate::lexer::Tok;

use anyhow::{anyhow, bail, Result};
use std::collections::HashMap;

use luminal::prelude::*;

pub struct Parser<'a> {
    src: &'a str,
    toks: Vec<Tok>,
    i: usize,
}

impl<'a> Parser<'a> {
    pub fn new(src: &'a str, toks: Vec<Tok>) -> Self {
        Self { src, toks, i: 0 }
    }

    pub fn parse_operation(&mut self) -> Result<Operation> {
        let result_name = self.expect_percent_ident()?;
        self.expect(Tok::Eq)?;
        let name = self.expect_ident()?;

        // Parse operand list
        let mut operands = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::PercentIdent(_)) => operands.push(self.expect_percent_ident()?),
                Some(Tok::Colon) => {
                    self.bump();
                    break;
                }
                Some(Tok::Ident(s))
                    if s == "dim"
                        || s == "dims"
                        || s == "dim_numbers"
                        || s == "apply"
                        || s == "applies"
                        || s == "dense"
                        || s == "window_dimensions"
                        || s == "window_strides"
                        || s == "base_dilations"
                        || s == "window_dilations"
                        || s == "padding"
                        || s == "batch_group_count"
                        || s == "feature_group_count"
                        || s == "contracting_dims"
                        || s == "batching_dims"
                        || s == "NE"
                        || s == "GT"
                        || s == "GE"
                        || s == "LT"
                        || s == "LE"
                        || s == "EQ" =>
                {
                    break;
                }
                Some(Tok::LBracket) if name == "stablehlo.slice" => {
                    break;
                }
                Some(Tok::Ident(s)) if s == "init" => {
                    self.bump();
                    self.bump();
                }
                Some(Tok::Comma) | Some(Tok::LParen) | Some(Tok::RParen) | Some(Tok::RBracket)
                | Some(Tok::Less) | Some(Tok::Greater) | Some(Tok::LBrace) | Some(Tok::RBrace) => {
                    self.bump();
                }
                other => bail!("unexpected token in operand list: {:?}", other),
            }
        }

        let mut attrs: AttrMap = HashMap::new();

        while !matches!(self.peek(), None | Some(Tok::Arrow)) {
            match self.peek() {
                Some(Tok::Ident(s)) if s == "dims" => {
                    self.bump();
                    self.expect(Tok::Eq)?;
                    let v = self.parse_intvec()?;
                    attrs.insert("dims".into(), Attr::IntVec(v));
                }
                Some(Tok::Ident(s)) if s == "dim" => {
                    self.bump();
                    self.expect(Tok::Eq)?;
                    let v = self.expect_integer()?;
                    attrs.insert("dim".into(), Attr::Int(v));
                }
                Some(Tok::Ident(s)) if s == "applies" => {
                    self.bump();
                    let id = self.expect_ident()?;
                    attrs.insert("apply".into(), Attr::Id(id));
                    if let Some(Tok::Ident(s2)) = self.peek() {
                        if s2 == "across" {
                            self.bump();
                        }
                    }
                    if let Some(Tok::Ident(s3)) = self.peek() {
                        if s3 == "dimensions" {
                            self.bump();
                        }
                    }
                    if let Some(Tok::Eq) = self.peek() {
                        self.bump();
                        let v = self.parse_intvec()?;
                        attrs.insert("dimensions".into(), Attr::IntVec(v));
                    }
                }
                Some(Tok::Ident(s)) if s == "dense" => {
                    self.bump();
                    self.expect(Tok::Less)?;
                    if let Some(Tok::Float(v)) = self.peek() {
                        attrs.insert("dense".into(), Attr::Float(v.clone()));
                    }
                    if let Some(Tok::Integer(v)) = self.peek() {
                        attrs.insert("dense".into(), Attr::Int(v.clone()));
                    }
                    if let Some(Tok::Ident(s)) = self.peek() {
                        if s == "true" {
                            attrs.insert("dense".into(), Attr::Int(1));
                        } else if s == "false" {
                            attrs.insert("dense".into(), Attr::Int(0));
                        }
                    }
                }
                Some(Tok::LBracket) if name == "stablehlo.slice" => {
                    self.expect(Tok::LBracket)?;

                    let mut starts: Vec<usize> = Vec::new();
                    let mut limits: Vec<usize> = Vec::new();

                    loop {
                        let start = self.expect_integer()? as usize;
                        self.expect(Tok::Colon)?;
                        let limit = self.expect_integer()? as usize;

                        starts.push(start);
                        limits.push(limit);

                        match self.peek() {
                            Some(Tok::Comma) => {
                                self.bump();
                            }
                            Some(Tok::RBracket) => {
                                self.bump();
                                break;
                            }
                            other => {
                                bail!("expected ',' or ']' in slice ranges, found {:?}", other);
                            }
                        }
                    }

                    attrs.insert("start_indices".into(), Attr::IntVec(starts));
                    attrs.insert("end_indices".into(), Attr::IntVec(limits));
                }
                Some(Tok::Ident(s))
                    if s == "NE"
                        || s == "GT"
                        || s == "GE"
                        || s == "LT"
                        || s == "LE"
                        || s == "EQ" =>
                {
                    self.parse_compare_attrs(&mut operands, &mut attrs)?;
                }
                Some(Tok::Ident(s)) if s == "contracting_dims" => {
                    self.bump();
                    self.expect(Tok::Eq)?;

                    let lhs = self.parse_intvec()?;
                    self.expect(Tok::Ident("x".to_string()))?;
                    let rhs = self.parse_intvec()?;

                    attrs.insert("lhs_contracting_dims".into(), Attr::IntVec(lhs));
                    attrs.insert("rhs_contracting_dims".into(), Attr::IntVec(rhs));
                }
                Some(Tok::Ident(s)) if s == "batching_dims" => {
                    self.bump();
                    self.expect(Tok::Eq)?;

                    let lhs = self.parse_intvec()?;
                    self.expect(Tok::Ident("x".to_string()))?;
                    let rhs = self.parse_intvec()?;

                    attrs.insert("lhs_batching_dims".into(), Attr::IntVec(lhs));
                    attrs.insert("rhs_batching_dims".into(), Attr::IntVec(rhs));
                }
                _ => {
                    self.bump();
                }
            }
        }

        if name == "stablehlo.convolution" {
            self.parse_convolution_attrs(&mut attrs)?;
        } else if name == "stablehlo.reduce_window" {
            self.parse_reduce_window_attrs(&mut attrs)?;
        }

        let mut result_type_src = String::new();
        if matches!(self.peek(), Some(Tok::Arrow)) {
            if let Some(pos) = self.src.find("->") {
                result_type_src = self.src[pos + 2..].trim().to_string();
            }
        } else if let Some(pos) = self.src.find(":") {
            result_type_src = self.src[pos + 1..].trim().to_string();
        }

        Ok(Operation {
            result_name,
            name,
            operands,
            attributes: attrs,
            result_type_src,
        })
    }

    pub fn parse_return(&mut self) -> Result<Operation> {
        let name = String::from("return");
        let mut ret = String::new();
        while let Some(tok) = self.bump() {
            if let Tok::PercentIdent(s) = tok {
                ret = s;
                break;
            }
        }
        if ret.is_empty() {
            bail!("return missing %ident");
        }
        Ok(Operation {
            result_name: "%_ret".into(),
            name,
            operands: vec![ret],
            attributes: HashMap::new(),
            result_type_src: String::new(),
        })
    }

    fn parse_convolution_attrs(&self, attrs: &mut AttrMap) -> Result<()> {
        // parse dim_numbers = [b, f, ...]x[o, i, ...]->[b, f, ...]
        if let Some(idx) = self.src.find("dim_numbers") {
            if let Some(start) = self.src[idx..].find('[') {
                let start = idx + start;
                let (a, p1) = extract_bracket_list(self.src, start)?;
                let after_a = &self.src[p1..];
                let x_pos = after_a
                    .find('x')
                    .ok_or_else(|| anyhow!("dim_numbers: missing 'x'"))?
                    + p1;
                let (b, p2) = extract_bracket_list(self.src, next_bracket_after(self.src, x_pos)?)?;
                let arrow_pos = self.src[p2..]
                    .find("->")
                    .ok_or_else(|| anyhow!("dim_numbers: missing '->'"))?
                    + p2;
                let (c, _p3) =
                    extract_bracket_list(self.src, next_bracket_after(self.src, arrow_pos)?)?;
                attrs.insert(
                    "dim_numbers".into(),
                    Attr::DimNumbers {
                        input: split_tags(a),
                        kernel: split_tags(b),
                        output: split_tags(c),
                    },
                );
            }
        }

        // parse window
        if let Some(wi) = self.src.find("window") {
            if let Some(open) = self.src[wi..].find('{') {
                let open = wi + open;
                if let Some(close_rel) = find_matching_brace(self.src, open) {
                    let body = &self.src[open + 1..close_rel];
                    if let Some(pi) = body.find("pad") {
                        if let Some(lb) = body[pi..].find('[') {
                            let pad_start = pi + lb;
                            let v: Vec<Vec<usize>> = serde_json::from_str(&body[pad_start..])?;
                            let pads: Vec<(usize, usize)> = v
                                .into_iter()
                                .map(|pair| (pair[0], pair[1]))
                                .collect::<Vec<(usize, usize)>>();
                            attrs.insert("window_pad".into(), Attr::PadPairs(pads));
                        }
                    }
                    if let Some(si) = body.find("stride") {
                        if let Some(lb) = body[si..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[si + lb..])?;
                            attrs.insert("stride".into(), Attr::IntVec(vec));
                        }
                    }
                    if let Some(bd) = body.find("base_dilations") {
                        if let Some(lb) = body[bd..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[bd + lb..])?;
                            attrs.insert("base_dilations".into(), Attr::IntVec(vec));
                        }
                    }
                    if let Some(wd) = body.find("window_dilations") {
                        if let Some(lb) = body[wd..].find('[') {
                            let (vec, _) = parse_bracket_intvec(&body[wd + lb..])?;
                            attrs.insert("window_dilations".into(), Attr::IntVec(vec));
                        }
                    }
                }
            }
        }

        // group counts
        if let Some(bi) = self.src.find("batch_group_count") {
            if let Some((v, _)) = parse_trailing_int(&self.src[bi..]) {
                attrs.insert("batch_group_count".into(), Attr::Int(v as i64));
            }
        }
        if let Some(fi) = self.src.find("feature_group_count") {
            if let Some((v, _)) = parse_trailing_int(&self.src[fi..]) {
                attrs.insert("feature_group_count".into(), Attr::Int(v as i64));
            }
        }

        Ok(())
    }

    fn parse_reduce_window_attrs(&self, attrs: &mut AttrMap) -> Result<()> {
        // 1) window_dimensions
        if let Some(v) = find_array_i64(self.src, "window_dimensions")? {
            attrs.insert("window_dimensions".into(), Attr::IntVec(v));
        }
        // 2) window_strides (optional)
        if let Some(v) = find_array_i64(self.src, "window_strides")? {
            attrs.insert("window_strides".into(), Attr::IntVec(v));
        }
        // 3) base_dilations (optional, default 1s)
        if let Some(v) = find_array_i64(self.src, "base_dilations")? {
            attrs.insert("base_dilations".into(), Attr::IntVec(v));
        }
        // 4) window_dilations (optional, default 1s)
        if let Some(v) = find_array_i64(self.src, "window_dilations")? {
            attrs.insert("window_dilations".into(), Attr::IntVec(v));
        }
        // 5) padding = dense<[[lo,hi], ...]> : tensor<rankx2xi64>
        if let Some(pads) = find_padding_dense_pairs(self.src, "padding")? {
            attrs.insert("padding".into(), Attr::PadPairs(pads));
        }

        // 6) combiner: sniff the region body for a single op
        if let Some(body) = extract_region_body(self.src) {
            if body.contains("stablehlo.maximum") {
                attrs.insert("apply".into(), Attr::Id("stablehlo.maximum".into()));
            } else if body.contains("stablehlo.minimum") {
                attrs.insert("apply".into(), Attr::Id("stablehlo.minimum".into()));
            } else if body.contains("stablehlo.add") {
                attrs.insert("apply".into(), Attr::Id("stablehlo.add".into()));
            } else if body.contains("stablehlo.multiply") {
                attrs.insert("apply".into(), Attr::Id("stablehlo.multiply".into()));
            }
        }

        Ok(())
    }

    fn parse_compare_attrs(
        &mut self,
        operands: &mut Vec<String>,
        attrs: &mut AttrMap,
    ) -> Result<()> {
        // 1) comparison_direction enum
        let dir = match self.bump() {
            Some(Tok::Ident(s)) => match s.as_str() {
                "EQ" | "NE" | "GE" | "GT" | "LE" | "LT" => s,
                _ => bail!("stablehlo.compare: invalid comparison_direction '{}'", s),
            },
            other => bail!(
                "stablehlo.compare: expected comparison_direction, got {:?}",
                other
            ),
        };
        attrs.insert("comparison_direction".into(), Attr::Id(dir));

        self.expect(Tok::Comma)?;

        operands.push(self.expect_percent_ident()?);

        self.expect(Tok::Comma)?;

        operands.push(self.expect_percent_ident()?);

        if self.peek() == Some(&Tok::Colon) {
            return Ok(());
        }

        self.expect(Tok::Comma)?;

        let cty = match self.bump() {
            Some(Tok::Ident(s)) => match s.as_str() {
                "FLOAT" | "TOTALORDER" | "SIGNED" | "UNSIGNED" => s,
                _ => bail!("stablehlo.compare: invalid compare_type '{}'", s),
            },
            other => bail!("stablehlo.compare: expected compare_type, got {:?}", other),
        };
        attrs.insert("compare_type".into(), Attr::Id(cty));

        self.expect(Tok::Colon)?;

        Ok(())
    }

    fn parse_intvec(&mut self) -> Result<Vec<usize>> {
        self.expect(Tok::LBracket)?;
        let mut out = Vec::new();
        loop {
            match self.peek() {
                Some(Tok::Integer(i)) => {
                    let v = *i as usize;
                    self.bump();
                    out.push(v);
                }
                Some(Tok::Comma) => {
                    self.bump();
                }
                Some(Tok::RBracket) => {
                    self.bump();
                    break;
                }
                other => bail!("bad intvec token: {:?}", other),
            }
        }
        Ok(out)
    }

    fn expect_integer(&mut self) -> Result<i64> {
        match self.bump() {
            Some(Tok::Integer(i)) => Ok(i),
            other => bail!("expected integer, got {:?}", other),
        }
    }

    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.i)
    }

    fn bump(&mut self) -> Option<Tok> {
        if self.i < self.toks.len() {
            let t = self.toks[self.i].clone();
            self.i += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect_percent_ident(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::PercentIdent(s)) => Ok(s),
            _ => bail!("expected %ident"),
        }
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.bump() {
            Some(Tok::Ident(s)) => Ok(s),
            _ => bail!("expected ident"),
        }
    }

    fn expect(&mut self, want: Tok) -> Result<()> {
        match self.bump() {
            Some(t) if t == want => Ok(()),
            other => bail!("expected {:?}, got {:?}", want, other),
        }
    }
}

pub fn parse_func_args_line(line: &str, cx: &mut Graph, env: &mut HashMap<String, GraphTensor>) {
    if let Some((start_idx, end_idx)) = line.find('(').zip(line.find(')')) {
        let args_str = &line[start_idx + 1..end_idx];
        for arg in args_str.split(',') {
            let arg_tokens: Vec<&str> = arg.trim().split(':').collect();
            if let [arg_name, tensor_shape_str] = arg_tokens.as_slice() {
                let arg_name = arg_name.trim();
                let tensor_shape_str = tensor_shape_str.trim();
                let tensor_shape = parse_tensor_shape(tensor_shape_str);
                // TODO: Use named_tensor instead of tensor
                let tensor = cx.tensor(tensor_shape);
                env.insert(arg_name.to_string(), tensor);
            }
        }
    }
}

pub fn parse_output_shape_from_op(op_line: &str) -> Vec<usize> {
    if let Some(tensor_start) = op_line.find("tensor<") {
        let tensor_end = op_line[tensor_start..]
            .find('>')
            .map(|pos| tensor_start + pos + 1)
            .unwrap_or(op_line.len());
        let tensor_type = &op_line[tensor_start..tensor_end];
        parse_tensor_shape(tensor_type)
    } else {
        panic!("No tensor type found after '->' in: {}", op_line);
    }
}

pub fn parse_tensor_shape(tensor_type_str: &str) -> Vec<usize> {
    if let Some(start) = tensor_type_str.find('<') {
        if let Some(end) = tensor_type_str.find('>') {
            let shape_str = &tensor_type_str[start + 1..end];

            if !shape_str.contains('x')
                && (shape_str.ends_with("f32")
                    || shape_str.ends_with("f16")
                    || shape_str.ends_with("i32")
                    || shape_str.ends_with("i64"))
            {
                return vec![1];
            }

            let dims: Vec<usize> = shape_str
                .split('x')
                .filter_map(|s| {
                    let s = s.trim();
                    if s.ends_with("f32")
                        || s.ends_with("f16")
                        || s.ends_with("i32")
                        || s.ends_with("i64")
                    {
                        None
                    } else {
                        s.parse::<usize>().ok()
                    }
                })
                .collect();

            if dims.is_empty() {
                vec![1]
            } else {
                dims
            }
        } else {
            panic!("Malformed tensor type: missing '>' in {}", tensor_type_str);
        }
    } else {
        panic!("Malformed tensor type: missing '<' in {}", tensor_type_str);
    }
}

fn extract_bracket_list(s: &str, start: usize) -> Result<(String, usize)> {
    let mut depth = 0usize;
    let mut i = start;
    let bytes = s.as_bytes();
    while i < s.len() {
        let c = bytes[i] as char;
        if c == '[' {
            depth += 1;
            if depth == 1 {
                i += 1;
                let j = i;
                while i < s.len() {
                    let c2 = bytes[i] as char;
                    if c2 == ']' {
                        depth -= 1;
                        if depth == 0 {
                            let body = &s[j..i];
                            return Ok((body.trim().to_string(), i + 1));
                        }
                    }
                    i += 1;
                }
                break;
            }
        }
        i += 1;
    }
    Err(anyhow!("unclosed bracket list"))
}

fn next_bracket_after(s: &str, from: usize) -> Result<usize> {
    let rest = &s[from..];
    let off = rest.find('[').ok_or_else(|| anyhow!("expected '['"))?;
    Ok(from + off)
}

fn find_matching_brace(s: &str, open: usize) -> Option<usize> {
    let mut depth = 0;
    let b = s.as_bytes();
    for i in open..s.len() {
        let c = b[i] as char;
        if c == '{' {
            depth += 1;
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                return Some(i);
            }
        }
    }
    None
}

fn split_tags(s: String) -> Vec<String> {
    s.split(',')
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect()
}

fn parse_bracket_intvec(s: &str) -> Result<(Vec<usize>, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    let b = s.as_bytes();
    assert_eq!(b[0] as char, '[');
    i += 1;
    let mut num = String::new();
    while i < s.len() {
        let c = b[i] as char;
        match c {
            '0'..='9' => num.push(c),
            ',' => {
                if !num.is_empty() {
                    out.push(num.parse::<usize>().unwrap());
                    num.clear();
                }
            }
            ']' => {
                if !num.is_empty() {
                    out.push(num.parse::<usize>().unwrap());
                }
                return Ok((out, i + 1));
            }
            ' ' => {}
            _ => break,
        }
        i += 1;
    }
    Err(anyhow!("bad intvec"))
}

fn parse_trailing_int(s: &str) -> Option<(usize, usize)> {
    let bytes = s.as_bytes();
    let mut i = 0;

    while i < bytes.len() && !bytes[i].is_ascii_digit() {
        i += 1;
    }

    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }

    if start < i {
        let v = s[start..i].parse::<usize>().ok()?;
        return Some((v, i));
    }

    None
}

fn find_array_i64(src: &str, key: &str) -> Result<Option<Vec<usize>>> {
    if let Some(k) = src.find(key) {
        if let Some(arr) = src[k..].find("array<i64:") {
            let start = k + arr + "array<i64:".len();
            if let Some(end_rel) = src[start..].find('>') {
                let nums = &src[start..start + end_rel];
                let v = nums
                    .split(',')
                    .filter_map(|s| s.trim().parse::<isize>().ok())
                    .map(|x| usize::try_from(x).unwrap_or(0))
                    .collect();
                return Ok(Some(v));
            }
        }
    }
    Ok(None)
}

fn find_padding_dense_pairs(src: &str, key: &str) -> Result<Option<Vec<(usize, usize)>>> {
    if let Some(k) = src.find(key) {
        if let Some(d) = src[k..].find("dense<[[") {
            let start = k + d + "dense<".len();
            // find matching '>' after starting at 'dense<'
            if let Some(close) = src[start..].find('>') {
                let inner = &src[start..start + close]; // e.g. [[1, 1], [1, 1]]
                let mut pads = Vec::new();
                for pair in inner
                    .trim()
                    .trim_start_matches('[')
                    .trim_end_matches(']')
                    .split("],")
                    .map(|p| p.trim().trim_start_matches('[').trim_end_matches(']'))
                {
                    let mut it = pair
                        .split(',')
                        .map(|x| x.trim().parse::<usize>().unwrap_or(0));
                    if let (Some(lo), Some(hi)) = (it.next(), it.next()) {
                        pads.push((lo, hi));
                    }
                }
                return Ok(Some(pads));
            }
        }
    }
    Ok(None)
}

fn extract_region_body(src: &str) -> Option<&str> {
    let open = src.find("({")?;
    let after = &src[open + 2..];
    let close = after.rfind("})")?;
    Some(&after[..close])
}

pub fn gather_op_blocks(src: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut cur = String::new();
    let mut paren = 0i32;
    let mut brace = 0i32;
    let mut bracket = 0i32;
    let mut in_string = false;

    let mut in_block = false;

    for line in src.lines() {
        let trimmed = line.trim_start();
        if !in_block {
            // start of an op/return
            if trimmed.starts_with('%') || trimmed.starts_with("return") {
                in_block = true;
                cur.clear();
                paren = 0;
                brace = 0;
                bracket = 0;
                in_string = false;
            } else {
                continue;
            }
        }

        cur.push_str(line);
        cur.push('\n');

        for ch in line.chars() {
            if ch == '"' {
                in_string = !in_string;
                continue;
            }
            if in_string {
                continue;
            }
            match ch {
                '(' => paren += 1,
                ')' => paren -= 1,
                '{' => brace += 1,
                '}' => brace -= 1,
                '[' => bracket += 1,
                ']' => bracket -= 1,
                _ => {}
            }
        }

        if in_block && paren == 0 && brace == 0 && bracket == 0 {
            blocks.push(cur.trim().to_string());
            in_block = false;
        }
    }

    blocks
}

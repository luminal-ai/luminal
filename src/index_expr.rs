//! Index expressions read from the e-graph — the SHARED IR-layer
//! vocabulary every runtime lowers from (ruling 2026-08-17: runtimes
//! own what they EXECUTE; how the e-graph is READ is shared). Moved
//! from the reference iota module, verbatim except visibility: the
//! numeric [`IotaExpr`] tree, its host evaluator, and the memoized,
//! cycle-tainting, owner-shape-guarded parser over extraction sites.

use crate::layout_ir::ExtractionSite;

type BinaryIotaBuilder = fn(Box<IotaExpr>, Box<IotaExpr>) -> IotaExpr;

/// A numeric IntExpr tree for reference evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IotaExpr {
    Lit(i64),
    /// A runtime dimension, retained when extraction uses interval bounds.
    Var(String),
    /// CoordVar axis, zero-based from the END over the OUT coordinates.
    Coord(usize),
    Add(Box<IotaExpr>, Box<IotaExpr>),
    Mul(Box<IotaExpr>, Box<IotaExpr>),
    /// Truncated (toward-zero) division — the preamble's IntTruncDiv.
    TruncDiv(Box<IotaExpr>, Box<IotaExpr>),
    /// Truncated remainder — the preamble's IntTruncRem.
    TruncRem(Box<IotaExpr>, Box<IotaExpr>),
    /// Ceiling division, including symbolic shape arithmetic.
    CeilDiv(Box<IotaExpr>, Box<IotaExpr>),
    Min(Box<IotaExpr>, Box<IotaExpr>),
    Max(Box<IotaExpr>, Box<IotaExpr>),
    /// The bool bridge's indicator: `(a < b) as i64` — IntCastFromBool
    /// over BoolLessThanInt.
    LessThanCast(Box<IotaExpr>, Box<IotaExpr>),
}

impl IotaExpr {
    /// Evaluate at the given OUT coordinates (front-indexed).
    pub fn eval(&self, coords: &[usize]) -> i64 {
        self.eval_with_dims(coords, &Default::default())
    }

    /// Evaluate coordinates and runtime dimension variables.
    pub fn eval_with_dims(&self, coords: &[usize], dims: &crate::shape::DynMap) -> i64 {
        let eval = |expr: &Self| expr.eval_with_dims(coords, dims);
        match self {
            IotaExpr::Lit(value) => *value,
            IotaExpr::Var(name) => i64::try_from(
                *dims
                    .get(&crate::shape::Symbol::from(name.as_str()))
                    .unwrap_or_else(|| panic!("unbound index-expression dimension `{name}`")),
            )
            .expect("dimension exceeds i64"),
            IotaExpr::Coord(axis_from_end) => coords[coords.len() - 1 - axis_from_end] as i64,
            IotaExpr::Add(a, b) => eval(a) + eval(b),
            IotaExpr::Mul(a, b) => eval(a) * eval(b),
            // Divisors are literal strides/extents (>= 1 by construction);
            // a zero here is a translator bug and deserves the loud panic.
            IotaExpr::TruncDiv(a, b) => eval(a) / eval(b),
            IotaExpr::TruncRem(a, b) => eval(a) % eval(b),
            IotaExpr::CeilDiv(a, b) => {
                let (a, b) = (eval(a), eval(b));
                a / b + i64::from(a % b != 0 && ((a > 0) == (b > 0)))
            }
            IotaExpr::Min(a, b) => eval(a).min(eval(b)),
            IotaExpr::Max(a, b) => eval(a).max(eval(b)),
            IotaExpr::LessThanCast(a, b) => (eval(a) < eval(b)) as i64,
        }
    }
}

/// Parse one IntExpr class into an [`IotaExpr`], preferring folded literals;
/// depth-guarded (saturated classes hold many equal representations — any
/// one denotes the same function). `None` = unsupported constructors.
/// `expected_shape` is the OWNER-SHAPE GUARD (Design A fold-in,
/// 2026-08-06): a CoordVar spelling counts only if its owner shape class
/// IS the consumer's out shape — a foreign shape's coordinate must never
/// be silently read as an out-coordinate; the kernel's loud refusal
/// carries the burden instead.
pub fn parse_int_expr(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
) -> Option<IotaExpr> {
    parse_int_expr_memo(
        site,
        class,
        depth,
        expected_shape,
        &mut std::collections::HashMap::new(),
    )
}

/// Memo entry: a finished parse, or the in-progress cycle guard.
#[derive(Clone)]
pub enum ParseMemo {
    InProgress,
    Done(Option<IotaExpr>),
}

/// Memoized worker: each class parses at most once — the subsumed-
/// spelling fallback widens the branching factor enough that the naive
/// backtracking walk goes exponential on fat saturated classes.
///
/// CYCLE-TAINT RULE (2026-08-06, found by the MoE map parse): hitting an
/// in-progress class fails THAT spelling, but the failure is contextual —
/// the same class can parse fine once its ancestor resolves. So a `None`
/// outcome whose walk touched the cycle guard is NOT cached (a later
/// query retries from a clean stack), while an untainted `None` — every
/// spelling genuinely outside the subset — caches as before, keeping the
/// exponential-blowup protection.
pub fn parse_int_expr_memo(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
) -> Option<IotaExpr> {
    parse_int_expr_tainting(site, class, depth, expected_shape, memo, &mut false)
}

fn parse_int_expr_tainting(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IotaExpr> {
    match memo.get(class) {
        Some(ParseMemo::Done(cached)) => return cached.clone(),
        Some(ParseMemo::InProgress) => {
            *tainted = true;
            return None;
        }
        None => {}
    }
    memo.insert(class.clone(), ParseMemo::InProgress);
    let mut local_taint = false;
    let parsed =
        parse_int_expr_uncached(site, class, depth, expected_shape, memo, &mut local_taint);
    if parsed.is_none() && local_taint {
        memo.remove(class);
        *tainted = true;
    } else {
        memo.insert(class.clone(), ParseMemo::Done(parsed.clone()));
    }
    parsed
}

fn parse_int_expr_uncached(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    expected_shape: Option<&egraph_serialize::ClassId>,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
    tainted: &mut bool,
) -> Option<IotaExpr> {
    if depth == 0 {
        return None;
    }
    if let Some(lit) = site.nodes_in_class_value(class, "IntLit").next() {
        let value_class = site.class_of_child(lit, 0)?;
        return Some(IotaExpr::Lit(site.node_in_class_parse_i64(&value_class)?));
    }
    for var in site.nodes_in_class_value(class, "IntVar") {
        let name_class = site.class_of_child(var, 0)?;
        if let Some(name) = site
            .index
            .nodes_of(&name_class)
            .iter()
            .filter_map(|id| site.egraph.nodes.get(id))
            .find_map(|n| n.op.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
        {
            return Some(IotaExpr::Var(name.to_string()));
        }
    }
    for coord in site.nodes_in_class_value(class, "CoordVar") {
        // Scoped coordinates: child 0 is the owner Shape, child 1 the
        // axis. The owner-shape guard: when the caller names its out
        // shape, a CoordVar owned by any OTHER shape is not an
        // out-coordinate and cannot parse (loud kernel refusal instead
        // of a silently misread axis).
        if let Some(expected) = expected_shape {
            let Some(owner_class) = site.class_of_child(coord, 0) else {
                continue;
            };
            if owner_class != *expected {
                continue;
            }
        }
        let Some(axis_class) = site.class_of_child(coord, 1) else {
            continue;
        };
        return Some(IotaExpr::Coord(
            site.node_in_class_parse_i64(&axis_class)? as usize
        ));
    }
    // Binary kinds: BACKTRACK across every representation in the class —
    // a saturated class holds many equal spellings, and the first node of
    // a kind may have children outside the parsed subset while a sibling
    // spelling parses fine.
    let binary_kinds: [(&str, BinaryIotaBuilder); 7] = [
        ("IntAdd", |a, b| IotaExpr::Add(a, b)),
        ("IntMul", |a, b| IotaExpr::Mul(a, b)),
        ("IntTruncDiv", |a, b| IotaExpr::TruncDiv(a, b)),
        ("IntTruncRem", |a, b| IotaExpr::TruncRem(a, b)),
        ("IntCeilDiv", |a, b| IotaExpr::CeilDiv(a, b)),
        ("IntMin", |a, b| IotaExpr::Min(a, b)),
        ("IntMax", |a, b| IotaExpr::Max(a, b)),
    ];
    for (kind, build) in binary_kinds {
        for node in site.nodes_in_class_value(class, kind) {
            let Some(lhs_class) = site.class_of_child(node, 0) else {
                continue;
            };
            let Some(rhs_class) = site.class_of_child(node, 1) else {
                continue;
            };
            let Some(lhs) =
                parse_int_expr_tainting(site, &lhs_class, depth - 1, expected_shape, memo, tainted)
            else {
                continue;
            };
            let Some(rhs) =
                parse_int_expr_tainting(site, &rhs_class, depth - 1, expected_shape, memo, tainted)
            else {
                continue;
            };
            return Some(build(Box::new(lhs), Box::new(rhs)));
        }
    }
    for cast in site.nodes_in_class_value(class, "IntCastFromBool") {
        let Some(bool_class) = site.class_of_child(cast, 0) else {
            continue;
        };
        let Some(less_than) = site
            .nodes_in_class_value(&bool_class, "BoolLessThanInt")
            .next()
        else {
            continue;
        };
        let Some(lhs_class) = site.class_of_child(less_than, 0) else {
            continue;
        };
        let Some(rhs_class) = site.class_of_child(less_than, 1) else {
            continue;
        };
        let Some(lhs) =
            parse_int_expr_tainting(site, &lhs_class, depth - 1, expected_shape, memo, tainted)
        else {
            continue;
        };
        let Some(rhs) =
            parse_int_expr_tainting(site, &rhs_class, depth - 1, expected_shape, memo, tainted)
        else {
            continue;
        };
        return Some(IotaExpr::LessThanCast(Box::new(lhs), Box::new(rhs)));
    }
    None
}

/// Walk a matched term's index-map metadata into numeric entries:
/// `IndexMapLit` → cons spine BY E-CLASS, one [`IotaExpr`] per parent axis.
/// The SHARED form of the per-runtime materialize parsers (how the e-graph
/// is READ is shared, ruling 2026-08-17). EXISTENTIAL AT EVERY LEVEL (the
/// R8/backtracking doctrine): a saturated map class holds several
/// `IndexMapLit` spellings and a list class several cons spellings, so the
/// walk tries every spelling and takes the first that parses all the way
/// down — all spellings of a class denote the same map, so any parseable
/// one is correct. `map_child`/`shape_child` are the matched enode's
/// metadata child indices (the map, and the OUT shape whose coordinates
/// the entries are functions of — the owner-shape guard). `None` = no
/// spelling parses; extraction stays infallible and the numeric consumer's
/// loud refusal carries the burden.
pub fn parse_index_map_entries(
    site: &ExtractionSite<'_>,
    map_child: usize,
    shape_child: usize,
) -> Option<Vec<IotaExpr>> {
    let map_class = site.child_class(map_child);
    let out_shape = site.child_class(shape_child);
    let mut memo = std::collections::HashMap::new();
    for map_node in site.nodes_in_class_value(&map_class, "IndexMapLit") {
        let Some(head) = site.class_of_child(map_node, 0) else {
            continue;
        };
        if let Some(entries) = parse_entry_list(site, &head, 64, &out_shape, &mut memo) {
            return Some(entries);
        }
    }
    None
}

fn parse_entry_list(
    site: &ExtractionSite<'_>,
    class: &egraph_serialize::ClassId,
    depth: usize,
    out_shape: &egraph_serialize::ClassId,
    memo: &mut std::collections::HashMap<egraph_serialize::ClassId, ParseMemo>,
) -> Option<Vec<IotaExpr>> {
    if depth == 0 {
        return None;
    }
    if site
        .nodes_in_class_value(class, "IntExprNil")
        .next()
        .is_some()
    {
        return Some(Vec::new());
    }
    for cons in site.nodes_in_class_value(class, "IntExprCons") {
        let Some(element) = site.class_of_child(cons, 0) else {
            continue;
        };
        let Some(tail) = site.class_of_child(cons, 1) else {
            continue;
        };
        let Some(expr) = parse_int_expr_memo(site, &element, 64, Some(out_shape), memo) else {
            continue;
        };
        if let Some(mut rest) = parse_entry_list(site, &tail, depth - 1, out_shape, memo) {
            rest.insert(0, expr);
            return Some(rest);
        }
    }
    None
}

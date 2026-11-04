//! Index expressions read from the e-graph — the SHARED IR-layer
//! vocabulary every runtime lowers from (ruling 2026-08-17: runtimes
//! own what they EXECUTE; how the e-graph is READ is shared). Moved
//! from the reference iota module, verbatim except visibility: the
//! numeric [`IotaExpr`] tree, its host evaluator, and the memoized,
//! cycle-tainting, owner-shape-guarded parser over extraction sites.

use crate::layout_ir::ExtractionSite;

type BinaryIotaBuilder = fn(Box<IotaExpr>, Box<IotaExpr>) -> IotaExpr;

/// A numeric IntExpr tree for reference evaluation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
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

    /// Replace every coordinate (`Coord(k)`, k from the END) by
    /// `replace(k)`: the beta reduction of a composed map, the inner's
    /// entries over the intermediate coordinates with the outer's entries
    /// substituted in. `None` when an axis has no replacement, which is a
    /// rank disagreement between the two levels — the decoder is
    /// existential and never loud.
    pub fn substitute_coords(
        &self,
        replace: &dyn Fn(usize) -> Option<IotaExpr>,
    ) -> Option<IotaExpr> {
        let go = |e: &IotaExpr| e.substitute_coords(replace).map(Box::new);
        Some(match self {
            IotaExpr::Lit(value) => IotaExpr::Lit(*value),
            IotaExpr::Var(name) => IotaExpr::Var(name.clone()),
            IotaExpr::Coord(axis_from_end) => return replace(*axis_from_end),
            IotaExpr::Add(a, b) => IotaExpr::Add(go(a)?, go(b)?),
            IotaExpr::Mul(a, b) => IotaExpr::Mul(go(a)?, go(b)?),
            IotaExpr::TruncDiv(a, b) => IotaExpr::TruncDiv(go(a)?, go(b)?),
            IotaExpr::TruncRem(a, b) => IotaExpr::TruncRem(go(a)?, go(b)?),
            IotaExpr::CeilDiv(a, b) => IotaExpr::CeilDiv(go(a)?, go(b)?),
            IotaExpr::Min(a, b) => IotaExpr::Min(go(a)?, go(b)?),
            IotaExpr::Max(a, b) => IotaExpr::Max(go(a)?, go(b)?),
            IotaExpr::LessThanCast(a, b) => IotaExpr::LessThanCast(go(a)?, go(b)?),
        })
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
            .find_map(|n| crate::shape::Symbol::decode_serialized(&n.op))
        {
            return Some(IotaExpr::Var(name));
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
    parse_index_map(site, &map_class, &out_shape, COMPOSITE_DEPTH)
}

/// How deep a composite may nest. The structural rule composes literal
/// pairs only, so a composite's children are literals (or opened
/// composites, which are literals too); the bound guards a hand-built
/// term, not a path the rules produce.
const COMPOSITE_DEPTH: usize = 8;

/// Parse an `IndexMap` class into one [`IotaExpr`] per parent axis over
/// `out_shape`'s coordinates. A literal spelling (`IndexMapLit`) parses
/// its entries directly. A closed composite spelling
/// (`IndexMapCompose inner outer`) is the view's lambda left unreduced by
/// the e-graph — the outer map takes the out coordinates into the
/// intermediate box (its tag), the inner takes the intermediate box into
/// the parent — and is BETA-REDUCED HERE, after extraction: the outer's
/// entries are parsed over `out_shape`, the inner's over the outer's tag,
/// and every intermediate coordinate in the inner's entries is replaced
/// by the outer's entry for it. Existential over spellings at every
/// level; literal spellings first (a class that holds both denotes one
/// map, and the literal is cheaper). Each level gets its own parse memo:
/// the memo is keyed by class alone, not by owner shape, and a class
/// refused under one owner may parse under another.
pub fn parse_index_map(
    site: &ExtractionSite<'_>,
    map_class: &egraph_serialize::ClassId,
    out_shape: &egraph_serialize::ClassId,
    depth: usize,
) -> Option<Vec<IotaExpr>> {
    if depth == 0 {
        return None;
    }
    for map_node in site.nodes_in_class_value(map_class, "IndexMapLit") {
        let Some(head) = site.class_of_child(map_node, 0) else {
            continue;
        };
        let mut memo = std::collections::HashMap::new();
        if let Some(entries) = parse_entry_list(site, &head, 64, out_shape, &mut memo) {
            return Some(entries);
        }
    }
    for composite in site.nodes_in_class_value(map_class, "IndexMapCompose") {
        let Some(inner) = site.class_of_child(composite, 0) else {
            continue;
        };
        let Some(outer) = site.class_of_child(composite, 1) else {
            continue;
        };
        let Some(outer_entries) = parse_index_map(site, &outer, out_shape, depth - 1) else {
            continue;
        };
        let Some(intermediate) = map_tag_class(site, &outer, depth - 1) else {
            continue;
        };
        let Some(inner_entries) = parse_index_map(site, &inner, &intermediate, depth - 1) else {
            continue;
        };
        let composed: Option<Vec<IotaExpr>> = inner_entries
            .iter()
            .map(|entry| {
                entry.substitute_coords(&|axis_from_end| {
                    let index = outer_entries.len().checked_sub(axis_from_end + 1)?;
                    outer_entries.get(index).cloned()
                })
            })
            .collect();
        if let Some(entries) = composed {
            return Some(entries);
        }
    }
    None
}

/// The source-shape tag of a map class: a literal's `Shape` child, a
/// composite's inner's (the e-graph's `index-map-tag`, read structurally
/// so the decoder does not depend on a function row being serialized).
/// Every spelling of a class carries one tag (the map-tag locks), so the
/// first found is the answer; the walk stays existential anyway.
fn map_tag_class(
    site: &ExtractionSite<'_>,
    map_class: &egraph_serialize::ClassId,
    depth: usize,
) -> Option<egraph_serialize::ClassId> {
    if depth == 0 {
        return None;
    }
    if let Some(tag) = site
        .nodes_in_class_value(map_class, "IndexMapLit")
        .find_map(|node| site.class_of_child(node, 1))
    {
        return Some(tag);
    }
    site.nodes_in_class_value(map_class, "IndexMapCompose")
        .find_map(|node| map_tag_class(site, &site.class_of_child(node, 0)?, depth - 1))
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

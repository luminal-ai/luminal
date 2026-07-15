use std::sync::LazyLock;

use super::api::*;
use crate::shape;
use egglog_static::egglog_static;
use rustc_hash::FxHashSet;

// The base egglog schema (datatypes + helper functions), declared once and
// type-checked at compile time by `egglog_header!`. Rewrite rulesets check
// their fragments against it via `egglog_static!(luminal_base; …)`, and
// `luminal_base_schema()` yields these declarations as commands to run.
egglog_static::egglog_header!(luminal_base
    (datatype*
        (Expression
            (MNum i64)
            (MFloat f64)
            (MIter)
            (MVar String)
            (MAdd Expression Expression)
            (MSub Expression Expression)
            (MMul Expression Expression)
            (MCeilDiv Expression Expression)
            (MDiv Expression Expression)
            (MMod Expression Expression)
            (MMin Expression Expression)
            (MMax Expression Expression)
            (MAnd Expression Expression)
            (MOr Expression Expression)
            (MGte Expression Expression)
            (MLt Expression Expression)
            (MFloorTo Expression Expression)
            (MReplace Expression Expression Expression))
        (EList
            (ECons Expression EList)
            (ENil)
            (MReplaceList EList Expression Expression)
            (ReplaceNthFromEnd EList Expression i64)
            (RemoveNthFromEnd EList i64)
            (RowMajor EList))
        (DType
            (F32) (F64) (F16) (Bf16) (Int) (Int64) (Bool)
            (F4E2M1) (F8E4M3) (F8E5M2) (F8UE8M0)
            (I4) (U4) (I8) (U8) (I16) (U16) (TF32) (F6E2M3) (F6E3M2)))
    (function lower (Expression) i64 :merge (max old new))
    (function upper (Expression) i64 :merge (min old new))
    (function len (EList) i64 :merge new)
    (function nth_from_end (EList i64) Expression :merge new)
    (function n_elements (EList) Expression :merge new)
);

// ---- Sort classes (pub const) ----

pub const IR: SortClass = SortClass::new("IR");
pub const OP_KIND: SortClass = SortClass::new("OpKind");
pub const ILIST: SortClass = SortClass::new("IList");
pub const EXPRESSION: SortClass = SortClass::new("Expression");
pub const ELIST: SortClass = SortClass::new("EList");
pub const DTYPE: SortClass = SortClass::new("DType");
pub const I64: SortClass = SortClass::new("i64");
pub const F64: SortClass = SortClass::new("f64");
pub const STRING: SortClass = SortClass::new("String");

pub static SORTS: LazyLock<BaseSorts> = LazyLock::new(BaseSorts::new);

// ---- Egglog primitive operations ----

pub fn padd(a: Term, b: Term) -> Term {
    app(&SORTS.p_add, vec![a, b])
}
pub fn psub(a: Term, b: Term) -> Term {
    app(&SORTS.p_sub, vec![a, b])
}
pub fn pmul(a: Term, b: Term) -> Term {
    app(&SORTS.p_mul, vec![a, b])
}
pub fn pdiv(a: Term, b: Term) -> Term {
    app(&SORTS.p_div, vec![a, b])
}
pub fn pmod(a: Term, b: Term) -> Term {
    app(&SORTS.p_mod, vec![a, b])
}
pub fn pmax(a: Term, b: Term) -> Term {
    app(&SORTS.p_max, vec![a, b])
}
pub fn pmin(a: Term, b: Term) -> Term {
    app(&SORTS.p_min, vec![a, b])
}
pub fn pand(a: Term, b: Term) -> Term {
    app(&SORTS.p_and, vec![a, b])
}
pub fn plt(a: Term, b: Term) -> Term {
    app(&SORTS.p_lt, vec![a, b])
}
pub fn pgte(a: Term, b: Term) -> Term {
    app(&SORTS.p_gte, vec![a, b])
}
pub fn peq(a: Term, b: Term) -> Term {
    eq(a, b)
}
pub fn pneq(a: Term, b: Term) -> Term {
    neq(a, b)
}
pub fn interval_lower(e: Term) -> Term {
    app(&func("lower", &["expr"]), vec![e])
}
pub fn interval_upper(e: Term) -> Term {
    app(&func("upper", &["expr"]), vec![e])
}

// ---- Egglog function applications ----

pub fn len_f(l: Term) -> Term {
    app(&SORTS.f_len, vec![l])
}
pub fn nth_f(l: Term, i: Term) -> Term {
    app(&SORTS.f_nth, vec![l, i])
}
pub fn nelem_f(l: Term) -> Term {
    app(&SORTS.f_nelem, vec![l])
}

// ---- Expression term constructors ----

pub fn num(val: Term) -> Term {
    SORTS.m_num.call(("n", val))
}
pub fn float(val: Term) -> Term {
    SORTS.m_float.call(("n", val))
}
pub fn iter() -> Term {
    SORTS.m_iter.call(())
}
pub fn mvar(name: Term) -> Term {
    SORTS.m_var.call(("name", name))
}
pub fn add(a: Term, b: Term) -> Term {
    SORTS.m_add.call([("a", a), ("b", b)])
}
pub fn sub(a: Term, b: Term) -> Term {
    SORTS.m_sub.call([("a", a), ("b", b)])
}
pub fn mul(a: Term, b: Term) -> Term {
    SORTS.m_mul.call([("a", a), ("b", b)])
}
pub fn ceildiv(a: Term, b: Term) -> Term {
    SORTS.m_ceildiv.call([("a", a), ("b", b)])
}
pub fn div(a: Term, b: Term) -> Term {
    SORTS.m_div.call([("a", a), ("b", b)])
}
pub fn modd(a: Term, b: Term) -> Term {
    SORTS.m_mod.call([("a", a), ("b", b)])
}
pub fn min(a: Term, b: Term) -> Term {
    SORTS.m_min.call([("a", a), ("b", b)])
}
pub fn max(a: Term, b: Term) -> Term {
    SORTS.m_max.call([("a", a), ("b", b)])
}
pub fn and(a: Term, b: Term) -> Term {
    SORTS.m_and.call([("a", a), ("b", b)])
}
pub fn or(a: Term, b: Term) -> Term {
    SORTS.m_or.call([("a", a), ("b", b)])
}
pub fn gte(a: Term, b: Term) -> Term {
    SORTS.m_gte.call([("a", a), ("b", b)])
}
pub fn lt(a: Term, b: Term) -> Term {
    SORTS.m_lt.call([("a", a), ("b", b)])
}
pub fn floorto(a: Term, b: Term) -> Term {
    SORTS.m_floorto.call([("a", a), ("b", b)])
}
pub fn replace(x: Term, from: Term, to: Term) -> Term {
    SORTS.m_replace.call([("x", x), ("from", from), ("to", to)])
}

// ---- EList term constructors ----

pub fn cons(head: Term, tail: Term) -> Term {
    SORTS.e_cons.call([("head", head), ("tail", tail)])
}
pub fn nil() -> Term {
    SORTS.e_nil.call(())
}
pub fn replace_list(list: Term, from: Term, to: Term) -> Term {
    SORTS
        .m_replace_list
        .call([("list", list), ("from", from), ("to", to)])
}
pub fn replace_nth(list: Term, to: Term, ind: Term) -> Term {
    SORTS
        .replace_nth_from_end
        .call([("list", list), ("to", to), ("ind", ind)])
}
pub fn remove_nth(list: Term, ind: Term) -> Term {
    SORTS
        .remove_nth_from_end
        .call([("list", list), ("ind", ind)])
}
pub fn rowmajor(list: Term) -> Term {
    SORTS.row_major.call(("list", list))
}

/// All sort classes, sort definitions, and convenience term constructors
/// for the base Expression/EList/DType egglog types.
pub struct BaseSorts {
    // Expression variants
    pub m_num: SortDef,
    pub m_float: SortDef,
    pub m_iter: SortDef,
    pub m_var: SortDef,
    pub m_add: SortDef,
    pub m_sub: SortDef,
    pub m_mul: SortDef,
    pub m_ceildiv: SortDef,
    pub m_div: SortDef,
    pub m_mod: SortDef,
    pub m_min: SortDef,
    pub m_max: SortDef,
    pub m_and: SortDef,
    pub m_or: SortDef,
    pub m_gte: SortDef,
    pub m_lt: SortDef,
    pub m_floorto: SortDef,
    pub m_replace: SortDef,

    // EList variants
    pub e_cons: SortDef,
    pub e_nil: SortDef,
    pub m_replace_list: SortDef,
    pub replace_nth_from_end: SortDef,
    pub remove_nth_from_end: SortDef,
    pub row_major: SortDef,

    // DType variants
    pub f32_dt: SortDef,
    pub f64_dt: SortDef,
    pub f16_dt: SortDef,
    pub bf16_dt: SortDef,
    pub int_dt: SortDef,
    /// Egglog sort for `DType::I64`. Named `"Int64"` (not `"I64"`) to avoid
    /// shadowing egglog's built-in `I64` primitive sort.
    pub int64_dt: SortDef,
    pub bool_dt: SortDef,
    pub f4e2m1_dt: SortDef,
    pub f8e4m3_dt: SortDef,
    pub f8e5m2_dt: SortDef,
    pub f8ue8m0_dt: SortDef,
    pub i4_dt: SortDef,
    pub u4_dt: SortDef,
    pub i8_dt: SortDef,
    pub u8_dt: SortDef,
    pub i16_dt: SortDef,
    pub u16_dt: SortDef,
    pub tf32_dt: SortDef,
    pub f6e2m3_dt: SortDef,
    pub f6e3m2_dt: SortDef,
    // Egglog builtin primitives (for term construction only)
    pub p_add: SortDef,
    pub p_sub: SortDef,
    pub p_mul: SortDef,
    pub p_div: SortDef,
    pub p_mod: SortDef,
    pub p_max: SortDef,
    pub p_min: SortDef,
    pub p_and: SortDef,
    pub p_lt: SortDef,
    pub p_gte: SortDef,

    // Egglog function defs (for term construction only)
    pub f_len: SortDef,
    pub f_nth: SortDef,
    pub f_nelem: SortDef,
}

impl Default for BaseSorts {
    fn default() -> Self {
        Self::new()
    }
}

impl BaseSorts {
    pub fn new() -> Self {
        Self {
            m_num: sort(EXPRESSION, "MNum", &[("n", I64)]),
            m_float: sort(EXPRESSION, "MFloat", &[("n", F64)]),
            m_iter: sort(EXPRESSION, "MIter", &[]),
            m_var: sort(EXPRESSION, "MVar", &[("name", STRING)]),
            m_add: sort(EXPRESSION, "MAdd", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_sub: sort(EXPRESSION, "MSub", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_mul: sort(EXPRESSION, "MMul", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_ceildiv: sort(
                EXPRESSION,
                "MCeilDiv",
                &[("a", EXPRESSION), ("b", EXPRESSION)],
            ),
            m_div: sort(EXPRESSION, "MDiv", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_mod: sort(EXPRESSION, "MMod", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_min: sort(EXPRESSION, "MMin", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_max: sort(EXPRESSION, "MMax", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_and: sort(EXPRESSION, "MAnd", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_or: sort(EXPRESSION, "MOr", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_gte: sort(EXPRESSION, "MGte", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_lt: sort(EXPRESSION, "MLt", &[("a", EXPRESSION), ("b", EXPRESSION)]),
            m_floorto: sort(
                EXPRESSION,
                "MFloorTo",
                &[("a", EXPRESSION), ("b", EXPRESSION)],
            ),
            m_replace: sort(
                EXPRESSION,
                "MReplace",
                &[("x", EXPRESSION), ("from", EXPRESSION), ("to", EXPRESSION)],
            ),

            e_cons: sort(ELIST, "ECons", &[("head", EXPRESSION), ("tail", ELIST)]),
            e_nil: sort(ELIST, "ENil", &[]),
            m_replace_list: sort(
                ELIST,
                "MReplaceList",
                &[("list", ELIST), ("from", EXPRESSION), ("to", EXPRESSION)],
            ),
            replace_nth_from_end: sort(
                ELIST,
                "ReplaceNthFromEnd",
                &[("list", ELIST), ("to", EXPRESSION), ("ind", I64)],
            ),
            remove_nth_from_end: sort(ELIST, "RemoveNthFromEnd", &[("list", ELIST), ("ind", I64)]),
            row_major: sort(ELIST, "RowMajor", &[("list", ELIST)]),

            f32_dt: sort(DTYPE, "F32", &[]),
            f64_dt: sort(DTYPE, "F64", &[]),
            f16_dt: sort(DTYPE, "F16", &[]),
            bf16_dt: sort(DTYPE, "Bf16", &[]),
            int_dt: sort(DTYPE, "Int", &[]),
            int64_dt: sort(DTYPE, "Int64", &[]),
            bool_dt: sort(DTYPE, "Bool", &[]),
            f4e2m1_dt: sort(DTYPE, "F4E2M1", &[]),
            f8e4m3_dt: sort(DTYPE, "F8E4M3", &[]),
            f8e5m2_dt: sort(DTYPE, "F8E5M2", &[]),
            f8ue8m0_dt: sort(DTYPE, "F8UE8M0", &[]),
            i4_dt: sort(DTYPE, "I4", &[]),
            u4_dt: sort(DTYPE, "U4", &[]),
            i8_dt: sort(DTYPE, "I8", &[]),
            u8_dt: sort(DTYPE, "U8", &[]),
            i16_dt: sort(DTYPE, "I16", &[]),
            u16_dt: sort(DTYPE, "U16", &[]),
            tf32_dt: sort(DTYPE, "TF32", &[]),
            f6e2m3_dt: sort(DTYPE, "F6E2M3", &[]),
            f6e3m2_dt: sort(DTYPE, "F6E3M2", &[]),
            p_add: func("+", &["a", "b"]),
            p_sub: func("-", &["a", "b"]),
            p_mul: func("*", &["a", "b"]),
            p_div: func("/", &["a", "b"]),
            p_mod: func("%", &["a", "b"]),
            p_max: func("max", &["a", "b"]),
            p_min: func("min", &["a", "b"]),
            p_and: func("&", &["a", "b"]),
            p_lt: func("<", &["a", "b"]),
            p_gte: func(">=", &["a", "b"]),

            f_len: func("len", &["list"]),
            f_nth: func("nth_from_end", &["list", "index"]),
            f_nelem: func("n_elements", &["list"]),
        }
    }
}

pub fn dtype(e: Term) -> Term {
    app(&func("dtype", &["inp"]), vec![e])
}

pub fn interval_facts_egglog(
    intervals: &shape::DynDimIntervals,
    vars: impl IntoIterator<Item = char>,
) -> String {
    let mut all_vars = FxHashSet::default();
    all_vars.extend(intervals.keys().copied());
    all_vars.extend(vars);

    let mut all_vars = all_vars.into_iter().collect::<Vec<_>>();
    all_vars.sort_unstable();

    let mut out = String::new();
    for var in all_vars {
        let interval = intervals
            .get(&var)
            .copied()
            .unwrap_or_else(shape::DimInterval::unbounded);
        let var_expr = mvar(str(&var.to_string()));
        out.push_str(&format!(
            "(set {} {})\n",
            interval_lower(var_expr.clone()),
            interval.min
        ));
        out.push_str(&format!(
            "(set {} {})\n",
            interval_upper(var_expr),
            interval.max
        ));
        if interval.min == interval.max {
            out.push_str(&format!(
                "(union {} {})\n",
                mvar(str(&var.to_string())),
                num(i64(interval.min))
            ));
        }
    }
    out
}

// ---- Normalized Op helpers ----

/// Build an `(Op kind inputs)` IR term.
pub fn op_term(kind: Term, inputs: Term) -> Term {
    call_named("Op", vec![kind, inputs])
}

/// Build an IList from IR terms: `(ICons t1 (ICons t2 (INil)))`.
pub fn ilist(terms: Vec<Term>) -> Term {
    terms
        .into_iter()
        .rev()
        .fold(call_named("INil", vec![]), |tail, head| {
            call_named("ICons", vec![head, tail])
        })
}

/// Construct a normalized Op call from an OpKind SortDef + named args + input terms.
/// Returns (args, full_op_term) where the op_term is `(Op (XxxKind ...) (ICons ...))`.
pub fn new_op_call(kind_sort: &SortDef, input_names: &[&str]) -> (Args, Term) {
    let (mut args, kind_term) = kind_sort.new_call();
    // Create variables for each input
    let prefix = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        format!("inp{}", COUNTER.fetch_add(1, Ordering::Relaxed))
    };
    let input_vars: Vec<Term> = input_names
        .iter()
        .map(|name| {
            let var = v(format!("{prefix}_{name}"));
            args.add(name, var.clone());
            var
        })
        .collect();
    let inputs_term = ilist(input_vars);
    args.add("__inputs", inputs_term.clone());
    let op = op_term(kind_term, inputs_term);
    (args, op)
}

pub fn base_expression_egglog() -> String {
    base_expression_egglog_impl(false)
}

pub fn base_expression_egglog_with_intervals() -> String {
    base_expression_egglog_impl(true)
}

/// The base Expression/EList/DType datatypes plus every algebraic rewrite,
/// replacement rule, and list helper — authored directly as egglog via the
/// `egglog!` quasiquote (this replaced a large hand-rolled `Program`/`Rule`
/// builder). Base constructors are only ever *built* positionally, so the
/// datatypes are declared positionally and a default parser suffices; the
/// parsed commands are rendered back to text so the existing text-concatenation
/// setup pipeline is unchanged.
fn base_expression_egglog_impl(use_interval_analysis: bool) -> String {
    // The datatypes + helper functions come from the shared `luminal_base`
    // schema (checked once at compile time). Run those, then extend with the
    // rewrite rules — each is `egglog_static!`-checked against that schema, so a
    // bad constructor or arity is a build error, not a run-time failure.
    let mut commands = luminal_base_schema().expect("base schema should typecheck");
    commands.extend(
        egglog_static!(
            luminal_base;
            (ruleset expr)
            (ruleset dtype_prop)
            (ruleset cleanup)
            (ruleset post_cleanup)
        (rule ((= ?__rw (MMul ?a ?b))) ((union ?__rw (MMul ?b ?a))) :ruleset expr :name "mul-comm")
        (rule ((= ?__rw (MAdd ?a ?b))) ((union ?__rw (MAdd ?b ?a))) :ruleset expr :name "add-comm")
        (rule ((= ?e (MAdd (MNum ?a) (MNum ?b))) (= ?ans (+ ?a ?b))) ((union ?e (MNum ?ans)) (subsume (MAdd (MNum ?a) (MNum ?b)))) :ruleset expr)
        (rule ((= ?__rw (MSub (MNum ?a) (MNum ?b)))) ((union ?__rw (MNum (- ?a ?b)))) :ruleset expr :name "sub-const")
        (rule ((= ?e (MMul (MNum ?a) (MNum ?b))) (= ?prod (* ?a ?b))) ((union ?e (MNum ?prod)) (subsume (MMul (MNum ?a) (MNum ?b)))) :ruleset expr)
        (rule ((= ?expr (MMul (MMul ?x (MNum ?a)) (MNum ?b))) (= ?prod (* ?a ?b))) ((union ?expr (MMul ?x (MNum ?prod)))) :ruleset expr :name "fold-right-associated-const-mul")
        (rule ((= ?expr (MMul (MNum ?b) (MMul ?x (MNum ?a)))) (= ?prod (* ?a ?b))) ((union ?expr (MMul ?x (MNum ?prod)))) :ruleset expr :name "fold-left-associated-const-mul")
        (rule ((= ?__rw (MDiv (MNum ?a) (MNum ?b))) (!= 0 ?b)) ((union ?__rw (MNum (/ ?a ?b)))) :ruleset expr :name "div-const")
        (rule ((= ?__rw (MDiv ?a ?a))) ((union ?__rw (MNum 1))) :ruleset expr :name "div-self")
        (rule ((= ?__rw (MDiv (MMul ?x (MNum ?n)) (MNum ?n))) (>= ?n 1)) ((union ?__rw ?x)) :ruleset expr :name "div-mul-num-self")
        (rule ((= ?__rw (MDiv (MAdd (MMul ?x (MNum ?n)) (MNum ?r)) (MNum ?n))) (>= ?n 1) (>= ?r 0) (< ?r ?n)) ((union ?__rw ?x)) :ruleset expr :name "div-mul-num-plus-rem")
        (rule ((= ?__rw (MCeilDiv (MNum ?a) (MNum ?b))) (!= 0 ?b) (= 0 (% ?a ?b))) ((union ?__rw (MNum (/ ?a ?b)))) :ruleset expr :name "ceildiv-const")
        (rule ((= ?__rw (MMax (MNum ?a) (MNum ?b)))) ((union ?__rw (MNum (max ?a ?b)))) :ruleset expr :name "max-const")
        (rule ((= ?__rw (MMin (MNum ?a) (MNum ?b)))) ((union ?__rw (MNum (min ?a ?b)))) :ruleset expr :name "min-const")
        (rule ((= ?__rw (MAnd (MNum ?a) (MNum ?b)))) ((union ?__rw (MNum (& ?a ?b)))) :ruleset expr :name "and-const")
        (rule ((= ?__rw (MFloat -1.0))) ((union ?__rw (MNum -1))) :ruleset expr :name "float-neg1-to-num")
        (rule ((= ?__rw (MNum -1))) ((union ?__rw (MFloat -1.0))) :ruleset expr :name "num-neg1-to-float")
        (rule ((= ?__rw (MAdd ?a (MNum 0)))) ((union ?__rw ?a)) :ruleset expr :name "add-zero")
        (rule ((= ?e (MMul ?a (MNum 1)))) ((union ?e ?a)) :ruleset expr)
        (rule ((= ?e (MMul ?a (MNum 0)))) ((union ?e (MNum 0)) (subsume (MMul ?a (MNum 0)))) :ruleset expr)
        (rule ((= ?__rw (MDiv ?a (MNum 1)))) ((union ?__rw ?a)) :ruleset expr :name "div-one")
        (rule ((= ?__rw (MMod (MMul ?x ?y) ?y))) ((union ?__rw (MNum 0))) :ruleset expr :name "mod-mul-self")
        (rule ((= ?__rw (MMod (MNum ?a) (MNum ?b))) (!= 0 ?b)) ((union ?__rw (MNum (% ?a ?b)))) :ruleset expr :name "mod-const")
        (rule ((= ?__rw (MMod (MAdd (MMul ?x (MNum ?n)) (MNum ?r)) (MNum ?n))) (>= ?n 1) (>= ?r 0) (< ?r ?n)) ((union ?__rw (MNum ?r))) :ruleset expr :name "mod-mul-num-plus-rem")
        (rule ((= ?__rw (MMod (MMod ?x (MNum ?y)) (MNum ?z))) (>= ?z ?y) (= 0 (% ?y ?z))) ((union ?__rw (MMod ?x (MNum ?y)))) :ruleset expr :name "mod-mod-larger")
        (rule ((= ?__rw (MMod (MMod ?x (MNum ?y)) (MNum ?z))) (>= ?y ?z) (= 0 (% ?z ?y))) ((union ?__rw (MMod ?x (MNum ?z)))) :ruleset expr :name "mod-mod-smaller")
        (rule ((= ?__rw (MAdd (MMul (MDiv ?z ?x) ?x) (MMod ?z ?x)))) ((union ?__rw ?z)) :ruleset expr :name "merge-dims")
        (rule ((= ?__rw (MDiv (MDiv ?a (MNum ?b)) (MNum ?c))) (>= ?b 1) (>= ?c 1) (< ?b 3037000500) (< ?c 3037000500)) ((union ?__rw (MDiv ?a (MNum (* ?b ?c))))) :ruleset expr :name "div-div-num")
        (rule ((= ?__rw (MAdd (MDiv ?a ?b) ?c))) ((union ?__rw (MDiv (MAdd ?a (MMul ?c ?b)) ?b))) :ruleset expr :name "add-div")
        (rule ((= ?__rw (MAdd ?a (MSub ?b ?a)))) ((union ?__rw ?b)) :ruleset expr :name "add-sub-cancel")
        (rule ((= ?__rw (MAdd (MSub ?b ?a) ?a))) ((union ?__rw ?b)) :ruleset expr :name "add-sub-cancel2")
        (rule ((= ?__rw (MSub ?a ?a))) ((union ?__rw (MNum 0))) :ruleset expr :name "sub-self")
        (rule ((= ?__rw (MAdd (MSub ?a (MNum ?b)) (MNum ?c)))) ((union ?__rw (MSub ?a (MNum (- ?b ?c))))) :ruleset expr :name "add-sub-const")
        (rule ((= ?__rw (MAdd (MNum ?c) (MSub ?a (MNum ?b))))) ((union ?__rw (MSub ?a (MNum (- ?b ?c))))) :ruleset expr :name "add-sub-const2")
        (rule ((= ?__rw (MSub (MAdd ?a (MNum ?b)) (MNum ?c)))) ((union ?__rw (MAdd ?a (MNum (- ?b ?c))))) :ruleset expr :name "sub-add-const")
        (rule ((= ?__rw (MSub (MSub ?a (MNum ?b)) (MNum ?c)))) ((union ?__rw (MSub ?a (MNum (+ ?b ?c))))) :ruleset expr :name "sub-sub-const")
        (rule ((= ?__rw (MAdd (MMul ?a ?b) (MMul ?a ?c)))) ((union ?__rw (MMul ?a (MAdd ?b ?c)))) :ruleset expr :name "factor")
        (rule ((= ?__rw (MAdd ?a ?a))) ((union ?__rw (MMul (MNum 2) ?a))) :ruleset expr :name "double")
        (rule ((= ?e (MAdd (MAdd ?a (MNum ?b)) (MNum ?c))) (= ?ans (+ ?b ?c))) ((union ?e (MAdd ?a (MNum ?ans))) (subsume (MAdd (MAdd ?a (MNum ?b)) (MNum ?c)))) :ruleset expr)
        (rule ((= ?__rw (MAdd (MAdd (MNum ?b) (MVar ?v)) (MNum ?c)))) ((union ?__rw (MAdd (MVar ?v) (MNum (+ ?b ?c))))) :ruleset expr :name "add-assoc-var")
        (rule ((= ?__rw (MAdd (MAdd (MNum ?b) (MMul ?n ?a)) (MNum ?c)))) ((union ?__rw (MAdd (MMul ?n ?a) (MNum (+ ?b ?c))))) :ruleset expr :name "add-assoc-mul")
        (rule ((= ?__rw (MAdd (MMul (MNum ?n) ?a) ?a))) ((union ?__rw (MMul (MNum (+ ?n 1)) ?a)) (subsume (MAdd (MMul (MNum ?n) ?a) ?a))) :ruleset expr :name "combine-like-1")
        (rule ((= ?__rw (MAdd ?a (MMul (MNum ?n) ?a)))) ((union ?__rw (MMul (MNum (+ ?n 1)) ?a)) (subsume (MAdd ?a (MMul (MNum ?n) ?a)))) :ruleset expr :name "combine-like-2")
        (rule ((= ?__rw (MAdd (MMul ?a (MNum ?n)) ?a))) ((union ?__rw (MMul (MNum (+ ?n 1)) ?a)) (subsume (MAdd (MMul ?a (MNum ?n)) ?a))) :ruleset expr :name "combine-like-3")
        (rule ((= ?__rw (MAdd ?a (MMul ?a (MNum ?n))))) ((union ?__rw (MMul (MNum (+ ?n 1)) ?a)) (subsume (MAdd ?a (MMul ?a (MNum ?n))))) :ruleset expr :name "combine-like-4")
        (rule ((= ?__rw (MAdd (MAdd ?a (MVar ?v)) (MVar ?v)))) ((union ?__rw (MAdd ?a (MMul (MNum 2) (MVar ?v)))) (subsume (MAdd (MAdd ?a (MVar ?v)) (MVar ?v)))) :ruleset expr :name "combine-var-1")
        (rule ((= ?__rw (MAdd (MAdd (MVar ?v) ?a) (MVar ?v)))) ((union ?__rw (MAdd ?a (MMul (MNum 2) (MVar ?v)))) (subsume (MAdd (MAdd (MVar ?v) ?a) (MVar ?v)))) :ruleset expr :name "combine-var-2")
        (rule ((= ?__rw (MAdd (MAdd (MMul (MNum ?n) ?a) ?b) ?a))) ((union ?__rw (MAdd (MMul (MNum (+ ?n 1)) ?a) ?b)) (subsume (MAdd (MAdd (MMul (MNum ?n) ?a) ?b) ?a))) :ruleset expr :name "accum-1")
        (rule ((= ?__rw (MAdd (MAdd ?b (MMul (MNum ?n) ?a)) ?a))) ((union ?__rw (MAdd ?b (MMul (MNum (+ ?n 1)) ?a))) (subsume (MAdd (MAdd ?b (MMul (MNum ?n) ?a)) ?a))) :ruleset expr :name "accum-2")
        (rule ((= ?__rw (MReplace ?x ?y ?z)) (= ?x ?y)) ((union ?__rw ?z)) :ruleset expr :name "replace-match")
        (rule ((= ?__rw (MReplace (MAdd ?a ?b) ?x ?y))) ((union ?__rw (MAdd (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MAdd")
        (rule ((= ?__rw (MReplace (MSub ?a ?b) ?x ?y))) ((union ?__rw (MSub (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MSub")
        (rule ((= ?__rw (MReplace (MMul ?a ?b) ?x ?y))) ((union ?__rw (MMul (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MMul")
        (rule ((= ?__rw (MReplace (MDiv ?a ?b) ?x ?y))) ((union ?__rw (MDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MDiv")
        (rule ((= ?__rw (MReplace (MCeilDiv ?a ?b) ?x ?y))) ((union ?__rw (MCeilDiv (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MCeilDiv")
        (rule ((= ?__rw (MReplace (MMod ?a ?b) ?x ?y))) ((union ?__rw (MMod (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MMod")
        (rule ((= ?__rw (MReplace (MMin ?a ?b) ?x ?y))) ((union ?__rw (MMin (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MMin")
        (rule ((= ?__rw (MReplace (MMax ?a ?b) ?x ?y))) ((union ?__rw (MMax (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MMax")
        (rule ((= ?__rw (MReplace (MFloorTo ?a ?b) ?x ?y))) ((union ?__rw (MFloorTo (MReplace ?a ?x ?y) (MReplace ?b ?x ?y)))) :ruleset expr :name "replace-MFloorTo")
        (rule ((= ?__rw (MReplace (MNum ?n) ?x ?y))) ((union ?__rw (MNum ?n))) :ruleset expr :name "replace-num")
        (rule ((= ?__rw (MReplace (MVar ?z) ?find ?replace)) (!= ?find (MVar ?z))) ((union ?__rw (MVar ?z))) :ruleset expr :name "replace-var-miss")
        (rule ((= ?__rw (MReplace (MIter) ?find ?replace)) (!= ?find (MIter))) ((union ?__rw (MIter))) :ruleset expr :name "replace-iter-miss")
        (rule ((= ?e (ENil))) ((set (len ?e) 0)) :ruleset expr)
        (rule ((= ?e (ECons ?expr ?list)) (= ?prev_len (len ?list))) ((set (len ?e) (+ ?prev_len 1))) :ruleset expr)
        (rule ((= ?e (ECons ?expr ?list)) (= ?list_len (len ?list))) ((set (nth_from_end ?e ?list_len) ?expr)) :ruleset expr)
        (rule ((= ?e (ECons ?expr ?list)) (= ?other_nth (nth_from_end ?list ?n))) ((set (nth_from_end ?e ?n) ?other_nth)) :ruleset expr)
        (rule ((= ?e (ENil))) ((set (n_elements ?e) (MNum 1))) :ruleset expr)
        (rule ((= ?e (ECons ?dim ?other)) (= ?other_elems (n_elements ?other))) ((set (n_elements ?e) (MMul ?dim ?other_elems))) :ruleset expr)
        (rule ((= ?other (ECons ?other_dim ?other_other)) (= ?list (ECons ?d ?other)) (= ?e (RowMajor ?list)) (= ?n_elems (n_elements ?other))) ((union ?e (ECons (MMul ?n_elems (MIter)) (RowMajor ?other)))) :ruleset expr)
        (rule ((= ?__rw (RowMajor (ECons ?dim (ENil))))) ((union ?__rw (ECons (MIter) (ENil)))) :ruleset expr :name "rowmajor-base")
        (rule ((= ?__rw (MReplaceList (ECons ?expr ?list) ?from ?to))) ((union ?__rw (ECons (MReplace ?expr ?from ?to) (MReplaceList ?list ?from ?to)))) :ruleset expr :name "replace-list-cons")
        (rule ((= ?e (ReplaceNthFromEnd (ECons ?expr ?list) ?to ?ind)) (= ?ind (len ?list))) ((union ?e (ECons ?to ?list))) :ruleset expr)
        (rule ((= ?e (ReplaceNthFromEnd (ECons ?expr ?list) ?to ?ind)) (< ?ind (len ?list))) ((union ?e (ECons ?expr (ReplaceNthFromEnd ?list ?to ?ind)))) :ruleset expr)
        (rule ((= ?e (RemoveNthFromEnd (ECons ?expr ?list) ?ind)) (= ?ind (len ?list))) ((union ?e ?list)) :ruleset expr)
        (rule ((= ?e (RemoveNthFromEnd (ECons ?expr ?list) ?ind)) (< ?ind (len ?list))) ((union ?e (ECons ?expr (RemoveNthFromEnd ?list ?ind)))) :ruleset expr)
        )
        .expect("base rules should typecheck"),
    );

    if use_interval_analysis {
        commands.extend(
            egglog_static!(
                luminal_base;
                (ruleset interval_expr)
                (rule ((= ?e (MNum ?n))) ((set (lower ?e) ?n) (set (upper ?e) ?n)) :ruleset interval_expr :name "interval-num-exact")
                (rule ((= ?e (MAdd ?a ?b)) (= ?lo_a (lower ?a)) (= ?lo_b (lower ?b)) (= ?sum (+ ?lo_a ?lo_b)) (>= ?lo_a 0) (>= ?lo_b 0) (>= (- 9223372036854775807 ?lo_b) ?lo_a)) ((set (lower ?e) ?sum)) :ruleset interval_expr :name "interval-add-lower-nonnegative")
                (rule ((= ?e (MAdd ?a ?b)) (= ?hi_a (upper ?a)) (= ?hi_b (upper ?b)) (= ?sum (+ ?hi_a ?hi_b)) (< ?hi_a 9223372036854775807) (< ?hi_b 9223372036854775807) (>= (- 9223372036854775807 ?hi_b) ?hi_a)) ((set (upper ?e) ?sum)) :ruleset interval_expr :name "interval-add-upper-finite")
                (rule ((= ?e (MMin ?a ?b)) (= ?lo_a (lower ?a)) (= ?lo_b (lower ?b))) ((set (lower ?e) (min ?lo_a ?lo_b))) :ruleset interval_expr :name "interval-min-lower")
                (rule ((= ?e (MMin ?a ?b)) (= ?hi_a (upper ?a)) (= ?hi_b (upper ?b))) ((set (upper ?e) (min ?hi_a ?hi_b))) :ruleset interval_expr :name "interval-min-upper")
                (rule ((= ?e (MMax ?a ?b)) (= ?lo_a (lower ?a)) (= ?lo_b (lower ?b))) ((set (lower ?e) (max ?lo_a ?lo_b))) :ruleset interval_expr :name "interval-max-lower")
                (rule ((= ?e (MMax ?a ?b)) (= ?hi_a (upper ?a)) (= ?hi_b (upper ?b))) ((set (upper ?e) (max ?hi_a ?hi_b))) :ruleset interval_expr :name "interval-max-upper")
                (rule ((= ?__rw (MLt ?x (MNum ?n))) (= ?hi (upper ?x)) (< ?hi ?n)) ((union ?__rw (MNum 1))) :ruleset interval_expr :name "interval-lt-true")
                (rule ((= ?__rw (MLt ?x (MNum ?n))) (= ?lo (lower ?x)) (>= ?lo ?n)) ((union ?__rw (MNum 0))) :ruleset interval_expr :name "interval-lt-false")
                (rule ((= ?__rw (MGte ?x (MNum ?n))) (= ?lo (lower ?x)) (>= ?lo ?n)) ((union ?__rw (MNum 1))) :ruleset interval_expr :name "interval-gte-true")
                (rule ((= ?__rw (MGte ?x (MNum ?n))) (= ?hi (upper ?x)) (< ?hi ?n)) ((union ?__rw (MNum 0))) :ruleset interval_expr :name "interval-gte-false")
                (rule ((= ?__rw (MMin ?x (MNum ?n))) (= ?hi (upper ?x)) (>= ?n ?hi)) ((union ?__rw ?x)) :ruleset interval_expr :name "interval-min-right-identity")
                (rule ((= ?__rw (MMax ?x (MNum ?n))) (= ?lo (lower ?x)) (>= ?lo ?n)) ((union ?__rw ?x)) :ruleset interval_expr :name "interval-max-right-identity")
                (rule ((= ?__rw (MMod ?x (MNum ?n))) (>= ?n 1) (= ?lo (lower ?x)) (= ?hi (upper ?x)) (>= ?lo 0) (< ?hi ?n)) ((union ?__rw ?x)) :ruleset interval_expr :name "interval-mod-small")
                (rule ((= ?__rw (MDiv ?x (MNum ?n))) (>= ?n 1) (= ?lo (lower ?x)) (= ?hi (upper ?x)) (>= ?lo 0) (< ?hi ?n)) ((union ?__rw (MNum 0))) :ruleset interval_expr :name "interval-div-small")
            )
            .expect("base interval egglog program should parse"),
        );
    }

    commands
        .iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The base "cleanup" ruleset: rules that delete intermediate helper nodes
/// (MReplace, MReplaceList, ReplaceNthFromEnd, RemoveNthFromEnd, RowMajor, and
/// the helper functions len, nth_from_end, n_elements) once simplification has
/// consumed them. Authored via `egglog!` and rendered to text like the base
/// program above.
pub fn base_cleanup_egglog() -> String {
    egglog_static!(
        luminal_base;
        (ruleset base_cleanup)
        (rule ((= ?m (MReplace ?a ?b ?c))) ((delete (MReplace ?a ?b ?c))) :ruleset base_cleanup)
        (rule ((= ?m (MReplaceList ?a ?b ?c))) ((delete (MReplaceList ?a ?b ?c))) :ruleset base_cleanup)
        (rule ((= ?m (ReplaceNthFromEnd ?a ?b ?c))) ((delete (ReplaceNthFromEnd ?a ?b ?c))) :ruleset base_cleanup)
        (rule ((= ?m (RemoveNthFromEnd ?a ?b))) ((delete (RemoveNthFromEnd ?a ?b))) :ruleset base_cleanup)
        (rule ((= ?m (RowMajor ?x))) ((delete (RowMajor ?x))) :ruleset base_cleanup)
        (rule ((= ?m (len ?x))) ((delete (len ?x))) :ruleset base_cleanup)
        (rule ((= ?m (nth_from_end ?x ?y))) ((delete (nth_from_end ?x ?y))) :ruleset base_cleanup)
        (rule ((= ?m (n_elements ?x))) ((delete (n_elements ?x))) :ruleset base_cleanup)
    )
    .expect("base cleanup egglog program should parse")
    .iter()
    .map(|c| c.to_string())
    .collect::<Vec<_>>()
    .join("\n")
}

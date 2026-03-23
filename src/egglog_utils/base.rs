use std::sync::LazyLock;

use super::api::*;
use crate::shape::{self, ToShape};

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

// ---- Conversions from shape types to egglog terms ----

/// Convert a shape `Expression` into an egglog `Term`.
pub fn expr_to_term(expr: &shape::Expression) -> Term {
    let mut stack = Vec::new();
    for term in expr.terms.read().iter() {
        let t = match term {
            shape::Term::Num(n) => num(i64(*n)),
            shape::Term::Var(c) => mvar(str(&c.to_string())),
            op => {
                let a = stack.pop().unwrap();
                let b = stack.pop().unwrap();
                match op {
                    shape::Term::Add => add(a, b),
                    shape::Term::Sub => sub(a, b),
                    shape::Term::Mul => mul(a, b),
                    shape::Term::Div => div(a, b),
                    shape::Term::CeilDiv => ceildiv(a, b),
                    shape::Term::Mod => modd(a, b),
                    shape::Term::Min => min(a, b),
                    shape::Term::Max => max(a, b),
                    shape::Term::And => and(a, b),
                    shape::Term::Or => or(a, b),
                    shape::Term::Gte => gte(a, b),
                    shape::Term::Lt => lt(a, b),
                    _ => unreachable!(),
                }
            }
        };
        stack.push(t);
    }
    stack.pop().unwrap()
}

/// Convert a shape (anything implementing `ToShape`) into an egglog `EList` term.
pub fn shape_to_elist(shape: impl ToShape) -> Term {
    shape
        .to_shape()
        .iter()
        .rev()
        .fold(nil(), |acc, expr| cons(expr_to_term(expr), acc))
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
    pub f16_dt: SortDef,
    pub bf16_dt: SortDef,
    pub int_dt: SortDef,
    pub bool_dt: SortDef,
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
            f16_dt: sort(DTYPE, "F16", &[]),
            bf16_dt: sort(DTYPE, "Bf16", &[]),
            int_dt: sort(DTYPE, "Int", &[]),
            bool_dt: sort(DTYPE, "Bool", &[]),
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

    /// Register all sort classes and variants into a Program.
    pub fn register(&self, p: &mut Program) {
        p.add_class(EXPRESSION);
        p.add_class(ELIST);
        p.add_class(DTYPE);

        for s in [
            &self.m_num,
            &self.m_float,
            &self.m_iter,
            &self.m_var,
            &self.m_add,
            &self.m_sub,
            &self.m_mul,
            &self.m_ceildiv,
            &self.m_div,
            &self.m_mod,
            &self.m_min,
            &self.m_max,
            &self.m_and,
            &self.m_or,
            &self.m_gte,
            &self.m_lt,
            &self.m_floorto,
            &self.m_replace,
            &self.e_cons,
            &self.e_nil,
            &self.m_replace_list,
            &self.replace_nth_from_end,
            &self.remove_nth_from_end,
            &self.row_major,
            &self.f32_dt,
            &self.f16_dt,
            &self.bf16_dt,
            &self.int_dt,
            &self.bool_dt,
        ] {
            p.add_sort(s);
        }
    }
}

pub fn dtype(e: Term) -> Term {
    app(&func("dtype", &["inp"]), vec![e])
}

// ---- Normalized Op helpers ----

/// Build an `(Op kind inputs)` IR term.
pub fn op_term(kind: Term, inputs: Term) -> Term {
    Term::App {
        variant: "Op".to_string(),
        args: vec![kind, inputs],
    }
}

/// Build an IList from IR terms: `(ICons t1 (ICons t2 (INil)))`.
pub fn ilist(terms: Vec<Term>) -> Term {
    terms.into_iter().rev().fold(
        Term::App {
            variant: "INil".to_string(),
            args: vec![],
        },
        |tail, head| Term::App {
            variant: "ICons".to_string(),
            args: vec![head, tail],
        },
    )
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

/// Generate the egglog program equivalent to `base.egg`.
///
/// This builds the Expression, EList, and DType datatypes along with all
/// algebraic rewrites, replacement rules, and list helper functions.
pub fn base_expression_egglog() -> String {
    let s = BaseSorts::new();

    // Build the program
    let mut p = Program {
        mutual_recursive: true,
        ..Default::default()
    };

    // Rulesets
    p.add_ruleset("expr");
    p.add_ruleset("cleanup");
    p.add_ruleset("early");

    // Register all sorts
    s.register(&mut p);

    // ---- Algebraic rewrites ----
    // Commutativity
    p.add_rule(rewrite("mul-comm", mul(v("a"), v("b")), mul(v("b"), v("a"))).ruleset("expr"));
    p.add_rule(rewrite("add-comm", add(v("a"), v("b")), add(v("b"), v("a"))).ruleset("expr"));

    // Constant folding: add
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), add(num(v("a")), num(v("b")))),
                peq(v("?ans"), padd(v("a"), v("b"))),
            ])
            .actions(vec![
                Action::Union(v("?e"), num(v("?ans"))),
                Action::Subsume(add(num(v("a")), num(v("b")))),
            ])
            .ruleset("expr"),
    );

    // Constant folding: sub
    p.add_rule(
        rewrite(
            "sub-const",
            sub(num(v("a")), num(v("b"))),
            num(psub(v("a"), v("b"))),
        )
        .ruleset("expr"),
    );

    // Constant folding: mul
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), mul(num(v("?a")), num(v("?b")))),
                peq(v("?prod"), pmul(v("?a"), v("?b"))),
            ])
            .actions(vec![
                union(v("?e"), num(v("?prod"))),
                subsume(mul(num(v("?a")), num(v("?b")))),
            ])
            .ruleset("expr"),
    );

    // Constant folding: div (with conditions)
    p.add_rule(
        rewrite(
            "div-const",
            div(num(v("a")), num(v("b"))),
            num(pdiv(v("a"), v("b"))),
        )
        .when(vec![
            pneq(i64(0), v("b")),
            peq(i64(0), pmod(v("a"), v("b"))),
        ])
        .ruleset("expr"),
    );

    // Cancel common factor in division: (a*b)/(a*c) → b/c
    p.add_rule(
        rewrite(
            "div-cancel-factor",
            div(mul(v("a"), v("b")), mul(v("a"), v("c"))),
            div(v("b"), v("c")),
        )
        .ruleset("expr"),
    );

    // Division self-cancel: a/a → 1
    p.add_rule(rewrite("div-self", div(v("a"), v("a")), num(i64(1))).ruleset("expr"));

    // Constant folding: ceildiv
    p.add_rule(
        rewrite(
            "ceildiv-const",
            ceildiv(num(v("a")), num(v("b"))),
            num(pdiv(v("a"), v("b"))),
        )
        .when(vec![
            pneq(i64(0), v("b")),
            peq(i64(0), pmod(v("a"), v("b"))),
        ])
        .ruleset("expr"),
    );

    // Constant folding: max, min, and
    p.add_rule(
        rewrite(
            "max-const",
            max(num(v("a")), num(v("b"))),
            num(pmax(v("a"), v("b"))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "min-const",
            min(num(v("a")), num(v("b"))),
            num(pmin(v("a"), v("b"))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "and-const",
            and(num(v("a")), num(v("b"))),
            num(pand(v("a"), v("b"))),
        )
        .ruleset("expr"),
    );

    // Float <-> Num for -1
    p.add_rule(rewrite("float-neg1-to-num", float(f64(-1.0)), num(i64(-1))).ruleset("expr"));
    p.add_rule(rewrite("num-neg1-to-float", num(i64(-1)), float(f64(-1.0))).ruleset("expr"));

    // Identity/zero rules
    p.add_rule(rewrite("add-zero", add(v("a"), num(i64(0))), v("a")).ruleset("expr"));
    p.add_rule(
        Rule::new()
            .fact(peq(v("?e"), mul(v("?a"), num(i64(1)))))
            .union(v("?e"), v("?a"))
            .ruleset("expr"),
    );
    p.add_rule(
        Rule::new()
            .fact(peq(v("?e"), mul(v("?a"), num(i64(0)))))
            .union(v("?e"), num(i64(0)))
            .subsume(mul(v("?a"), num(i64(0))))
            .ruleset("expr"),
    );
    p.add_rule(rewrite("div-one", div(v("a"), num(i64(1))), v("a")).ruleset("expr"));
    p.add_rule(
        rewrite(
            "mod-mul-self",
            modd(mul(v("?x"), v("?y")), v("?y")),
            num(i64(0)),
        )
        .ruleset("expr"),
    );

    p.add_rule(
        rewrite(
            "mod-mod-larger",
            modd(modd(v("?x"), num(v("?y"))), num(v("?z"))),
            modd(v("?x"), num(v("?y"))),
        )
        .when(vec![
            pgte(v("?z"), v("?y")),
            peq(i64(0), pmod(v("?y"), v("?z"))),
        ])
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "mod-mod-smaller",
            modd(modd(v("?x"), num(v("?y"))), num(v("?z"))),
            modd(v("?x"), num(v("?z"))),
        )
        .when(vec![
            pgte(v("?y"), v("?z")),
            peq(i64(0), pmod(v("?z"), v("?y"))),
        ])
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "merge-dims",
            add(mul(div(v("?z"), v("?x")), v("?x")), modd(v("?z"), v("?x"))),
            v("?z"),
        )
        .ruleset("expr"),
    );

    p.add_rule(
        rewrite(
            "div-div",
            div(div(v("a"), v("b")), v("c")),
            div(v("a"), mul(v("b"), v("c"))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "add-div",
            add(div(v("a"), v("b")), v("c")),
            div(add(v("a"), mul(v("c"), v("b"))), v("b")),
        )
        .ruleset("expr"),
    );
    p.add_rule(rewrite("add-sub-cancel", add(v("a"), sub(v("b"), v("a"))), v("b")).ruleset("expr"));
    p.add_rule(
        rewrite("add-sub-cancel2", add(sub(v("b"), v("a")), v("a")), v("b")).ruleset("expr"),
    );
    p.add_rule(rewrite("sub-self", sub(v("a"), v("a")), num(i64(0))).ruleset("expr"));
    p.add_rule(
        rewrite(
            "add-sub-const",
            add(sub(v("a"), num(v("?b"))), num(v("?c"))),
            sub(v("a"), num(psub(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "add-sub-const2",
            add(num(v("?c")), sub(v("a"), num(v("?b")))),
            sub(v("a"), num(psub(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "sub-add-const",
            sub(add(v("a"), num(v("?b"))), num(v("?c"))),
            add(v("a"), num(psub(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "sub-sub-const",
            sub(sub(v("a"), num(v("?b"))), num(v("?c"))),
            sub(v("a"), num(padd(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "factor",
            add(mul(v("a"), v("b")), mul(v("a"), v("c"))),
            mul(v("a"), add(v("b"), v("c"))),
        )
        .ruleset("expr"),
    );
    p.add_rule(rewrite("double", add(v("a"), v("a")), mul(num(i64(2)), v("a"))).ruleset("expr"));

    // Constant folding through associativity
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), add(add(v("?a"), num(v("?b"))), num(v("?c")))),
                peq(v("?ans"), padd(v("?b"), v("?c"))),
            ])
            .union(v("?e"), add(v("?a"), num(v("?ans"))))
            .subsume(add(add(v("?a"), num(v("?b"))), num(v("?c"))))
            .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "add-assoc-var",
            add(add(num(v("?b")), mvar(v("?v"))), num(v("?c"))),
            add(mvar(v("?v")), num(padd(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "add-assoc-mul",
            add(add(num(v("?b")), mul(v("?n"), v("?a"))), num(v("?c"))),
            add(mul(v("?n"), v("?a")), num(padd(v("?b"), v("?c")))),
        )
        .ruleset("expr"),
    );

    // Combine like terms: (n*a) + a -> (n+1)*a
    p.add_rule(
        rewrite(
            "combine-like-1",
            add(mul(num(v("?n")), v("?a")), v("?a")),
            mul(num(padd(v("?n"), i64(1))), v("?a")),
        )
        .subsume(add(mul(num(v("?n")), v("?a")), v("?a")))
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "combine-like-2",
            add(v("?a"), mul(num(v("?n")), v("?a"))),
            mul(num(padd(v("?n"), i64(1))), v("?a")),
        )
        .subsume(add(v("?a"), mul(num(v("?n")), v("?a"))))
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "combine-like-3",
            add(mul(v("?a"), num(v("?n"))), v("?a")),
            mul(num(padd(v("?n"), i64(1))), v("?a")),
        )
        .subsume(add(mul(v("?a"), num(v("?n"))), v("?a")))
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "combine-like-4",
            add(v("?a"), mul(v("?a"), num(v("?n")))),
            mul(num(padd(v("?n"), i64(1))), v("?a")),
        )
        .subsume(add(v("?a"), mul(v("?a"), num(v("?n")))))
        .ruleset("expr"),
    );

    // Combine repeated variables: ((a + v) + v) -> (a + 2*v)
    p.add_rule(
        rewrite(
            "combine-var-1",
            add(add(v("?a"), mvar(v("?v"))), mvar(v("?v"))),
            add(v("?a"), mul(num(i64(2)), mvar(v("?v")))),
        )
        .subsume(add(add(v("?a"), mvar(v("?v"))), mvar(v("?v"))))
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "combine-var-2",
            add(add(mvar(v("?v")), v("?a")), mvar(v("?v"))),
            add(v("?a"), mul(num(i64(2)), mvar(v("?v")))),
        )
        .subsume(add(add(mvar(v("?v")), v("?a")), mvar(v("?v"))))
        .ruleset("expr"),
    );

    // Accumulate: ((n*a + b) + a) -> ((n+1)*a + b)
    p.add_rule(
        rewrite(
            "accum-1",
            add(add(mul(num(v("?n")), v("?a")), v("?b")), v("?a")),
            add(mul(num(padd(v("?n"), i64(1))), v("?a")), v("?b")),
        )
        .subsume(add(add(mul(num(v("?n")), v("?a")), v("?b")), v("?a")))
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "accum-2",
            add(add(v("?b"), mul(num(v("?n")), v("?a"))), v("?a")),
            add(v("?b"), mul(num(padd(v("?n"), i64(1))), v("?a"))),
        )
        .subsume(add(add(v("?b"), mul(num(v("?n")), v("?a"))), v("?a")))
        .ruleset("expr"),
    );

    // ---- Replacement over expressions ----
    p.add_rule(
        rewrite("replace-match", replace(v("?x"), v("?y"), v("?z")), v("?z"))
            .when(vec![peq(v("?x"), v("?y"))])
            .ruleset("expr"),
    );

    // Replacement distributes over binary ops
    #[allow(clippy::type_complexity)]
    let binary_ops: Vec<(&str, Box<dyn Fn(Term, Term) -> Term>)> = vec![
        ("MAdd", Box::new(&add)),
        ("MSub", Box::new(&sub)),
        ("MMul", Box::new(&mul)),
        ("MDiv", Box::new(&div)),
        ("MCeilDiv", Box::new(&ceildiv)),
        ("MMod", Box::new(&modd)),
        ("MMin", Box::new(&min)),
        ("MMax", Box::new(&max)),
        ("MFloorTo", Box::new(&floorto)),
    ];
    for (name, op) in &binary_ops {
        p.add_rule(
            rewrite(
                &format!("replace-{}", name),
                replace(op(v("?a"), v("?b")), v("?x"), v("?y")),
                op(
                    replace(v("?a"), v("?x"), v("?y")),
                    replace(v("?b"), v("?x"), v("?y")),
                ),
            )
            .ruleset("expr"),
        );
    }

    p.add_rule(
        rewrite(
            "replace-num",
            replace(num(v("?n")), v("?x"), v("?y")),
            num(v("?n")),
        )
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "replace-var-miss",
            replace(mvar(v("?z")), v("?find"), v("?replace")),
            mvar(v("?z")),
        )
        .when(vec![pneq(v("?find"), mvar(v("?z")))])
        .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "replace-iter-miss",
            replace(iter(), v("?find"), v("?replace")),
            iter(),
        )
        .when(vec![pneq(v("?find"), iter())])
        .ruleset("expr"),
    );

    // ---- EList helper functions ----
    p.add_function(FunctionDef {
        name: "len".into(),
        args: vec!["EList".into()],
        ret: "i64".into(),
        merge: Some("new".into()),
    });
    p.add_rule(
        Rule::new()
            .fact(peq(v("?e"), nil()))
            .action(Action::Set(len_f(v("?e")), i64(0)))
            .ruleset("expr"),
    );
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), cons(v("?expr"), v("?list"))),
                peq(v("?prev_len"), len_f(v("?list"))),
            ])
            .action(Action::Set(len_f(v("?e")), padd(v("?prev_len"), i64(1))))
            .ruleset("expr"),
    );

    p.add_function(FunctionDef {
        name: "nth_from_end".into(),
        args: vec!["EList".into(), "i64".into()],
        ret: "Expression".into(),
        merge: Some("new".into()),
    });
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), cons(v("?expr"), v("?list"))),
                peq(v("?list_len"), len_f(v("?list"))),
            ])
            .action(Action::Set(nth_f(v("?e"), v("?list_len")), v("?expr")))
            .ruleset("expr"),
    );
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), cons(v("?expr"), v("?list"))),
                peq(v("?other_nth"), nth_f(v("?list"), v("?n"))),
            ])
            .action(Action::Set(nth_f(v("?e"), v("?n")), v("?other_nth")))
            .ruleset("expr"),
    );

    p.add_function(FunctionDef {
        name: "n_elements".into(),
        args: vec!["EList".into()],
        ret: "Expression".into(),
        merge: Some("new".into()),
    });
    p.add_rule(
        Rule::new()
            .fact(peq(v("?e"), nil()))
            .action(Action::Set(nelem_f(v("?e")), num(i64(1))))
            .ruleset("expr"),
    );
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), cons(v("?dim"), v("?other"))),
                peq(v("?other_elems"), nelem_f(v("?other"))),
            ])
            .action(Action::Set(
                nelem_f(v("?e")),
                mul(v("?dim"), v("?other_elems")),
            ))
            .ruleset("expr"),
    );

    // RowMajor rules (z-strides: base stride is MIter/'z', not 1)
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?other"), cons(v("?other_dim"), v("?other_other"))),
                peq(v("?list"), cons(v("?d"), v("?other"))),
                peq(v("?e"), rowmajor(v("?list"))),
                peq(v("?n_elems"), nelem_f(v("?other"))),
            ])
            .action(Action::Union(
                v("?e"),
                cons(mul(v("?n_elems"), iter()), rowmajor(v("?other"))),
            ))
            .ruleset("expr"),
    );
    p.add_rule(
        rewrite(
            "rowmajor-base",
            rowmajor(cons(v("?dim"), nil())),
            cons(iter(), nil()),
        )
        .ruleset("expr"),
    );

    // MReplaceList / ReplaceNthFromEnd / RemoveNthFromEnd
    p.add_rule(
        rewrite(
            "replace-list-cons",
            replace_list(cons(v("?expr"), v("?list")), v("?from"), v("?to")),
            cons(
                replace(v("?expr"), v("?from"), v("?to")),
                replace_list(v("?list"), v("?from"), v("?to")),
            ),
        )
        .ruleset("expr"),
    );

    // ReplaceNthFromEnd: match case (ind == len list)
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(
                    v("?e"),
                    replace_nth(cons(v("?expr"), v("?list")), v("?to"), v("?ind")),
                ),
                peq(v("?ind"), len_f(v("?list"))),
            ])
            .action(Action::Union(v("?e"), cons(v("?to"), v("?list"))))
            .ruleset("expr"),
    );
    // ReplaceNthFromEnd: recurse case (ind < len list)
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(
                    v("?e"),
                    replace_nth(cons(v("?expr"), v("?list")), v("?to"), v("?ind")),
                ),
                plt(v("?ind"), len_f(v("?list"))),
            ])
            .action(Action::Union(
                v("?e"),
                cons(v("?expr"), replace_nth(v("?list"), v("?to"), v("?ind"))),
            ))
            .ruleset("expr"),
    );

    // RemoveNthFromEnd: match case (ind == len list)
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), remove_nth(cons(v("?expr"), v("?list")), v("?ind"))),
                peq(v("?ind"), len_f(v("?list"))),
            ])
            .action(Action::Union(v("?e"), v("?list")))
            .ruleset("expr"),
    );
    // RemoveNthFromEnd: recurse case (ind < len list)
    p.add_rule(
        Rule::new()
            .facts(vec![
                peq(v("?e"), remove_nth(cons(v("?expr"), v("?list")), v("?ind"))),
                plt(v("?ind"), len_f(v("?list"))),
            ])
            .action(Action::Union(
                v("?e"),
                cons(v("?expr"), remove_nth(v("?list"), v("?ind"))),
            ))
            .ruleset("expr"),
    );

    p.to_egglog_string()
}

/// Generate the cleanup rules that delete intermediate helper nodes
/// (MReplace, MReplaceList, ReplaceNthFromEnd, RemoveNthFromEnd, RowMajor,
/// and the helper functions len, nth_from_end, n_elements).
pub fn base_cleanup_egglog() -> String {
    let mut p = Program::default();
    p.add_ruleset("base_cleanup");

    // Delete sort-based intermediates
    #[allow(clippy::type_complexity)]
    let sort_cleanups: &[(&str, &dyn Fn(Vec<Term>) -> Term, &[&str])] = &[
        (
            "MReplace",
            &|a| replace(a[0].clone(), a[1].clone(), a[2].clone()),
            &["a", "b", "c"],
        ),
        (
            "MReplaceList",
            &|a| replace_list(a[0].clone(), a[1].clone(), a[2].clone()),
            &["a", "b", "c"],
        ),
        (
            "ReplaceNthFromEnd",
            &|a| replace_nth(a[0].clone(), a[1].clone(), a[2].clone()),
            &["a", "b", "c"],
        ),
        (
            "RemoveNthFromEnd",
            &|a| remove_nth(a[0].clone(), a[1].clone()),
            &["a", "b"],
        ),
        ("RowMajor", &|a| rowmajor(a[0].clone()), &["x"]),
    ];
    for (name, ctor, vars) in sort_cleanups {
        let args: Vec<Term> = vars.iter().map(v).collect();
        let term = ctor(args);
        p.add_rule(
            Rule::new()
                .fact(peq(v("?m"), term.clone()))
                .action(Action::Delete(term))
                .ruleset("base_cleanup"),
        );
        let _ = name; // used only for clarity
    }

    // Delete function-based intermediates
    #[allow(clippy::type_complexity)]
    let fn_cleanups: &[(&str, fn(Vec<Term>) -> Term, usize)] = &[
        ("len", |a| len_f(a[0].clone()), 1),
        ("nth_from_end", |a| nth_f(a[0].clone(), a[1].clone()), 2),
        ("n_elements", |a| nelem_f(a[0].clone()), 1),
    ];
    for (_name, ctor, arity) in fn_cleanups {
        let var_names: Vec<&str> = match arity {
            1 => vec!["?x"],
            2 => vec!["?x", "?y"],
            _ => unreachable!(),
        };
        let args: Vec<Term> = var_names.iter().map(v).collect();
        let term = ctor(args);
        p.add_rule(
            Rule::new()
                .fact(peq(v("?m"), term.clone()))
                .action(Action::Delete(term))
                .ruleset("base_cleanup"),
        );
    }

    p.to_egglog_string()
}

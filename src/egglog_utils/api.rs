use std::collections::HashMap;

use egglog::ast::{Literal, Sexp, Span, atom_to_sexp};

// ========== Core Types ==========

/// A sort class (type) — either a builtin like `i64` or a user-defined datatype like `Expr`.
#[derive(Clone, Copy, Debug)]
pub struct SortClass {
    pub name: &'static str,
}

impl SortClass {
    pub const fn new(name: &'static str) -> Self {
        Self { name }
    }
}

#[derive(Clone, Debug)]
pub struct Field {
    pub name: String,
    pub sort: String,
}

/// The core term type for building egglog expressions.
///
/// This is a direct alias for egglog's pre-parse [`Sexp`]. Builders below
/// construct `Sexp` values that splice straight into `egglog!`/`sexp!`
/// quasiquotes via `ToSexp` (identity for `Sexp`), and render back to source
/// text via `Sexp`'s `Display`. (Formerly a bespoke luminal `Term` enum with a
/// hand-rolled `Rule`/`Program`/`term_to_egglog` DSL — all now deleted in favor
/// of the quasiquote.)
pub type Term = Sexp;

/// A fresh, source-less span for terms built programmatically (not parsed from
/// text). Every `Sexp` node carries a span; for constructed terms it is only
/// used in diagnostics.
fn sp() -> Span {
    egglog::span!()
}

// ========== Args ==========

/// Named argument list for sort/function calls.
///
/// Supports adding arguments by name, indexing by field name to retrieve
/// the generated variable, and passing directly to `SortDef::call`.
///
/// (`Sexp` — and therefore `Term` — is not `Debug`, so neither is `Args`.)
#[derive(Clone)]
pub struct Args {
    entries: Vec<(String, Term)>,
}

impl Default for Args {
    fn default() -> Self {
        Self::new()
    }
}

impl Args {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a named argument.
    pub fn add(&mut self, name: impl ToString, value: Term) {
        self.entries.push((name.to_string(), value));
    }

    /// Get the term for a field name. Panics if not found.
    pub fn get(&self, name: &str) -> &Term {
        self.entries
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, t)| t)
            .unwrap_or_else(|| panic!("no argument named `{}`", name))
    }

    /// Remove and return the term for a field name. Panics if not found.
    pub fn remove(&mut self, name: &str) -> Term {
        let idx = self
            .entries
            .iter()
            .position(|(n, _)| n == name)
            .unwrap_or_else(|| panic!("no argument named `{}`", name));
        self.entries.remove(idx).1
    }

    /// Extend with entries from anything convertible to `Args`.
    pub fn extend(&mut self, other: impl IntoArgs) {
        self.entries.extend(other.into_args().entries);
    }
}

impl std::ops::Index<&str> for Args {
    type Output = Term;
    fn index(&self, name: &str) -> &Term {
        self.get(name)
    }
}

/// Trait for types that can be converted into an `Args`.
pub trait IntoArgs {
    fn into_args(self) -> Args;
}

impl IntoArgs for Args {
    fn into_args(self) -> Args {
        self
    }
}

impl IntoArgs for &Args {
    fn into_args(self) -> Args {
        self.to_owned()
    }
}

impl<S: ToString> IntoArgs for (S, Term) {
    fn into_args(self) -> Args {
        let mut args = Args::new();
        args.add(self.0, self.1);
        args
    }
}

impl IntoArgs for () {
    fn into_args(self) -> Args {
        Args::new()
    }
}

impl<S: ToString> IntoArgs for Vec<(S, Term)> {
    fn into_args(self) -> Args {
        let mut args = Args::new();
        for (name, term) in self {
            args.add(name, term);
        }
        args
    }
}

impl<S: ToString, const N: usize> IntoArgs for [(S, Term); N] {
    fn into_args(self) -> Args {
        let mut args = Args::new();
        for (name, term) in self {
            args.add(name, term);
        }
        args
    }
}

impl<S: ToString> IntoArgs for &[(S, Term)] {
    fn into_args(self) -> Args {
        let mut args = Args::new();
        for (name, term) in self {
            args.add(name.to_string(), term.clone());
        }
        args
    }
}

impl std::ops::Deref for Args {
    type Target = [(String, Term)];
    fn deref(&self) -> &Self::Target {
        &self.entries
    }
}

// ========== Free-standing Sort Definition ==========

/// A sort variant definition that has not yet been registered into a program.
#[derive(Clone, Debug)]
pub struct SortDef {
    pub class: String,
    pub name: String,
    pub fields: Vec<Field>,
}

impl SortDef {
    /// Call this sort on fresh variables, returning the args and the application term.
    pub fn new_call(&self) -> (Args, Term) {
        let prefix = {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            format!("v{}", COUNTER.fetch_add(1, Ordering::Relaxed))
        };
        let mut args = Args::new();
        for f in &self.fields {
            args.add(&f.name, v(format!("{prefix}_{}", f.name)));
        }
        let term = self.call(&args);
        (args, term)
    }

    /// Construct an application term from this sort definition with named arguments.
    pub fn call(&self, args: impl IntoArgs) -> Term {
        let args = args.into_args();
        assert_eq!(
            args.len(),
            self.fields.len(),
            "sort `{}` expects {} args, got {}",
            self.name,
            self.fields.len(),
            args.len()
        );

        let mut provided: HashMap<String, Term> = args
            .iter()
            .map(|(s, t)| (s.to_string(), t.clone()))
            .collect();

        let mut ordered = Vec::with_capacity(args.len());
        for field in &self.fields {
            let term = provided.remove(field.name.as_str()).unwrap_or_else(|| {
                panic!(
                    "missing argument `{}` in call to `{}`",
                    field.name, self.name
                )
            });
            ordered.push(term);
        }

        if !provided.is_empty() {
            let extra: Vec<_> = provided.keys().cloned().collect();
            panic!(
                "unexpected arguments in call to `{}`: {}",
                self.name,
                extra.join(", ")
            );
        }

        call_named(&self.name, ordered)
    }
}

// ========== Free-standing Builders ==========

/// Create a sort variant definition.
pub fn sort(class: SortClass, name: &str, args: &[(&str, SortClass)]) -> SortDef {
    let mut seen = std::collections::HashSet::new();
    let mut fields = Vec::with_capacity(args.len());
    for (arg_name, arg_sort) in args {
        if !seen.insert(*arg_name) {
            panic!("duplicate field name {} in variant {}", arg_name, name);
        }
        fields.push(Field {
            name: arg_name.to_string(),
            sort: arg_sort.name.to_string(),
        });
    }
    SortDef {
        class: class.name.to_string(),
        name: name.to_string(),
        fields,
    }
}

/// Build an application node `(head args...)` as a [`Sexp`]. The head is always
/// an atom (a variant/function name); values are already-built terms.
pub fn call_named(head: &str, args: Vec<Term>) -> Term {
    let mut items = Vec::with_capacity(args.len() + 1);
    items.push(Sexp::Atom(head.to_string(), sp()));
    items.extend(args);
    Sexp::List(items, sp())
}

/// Create an untyped pattern variable (sort is not tracked). A leading `?` is
/// added if missing, matching egglog's pattern-variable syntax.
pub fn v(name: impl ToString) -> Term {
    let n = name.to_string();
    let n = if n.starts_with('?') { n } else { format!("?{n}") };
    Sexp::Atom(n, sp())
}

pub fn i64(value: i64) -> Term {
    Sexp::Literal(Literal::Int(value), sp())
}

pub fn f64(value: f64) -> Term {
    // Ensure a decimal point so it reparses as a float, then classify.
    let s = value.to_string();
    let s = if s.contains('.') || s.contains('e') || s.contains("inf") || s.contains("NaN") {
        s
    } else {
        format!("{s}.0")
    };
    atom_to_sexp(&s, sp())
}

pub fn bool(value: bool) -> Term {
    Sexp::Literal(Literal::Bool(value), sp())
}

pub fn str(value: &str) -> Term {
    Sexp::Literal(Literal::String(value.to_string()), sp())
}

pub fn unit() -> Term {
    Sexp::List(vec![], sp())
}

/// Create a function/builtin definition (for term construction only, not registered as a sort).
pub fn func(name: &str, arg_names: &[&str]) -> SortDef {
    SortDef {
        class: String::new(),
        name: name.to_string(),
        fields: arg_names
            .iter()
            .map(|n| Field {
                name: n.to_string(),
                sort: String::new(),
            })
            .collect(),
    }
}

/// Sort/function application — builds a term from a `SortDef` and positional arguments.
pub fn app(sort: &SortDef, args: Vec<Term>) -> Term {
    assert_eq!(
        args.len(),
        sort.fields.len(),
        "`{}` expects {} args, got {}",
        sort.name,
        sort.fields.len(),
        args.len()
    );
    call_named(&sort.name, args)
}

/// Egglog equality fact: `(= a b)`
pub fn eq(a: Term, b: Term) -> Term {
    call_named("=", vec![a, b])
}

/// Egglog inequality fact: `(!= a b)`
pub fn neq(a: Term, b: Term) -> Term {
    call_named("!=", vec![a, b])
}

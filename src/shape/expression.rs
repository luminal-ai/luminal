use generational_box::{AnyStorage, GenerationalBox, Owner, SyncStorage};
use lru::LruCache;
use rustc_hash::FxHashMap;
use serde::{Serialize, Serializer};
use std::{
    fmt::Debug,
    hash::Hash,
    num::NonZeroUsize,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign,
        Neg, Rem, RemAssign, Sub, SubAssign,
    },
    sync::{Mutex, OnceLock},
};

use crate::{egglog_utils, graph::extract_expr, prelude::*};
use egglog::{prelude::RustSpan, var};
use egglog_ast::span::Span;

type ExprBox = GenerationalBox<Vec<Term>, SyncStorage>;

static EXPR_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();
static SIMPLIFY_CACHE: OnceLock<Mutex<LruCache<Expression, Expression>>> = OnceLock::new();

const MAX_CACHED_SIMPLIFICATIONS: usize = 100_000;

#[derive(Copy, Clone)]
pub struct Expression {
    pub terms: ExprBox,
}

impl Serialize for Expression {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // Access the Vec<Term> inside the GenerationalBox and serialize it
        self.terms.read().serialize(serializer)
    }
}

impl Expression {
    pub fn new(terms: Vec<Term>) -> Self {
        Self {
            terms: EXPR_OWNER.get_or_init(SyncStorage::owner).insert(terms),
        }
    }

    pub fn is_dynamic(&self) -> bool {
        self.terms.read().iter().any(|i| {
            if let Term::Var(v) = i {
                *v != 'z'
            } else {
                false
            }
        })
    }

    pub fn dyn_vars(&self) -> Vec<char> {
        self.terms
            .read()
            .iter()
            .filter_map(|i| {
                if let Term::Var(v) = i {
                    if *v != 'z' { Some(*v) } else { None }
                } else {
                    None
                }
            })
            .collect()
    }
}

impl Hash for Expression {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.terms.read().hash(state);
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::new(vec![])
    }
}

/// A single term of a symbolic expression such as a variable, number or operation.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Term {
    Num(i32),
    Var(char),
    Add,
    Sub,
    Mul,
    Div,
    CeilDiv,
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
}

impl std::fmt::Debug for Term {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Term::Num(n) => write!(f, "{n}"),
            Term::Var(c) => write!(f, "{c}"),
            Term::Add => write!(f, "+"),
            Term::Sub => write!(f, "-"),
            Term::Mul => write!(f, "*"),
            Term::Div => write!(f, "/"),
            Term::Mod => write!(f, "%"),
            Term::Min => write!(f, "min"),
            Term::CeilDiv => write!(f, "^/"),
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
        }
    }
}

impl Default for Term {
    fn default() -> Self {
        Self::Num(0)
    }
}

impl Term {
    pub fn as_op(self) -> Option<fn(i64, i64) -> Option<i64>> {
        match self {
            Term::Add => Some(|a, b| a.checked_add(b)),
            Term::Sub => Some(|a, b| a.checked_sub(b)),
            Term::Mul => Some(|a, b| a.checked_mul(b)),
            Term::Div => Some(|a, b| a.checked_div(b)),
            Term::Mod => Some(|a, b| a.checked_rem(b)),
            Term::Max => Some(|a, b| Some(a.max(b))),
            Term::Min => Some(|a, b| Some(a.min(b))),
            Term::And => Some(|a, b| Some((a != 0 && b != 0) as i64)),
            Term::Or => Some(|a, b| Some((a != 0 || b != 0) as i64)),
            Term::Gte => Some(|a, b| Some((a >= b) as i64)),
            Term::Lt => Some(|a, b| Some((a < b) as i64)),
            Term::CeilDiv => Some(|a, b| Some(if a % b != 0 { a / b + 1 } else { a / b })),
            _ => None,
        }
    }
    pub fn as_float_op(self) -> Option<fn(f64, f64) -> f64> {
        match self {
            Term::Add => Some(|a, b| a + b),
            Term::Sub => Some(|a, b| a - b),
            Term::Mul => Some(|a, b| a * b),
            Term::Div => Some(|a, b| a / b),
            Term::Mod => Some(|a, b| a % b),
            Term::Max => Some(|a, b| a.max(b)),
            Term::Min => Some(|a, b| a.min(b)),
            Term::And => Some(|a, b| (a.abs() > 1e-4 && b.abs() > 1e-4) as i32 as f64),
            Term::Or => Some(|a, b| (a.abs() > 1e-4 || b.abs() > 1e-4) as i32 as f64),
            Term::Gte => Some(|a, b| (a >= b) as i32 as f64),
            Term::Lt => Some(|a, b| (a < b) as i32 as f64),
            Term::CeilDiv => Some(|a, b| (a / b).ceil()),
            _ => None,
        }
    }
    pub fn to_egglog(self) -> String {
        match self {
            Term::Add => "MAdd",
            Term::Sub => "MSub",
            Term::Mul => "MMul",
            Term::Div => "MDiv",
            Term::Mod => "MMod",
            Term::Max => "MMax",
            Term::Min => "MMin",
            Term::CeilDiv => "MCeilDiv",
            Term::Gte => "MGte",
            Term::Lt => "MLt",
            _ => panic!("egglog doesn't implement {self:?}"),
        }
        .to_string()
    }
}

impl<T> PartialEq<T> for Expression
where
    for<'a> &'a T: Into<Expression>,
{
    fn eq(&self, other: &T) -> bool {
        // Equals-approximation. For proper equality checking, use .egglog_equals (more expensive)
        *self.terms.read() == *other.into().terms.read()
    }
}

impl From<&Expression> for Expression {
    fn from(value: &Expression) -> Self {
        *value
    }
}

impl Eq for Expression {}

impl Debug for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => c.to_string(),
                Term::Max => format!(
                    "max({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min({}, {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        write!(f, "{}", symbols.pop().unwrap_or_default())
    }
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Expression {
    pub fn to_egglog(&self) -> String {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => format!("(MNum {n})"),
                Term::Var(c) => format!("(MVar \"{c}\")"),
                Term::Max => format!(
                    "(MMax {} {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "(MMin {} {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                _ => format!(
                    "({} {} {})",
                    term.to_egglog(),
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        symbols.pop().unwrap_or_default()
    }

    pub fn to_kernel(&self) -> String {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => n.to_string(),
                Term::Var(c) => format!("{}const_{c}", if *c == 'z' { "" } else { "*" }),
                Term::Max => format!(
                    "max((int){}, (int){})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Min => format!(
                    "min((int){}, (int){})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Lt => format!(
                    "(int)({} < {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::Gte => format!(
                    "(int)({} >= {})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
                Term::CeilDiv => {
                    let a = symbols.pop().unwrap();
                    let b = symbols.pop().unwrap();
                    format!("(({a} + {b} - 1) / {b})")
                }
                Term::Div => format!("({} / {})", symbols.pop().unwrap(), symbols.pop().unwrap()),
                _ => format!(
                    "({}{term:?}{})",
                    symbols.pop().unwrap(),
                    symbols.pop().unwrap()
                ),
            };
            symbols.push(new_symbol);
        }
        symbols.pop().unwrap_or_default()
    }
    /// Simplify the expression to its minimal terms
    #[tracing::instrument(skip_all)]
    pub fn simplify(self) -> Self {
        if self.terms.read().len() == 1 {
            return self;
        }

        // Early exit for ((M*X)+N) pattern where M and N are integers, X is var
        {
            let terms = self.terms.read();
            if terms.len() == 5 {
                if let (Term::Num(_), Term::Var(_), Term::Num(_), Term::Mul, Term::Add) =
                    (terms[0], terms[1], terms[2], terms[3], terms[4])
                {
                    return self;
                }
            }
        }

        egglog_simplify(self)
    }
    pub fn as_num(&self) -> Option<i32> {
        if let Term::Num(n) = self.terms.read()[0] {
            if self.terms.read().len() == 1 {
                return Some(n);
            }
        }
        None
    }
    pub fn len(&self) -> usize {
        self.terms.read().len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Minimum
    pub fn min(self, rhs: impl Into<Self>) -> Self {
        let rhs = rhs.into();
        if rhs == self || rhs == i32::MAX {
            return self;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.min(b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Min);
        Expression::new(terms)
    }
    /// Maximum
    pub fn max<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self || self == i32::MAX {
            return self;
        }
        if rhs == i32::MAX {
            return rhs;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return a.max(b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Max);
        Expression::new(terms)
    }
    /// Greater than or equals
    pub fn gte<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self {
            return true.into();
        }
        if rhs == i32::MAX {
            return false.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a >= b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Gte);
        Expression::new(terms)
    }
    /// Ceil Division
    pub fn ceil_div<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::CeilDiv);
        Expression::new(terms)
    }
    /// Less than
    pub fn lt<E: Into<Expression>>(self, rhs: E) -> Self {
        let rhs = rhs.into();
        if rhs == self {
            return false.into();
        }
        if let Term::Num(n) = rhs.terms.read()[0] {
            if self.terms.read()[self.terms.read().len() - 1] == Term::Mod
                && self.terms.read()[0] == Term::Num(n)
            {
                return true.into();
            }
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a < b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Lt);
        Expression::new(terms)
    }
    /// Substitute an expression for a variable
    pub fn substitute(self, var: char, expr: impl Into<Expression>) -> Self {
        let mut new_terms = vec![];
        let t = expr.into().terms.read();
        for term in self.terms.read().iter() {
            match term {
                Term::Var(c) if *c == var => {
                    for t in t.iter() {
                        new_terms.push(*t);
                    }
                }
                _ => {
                    new_terms.push(*term);
                }
            }
        }
        Expression::new(new_terms)
    }
    /// Evaluate the expression with no variables. Returns Some(value) if no variables are required, otherwise returns None.
    pub fn to_usize(&self) -> Option<usize> {
        self.exec(&FxHashMap::default())
    }
    /// Evaluate the expression with one value for all variables.
    pub fn exec_single_var(&self, value: usize) -> usize {
        let mut stack = Vec::new();
        self.exec_single_var_stack(value, &mut stack)
    }
    /// Evaluate the expression with one value for all variables. Uses a provided stack
    pub fn exec_single_var_stack(&self, value: usize, stack: &mut Vec<i64>) -> usize {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Var(_) => stack.push(value as i64),
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
                }
            }
        }
        stack.pop().unwrap() as usize
    }
    /// Evaluate the expression given variables.
    pub fn exec(&self, variables: &FxHashMap<char, usize>) -> Option<usize> {
        self.exec_stack(variables, &mut Vec::new())
    }
    /// Evaluate the expression given variables. This function requires a stack to be given for use as storage
    pub fn exec_stack(
        &self,
        variables: &FxHashMap<char, usize>,
        stack: &mut Vec<i64>,
    ) -> Option<usize> {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as i64),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as i64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_op().unwrap()(a, b).unwrap());
                }
            }
        }
        stack.pop().map(|i| i as usize)
    }
    /// Retrieve all symbols in the expression.
    pub fn to_symbols(&self) -> Vec<char> {
        self.terms
            .read()
            .iter()
            .filter_map(|t| match t {
                Term::Var(c) => Some(*c),
                _ => None,
            })
            .collect()
    }
    /// Resolve all known variables from dyn map into real values
    pub fn resolve_vars(&mut self, dyn_map: &FxHashMap<char, usize>) {
        for term in self.terms.write().iter_mut() {
            if let Term::Var(v) = *term
                && let Some(val) = dyn_map.get(&v)
            {
                *term = Term::Num(*val as i32);
            }
        }
    }
    /// Run proper equality check inside egglog
    #[tracing::instrument(skip_all)]
    pub fn egglog_equal(self, rhs: impl Into<Expression>) -> bool {
        let lhs_expr = self.to_egglog();
        let rhs_expr = rhs.into().to_egglog();
        let mut program = String::new();
        program.push_str(egglog_utils::BASE);
        program.push('\n');
        program.push_str(egglog_utils::BASE_CLEANUP);
        program.push('\n');
        program.push_str(&format!("(let expr_lhs {lhs_expr})\n"));
        program.push_str(&format!("(let expr_rhs {rhs_expr})\n"));
        program.push_str(
            "(run-schedule
                (saturate expr)
                (saturate base_cleanup)
                (saturate cleanup)
            )",
        );
        program.push('\n');
        program.push_str("(check (= expr_lhs expr_rhs))\n");

        let mut egraph = egglog::EGraph::default();
        let commands = egraph
            .parser
            .get_program_from_string(None, &program)
            .expect("failed to parse egglog program");
        let span = tracing::span!(tracing::Level::INFO, "to_egglog");
        let _entered = span.enter();
        match egraph.run_program(commands) {
            Ok(_) => true,
            Err(err) => {
                if matches!(err, egglog::Error::CheckError(_, _)) {
                    return false;
                }
                panic!("failed to run egglog program: {err}");
            }
        }
    }
}

impl From<Term> for Expression {
    fn from(value: Term) -> Self {
        Expression::new(vec![value])
    }
}

impl From<char> for Expression {
    fn from(value: char) -> Self {
        Expression::new(vec![Term::Var(value)])
    }
}

impl From<&char> for Expression {
    fn from(value: &char) -> Self {
        Expression::new(vec![Term::Var(*value)])
    }
}

impl From<usize> for Expression {
    fn from(value: usize) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&usize> for Expression {
    fn from(value: &usize) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl From<i32> for Expression {
    fn from(value: i32) -> Self {
        Expression::new(vec![Term::Num(value)])
    }
}

impl From<&i32> for Expression {
    fn from(value: &i32) -> Self {
        Expression::new(vec![Term::Num(*value)])
    }
}

impl From<bool> for Expression {
    fn from(value: bool) -> Self {
        Expression::new(vec![Term::Num(value as i32)])
    }
}

impl From<&bool> for Expression {
    fn from(value: &bool) -> Self {
        Expression::new(vec![Term::Num(*value as i32)])
    }
}

impl Add<Expression> for usize {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for usize {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for usize {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for usize {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for usize {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for usize {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for usize {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl Add<Expression> for i32 {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        rhs + self
    }
}

impl Sub<Expression> for i32 {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

impl Mul<Expression> for i32 {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

impl Div<Expression> for i32 {
    type Output = Expression;
    fn div(self, rhs: Expression) -> Self::Output {
        Expression::from(self) / rhs
    }
}

impl Rem<Expression> for i32 {
    type Output = Expression;
    fn rem(self, rhs: Expression) -> Self::Output {
        Expression::from(self) % rhs
    }
}

impl BitAnd<Expression> for i32 {
    type Output = Expression;
    fn bitand(self, rhs: Expression) -> Self::Output {
        rhs & self
    }
}

impl BitOr<Expression> for i32 {
    type Output = Expression;
    fn bitor(self, rhs: Expression) -> Self::Output {
        rhs | self
    }
}

impl Neg for Expression {
    type Output = Expression;
    fn neg(self) -> Self::Output {
        self * -1
    }
}

impl<E: Into<Expression>> Add<E> for Expression {
    type Output = Self;
    fn add(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 {
            return self;
        }
        if self == 0 {
            return rhs;
        }
        if self == rhs {
            return self * 2;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a + b).into();
        }

        // Shortcut: if we're adding an integer to an Add expression with an integer operand, fold them together: ((...)+X)+Y -> (...+(X+Y))
        if let Some(z) = rhs.as_num() {
            let self_terms = self.terms.read();
            if self_terms.last() == Some(&Term::Add) {
                if let Some(Term::Num(n)) = self_terms.first() {
                    if let Some(folded) = n.checked_add(z) {
                        let mut new_terms = self_terms.clone();
                        new_terms[0] = Term::Num(folded);
                        return Expression::new(new_terms);
                    }
                }
            }
        }

        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Add);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Sub<E> for Expression {
    type Output = Self;
    fn sub(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 {
            return self;
        }
        if self == rhs {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a - b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Sub);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Mul<E> for Expression {
    type Output = Self;
    fn mul(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            if let Some(c) = a.checked_mul(b) {
                return c.into();
            }
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mul);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Div<E> for Expression {
    type Output = Self;
    fn div(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 {
            return self;
        }
        if self == rhs {
            return 1.into();
        }
        if self == 0 {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            if a % b == 0 {
                if let Some(c) = a.checked_div(b) {
                    return c.into();
                }
            }
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Div);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> Rem<E> for Expression {
    type Output = Self;
    fn rem(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 || rhs == self {
            return 0.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a % b).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Mod);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitAnd<E> for Expression {
    type Output = Self;
    fn bitand(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 0 || self == 0 {
            return 0.into();
        }
        if rhs == 1 {
            return self;
        }
        if self == 1 {
            return rhs;
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 && b != 0).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::And);
        Expression::new(terms)
    }
}

impl<E: Into<Expression>> BitOr<E> for Expression {
    type Output = Self;
    fn bitor(self, rhs: E) -> Self::Output {
        let rhs = rhs.into();
        if rhs == 1 || self == 1 {
            return 1.into();
        }
        if let (Some(a), Some(b)) = (self.as_num(), rhs.as_num()) {
            return (a != 0 || b != 0).into();
        }
        let mut terms = rhs.terms.read().clone();
        terms.extend(self.terms.read().iter().copied());
        terms.push(Term::Or);
        Expression::new(terms)
    }
}

impl std::iter::Product for Expression {
    fn product<I: Iterator<Item = Expression>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p *= n;
        }
        p
    }
}

impl std::iter::Sum for Expression {
    fn sum<I: Iterator<Item = Expression>>(mut iter: I) -> Self {
        let Some(mut p) = iter.next() else {
            return 0.into();
        };
        for n in iter {
            p += n;
        }
        p
    }
}

impl<E: Into<Expression>> AddAssign<E> for Expression {
    fn add_assign(&mut self, rhs: E) {
        *self = *self + rhs;
    }
}

impl<E: Into<Expression>> SubAssign<E> for Expression {
    fn sub_assign(&mut self, rhs: E) {
        *self = *self - rhs;
    }
}

impl<E: Into<Expression>> MulAssign<E> for Expression {
    fn mul_assign(&mut self, rhs: E) {
        *self = *self * rhs;
    }
}

impl<E: Into<Expression>> DivAssign<E> for Expression {
    fn div_assign(&mut self, rhs: E) {
        *self = *self / rhs;
    }
}

impl<E: Into<Expression>> RemAssign<E> for Expression {
    fn rem_assign(&mut self, rhs: E) {
        *self = *self % rhs;
    }
}

impl<E: Into<Expression>> BitAndAssign<E> for Expression {
    fn bitand_assign(&mut self, rhs: E) {
        *self = *self & rhs;
    }
}

impl<E: Into<Expression>> BitOrAssign<E> for Expression {
    fn bitor_assign(&mut self, rhs: E) {
        *self = *self | rhs;
    }
}

#[tracing::instrument(skip_all)]
fn egglog_simplify(e: Expression) -> Expression {
    let cache = SIMPLIFY_CACHE.get_or_init(|| {
        Mutex::new(LruCache::<Expression, Expression>::new(
            NonZeroUsize::new(MAX_CACHED_SIMPLIFICATIONS).unwrap(),
        ))
    });

    if let Some(out) = cache.lock().unwrap().get(&e).copied() {
        return out;
    }
    let expr = e.to_egglog();
    let mut program = String::new();
    program.push_str(egglog_utils::BASE);
    program.push('\n');
    program.push_str(egglog_utils::BASE_CLEANUP);
    program.push('\n');
    program.push_str(&format!("(let expr_root {expr})\n"));
    program.push_str(
        "(run-schedule
            (saturate expr)
            (saturate base_cleanup)
            (saturate cleanup)
        )",
    );
    let mut egraph = egglog::EGraph::default();
    let commands = egraph
        .parser
        .get_program_from_string(None, &program)
        .unwrap();
    egraph.run_program(commands).unwrap();
    let (sort, value) = egraph.eval_expr(&var!("expr_root")).unwrap();
    let serialized = SerializedEGraph::new(&egraph, vec![(sort, value)]);
    let simplified = extract_expr(
        &serialized,
        &serialized.eclasses[serialized.roots.first().unwrap()].1[0],
        &mut FxHashMap::default(),
    )
    .unwrap_or(e);
    cache.lock().unwrap().push(e, simplified);
    simplified
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_basic_simplifications() {
        let x = Expression::from('x');
        let a = Expression::from('a');
        // Identity operations simplify away
        assert_eq!(((a * 1) + 0) / 1 + (1 - 1), a);
        // Evaluation after simplification
        let n = (x + (256 - (x % 256))).simplify();
        assert_eq!(n.exec(&[('x', 767)].into_iter().collect()).unwrap(), 768);
    }

    #[test]
    fn test_substitution() {
        let x = Expression::from('x');
        let new = (x - 255).substitute('x', x / 2).simplify();
        assert_eq!(new.len(), 5);
    }

    #[test]
    fn test_group_terms() {
        let s = Expression::from('s');
        let expr = (s * ((s - 4) + 1)) + (((s + 1) * ((s - 4) + 1)) - (s * ((s - 4) + 1)));
        assert_eq!(expr.simplify().len(), 7);
    }

    #[test]
    fn test_egglog_equality() {
        let a = Expression::from('a');
        let b = Expression::from('b');
        assert!((a + (b - a)).egglog_equal(b));
        assert!(!(a + 1).egglog_equal(a + 2));
    }

    #[test]
    fn test_simplify() {
        let (z, w, h, s) = (
            Expression::from('z'),
            Expression::from('w'),
            Expression::from('h'),
            Expression::from('s'),
        );
        // Nested divisions combine: ((((w + 3) / 2) + 2) / 2) -> (w + 7) / 4
        assert_eq!(((((w + 3) / 2) + 2) / 2).simplify(), (w + 7) / 4);
        // Complex division simplification
        let o = (z
            / ((-5 + (((((-5 + ((((((w + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
                * (-5 + (((9 + (4 * (-5 + ((((((153 + h) / 2) / 2) / 2) / 2) / 2)))) / 2) / 2))))
            % 64;
        assert!(o.simplify().len() <= 27);
        // // Mul-div simplification
        // let x = z % (((((153 + h) / 8) + -31) * ((((w + 153) / 8) + -31) / 16)) * 64);
        // assert!(x.simplify().len() < 15);
        // Like-term combining: 1+s+8+s+12+s+1+s+3+s+8+s+3+s+11+s+15+s+8+s+19 -> 10*s + 89
        let x: Expression =
            (((((((((((((((((((1 + s) + 8) + s) + 12) + s) + 1) + s) + 3) + s) + 8) + s)
                + 3)
                + s)
                + 11)
                + s)
                + 15)
                + s)
                + 8)
                + s)
                + 19;
        assert!(x.simplify().egglog_equal((s * 10) + 89));
    }

    #[test]
    fn test_no_explode() {
        // This expression previously caused e-graph explosion with naive associativity rules
        let x: Expression = 1 + ((8 / Expression::from(32)) + 27);
        x.simplify();
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(10))]
        #[test]
        fn test_simplify_preserves_eval(x_val in 0usize..100, y_val in 0usize..100, z_val in 0usize..100) {
            let (x, y, z) = (Expression::from('x'), Expression::from('y'), Expression::from('z'));
            // Simplification preserves evaluation
            let expr = ((x + 3) * 2) - (x * 2) + (y % 5);
            let env = [('x', x_val), ('y', y_val)].into_iter().collect();
            assert_eq!(expr.exec(&env).unwrap(), expr.simplify().exec(&env).unwrap());
            // Substitution + simplification preserves evaluation
            let expr = (x + y) * (y - x);
            let substituted = expr.substitute('x', z + 1).substitute('y', z - 1);
            let env = [('z', z_val)].into_iter().collect();
            assert_eq!(substituted.exec(&env).unwrap(), substituted.simplify().exec(&env).unwrap());
        }
    }
}

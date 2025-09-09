use egg::*;
use generational_box::{AnyStorage, GenerationalBox, Owner, SyncStorage};
use rustc_hash::FxHashMap;
use serde::{Serialize, Serializer};
use std::{
    fmt::Debug,
    hash::Hash,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, Div, DivAssign, Mul, MulAssign,
        Rem, RemAssign, Sub, SubAssign,
    },
    sync::OnceLock,
};
use symbolic_expressions::Sexp;

type ExprBox = GenerationalBox<Vec<Term>, SyncStorage>;

static EXPR_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();

pub fn expression_owner() -> &'static Owner<SyncStorage> {
    EXPR_OWNER.get_or_init(SyncStorage::owner)
}

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
            terms: expression_owner().insert(terms),
        }
    }

    pub fn is_acc(&self) -> bool {
        self.terms.read().iter().any(|i| matches!(i, Term::Acc(_)))
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
    Mod,
    Min,
    Max,
    And,
    Or,
    Gte,
    Lt,
    Acc(char),
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
            Term::Max => write!(f, "max"),
            Term::And => write!(f, "&&"),
            Term::Or => write!(f, "||"),
            Term::Gte => write!(f, ">="),
            Term::Lt => write!(f, "<"),
            Term::Acc(s) => write!(f, "{s}"),
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
                Term::Acc(c) => format!("Acc({c})"),
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

impl Expression {
    pub fn to_egglog(&self) -> String {
        let mut symbols = vec![];
        for term in self.terms.read().iter() {
            let new_symbol = match term {
                Term::Num(n) => format!("(MNum {n})"),
                Term::Var(c) => format!("(MVar \"{c}\")"),
                Term::Acc(s) => format!("(MAccum \"{s}\")"),
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
                Term::Var(c) => format!("const_{c}"),
                Term::Acc(_) => unreachable!(),
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
}

impl std::fmt::Display for Expression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Expression {
    /// Simplify the expression to its minimal terms
    pub fn simplify(self) -> Self {
        if self.terms.read().len() == 1 {
            return self;
        }
        egg_simplify(self, false)
    }

    /// Simplify the expression to its minimal terms, using a cache to retrieve / store the simplification
    #[allow(clippy::mutable_key_type)]
    pub fn simplify_cache(self, cache: &mut FxHashMap<Expression, Expression>) -> Self {
        if let Some(s) = cache.get(&self) {
            *s
        } else {
            let simplified = self.simplify();
            cache.insert(self, simplified);
            simplified
        }
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
        if self == 0 || rhs == i32::MAX {
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
}

impl Expression {
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
                Term::Acc(_) => stack.push(1),
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
                Term::Acc(_) => stack.push(1),
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
    /// Evaluate the expression given variables.
    pub fn exec_float(&self, variables: &FxHashMap<char, usize>) -> Option<f64> {
        self.exec_stack_float(variables, &mut Vec::new())
    }
    /// Evaluate the expression given variables. This function requires a stack to be given for use as storage
    pub fn exec_stack_float(
        &self,
        variables: &FxHashMap<char, usize>,
        stack: &mut Vec<f64>,
    ) -> Option<f64> {
        for term in self.terms.read().iter() {
            match term {
                Term::Num(n) => stack.push(*n as f64),
                Term::Acc(_) => stack.push(1.0),
                Term::Var(c) =>
                {
                    #[allow(clippy::needless_borrow)]
                    if let Some(n) = variables.get(&c) {
                        stack.push(*n as f64)
                    } else {
                        return None;
                    }
                }
                _ => {
                    let a = stack.pop().unwrap();
                    let b = stack.pop().unwrap();
                    stack.push(term.as_float_op().unwrap()(a, b));
                }
            }
        }
        stack.pop()
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

    /// Check if the '-' variable exists in the expression.
    pub fn is_unknown(&self) -> bool {
        self.terms
            .read()
            .iter()
            .any(|t| matches!(t, Term::Var('-')))
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

define_language! {
    enum SimpleLanguage {
        Num(i32),
        "+" = Add([Id; 2]),
        "*" = Mul([Id; 2]),
        Symbol(Symbol),
    }
}

fn luminal_to_egg(expr: &Expression) -> RecExpr<Math> {
    let mut stack = Vec::new();

    for term in expr.terms.read().iter() {
        match term {
            Term::Num(_) | Term::Var(_) => {
                stack.push(symbolic_expressions::Sexp::String(format!("{term:?}")))
            }
            Term::Acc(_) => stack.push(symbolic_expressions::Sexp::String("1".to_string())),
            _ => {
                let left = stack.pop().unwrap();
                let right = stack.pop().unwrap();
                let subexpr = symbolic_expressions::Sexp::List(vec![
                    symbolic_expressions::Sexp::String(format!("{term:?}")),
                    left,
                    right,
                ]);
                stack.push(subexpr);
            }
        }
    }
    fn parse_sexp_into<L: FromOp>(
        sexp: &Sexp,
        expr: &mut RecExpr<L>,
    ) -> Result<Id, RecExprParseError<L::Error>> {
        match sexp {
            Sexp::Empty => Err(egg::RecExprParseError::EmptySexp),
            Sexp::String(s) => {
                let node = L::from_op(s, vec![]).map_err(egg::RecExprParseError::BadOp)?;
                Ok(expr.add(node))
            }
            Sexp::List(list) if list.is_empty() => Err(egg::RecExprParseError::EmptySexp),
            Sexp::List(list) => match &list[0] {
                Sexp::Empty => unreachable!("Cannot be in head position"),
                list @ Sexp::List(..) => Err(egg::RecExprParseError::HeadList(list.to_owned())),
                Sexp::String(op) => {
                    let arg_ids: Vec<Id> = list[1..]
                        .iter()
                        .map(|s| parse_sexp_into(s, expr))
                        .collect::<Result<_, _>>()?;
                    let node = L::from_op(op, arg_ids).map_err(egg::RecExprParseError::BadOp)?;
                    Ok(expr.add(node))
                }
            },
        }
    }

    let sexp = stack.pop().unwrap();
    let mut expr = RecExpr::default();
    parse_sexp_into(&sexp, &mut expr).unwrap();
    expr
}

fn egg_to_luminal(expr: RecExpr<Math>) -> Expression {
    fn create_postfix(expr: &[Math]) -> Vec<Term> {
        match expr.last().unwrap() {
            Math::Num(i) => vec![Term::Num(*i)],
            Math::Add([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Add],
            ]
            .concat(),
            Math::Sub([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Sub],
            ]
            .concat(),
            Math::Mul([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Mul],
            ]
            .concat(),
            Math::Div([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Div],
            ]
            .concat(),
            Math::Mod([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Mod],
            ]
            .concat(),
            Math::Min([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Min],
            ]
            .concat(),
            Math::Max([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Max],
            ]
            .concat(),
            Math::And([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::And],
            ]
            .concat(),
            Math::Or([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Or],
            ]
            .concat(),
            Math::LessThan([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Lt],
            ]
            .concat(),
            Math::GreaterThanEqual([a, b]) => [
                create_postfix(&expr[..usize::from(*b) + 1]),
                create_postfix(&expr[..usize::from(*a) + 1]),
                vec![Term::Gte],
            ]
            .concat(),
            Math::Symbol(s) => vec![Term::Var(s.as_str().chars().next().unwrap())],
        }
    }
    let mut terms = vec![];
    terms.extend(create_postfix(expr.as_ref()));
    Expression::new(terms)
}

type EGraph = egg::EGraph<Math, ConstantFold>;
type Rewrite = egg::Rewrite<Math, ConstantFold>;

define_language! {
    enum Math {
        Num(i32),
        "+" = Add([Id; 2]),
        "-" = Sub([Id; 2]),
        "*" = Mul([Id; 2]),
        "/" = Div([Id; 2]),
        "%" = Mod([Id; 2]),
        "min" = Min([Id; 2]),
        "max" = Max([Id; 2]),
        "&&" = And([Id; 2]),
        "||" = Or([Id; 2]),
        "<" = LessThan([Id; 2]),
        ">=" = GreaterThanEqual([Id; 2]),
        Symbol(Symbol),
    }
}

#[derive(Default)]
pub struct ConstantFold;
impl Analysis<Math> for ConstantFold {
    type Data = Option<i32>;

    fn make(egraph: &egg::EGraph<Math, ConstantFold>, enode: &Math) -> Self::Data {
        let x = |i: &Id| egraph[*i].data.as_ref().map(|d| *d);
        Some(match enode {
            Math::Num(c) => *c,
            Math::Add([a, b]) => x(a)?.checked_add(x(b)?)?,
            Math::Sub([a, b]) => x(a)?.checked_sub(x(b)?)?,
            Math::Mul([a, b]) => x(a)?.checked_mul(x(b)?)?,
            Math::Div([a, b]) if x(b) != Some(0) => {
                let (a, b) = (x(a)?, x(b)?);
                if a % b != 0 {
                    return None;
                } else {
                    a.checked_div(b)?
                }
            }
            Math::Mod([a, b]) => x(a)?.checked_rem(x(b)?)?,
            Math::Min([a, b]) => x(a)?.min(x(b)?),
            Math::Max([a, b]) => x(a)?.max(x(b)?),
            Math::And([a, b]) => (x(a)? != 0 && x(b)? != 0) as i32,
            Math::Or([a, b]) => (x(a)? != 0 || x(b)? != 0) as i32,
            Math::LessThan([a, b]) => (x(a)? < x(b)?) as i32,
            Math::GreaterThanEqual([a, b]) => (x(a)? >= x(b)?) as i32,
            _ => return None,
        })
    }

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        merge_option(to, from, |a, b| {
            assert_eq!(*a, b, "Merged non-equal constants");
            DidMerge(false, false)
        })
    }

    fn modify(egraph: &mut EGraph, id: Id) {
        let data = egraph[id].data;
        if let Some(c) = data {
            let added = egraph.add(Math::Num(c));
            egraph.union(id, added);
            egraph[id].nodes.retain(|n| n.is_leaf());

            #[cfg(debug_assertions)]
            egraph[id].assert_unique_leaves();
        }
    }
}

fn is_not_zero(var: &str) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let var = var.parse().unwrap();
    move |egraph, _, subst| egraph[subst[var]].data.map(|i| i != 0).unwrap_or(true)
}

fn is_const_positive(vars: &[&str]) -> impl Fn(&mut EGraph, Id, &Subst) -> bool {
    let vars: Vec<Var> = vars.iter().map(|i| i.parse().unwrap()).collect::<Vec<_>>();
    move |egraph, _, subst| {
        vars.iter()
            .all(|i| egraph[subst[*i]].data.map(|i| i >= 0).unwrap_or(false))
    }
}

fn make_rules(lower_bound_zero: bool) -> Vec<Rewrite> {
    let mut v = vec![
        // Communative properties
        rewrite!("commute-add"; "(+ ?a ?b)" => "(+ ?b ?a)"),
        rewrite!("commute-mul"; "(* ?a ?b)" => "(* ?b ?a)"),
        rewrite!("commute-min"; "(min ?a ?b)" => "(min ?b ?a)"),
        rewrite!("commute-max"; "(max ?a ?b)" => "(max ?b ?a)"),
        rewrite!("commute-and"; "(&& ?a ?b)" => "(&& ?b ?a)"),
        rewrite!("commute-or"; "(|| ?a ?b)" => "(|| ?b ?a)"),
        // Associative properties
        rewrite!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rewrite!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rewrite!("assoc-div"; "(/ (/ ?a ?b) ?c)" => "(/ ?a (* ?b ?c))"),
        rewrite!("mul-div-associative"; "(/ (* ?a ?b) ?c)" => "(* ?a (/ ?b ?c))"),
        // rewrite!("mul-div-associative-rev"; "(* ?a (/ ?b ?c))" => "(/ (* ?a ?b) ?c)"), // BAD? Makes test_pool_1d fail
        rewrite!("sub-canon"; "(- ?a ?b)" => "(+ ?a (* -1 ?b))"),
        // Distributive
        rewrite!("distribute-mul"; "(* ?a (+ ?b ?c))" => "(+ (* ?a ?b) (* ?a ?c))"),
        rewrite!("distribute-div"; "(/ (+ ?a ?b) ?c)" => "(+ (/ ?a ?c) (/ ?b ?c))"),
        rewrite!("distribute-max"; "(* ?a (max ?b ?c))" => "(max (* ?a ?b) (* ?a ?c))" if is_const_positive(&["?a"])),
        rewrite!("distribute-min"; "(* ?a (min ?b ?c))" => "(min (* ?a ?b) (* ?a ?c))"),
        // rewrite!("distribute-mod"; "(* (% ?b ?c) ?a)" => "(% (* ?b ?a) (* ?c ?a))"),
        // Factoring
        rewrite!("factor-mul"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        // rewrite!("factor-div"    ; "(+ (/ ?a ?b) (/ ?a ?c))" => "(/ ?a (+ ?b ?c))"),
        rewrite!("group-terms"; "(+ ?a ?a)" => "(* 2 ?a)"),
        // Other
        // rewrite!("explicit-truncate"; "(* (/ ?a ?b) ?b)" => "(- ?a (% ?a ?b))"),
        // rewrite!("mul-mod"; "(% (* ?a ?b) ?b)" => "0"),
        rewrite!("div-move-inside"; "(+ (/ ?a ?b) ?c)" => "(/ (+ ?a (* ?c ?b)) ?b)"),
        // rewrite!("mul-distribute"; "(* ?a (% (/ ?b ?c) ?d))" => "(% (/ ?b (* ?c ?a)) (* ?d ?a))"), // BAD
        // rewrite!("div-mod-mul"; "(% (/ ?a ?b) ?c)" => "(% ?a (* ?b ?c))"),
        // Simple binary reductions
        rewrite!("add-0"; "(+ ?a 0)" => "?a"),
        rewrite!("mul-0"; "(* ?a 0)" => "0"),
        rewrite!("mul-1"; "(* ?a 1)" => "?a"),
        rewrite!("div-1"; "(/ ?a 1)" => "?a"),
        rewrite!("div-self"; "(/ ?a ?a)" => "1"),
        rewrite!("and-0"; "(&& ?a 0)" => "0"),
        rewrite!("and-1"; "(&& ?a 1)" => "?a"),
        rewrite!("or-0"; "(|| ?a 0)" => "?a"),
        rewrite!("or-1"; "(|| ?a 1)" => "1"),
        rewrite!("min-i32-max"; "(min ?a 2147483647)" => "?a"),
        rewrite!("max-i32-max"; "(max ?a 2147483647)" => "2147483647"),
        rewrite!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero("?x")),
        rewrite!("add-zero"; "?a" => "(+ ?a 0)"),
        rewrite!("mul-one";  "?a" => "(* ?a 1)"),
        rewrite!("cancel-sub"; "(- ?a ?a)" => "0"),
        rewrite!("cancel-div"; "(/ ?a ?a)" => "1" if is_not_zero("?a")),
        rewrite!("dedup-max"; "(max ?a (max ?a ?b))" => "(max ?a ?b)"),
        rewrite!("dedup-min"; "(min ?a (min ?a ?b))" => "(min ?a ?b)"),
    ];
    if lower_bound_zero {
        v.push(rewrite!("max-zero"; "(max ?a 0)" => "?a"));
    }
    v
}

fn egg_simplify(e: Expression, lower_bound_zero: bool) -> Expression {
    // Convert to egg expression
    let expr = luminal_to_egg(&e);
    // Simplify
    let runner = Runner::default()
        // .with_iter_limit(1_000)
        // .with_time_limit(std::time::Duration::from_secs(30))
        // .with_node_limit(100_000_000)
        .with_expr(&expr)
        .run(&make_rules(lower_bound_zero));
    // runner.print_report();
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let (_, best) = extractor.find_best(runner.roots[0]);
    // Convert back to luminal expression
    egg_to_luminal(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_expressions() {
        let n = Expression::from('x') + (256 - (Expression::from('x') % 256));
        assert_eq!(
            n.simplify()
                .exec(&[('x', 767)].into_iter().collect())
                .unwrap(),
            768
        );
    }

    #[test]
    fn test_minimizations() {
        let expr = ((Expression::from('a') * 1) + 0) / 1 + (1 - 1);
        let reduced_expr = expr.simplify();
        assert_eq!(reduced_expr, 'a');
    }

    #[test]
    fn test_substitution() {
        let main = Expression::from('x') - 255;
        let sub = Expression::from('x') / 2;
        let new = main.substitute('x', sub).simplify();
        assert_eq!(new.len(), 5);
    }

    #[test]
    fn test_group_terms() {
        let s = Expression::from('s');
        let expr = (s * ((s - 4) + 1)) + (((s + 1) * ((s - 4) + 1)) - (s * ((s - 4) + 1)));
        assert_eq!(expr.simplify().len(), 7);
    }

    #[test]
    fn test_simple_div() {
        let w = Expression::from('w');
        let s = ((((w + 3) / 2) + 2) / 2).simplify();
        assert_eq!(s, (w + 7) / 4);
    }

    #[test]
    fn test_other() {
        let z = Expression::from('z');
        let w = Expression::from('w');
        let h = Expression::from('h');
        let o = (z
            / ((-5 + (((((-5 + ((((((w + 153) / 2) / 2) / 2) / 2) / 2)) * 4) + 9) / 2) / 2))
                * (-5 + (((9 + (4 * (-5 + ((((((153 + h) / 2) / 2) / 2) / 2) / 2)))) / 2) / 2))))
            % 64;
        let x = o.simplify();
        assert_eq!(x.len(), 23); // Should be 21 if we can re-enable mul-div-associative-rev
    }

    #[test]
    fn test_final() {
        let z = Expression::from('z');
        let w = Expression::from('w');
        let h = Expression::from('h');
        let x = (z % (((((153 + h) / 8) + -31) * ((((w + 153) / 8) + -31) / 16)) * 64)).simplify();
        assert_eq!(x.len(), 15); // Should be 11 if we can re-enable mul-div-associative-rev
    }
}

//! String-backed symbolic-dimension names (our landing of PR #396's
//! design, ruling 2026-08-13): `Symbol` is a Copy handle to one
//! process-global interned `&'static str`, so `Term` stays Copy while
//! names are arbitrary-length. Names validate against `[A-Za-z][A-Za-z0-9_]*`
//! with no doubled underscore and are REJECTED, never sanitized
//! (sanitizing is not injective — "a.b" and "a-b" must not collide).
//! The alphabet guarantees by construction that a name is a valid
//! egglog string literal and C identifier, so no codegen site
//! re-checks. Unlike main's PR, NO name is reserved: this branch
//! retired 'z' (z-var retirement, 2026-08-06) — every name is an
//! ordinary symbol.
//!
//! Equality, hashing, and ordering are by name, so any order-dependent
//! downstream behavior (such as backend slot assignment) is deterministic
//! in the name vocabulary, not in interning order. Construction interns one
//! leaked string per distinct name; this is the same bounded process-lifetime
//! storage contract as the old symbol interner, without an interior-mutable
//! handle inside map keys.

use rustc_hash::FxHashMap;
use std::sync::{
    OnceLock, RwLock,
    atomic::{AtomicU64, Ordering},
};

static NAME_INTERNER: OnceLock<RwLock<FxHashMap<String, &'static str>>> = OnceLock::new();
static FRESH_COUNTER: AtomicU64 = AtomicU64::new(0);

fn interner() -> &'static RwLock<FxHashMap<String, &'static str>> {
    NAME_INTERNER.get_or_init(|| RwLock::new(FxHashMap::default()))
}

fn is_well_formed(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    first.is_ascii_alphabetic()
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains("__")
}

/// A name that cannot be a dimension — the REPORTED form of the
/// rejection [`Symbol::new`] panics on. Main's PR #396 spells this as a
/// two-variant enum (`Malformed` | `Reserved`); this branch retired the
/// reserved index with 'z' (z-var retirement, 2026-08-06), so
/// malformedness is the only way a name can fail here and the type is a
/// struct. The `Display` text is the panic text verbatim, so the two
/// doors report identically.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidSymbolName(String);

impl std::fmt::Display for InvalidSymbolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "symbol name {:?} must match [A-Za-z][A-Za-z0-9_]* with no \
             doubled underscore (reject, never sanitize)",
            self.0
        )
    }
}

impl std::error::Error for InvalidSymbolName {}

/// An interned symbolic-dimension name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(&'static str);

impl Symbol {
    /// Intern a validated name — panics loudly on malformed input.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self::try_new_dim(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Intern a validated name, REPORTING the rejection instead of
    /// unwinding (main's `Symbol::try_new_dim`, PR #396).
    ///
    /// This is the door for a name the caller did not choose. A
    /// frontend importing someone else's graph — PT2 hands over names
    /// like `s77`, `u0`, and occasionally things that are not
    /// identifiers at all — must be able to see the rejection and remap,
    /// because DROPPING an unusable dim is the worst outcome available:
    /// a dim absent from the symbol map never gets a value, so it
    /// freezes at the export hint while the frontend, told it was
    /// dynamic, declines to recompile. Names are still rejected, never
    /// sanitized: sanitizing is not injective, so `a.b` and `a-b` would
    /// land on one symbol.
    pub fn try_new_dim(name: impl AsRef<str>) -> Result<Self, InvalidSymbolName> {
        let name = name.as_ref();
        if !is_well_formed(name) {
            return Err(InvalidSymbolName(name.to_string()));
        }
        Ok(Self::intern(name))
    }

    fn intern(name: &str) -> Self {
        // Fast path: the name is already interned (read lock only).
        if let Some(&existing) = interner().read().unwrap().get(name) {
            return Symbol(existing);
        }
        // Slow path: insert (write lock), double-checked because another
        // thread may have interned the name between the two locks.
        let mut guard = interner().write().unwrap();
        if let Some(&existing) = guard.get(name) {
            return Symbol(existing);
        }
        let interned: &'static str = Box::leak(name.to_string().into_boxed_str());
        guard.insert(name.to_string(), interned);
        Symbol(interned)
    }

    /// A fresh symbol no prior name can collide with — replaces the old
    /// private-use-char trick for internal temporaries.
    pub fn fresh(stem: &str) -> Self {
        let n = FRESH_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
        Self::new(format!("{stem}{n}"))
    }

    pub fn name(&self) -> &'static str {
        self.0
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl From<char> for Symbol {
    fn from(c: char) -> Self {
        Symbol::new(c.to_string())
    }
}

impl From<&char> for Symbol {
    fn from(c: &char) -> Self {
        Symbol::from(*c)
    }
}

impl From<&str> for Symbol {
    fn from(s: &str) -> Self {
        Symbol::new(s)
    }
}

impl serde::Serialize for Symbol {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0)
    }
}

impl<'de> serde::Deserialize<'de> for Symbol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        if !is_well_formed(&name) {
            return Err(serde::de::Error::custom(format!(
                "unusable symbol name {name:?}"
            )));
        }
        Ok(Symbol::intern(&name))
    }
}

/// The dynamic-dimension binding map (PR #396 vocabulary).
pub type DynMap = FxHashMap<Symbol, usize>;

#[cfg(test)]
mod tests {
    use super::Symbol;

    #[test]
    fn interning_equality_and_name_order() {
        let a = Symbol::new("seq");
        let b = Symbol::from("seq");
        assert_eq!(a, b);
        assert_eq!(a.name(), "seq");
        assert!(Symbol::new("a") < Symbol::new("b"), "Ord is by name");
        assert_eq!(Symbol::from('s').name(), "s");
    }

    #[test]
    #[should_panic(expected = "reject, never sanitize")]
    fn malformed_names_are_rejected() {
        Symbol::new("a.b");
    }

    /// The fallible door reports exactly what the panicking one panics
    /// with, and accepts exactly what it accepts. No name is reserved on
    /// this branch, so a bare `"z"` is an ordinary dimension.
    #[test]
    fn try_new_dim_reports_instead_of_unwinding() {
        assert_eq!(
            Symbol::try_new_dim("seq_len").unwrap(),
            Symbol::new("seq_len")
        );
        assert_eq!(Symbol::try_new_dim("s77").unwrap().name(), "s77");
        assert_eq!(Symbol::try_new_dim("z").unwrap().name(), "z");

        for bad in ["a.b", "a-b", "", "1st", "a__b"] {
            let error = Symbol::try_new_dim(bad).unwrap_err().to_string();
            assert!(
                error.contains("reject, never sanitize") && error.contains(&format!("{bad:?}")),
                "the report must name the offending string and the policy, got: {error}"
            );
        }
    }

    #[test]
    fn fresh_symbols_never_collide() {
        assert_ne!(Symbol::fresh("tmp"), Symbol::fresh("tmp"));
    }

    #[test]
    fn serde_round_trips_by_name() {
        let s = Symbol::new("seq_len");
        let json = serde_json::to_string(&s).unwrap();
        assert_eq!(json, "\"seq_len\"");
        let back: Symbol = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
    }
}

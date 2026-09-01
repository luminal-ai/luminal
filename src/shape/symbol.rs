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

/// An interned symbolic-dimension name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(&'static str);

impl Symbol {
    /// Intern a validated name — panics loudly on malformed input.
    pub fn new(name: impl AsRef<str>) -> Self {
        let name = name.as_ref();
        assert!(
            is_well_formed(name),
            "symbol name {name:?} must match [A-Za-z][A-Za-z0-9_]* with no \
             doubled underscore (reject, never sanitize)"
        );
        Self::intern(name)
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

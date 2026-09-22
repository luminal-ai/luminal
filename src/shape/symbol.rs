//! String-backed symbolic-dimension names (our landing of PR #396's
//! design, ruling 2026-08-13): `Symbol` is a Copy handle to one
//! process-global interned `&'static str`, so `Term` stays Copy while
//! names are arbitrary-length.
//!
//! Any non-empty string is a name (ruling 2026-09-22): a frontend hands
//! over whatever it calls its dimensions and the spelling is kept
//! verbatim, so `set_dim` by that spelling always finds it. The name is
//! never used as an identifier: codegen hex-encodes it and egglog gets a
//! quoted literal, so [`Symbol::egglog_literal`] is the one place the
//! quoting lives and [`Symbol::decode_serialized`] the one place it is
//! undone. Unlike main's PR, NO name is reserved: this branch retired
//! 'z' (z-var retirement, 2026-08-06) — every name is an ordinary symbol.
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
    !name.is_empty()
}

/// A name that cannot be a dimension — the REPORTED form of the
/// rejection [`Symbol::new`] panics on. Only the empty string fails: a
/// dimension has to be addressable by its name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvalidSymbolName(String);

impl std::fmt::Display for InvalidSymbolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "symbol name {:?} must be non-empty", self.0)
    }
}

impl std::error::Error for InvalidSymbolName {}

/// An interned symbolic-dimension name.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(&'static str);

impl Symbol {
    /// Intern a name — panics loudly on the empty string.
    pub fn new(name: impl AsRef<str>) -> Self {
        Self::try_new_dim(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Intern a name, REPORTING the rejection instead of unwinding
    /// (main's `Symbol::try_new_dim`, PR #396). This is the door for a
    /// name the caller did not choose; a frontend importing someone else's
    /// graph must see a rejection rather than drop the dim, because a dim
    /// absent from the symbol map never gets a value and freezes at its
    /// export hint.
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

    /// The name as an egglog string literal, quotes included: the
    /// spelling every `(IntVar ...)` and `(MVar ...)` renderer emits.
    pub fn egglog_literal(&self) -> String {
        Self::escape_egglog(self.0)
    }

    /// Quote a string for egglog's lexer, which reads `\"`, `\\`, `\n`
    /// and `\t` and takes every other character as itself.
    pub fn escape_egglog(name: &str) -> String {
        let mut out = String::with_capacity(name.len() + 2);
        out.push('"');
        for c in name.chars() {
            match c {
                '\\' => out.push_str("\\\\"),
                '"' => out.push_str("\\\""),
                '\n' => out.push_str("\\n"),
                '\t' => out.push_str("\\t"),
                c => out.push(c),
            }
        }
        out.push('"');
        out
    }

    /// The name inside a serialized string node. The serializer prints a
    /// string value with Rust's `{:?}`, so this undoes that escaping;
    /// `None` when `op` is not a quoted string.
    pub fn decode_serialized(op: &str) -> Option<String> {
        let inner = op.strip_prefix('"')?.strip_suffix('"')?;
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c != '\\' {
                out.push(c);
                continue;
            }
            match chars.next()? {
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                '0' => out.push('\0'),
                'u' => {
                    if chars.next()? != '{' {
                        return None;
                    }
                    let hex: String = chars.by_ref().take_while(|c| *c != '}').collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                _ => return None,
            }
        }
        Some(out)
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
    #[should_panic(expected = "must be non-empty")]
    fn the_empty_name_is_rejected() {
        Symbol::new("");
    }

    /// Any non-empty spelling is a name and is kept verbatim: a frontend's
    /// dimension names are addressed by exactly what the frontend called
    /// them. No name is reserved, so a bare `"z"` is an ordinary dimension.
    #[test]
    fn every_non_empty_spelling_is_a_name() {
        for name in [
            "seq_len",
            "s77",
            "z",
            "_batch",
            "a__b",
            "a.b",
            "a-b",
            "1st",
            "seq len",
            "L['x'].size()[0]",
            "q\"uote",
            "back\\slash",
            "größe",
            "#1",
        ] {
            assert_eq!(Symbol::try_new_dim(name).unwrap().name(), name);
        }
        let error = Symbol::try_new_dim("").unwrap_err().to_string();
        assert!(error.contains("must be non-empty"), "{error}");
    }

    /// The egglog spelling quotes what egglog's lexer needs quoted and
    /// nothing else; the serialized form comes back to the same name.
    #[test]
    fn egglog_literal_round_trips_through_the_serialized_form() {
        assert_eq!(Symbol::new("_batch").egglog_literal(), "\"_batch\"");
        assert_eq!(
            Symbol::new("q\"uote\\back\nline\ttab").egglog_literal(),
            "\"q\\\"uote\\\\back\\nline\\ttab\""
        );
        for name in [
            "_batch",
            "a.b",
            "q\"uote",
            "back\\slash",
            "new\nline",
            "größe",
            "nul\0",
        ] {
            let serialized = format!("{name:?}");
            assert_eq!(
                Symbol::decode_serialized(&serialized).as_deref(),
                Some(name),
                "{serialized}"
            );
        }
        assert_eq!(Symbol::decode_serialized("IntVar"), None);
        assert_eq!(Symbol::decode_serialized("\"unterminated"), None);
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

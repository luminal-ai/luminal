//! Symbolic dimension variables.
//!
//! A [`Symbol`] names one dynamic dimension — the `s` in a `[s, 128]` shape.
//! Everything that means "a dim variable" says `Symbol`, and every
//! symbol-to-concrete-size mapping says [`DynMap`].
//!
//! A `Symbol` is a `Copy` handle into a generational arena, the same shape as
//! [`Expression`](crate::shape::Expression) and its `Vec<Term>`: the name lives
//! in the arena, and the handle is a pointer plus a generation. That is what
//! keeps [`Term`](crate::shape::Term) `Copy` while a dim name is an owned,
//! arbitrary-length string.
//!
//! `GenerationalBox` supplies only `Copy`, `Clone` and `Debug`, so equality,
//! ordering and hashing are written by hand here and read *through* the handle,
//! comparing names rather than arena slots. `Expression` does the same for its
//! `Hash` and `PartialEq`.
//!
//! `Ord` compares names, not handles, because CUDA assigns each dim a
//! `dyn_dims[]` slot by sorting the dim set and the host uploads values in that
//! same order. Sorting by name keeps that a function of the graph alone; handle
//! order is allocation order.
//!
//! One name is reserved. `"z"` is the runtime loop index, not a dimension:
//! [`Expression::dyn_vars`](crate::shape::Expression::dyn_vars) filters it out
//! and generated kernels declare their own `long long const_z`, so a dimension
//! named `z` would vanish from the buffer planner and collide with that local.
//! See [`RESERVED_INDEX_NAME`] and [`Symbol::try_new_dim`].

use std::fmt;
use std::sync::{OnceLock, RwLock};

use generational_box::{AnyStorage, GenerationalBox, Owner, SyncStorage};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// A `Copy` handle to a name in the arena, as `ExprBox` is to a `Vec<Term>`.
type NameBox = GenerationalBox<String, SyncStorage>;

/// Names one symbolic dimension.
///
/// `Copy`, and compares/hashes/orders by name.
///
/// Most code never names a constructor: the dim-taking methods on `Graph` and
/// `CompileOptions` are generic over `impl Into<Symbol>`, so `set_dim("seq", 8)`
/// and `set_dim('s', 8)` both work. Reach for [`sym`] when you need a value
/// directly, or [`Symbol::try_new_dim`] for a name from a frontend.
#[derive(Clone, Copy)]
pub struct Symbol(NameBox);

/// Concrete sizes for the symbolic dimensions of a graph, resolved at runtime.
pub type DynMap = FxHashMap<Symbol, usize>;

/// The runtime loop index. Reserved — never hand this out as a dimension.
///
/// A `&str` rather than a `Symbol` because a `Symbol` is an arena handle and so
/// cannot be a `const`. Use [`Symbol::reserved_index`] for the value.
const RESERVED_INDEX_NAME: &str = "z";

/// A name that cannot be a dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidSymbolName {
    /// Not `[A-Za-z][A-Za-z0-9_]*`.
    ///
    /// The alphabet makes `const_<name>` a usable identifier and `"<name>"` a
    /// valid egglog literal by construction, so no codegen site re-checks. The
    /// `__` rule is C++'s, not C's: C++ reserves every identifier containing a
    /// doubled underscore, not just leading ones.
    ///
    /// Rejected rather than sanitized — sanitizing is not injective, so `a.b`
    /// and `a-b` would land on one `#define` and one `MVar`.
    Malformed(String),
    /// The reserved loop index — see [`RESERVED_INDEX_NAME`].
    Reserved,
}

impl fmt::Display for InvalidSymbolName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Malformed(name) => write!(
                f,
                "dim name {name:?} must match [A-Za-z][A-Za-z0-9_]* with no \
                 doubled underscore, so that const_<name> is never an identifier \
                 C++ reserves to the implementation, and <name> is a valid \
                 egglog literal"
            ),
            Self::Reserved => write!(
                f,
                "{RESERVED_INDEX_NAME:?} is the reserved runtime loop index, not a \
                 dimension: kernels declare their own const_{RESERVED_INDEX_NAME} \
                 local and dyn_vars() filters it out, so a dim with this name would \
                 be dropped from the buffer planner"
            ),
        }
    }
}

static NAME_OWNER: OnceLock<Owner<SyncStorage>> = OnceLock::new();
static NAME_INTERNER: OnceLock<RwLock<FxHashMap<String, NameBox>>> = OnceLock::new();

/// One arena slot per distinct name.
///
/// Not needed for correctness — equality reads through the handle and compares
/// names — but without it every `Symbol::new` would claim a fresh slot that is
/// never reclaimed. Mirrors `EXPRESSION_INTERNER`.
fn intern(name: &str) -> NameBox {
    let interner = NAME_INTERNER.get_or_init(|| RwLock::new(FxHashMap::default()));
    if let Some(existing) = interner.read().unwrap().get(name) {
        return *existing;
    }
    let mut guard = interner.write().unwrap();
    if let Some(existing) = guard.get(name) {
        return *existing;
    }
    let boxed = NAME_OWNER
        .get_or_init(SyncStorage::owner)
        .insert(name.to_string());
    guard.insert(name.to_string(), boxed);
    boxed
}

fn is_well_formed(name: &str) -> bool {
    // No is_empty check: starts_with already rejects "".
    name.starts_with(|c: char| c.is_ascii_alphabetic())
        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !name.contains("__")
}

impl Symbol {
    /// The reserved runtime loop index.
    pub fn reserved_index() -> Symbol {
        Symbol(intern(RESERVED_INDEX_NAME))
    }

    /// Build a symbol, allowing the reserved index.
    ///
    /// Panics if the name is malformed. Prefer [`Symbol::try_new_dim`] for
    /// names arriving from a frontend: it also rejects the reserved index, and
    /// reports rather than unwinding.
    pub fn new(name: &str) -> Symbol {
        Self::try_new(name).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Build a symbol, allowing the reserved index.
    fn try_new(name: &str) -> Result<Symbol, InvalidSymbolName> {
        if !is_well_formed(name) {
            return Err(InvalidSymbolName::Malformed(name.to_string()));
        }
        Ok(Symbol(intern(name)))
    }

    /// Build a symbol for a *dimension*, rejecting the reserved loop index.
    ///
    /// This is the guard a frontend should use on a name it did not choose.
    pub fn try_new_dim(name: &str) -> Result<Symbol, InvalidSymbolName> {
        let sym = Self::try_new(name)?;
        if sym.is_reserved() {
            return Err(InvalidSymbolName::Reserved);
        }
        Ok(sym)
    }

    /// Whether this is the reserved loop index rather than a real dimension.
    pub fn is_reserved(&self) -> bool {
        *self.0.read() == RESERVED_INDEX_NAME
    }
}

/// Build a dim symbol. Shorthand for [`Symbol::new`].
pub fn sym(name: &str) -> Symbol {
    Symbol::new(name)
}

// GenerationalBox supplies only Copy/Clone/Debug, and a derived PartialEq would
// compare arena slot and generation — so two symbols named `s` would differ.
// These read through the handle and compare names.

impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        *self.0.read() == *other.0.read()
    }
}

impl Eq for Symbol {}

impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.read().cmp(&other.0.read())
    }
}

impl std::hash::Hash for Symbol {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Hash the name, not the handle: two symbols named `s` must agree.
        (**self.0.read()).hash(state);
    }
}

// No `Borrow<str>`: probing a DynMap with a bare `&str` would need one borrowed
// from the Symbol, but the name lives behind a read guard that outlives nothing.
// Callers build a Symbol to look up.

impl fmt::Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.read())
    }
}

/// Prints the bare name, matching how `Term` and `Expression` render dims.
impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.read())
    }
}

/// `expr('s')` reads better than `expr(sym("s"))` in hand-written models.
/// Validated like any other name, so `'%'` panics rather than minting `const_%`.
impl From<char> for Symbol {
    fn from(c: char) -> Self {
        let mut buf = [0u8; 4];
        Symbol::new(c.encode_utf8(&mut buf))
    }
}

/// Serializes as the name, so the encoding is unchanged from when a dim was a
/// `char`: `Term::Var('s')` was `{"Var":"s"}` and still is, and artifacts
/// written before this change still load.
impl Serialize for Symbol {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0.read())
    }
}

impl<'de> Deserialize<'de> for Symbol {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        // try_new, not try_new_dim: the reserved index appears in serialized
        // strides and has to round-trip.
        Symbol::try_new(&name).map_err(serde::de::Error::custom)
    }
}

/// The C identifier a symbol takes in generated kernel source.
///
/// Kernels also declare a *local* `const_z` for the thread index, so this
/// namespace is shared with generated code — see [`RESERVED_INDEX_NAME`].
pub fn kernel_const_name(sym: &Symbol) -> String {
    format!("const_{sym}")
}

/// The name a symbol carries inside egglog, as the payload of `(MVar "…")`.
///
/// Kept separate from [`kernel_const_name`] even though both are the bare
/// spelling: the `.egg` rewrite rules match these literally, so this one is a
/// wire format that rules depend on, not a display detail.
pub fn egglog_var_name(sym: &Symbol) -> String {
    sym.to_string()
}

/// Inverse of [`egglog_var_name`], for names read back out of the e-graph.
///
/// Validates rather than trusting. A name that does not survive the round trip
/// means the boundary mangled it, and a mangled name misses in `dyn_map`, where
/// a miss silently becomes a dimension of 0.
pub fn symbol_from_egglog_name(name: &str) -> Symbol {
    Symbol::try_new(name).unwrap_or_else(|e| panic!("bad dim name out of egglog: {e}"))
}
#[cfg(test)]
mod tests {
    use crate::prelude::*;

    /// The whole reason `to_kernel_with_index` exists. Callers used to string-
    /// replace `const_z` in generated source, which is prefix-unsafe once dim
    /// names can exceed one char: `const_z` is a prefix of `const_zip`, so a
    /// dim named `zip` would be rewritten mid-name to `iip`.
    ///
    /// Substituting on the term means only the reserved index moves and every
    /// dim is emitted untouched. `zip` is not spellable while `Symbol` is a
    /// char, so the prefix case itself gets pinned when the newtype lands.
    #[test]
    fn index_substitution_only_touches_the_reserved_index() {
        let e = expr('z') + expr('a');
        assert_eq!(e.to_kernel_with_index("i"), "(i+const_a)");
        assert!(!e.to_kernel_with_index("i").contains("const_z"));
    }

    #[test]
    fn to_kernel_defaults_to_const_z_for_the_index() {
        assert_eq!((expr('z') + expr('a')).to_kernel(), "(const_z+const_a)");
    }

    /// Generated kernels declare a local `long long const_z` for the thread
    /// index, so a dim may never be named `z` — it would `#define` over that
    /// local.
    #[test]
    fn reserved_index_is_z_and_is_not_a_dyn_var() {
        assert_eq!(Symbol::reserved_index().to_string(), "z");
        assert_eq!(kernel_const_name(&Symbol::reserved_index()), "const_z");
        assert!(expr('z').dyn_vars().is_empty());
        assert_eq!(expr('a').dyn_vars(), vec![sym("a")]);
    }

    #[cfg(test)]
    mod no_const_prefix_surgery {
        use std::path::Path;

        fn scan(dir: &Path, needle: &str, hits: &mut Vec<String>) {
            let Ok(entries) = std::fs::read_dir(dir) else {
                return;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                let name = entry.file_name();
                let name = name.to_string_lossy();
                if path.is_dir() {
                    if name != "target" && name != ".git" {
                        scan(&path, needle, hits);
                    }
                } else if name.ends_with(".rs")
                    && let Ok(src) = std::fs::read_to_string(&path)
                    && src.contains(needle)
                {
                    hits.push(path.display().to_string());
                }
            }
        }

        /// Rewriting generated kernel source by string-replacing a `const_`-prefixed
        /// name is prefix-unsafe: `const_z` sits inside `const_zip`, so the surgery
        /// lands in the middle of an unrelated dim's name and silently miscompiles.
        ///
        /// `Expression::to_kernel_with_index` / `to_kernel_with` substitute on the
        /// term instead, before any string exists. Nothing stops someone reaching
        /// for the old pattern again, so fail the build if they do.
        #[test]
        fn no_source_file_string_replaces_a_const_prefix() {
            // Assembled so this file does not match its own needles.
            //
            // Two forms, because the original bug had the second one: Metal built
            // the name with `format!("const_{symbol}")` and then replaced *that*,
            // so a needle matching only a string literal would have missed it.
            let literal = format!("{}(\"const_", "replace");
            let built = format!("{}!(\"const_", "format");
            let root = Path::new(env!("CARGO_MANIFEST_DIR"));

            let mut hits = Vec::new();
            for sub in ["src", "crates", "examples"] {
                let dir = root.join(sub);
                assert!(dir.is_dir(), "expected to scan {}", dir.display());
                scan(&dir, &literal, &mut hits);
                // Building the name is only a problem when it feeds a rewrite, so
                // require a rewrite call in the same file rather than flagging
                // every construction site — kernel_const_name is a legitimate one.
                // Needles are assembled rather than written literally, or this
                // file matches itself.
                let mut built_hits = Vec::new();
                scan(&dir, &built, &mut built_hits);
                // Assembled, or this file matches its own filter: it contains
                // kernel_const_name's legitimate `format!("const_…")`.
                let rewrite = format!(".{}(", "replace");
                let rewrite_n = format!(".{}(", "replacen");
                hits.extend(built_hits.into_iter().filter(|f| {
                    std::fs::read_to_string(f)
                        .map(|src| src.contains(&rewrite) || src.contains(&rewrite_n))
                        .unwrap_or(false)
                }));
            }
            hits.sort();
            hits.dedup();

            // A lint that cannot tell "clean" from "never looked" is worse than no
            // lint: scan() swallows read_dir errors and the needles are assembled
            // at runtime, so prove the scanner finds something it should.
            let mut control = Vec::new();
            scan(&root.join("src"), "fn is_well_formed", &mut control);
            assert!(
                !control.is_empty(),
                "scanner found nothing at all — the lint is not actually looking"
            );

            assert!(
                hits.is_empty(),
                "these files rewrite generated source with {literal}…) or {built}…), which is \
             prefix-unsafe — a dim named `zip` emits `const_zip` and the replace \
             lands mid-name. Use Expression::to_kernel_with_index / to_kernel_with \
             instead, which substitute on the term:\n  {}",
                hits.join("\n  ")
            );
        }
    }

    /// Symbol #26 used to land on 'z', the reserved loop index, which
    /// `dyn_vars()` filters out — so the dim vanished from the buffer planner
    /// and every search candidate died evaluating its size.
    #[test]
    fn reserved_index_cannot_be_minted_as_a_dim() {
        assert_eq!(Symbol::try_new_dim("z"), Err(InvalidSymbolName::Reserved));
        assert!(Symbol::try_new("z").is_ok(), "still nameable as the index");
    }

    #[test]
    #[should_panic(expected = "reserved runtime loop index")]
    fn set_dim_rejects_the_reserved_index() {
        let mut cx = Graph::new();
        cx.set_dim('z', 4);
    }

    /// `set_dim` is not the only way a dim gets named — a shape can name one
    /// directly, and that path had no guard. `cx.tensor(('z', 4))` built fine,
    /// reported no dyn_vars (so the buffer planner never saw the dim), and
    /// emitted `const_z`, which in generated CUDA is the thread index.
    #[test]
    #[should_panic(expected = "reserved runtime loop index")]
    fn a_shape_cannot_be_sized_by_the_loop_index() {
        let mut cx = Graph::default();
        let _ = cx.tensor(('z', 4));
    }

    /// ...but a *stride* legitimately uses it, which is what makes the guard
    /// specific to dimension sizes rather than to expressions generally.
    #[test]
    fn strides_may_use_the_loop_index() {
        let cx_shape = ShapeTracker::new((4, 8));
        assert!(
            cx_shape.strides.iter().any(|s| s.uses_reserved_index()),
            "strides are indexed by the loop var"
        );
        assert!(!cx_shape.dims.iter().any(|d| d.uses_reserved_index()));
    }

    /// Equality is by name, so a char literal and a string name a single dim.
    /// The `.egg` rewrite rules match `(MVar "s")` literally, so this is what
    /// keeps hand-written models hitting them.
    #[test]
    fn char_and_str_spellings_name_the_same_dim() {
        assert_eq!(Symbol::from('s'), sym("s"));
        assert_eq!(expr('s'), expr(sym("s")));
        assert_eq!(expr('s').to_egglog(), "(MVar \"s\")");
    }

    /// Ordering drives `dyn_dims[]` slot assignment, and the host uploads in
    /// that same order. Sorting by name keeps it a function of the graph — for
    /// single-char names, byte-identical to the char ordering this replaced.
    #[test]
    fn ordering_is_by_name_and_matches_the_old_char_order() {
        let mut got: Vec<Symbol> = ['s', 'a', 'Z', 'A', 'c'].map(Symbol::from).to_vec();
        got.sort();
        let mut want = ['s', 'a', 'Z', 'A', 'c'];
        want.sort();
        assert_eq!(
            got,
            want.iter().copied().map(Symbol::from).collect::<Vec<_>>()
        );

        // Independent of the order the names were first interned in.
        let _ = sym("zz9");
        let _ = sym("aa9");
        let mut a = [sym("zz9"), sym("aa9")];
        a.sort();
        assert_eq!(a, [sym("aa9"), sym("zz9")]);
    }

    /// `From<char>` used to hand back any ASCII byte straight from the name
    /// static ASCII table this branch later deleted, skipping validation
    /// entirely — so `expr('%')` produced
    /// `const_%`, not a C identifier, and `Symbol::from('"')` produced
    /// `(MVar """)`, not an egglog literal. The module claims the alphabet
    /// holds *by construction* so codegen need not re-check; this is what
    /// makes that true.
    #[test]
    fn char_conversion_is_validated_too() {
        for c in ['a', 'z', 'A', 'Z'] {
            assert_eq!(Symbol::from(c).to_string(), c.to_string());
        }
        for c in ['%', '"', '-', '7', ' ', '\\', '_'] {
            assert!(
                std::panic::catch_unwind(|| Symbol::from(c)).is_err(),
                "Symbol::from({c:?}) must not mint an unusable name"
            );
        }
    }

    /// The seam this whole change exists to fix, joined end to end.
    ///
    /// `simplify()` round-trips through egglog: `to_egglog` encodes the dim as
    /// `(MVar "s77")`, egglog runs, and extraction decodes it back. Extraction
    /// used to take `name.chars().next()`, so `s77` came back as `s` — a
    /// *different, possibly live* dim — and the wrong name then missed in
    /// `dyn_map`, where cuda_lite turns a miss into a dimension of 0. Wrong
    /// numerics, no diagnostic.
    ///
    /// The decoder is unit-tested in egglog_utils, but nothing ran a
    /// multi-char name through the real encode/egglog/decode path until this.
    #[test]
    fn multichar_dim_survives_a_simplify_round_trip() {
        let s = expr(sym("s77"));

        // More than one term, so this takes the egglog path rather than
        // Expression::simplify's single-term shortcut.
        let simplified = ((s * 2) / 2).simplify();

        assert_eq!(
            simplified.dyn_vars(),
            vec![sym("s77")],
            "name must survive egglog; truncation would yield s"
        );

        let mut env = DynMap::default();
        env.insert(sym("s77"), 12);
        assert_eq!(
            simplified.exec(&env),
            Some(12),
            "a truncated name misses in dyn_map and resolves to nothing"
        );
    }

    /// A dim is emitted as `const_<name>`, so the alphabet has to keep that a
    /// valid C identifier. Rejected, never sanitized: a sanitizer maps `a.b`
    /// and `a-b` both to `a_b`, silently collapsing two dims onto one #define.
    #[test]
    fn malformed_names_are_rejected_not_mangled() {
        for bad in ["s-77", "_x", "a__b", "7s", "a b", "a\"b", ""] {
            assert!(
                Symbol::try_new(bad).is_err(),
                "{bad:?} should not be a dim name"
            );
        }
        assert_eq!(kernel_const_name(&sym("s77")), "const_s77");
    }

    /// The prefix hazard that `to_kernel_with_index` exists to prevent, now
    /// that a dim named `zip` is actually spellable.
    #[test]
    fn index_substitution_does_not_clobber_a_z_prefixed_dim() {
        let e = expr('z') + expr(sym("zip"));
        let out = e.to_kernel_with_index("i");
        assert_eq!(out, "(i+const_zip)");
    }
}

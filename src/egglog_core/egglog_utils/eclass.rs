//! READING A SERIALIZED E-CLASS, one named constructor at a time.
//!
//! This module knows nothing about layouts, dtypes, or any particular
//! egglog sort. It is the generic surface over a serialized e-graph:
//!
//!  * [`EGraphView`] — a serialized e-graph plus the decoders that know
//!    its constructors;
//!  * [`EClass`] / [`ENode`] — a class and its e-nodes, with children
//!    resolved back to CLASSES (the serializer's contract: a node's
//!    `children[i]` is a node inside argument `i`'s class);
//!  * [`EgglogConstructor`] — one egglog constructor mirrored as a Rust
//!    struct that can decode ONE e-node of that constructor;
//!  * [`Sort`] / [`DynFacts`] — a decoded sort's name and the erased
//!    fact surface its constructors share;
//!  * [`ConstructorDecoder`] / [`ConstructorRegistry`] — the registered
//!    `(sort, constructor)` decoders, and the assembly TRIPWIRE that
//!    proves a program's every constructor of a decoded sort has exactly
//!    one.
//!
//! THERE IS NO PREFERENCE ORDER HERE, and there is no closed enum of
//! constructors. A caller that wants one spelling names it —
//! `first::<C>()`, `has::<C>()`, `require::<C>(who)` — and a caller that
//! wants "whatever facts this class discloses" asks the registry through
//! `spellings::<S>()`. Which spelling a call site prefers is that call
//! site's business; the reader never chooses for it.
//!
//! Ordering is decided in exactly one place, [`EClass::nodes`]:
//! UNSUBSUMED nodes first, then by `NodeId`. Subsumed nodes are kept — a
//! subsumed constructor is still a true member of its class, and
//! saturation can subsume every constructor spelling of a class.
//!
//! NOTE ON THE NAME `Sort`: egglog has its own `egglog::sort::Sort`
//! trait. This module never imports it (it reaches egglog's sorts only
//! through `ArcSort::name()`), so the collision is a reading hazard and
//! not a compile one.

use anyhow::{Result, anyhow, bail};
use egraph_serialize::{ClassId, EGraph, Node, NodeId};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

// =============================================================================
// The view, the class, the node
// =============================================================================

/// Read-only view of one serialized e-graph plus the decoders that know
/// its constructors. Two references; `Copy`.
#[derive(Clone, Copy)]
pub struct EGraphView<'g> {
    egraph: &'g EGraph,
    decoders: &'g ConstructorRegistry,
}

impl<'g> EGraphView<'g> {
    pub fn new(egraph: &'g EGraph, decoders: &'g ConstructorRegistry) -> Self {
        Self { egraph, decoders }
    }

    /// TOTAL: an id with no nodes yields a class whose [`EClass::nodes`]
    /// is empty (asking about an absent class is a question, not a bug).
    pub fn class(&self, id: &ClassId) -> EClass<'g> {
        EClass {
            view: *self,
            id: id.clone(),
        }
    }

    pub fn egraph(&self) -> &'g EGraph {
        self.egraph
    }

    pub fn decoders(&self) -> &'g ConstructorRegistry {
        self.decoders
    }
}

impl std::fmt::Debug for EGraphView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EGraphView")
            .field("nodes", &self.egraph.nodes.len())
            .field("decoders", &self.decoders.len())
            .finish()
    }
}

/// One e-class of the view. Cheap to clone (the id is an `Arc<str>`).
#[derive(Clone)]
pub struct EClass<'g> {
    view: EGraphView<'g>,
    id: ClassId,
}

impl std::fmt::Debug for EClass<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EClass")
            .field("id", &self.id)
            .field("sort", &self.sort_name())
            .field("ops", &self.ops())
            .finish()
    }
}

impl<'g> EClass<'g> {
    pub fn id(&self) -> &ClassId {
        &self.id
    }

    pub fn view(&self) -> EGraphView<'g> {
        self.view
    }

    /// `class_data[id].typ` — the egglog sort name the serializer
    /// stamped on this class.
    pub fn sort_name(&self) -> Option<&'g str> {
        self.view
            .egraph
            .class_data
            .get(&self.id)
            .and_then(|data| data.typ.as_deref())
    }

    /// Every e-node in the class: UNSUBSUMED FIRST, then by `NodeId`.
    /// THE ONLY PLACE ORDER IS DECIDED.
    pub fn nodes(&self) -> Vec<ENode<'g>> {
        let Some(class) = self.view.egraph.classes().get(&self.id) else {
            return Vec::new();
        };
        let mut ids: Vec<&'g NodeId> = class.nodes.iter().collect();
        ids.sort();
        let mut out: Vec<ENode<'g>> = Vec::with_capacity(ids.len());
        for want_subsumed in [false, true] {
            for id in &ids {
                let Some(node) = self.view.egraph.nodes.get(*id) else {
                    continue;
                };
                if node.subsumed == want_subsumed {
                    out.push(ENode {
                        view: self.view,
                        id,
                        node,
                    });
                }
            }
        }
        out
    }

    /// The class's nodes whose op is exactly `op`, in [`EClass::nodes`]
    /// order.
    pub fn nodes_named(&self, op: &str) -> impl Iterator<Item = ENode<'g>> + '_ {
        let op = op.to_string();
        self.nodes().into_iter().filter(move |n| n.op() == op)
    }

    /// Every distinct `op` in the class, in [`EClass::nodes`] order —
    /// diagnostics.
    pub fn ops(&self) -> Vec<String> {
        let mut seen: Vec<String> = Vec::new();
        for node in self.nodes() {
            if !seen.iter().any(|op| op == node.op()) {
                seen.push(node.op().to_string());
            }
        }
        seen
    }

    /// Any node in the class whose op parses as an i64 (the serializer
    /// renders base values with `{:?}`, so an i64 literal is a digit
    /// string).
    pub fn i64_literal(&self) -> Option<i64> {
        self.nodes()
            .into_iter()
            .find_map(|n| n.op().parse::<i64>().ok())
    }

    /// Any node in the class whose op is a quoted string literal.
    pub fn string_literal(&self) -> Option<String> {
        self.nodes().into_iter().find_map(|n| {
            n.op()
                .strip_prefix('"')?
                .strip_suffix('"')
                .map(str::to_string)
        })
    }

    // ---- typed, registry-free: the call site names the constructor ----

    /// Every node named `C::NAME` that DECODES, in [`EClass::nodes`]
    /// order.
    pub fn all<C: EgglogConstructor>(&self) -> Vec<C> {
        self.nodes_named(C::NAME)
            .filter_map(|node| C::decode(&node).ok())
            .collect()
    }

    /// The first node named `C::NAME` that decodes.
    pub fn first<C: EgglogConstructor>(&self) -> Option<C> {
        self.nodes_named(C::NAME)
            .find_map(|node| C::decode(&node).ok())
    }

    /// A DECODABLE `C` spelling is present in the class.
    pub fn has<C: EgglogConstructor>(&self) -> bool {
        self.first::<C>().is_some()
    }

    /// [`EClass::first`] or a refusal naming who asked, the class, its
    /// sort, `C::NAME`, and what the class does hold.
    pub fn require<C: EgglogConstructor>(&self, who: &str) -> Result<C> {
        self.first::<C>().ok_or_else(|| {
            anyhow!(
                "{who}: e-class {} (sort {}) holds no decodable `{}` spelling — \
                 registered {} constructors present: {:?}; every op in the class: {:?}",
                self.id,
                self.sort_name().unwrap_or("?"),
                C::NAME,
                <C::Sort as Sort>::NAME,
                self.present::<C::Sort>(),
                self.ops()
            )
        })
    }

    /// [`EClass::require`] as a panic — ONLY after validation.
    pub fn expect<C: EgglogConstructor>(&self, why: &str) -> C {
        match self.require::<C>(why) {
            Ok(found) => found,
            Err(err) => panic!("{err:#}"),
        }
    }

    // ---- erased, registry-driven: generic code over a sort ----

    /// Registered constructor names of sort `S` that APPEAR in the class
    /// (at least one node, decodable or not), in REGISTRY order.
    pub fn present<S: Sort>(&self) -> Vec<&'static str> {
        let ops = self.ops();
        self.view
            .decoders
            .constructors_of(S::NAME)
            .filter(|name| ops.iter().any(|op| op == name))
            .collect()
    }

    /// Every registered constructor of `S` present in the class,
    /// decoded — registry order, then node order.
    pub fn spellings<S: Sort>(&self) -> Spellings<S> {
        let nodes = self.nodes();
        let mut decoded: Vec<Arc<S::Facts>> = Vec::new();
        let mut present: Vec<&'static str> = Vec::new();
        let mut failed: Vec<(&'static str, String)> = Vec::new();
        for name in self.view.decoders.constructors_of(S::NAME) {
            let decoder = self
                .view
                .decoders
                .get(S::NAME, name)
                .expect("constructors_of only names registered decoders");
            let mut seen = false;
            for node in nodes.iter().filter(|n| n.op() == name) {
                seen = true;
                match (decoder.decode)(node) {
                    Ok(erased) => match erased.downcast::<Arc<S::Facts>>() {
                        Ok(facts) => decoded.push(*facts),
                        Err(_) => panic!(
                            "decoder ({}, {name}) produced a value that is not an \
                             Arc of the sort's fact type — a registry bug",
                            S::NAME
                        ),
                    },
                    Err(err) => failed.push((name, format!("{err:#}"))),
                }
            }
            if seen {
                present.push(name);
            }
        }
        Spellings {
            decoded,
            present,
            failed,
        }
    }

    /// [`EClass::spellings`] items, for the "I just need a fact" shape.
    pub fn facts<S: Sort>(&self) -> std::vec::IntoIter<Arc<S::Facts>> {
        self.spellings::<S>().decoded.into_iter()
    }

    /// Ops in the class with NO decoder registered under sort `S`. A
    /// decoded sort's class legitimately holds such rows (`Custom`
    /// function rows whose output sort is `S`, the serializer's `[...]`
    /// truncation placeholder), so this is DIAGNOSTICS ONLY — the
    /// tripwire, not this list, is what proves the constructors are
    /// covered.
    pub fn unknown<S: Sort>(&self) -> Vec<String> {
        self.ops()
            .into_iter()
            .filter(|op| self.view.decoders.get(S::NAME, op).is_none())
            .collect()
    }
}

/// One e-node: its op name plus its children as CLASSES.
#[derive(Clone, Copy)]
pub struct ENode<'g> {
    view: EGraphView<'g>,
    id: &'g NodeId,
    node: &'g Node,
}

impl std::fmt::Debug for ENode<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ENode")
            .field("id", self.id)
            .field("op", &self.node.op)
            .field("arity", &self.node.children.len())
            .finish()
    }
}

impl<'g> ENode<'g> {
    pub fn id(&self) -> &'g NodeId {
        self.id
    }

    pub fn op(&self) -> &'g str {
        &self.node.op
    }

    /// The class this node belongs to.
    pub fn class(&self) -> EClass<'g> {
        self.view.class(&self.node.eclass)
    }

    pub fn is_subsumed(&self) -> bool {
        self.node.subsumed
    }

    pub fn arity(&self) -> usize {
        self.node.children.len()
    }

    /// `children[i]` resolved to that child node's e-class — the
    /// serializer's contract (an argument is rendered as SOME node
    /// inside the argument's class).
    pub fn child(&self, index: usize) -> Option<EClass<'g>> {
        let child = self.node.children.get(index)?;
        let class = &self.view.egraph.nodes.get(child)?.eclass;
        Some(self.view.class(class))
    }

    /// [`ENode::child`] or a refusal naming the constructor and index —
    /// an arity the decoder assumed and the program does not have.
    pub fn child_or_bail(&self, index: usize) -> Result<EClass<'g>> {
        self.child(index).ok_or_else(|| {
            anyhow!(
                "`{}` node {} has no child {index} (arity {})",
                self.op(),
                self.id,
                self.arity()
            )
        })
    }

    pub fn children(&self) -> Vec<EClass<'g>> {
        (0..self.arity()).filter_map(|i| self.child(i)).collect()
    }
}

// =============================================================================
// Sorts, constructors, erased facts
// =============================================================================

/// A DECODED SORT: its egglog name and the erased fact surface its
/// constructors share (e.g. `Layout: Sort<Facts = dyn LayoutFacts>`).
pub trait Sort: 'static {
    const NAME: &'static str;
    type Facts: ?Sized + DynFacts;
}

/// The sort-agnostic erased surface every decoded constructor exposes.
/// Blanket-implemented for every [`EgglogConstructor`] that is
/// `PartialEq + Debug + Send + Sync`, so constructor structs write NO
/// boilerplate for it.
pub trait DynFacts: Any + std::fmt::Debug + Send + Sync {
    /// The egglog constructor this value was decoded from — a NAME, for
    /// diagnostics. Nothing branches on it to change behaviour.
    fn constructor(&self) -> &'static str;
    /// The downcast door (`as_any().downcast_ref::<C>()`), spelled
    /// explicitly because trait upcasting needs Rust 1.86 and the
    /// workspace floor is 1.85.
    fn as_any(&self) -> &dyn Any;
    /// Structural equality against another erased value.
    fn dyn_eq(&self, other: &dyn Any) -> bool;
}

impl<T: EgglogConstructor + PartialEq + std::fmt::Debug + Send + Sync> DynFacts for T {
    fn constructor(&self) -> &'static str {
        T::NAME
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn dyn_eq(&self, other: &dyn Any) -> bool {
        other.downcast_ref::<T>() == Some(self)
    }
}

/// ONE EGGLOG CONSTRUCTOR, mirrored as a Rust struct.
pub trait EgglogConstructor: Sized + 'static {
    /// The full egglog constructor name, e.g.
    /// `"RightMajorContiguousElementLayoutLit"`.
    const NAME: &'static str;
    /// The sort this constructor produces.
    type Sort: Sort;

    /// Decode ONE e-node of this constructor. `node.op() == Self::NAME`
    /// is guaranteed by the callers; children are read by index.
    /// Failure means THIS SPELLING does not parse (a foreign-shape
    /// coordinate under the owner-shape guard, a cons spine with no
    /// parsing spelling, a cycle) — the caller moves on to the next
    /// node, exactly as the spelling walk always has.
    fn decode(node: &ENode<'_>) -> Result<Self>;

    /// Erase into the sort's shared fact object. One line per struct
    /// (`Arc::new(self)`): the unsizing coercion is only expressible
    /// where the concrete type and the target `dyn` type are both known.
    fn erase(self) -> Arc<<Self::Sort as Sort>::Facts>;
}

// =============================================================================
// The registry and THE TRIPWIRE
// =============================================================================

/// A registered `(sort, constructor)` decoder. `decode` yields a
/// `Box<dyn Any>` that HOLDS an `Arc<<S as Sort>::Facts>` — the
/// convention [`Spellings`] downcasts back through.
pub struct ConstructorDecoder {
    pub sort: &'static str,
    pub name: &'static str,
    pub decode: fn(&ENode<'_>) -> Result<Box<dyn Any>>,
}

impl ConstructorDecoder {
    pub fn of<C: EgglogConstructor>() -> Self {
        Self {
            sort: <C::Sort as Sort>::NAME,
            name: C::NAME,
            decode: |node| C::decode(node).map(|c| Box::new(c.erase()) as Box<dyn Any>),
        }
    }
}

impl std::fmt::Debug for ConstructorDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstructorDecoder")
            .field("sort", &self.sort)
            .field("name", &self.name)
            .finish()
    }
}

/// Decoders keyed by `(sort, constructor)`, in INSERTION ORDER (core
/// built-ins first, then matcher contributions in registry order). That
/// order is the only order [`Spellings`] and [`EClass::present`] speak
/// in — it is a listing order, never a preference.
#[derive(Debug, Default)]
pub struct ConstructorRegistry {
    entries: Vec<ConstructorDecoder>,
    index: HashMap<(&'static str, &'static str), usize>,
}

impl ConstructorRegistry {
    /// Build a registry. A duplicate `(sort, name)` is REFUSED: "exactly
    /// one decoder" is the contract, and a duplicate is a registration
    /// bug, never a preference to resolve.
    pub fn new(decoders: impl IntoIterator<Item = ConstructorDecoder>) -> Result<Self> {
        let mut registry = Self::default();
        for decoder in decoders {
            let key = (decoder.sort, decoder.name);
            if registry.index.contains_key(&key) {
                bail!(
                    "sort {}: two decoders registered for `{}` — every constructor \
                     of a decoded sort has exactly one decoder, registered beside \
                     the snippet that declares it",
                    decoder.sort,
                    decoder.name
                );
            }
            registry.index.insert(key, registry.entries.len());
            registry.entries.push(decoder);
        }
        Ok(registry)
    }

    pub fn get(&self, sort: &str, name: &str) -> Option<&ConstructorDecoder> {
        self.entries
            .iter()
            .find(|d| d.sort == sort && d.name == name)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Every DECODED sort name, in registry order, once each.
    pub fn sorts(&self) -> impl Iterator<Item = &'static str> + '_ {
        let mut seen: Vec<&'static str> = Vec::new();
        for entry in &self.entries {
            if !seen.contains(&entry.sort) {
                seen.push(entry.sort);
            }
        }
        seen.into_iter()
    }

    /// The registered constructors of one sort, in registry order.
    pub fn constructors_of<'a>(&'a self, sort: &'a str) -> impl Iterator<Item = &'static str> + 'a {
        self.entries
            .iter()
            .filter(move |d| d.sort == sort)
            .map(|d| d.name)
    }

    /// THE ASSEMBLY TRIPWIRE. Run against the LIVE e-graph right after
    /// the assembled program parses and before it is serialized: every
    /// `(constructor ...)` (and `(datatype ...)` variant) whose OUTPUT
    /// sort is a sort this registry decodes must have a decoder, and
    /// every decoder must name a constructor the program declares under
    /// that sort.
    ///
    /// `Custom` function rows are SKIPPED — a `(function ... :no-merge)`
    /// row such as `layout-of` has a `Layout` output but is not a
    /// spelling of one.
    pub fn check(&self, egraph: &egglog::EGraph) -> Result<()> {
        use egglog::ast::FunctionSubtype;
        let mut problems: Vec<String> = Vec::new();
        // (1) every declared constructor of a decoded sort has a decoder.
        for (name, func) in egraph.functions_iter() {
            let func_type = func.func_type();
            if func_type.subtype != FunctionSubtype::Constructor {
                continue;
            }
            let sort = func_type.output.name();
            if !self.sorts().any(|registered| registered == sort) {
                continue;
            }
            if self.get(sort, name).is_none() {
                problems.push(format!(
                    "sort {sort}: constructor `{name}` is declared by the program but \
                     has no registered decoder"
                ));
            }
        }
        // (2) every decoder names a constructor the program declares,
        //     under the sort it was registered for.
        for decoder in &self.entries {
            match egraph.get_function(decoder.name) {
                None => problems.push(format!(
                    "sort {}: decoder for `{}` names a constructor the program does \
                     not declare",
                    decoder.sort, decoder.name
                )),
                Some(func) => {
                    let func_type = func.func_type();
                    if func_type.subtype != FunctionSubtype::Constructor
                        || func_type.output.name() != decoder.sort
                    {
                        problems.push(format!(
                            "sort {}: decoder for `{}` is registered under sort {} but \
                             the program declares it as {} with output sort {}",
                            decoder.sort,
                            decoder.name,
                            decoder.sort,
                            func_type.subtype.label(),
                            func_type.output.name()
                        ));
                    }
                }
            }
        }
        if problems.is_empty() {
            return Ok(());
        }
        bail!(
            "egglog constructor decoders are incomplete for the assembled program:\n  - {}\n\
             Every constructor of a decoded sort needs exactly one decoder, registered \
             beside the snippet that declares it (OpMatcher::decoders); see \
             docs/design/eclass-decoder.md.",
            problems.join("\n  - ")
        )
    }
}

// =============================================================================
// The decoded spelling set of one class
// =============================================================================

/// The decoded constructor instances of sort `S` found in ONE class —
/// every registered spelling the class holds, never a chosen one.
pub struct Spellings<S: Sort> {
    decoded: Vec<Arc<S::Facts>>,
    present: Vec<&'static str>,
    failed: Vec<(&'static str, String)>,
}

impl<S: Sort> Spellings<S> {
    /// A hand-built spelling set (test fixtures and plan literals): the
    /// listed facts, `present` derived from what they say they are.
    pub fn from_decoded(decoded: Vec<Arc<S::Facts>>) -> Self {
        let mut present: Vec<&'static str> = Vec::new();
        for item in &decoded {
            let name = item.constructor();
            if !present.contains(&name) {
                present.push(name);
            }
        }
        Self {
            decoded,
            present,
            failed: Vec::new(),
        }
    }

    pub fn first<C: EgglogConstructor<Sort = S>>(&self) -> Option<&C> {
        self.decoded
            .iter()
            .find_map(|item| item.as_any().downcast_ref::<C>())
    }

    pub fn all<C: EgglogConstructor<Sort = S>>(&self) -> Vec<&C> {
        self.decoded
            .iter()
            .filter_map(|item| item.as_any().downcast_ref::<C>())
            .collect()
    }

    pub fn has<C: EgglogConstructor<Sort = S>>(&self) -> bool {
        self.first::<C>().is_some()
    }

    pub fn require<C: EgglogConstructor<Sort = S>>(&self, who: &str) -> Result<&C> {
        self.first::<C>().ok_or_else(|| {
            anyhow!(
                "{who}: no decodable `{}` spelling — the {} spellings present are {:?}",
                C::NAME,
                S::NAME,
                self.present
            )
        })
    }

    /// The constructor NAMES with at least one node in the class, in
    /// registry order.
    pub fn present(&self) -> &[&'static str] {
        &self.present
    }

    /// `(constructor, reason)` for every node that named a registered
    /// constructor and did not parse.
    pub fn failed(&self) -> &[(&'static str, String)] {
        &self.failed
    }

    pub fn iter(&self) -> impl Iterator<Item = &S::Facts> {
        self.decoded.iter().map(|item| &**item)
    }

    /// The first decoded spelling in registry order — for the
    /// class-INVARIANT facts (the domain, the width), where every
    /// spelling of a class must agree.
    pub fn any(&self) -> Option<&S::Facts> {
        self.decoded.first().map(|item| &**item)
    }

    pub fn len(&self) -> usize {
        self.decoded.len()
    }

    pub fn is_empty(&self) -> bool {
        self.decoded.is_empty()
    }
}

impl<S: Sort> Clone for Spellings<S> {
    fn clone(&self) -> Self {
        Self {
            decoded: self.decoded.clone(),
            present: self.present.clone(),
            failed: self.failed.clone(),
        }
    }
}

impl<S: Sort> PartialEq for Spellings<S> {
    fn eq(&self, other: &Self) -> bool {
        self.decoded.len() == other.decoded.len()
            && self
                .decoded
                .iter()
                .zip(&other.decoded)
                .all(|(a, b)| a.dyn_eq(b.as_any()))
    }
}

impl<S: Sort> Eq for Spellings<S> {}

impl<S: Sort> std::fmt::Debug for Spellings<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Spellings")
            .field("sort", &S::NAME)
            .field("present", &self.present)
            .field("decoded", &self.decoded.iter().collect::<Vec<_>>())
            .finish()
    }
}

// =============================================================================
// Tests — over a HAND-BUILT serialized e-graph, with a throwaway sort.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // A throwaway sort and two constructors of it, so the generic
    // surface is tested with no layout vocabulary in sight.
    #[derive(Debug)]
    pub struct Pair;
    impl Sort for Pair {
        const NAME: &'static str = "Pair";
        type Facts = dyn PairFacts;
    }

    pub trait PairFacts: DynFacts {
        fn left(&self) -> i64;
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Point {
        x: i64,
        y: i64,
    }
    impl EgglogConstructor for Point {
        const NAME: &'static str = "PointLit";
        type Sort = Pair;
        fn decode(node: &ENode<'_>) -> Result<Self> {
            let x = node
                .child_or_bail(0)?
                .i64_literal()
                .ok_or_else(|| anyhow!("child 0 is not an i64 literal"))?;
            let y = node
                .child_or_bail(1)?
                .i64_literal()
                .ok_or_else(|| anyhow!("child 1 is not an i64 literal"))?;
            Ok(Self { x, y })
        }
        fn erase(self) -> Arc<dyn PairFacts> {
            Arc::new(self)
        }
    }
    impl PairFacts for Point {
        fn left(&self) -> i64 {
            self.x
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    struct Named {
        name: String,
    }
    impl EgglogConstructor for Named {
        const NAME: &'static str = "NamedLit";
        type Sort = Pair;
        fn decode(node: &ENode<'_>) -> Result<Self> {
            let name = node
                .child_or_bail(0)?
                .string_literal()
                .ok_or_else(|| anyhow!("child 0 is not a string literal"))?;
            Ok(Self { name })
        }
        fn erase(self) -> Arc<dyn PairFacts> {
            Arc::new(self)
        }
    }
    impl PairFacts for Named {
        fn left(&self) -> i64 {
            -1
        }
    }

    fn node(op: &str, children: &[&str], eclass: &str, subsumed: bool) -> Node {
        Node {
            op: op.to_string(),
            children: children.iter().map(|c| NodeId::from(*c)).collect(),
            eclass: ClassId::from(eclass),
            cost: egraph_serialize::Cost::new(1.0).unwrap(),
            subsumed,
        }
    }

    /// One class holding BOTH constructors plus their literal children.
    fn two_spelling_graph() -> EGraph {
        let mut egraph = EGraph::default();
        egraph.add_node("n-x", node("2", &[], "c-x", false));
        egraph.add_node("n-y", node("3", &[], "c-y", false));
        egraph.add_node("n-s", node("\"hello\"", &[], "c-s", false));
        egraph.add_node(
            "n-point",
            node("PointLit", &["n-x", "n-y"], "c-pair", false),
        );
        egraph.add_node("n-named", node("NamedLit", &["n-s"], "c-pair", false));
        egraph
    }

    fn registry() -> ConstructorRegistry {
        ConstructorRegistry::new([
            ConstructorDecoder::of::<Point>(),
            ConstructorDecoder::of::<Named>(),
        ])
        .expect("no duplicates")
    }

    #[test]
    fn a_class_answers_for_each_constructor_it_is_asked_about() {
        let egraph = two_spelling_graph();
        let decoders = registry();
        let view = EGraphView::new(&egraph, &decoders);
        let class = view.class(&ClassId::from("c-pair"));
        assert_eq!(class.first::<Point>(), Some(Point { x: 2, y: 3 }));
        assert_eq!(
            class.first::<Named>(),
            Some(Named {
                name: "hello".to_string()
            })
        );
        assert!(class.has::<Point>() && class.has::<Named>());
        assert_eq!(
            class.present::<Pair>(),
            vec!["PointLit", "NamedLit"],
            "present() lists in REGISTRY order, not node order"
        );
        // The erased path finds the same two, and the fact surface works.
        let spellings = class.spellings::<Pair>();
        assert_eq!(spellings.len(), 2);
        assert_eq!(spellings.first::<Point>().map(|p| p.x), Some(2));
        assert_eq!(
            spellings.iter().map(|f| f.left()).collect::<Vec<_>>(),
            vec![2, -1]
        );
        assert!(spellings.failed().is_empty());
    }

    /// An absent class is a question, not a bug; and a class that does
    /// not hold the asked-for constructor refuses by NAME.
    #[test]
    fn absent_classes_and_absent_constructors() {
        let egraph = two_spelling_graph();
        let decoders = registry();
        let view = EGraphView::new(&egraph, &decoders);
        let missing = view.class(&ClassId::from("c-nope"));
        assert!(missing.nodes().is_empty());
        assert!(missing.first::<Point>().is_none());
        let x_class = view.class(&ClassId::from("c-x"));
        let err = format!("{:#}", x_class.require::<Point>("a caller").unwrap_err());
        assert!(err.contains("a caller"), "{err}");
        assert!(err.contains("PointLit"), "{err}");
        assert!(err.contains("Pair"), "{err}");
    }

    /// A node that does not parse is SKIPPED, recorded in `failed`, and
    /// never fatal on its own.
    #[test]
    fn an_undecodable_spelling_is_recorded_not_fatal() {
        let mut egraph = two_spelling_graph();
        // A second PointLit whose second child is a string: it names the
        // constructor and does not parse.
        egraph.add_node("n-bad", node("PointLit", &["n-x", "n-s"], "c-pair2", false));
        egraph.add_node("n-good", node("NamedLit", &["n-s"], "c-pair2", false));
        let decoders = registry();
        let view = EGraphView::new(&egraph, &decoders);
        let class = view.class(&ClassId::from("c-pair2"));
        assert!(class.first::<Point>().is_none());
        assert_eq!(
            class.present::<Pair>(),
            vec!["PointLit", "NamedLit"],
            "present() names it — the node is there; `has` says it does not decode"
        );
        let spellings = class.spellings::<Pair>();
        assert_eq!(spellings.len(), 1);
        assert_eq!(spellings.failed().len(), 1);
        assert_eq!(spellings.failed()[0].0, "PointLit");
    }

    /// UNSUBSUMED FIRST, then by `NodeId` — the only ordering rule.
    #[test]
    fn nodes_are_unsubsumed_first_then_by_id() {
        let mut egraph = EGraph::default();
        egraph.add_node("n-a", node("2", &[], "c-lit", false));
        egraph.add_node("n-b", node("5", &[], "c-lit", false));
        // Insertion order deliberately puts the subsumed one first and
        // the later id first.
        egraph.add_node("z-sub", node("SubsumedOp", &[], "c-many", true));
        egraph.add_node("m-two", node("SecondOp", &[], "c-many", false));
        egraph.add_node("b-one", node("FirstOp", &[], "c-many", false));
        let decoders = ConstructorRegistry::default();
        let view = EGraphView::new(&egraph, &decoders);
        let class = view.class(&ClassId::from("c-many"));
        assert_eq!(class.ops(), vec!["FirstOp", "SecondOp", "SubsumedOp"]);
        assert!(class.nodes()[2].is_subsumed());
    }

    #[test]
    fn a_duplicate_registration_is_refused() {
        let err = ConstructorRegistry::new([
            ConstructorDecoder::of::<Point>(),
            ConstructorDecoder::of::<Point>(),
        ])
        .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("two decoders registered for `PointLit`"),
            "{msg}"
        );
    }

    /// Structural equality of a spelling SET, and the literal readers.
    #[test]
    fn spelling_sets_compare_structurally_and_literals_read() {
        let egraph = two_spelling_graph();
        let decoders = registry();
        let view = EGraphView::new(&egraph, &decoders);
        let class = view.class(&ClassId::from("c-pair"));
        assert_eq!(class.spellings::<Pair>(), class.spellings::<Pair>());
        let hand: Spellings<Pair> = Spellings::from_decoded(vec![Point { x: 2, y: 3 }.erase()]);
        assert_ne!(hand, class.spellings::<Pair>());
        assert_eq!(hand.present(), ["PointLit"]);
        assert_eq!(view.class(&ClassId::from("c-x")).i64_literal(), Some(2));
        assert_eq!(
            view.class(&ClassId::from("c-s"))
                .string_literal()
                .as_deref(),
            Some("hello")
        );
    }
}

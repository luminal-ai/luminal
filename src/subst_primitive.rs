//! VENDORED from egglog-experimental PR #60 (branch `unstable-subst`,
//! head eaf27ca, author oflatt) — verbatim below this header. Vendored
//! rather than depended-on while the PR is unmerged (ruling: take
//! exactly the audited contract; drop this file for the upstream crate
//! once it merges). Registered by `new_egraph()` in egglog_snippet.rs.
//!
//! Substitution over a reachable sub-e-graph: the `unstable-subst` primitive.
//!
//! `(unstable-subst root map)` takes an e-class `root` of any eq-sort and a
//! `Map` whose key and value sorts are the same eq-sort. It walks the
//! constructor rows reachable from `root`, copies the part of that sub-e-graph
//! the substitution actually touches while replacing each key e-class with its
//! mapped value, and returns the e-class of the copied root.
//!
//! The walk follows constructor rows only — `function` rows are analyses over
//! the term structure, not part of it — and reaches through container-valued
//! children, rebuilding them with substituted contents.
//!
//! Unaffected e-classes are shared with the original rather than copied, so
//! substituting an empty map returns `root` itself and writes nothing.
//!
//! # Warning: `root` must already exist
//!
//! Pass a root the rule's query bound, or one from an earlier command — not a
//! term the enclosing action just built.
//!
//! The walk reads committed table contents, and an action's writes stay staged
//! until it finishes. A root this action built has no rows yet, so the walk
//! finds no e-nodes under it, nothing is affected, and it comes back
//! **unchanged and without an error**. `(unstable-subst (Mul x y) m)`, building
//! its own argument, silently does nothing. The same applies to any term under
//! the root that the action just built: it is not there to be substituted.
//!
//! Replacements are not affected — a map's values are spliced into the copy
//! without being walked, so those can be built in the same action.
//!
//! # Other properties worth knowing
//!
//! - The region's equations are substituted along with its terms. Copying an
//!   e-class copies every one of its e-nodes, so `t1 = t2` in the original
//!   becomes `σ(t1) = σ(t2)` in the copy — and an e-node with no substituted
//!   children copies to itself, merging the copy back into the original class.
//!   That is correct for equations that hold for every value of the substituted
//!   classes (anything a rewrite rule derived) and wrong for a ground `union`
//!   pinning one of them down, so only substitute classes that behave like
//!   universally quantified variables.
//! - The snapshot comes from live table contents, so this is a `Context::Full`
//!   primitive: top-level actions and `:naive` rule heads only.
//!
//! Copies are named by `lookup_or_insert`, the same way `(Add a b)` in an
//! action is, so no e-class id is ever invented here. A cyclic e-class can
//! therefore only be copied if it has an e-node whose children all lie outside
//! the cycle to name it first — `x = {Var "x", Add x (Num 0)}` does, and works.
//! A cycle with no such e-node is an error rather than a silent partial copy.

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use egglog::api::RawValues;
use egglog::ast::Span;
use egglog::constraint::{self, Constraint, ImpossibleConstraint, TypeConstraint};
use egglog::sort::MapContainer;
use egglog::{
    ArcSort, Atom, AtomTerm, Core, EGraph, Error, FullPrim, FullState, FuncType, Primitive, Read,
    TypeInfo, Value, Write,
};

/// The name of the primitive, as written in an egglog program.
pub const SUBST: &str = "unstable-subst";

/// A constructor the walk can follow.
type Constructor<'a> = &'a FuncType;

/// What a column of a given sort holds, as far as the walk cares.
enum Kind {
    /// An e-class: walk into it, and copy it if it is affected.
    Eclass,
    /// A container with e-classes somewhere inside: walk into its contents and
    /// rebuild it if any of them change. Carries the Rust [`TypeId`] its values
    /// are interned under.
    Container(TypeId),
    /// Nothing an e-class can hide in: copied through untouched.
    Opaque,
}

fn kind_of(sort: &ArcSort) -> Kind {
    if sort.is_eq_sort() {
        Kind::Eclass
    } else if sort.is_container_sort() {
        // Deliberately not `is_eq_container_sort`, which answers from the
        // sort's declared element sorts. For an `unstable-fn` value those are
        // the arguments still to be applied, not the ones it captured — a
        // `(UnstableFn () Math)` holding an e-class reports no eq-sort
        // elements. Walking every container and asking the value what it holds
        // costs an `inner_values` call on containers that turn out to have no
        // e-classes inside, and cannot miss one.
        Kind::Container(
            sort.value_type()
                .expect("a container sort has a value type"),
        )
    } else {
        Kind::Opaque
    }
}

/// One reachable constructor row, keyed by the constructor it came from so the
/// build pass can re-apply it.
struct ENode {
    ctor: usize,
    children: Vec<Value>,
    /// Copied rows of subsumed sources are subsumed in turn (vendoring
    /// delta): our identity folds retire a class's grounding spellings
    /// by :subsume, so skipping them entirely (upstream behavior)
    /// reports spurious ungrounded cycles. Subsumption is a matching
    /// fence, not semantic non-membership — the copy keeps the fence.
    subsumed: bool,
}

/// The reachable sub-e-graph.
#[derive(Default)]
struct Snapshot {
    /// E-nodes of each reachable e-class, including subsumed rows — their
    /// copies are subsumed in turn, so nothing is resurrected live.
    nodes: HashMap<Value, Vec<ENode>>,
    /// Contents of each reachable container value, with the [`TypeId`] to
    /// rebuild it under.
    containers: HashMap<Value, (TypeId, Vec<(ArcSort, Value)>)>,
    /// The e-classes each e-class references, flattened through containers.
    /// Drives both the affected fixpoint and the build order.
    deps: HashMap<Value, Vec<Value>>,
}

struct Walk<'a> {
    ctors: Vec<Constructor<'a>>,
    /// Indices into `ctors`, by output sort name.
    by_output: HashMap<String, Vec<usize>>,
    map: &'a BTreeMap<Value, Value>,
    snapshot: Snapshot,
    /// E-classes the substitution changes: a key, or a reference to an
    /// affected e-class.
    affected: HashSet<Value>,
    /// The copy of each affected e-class that has been named so far.
    images: HashMap<Value, Value>,
    container_images: HashMap<Value, Value>,
    /// Memo for [`Walk::container_leaves`].
    container_leaves: HashMap<Value, Vec<(ArcSort, Value)>>,
}

/// The constructors with an eq-sort output: the rows that make up the term
/// structure. Resolved per call, since an e-graph gains constructors over time.
fn constructors<'db>(
    state: &FullState<'_, 'db>,
    skip: &std::collections::HashSet<String>,
) -> Vec<Constructor<'db>> {
    let names: Vec<String> = state
        .table_sizes()
        .into_iter()
        .map(|(name, _)| name.to_owned())
        .filter(|name| !skip.contains(name))
        .collect();
    names
        .into_iter()
        .filter_map(|name| {
            // `constructor_schema` rejects the function tables, which is also
            // what keeps globals out: they lower to function tables.
            let func_type = state.constructor_schema(&name).ok()?;
            func_type.output.is_eq_sort().then_some(func_type)
        })
        .collect()
}

/// Substitute `map` through the sub-e-graph reachable from `root`, returning
/// the root of the copy. See the module docs for the semantics.
///
/// `root` must be an e-class that already has rows: one the query bound, or
/// one from an earlier command. A root the enclosing action just built is not
/// in the tables yet and comes back unchanged, with no error.
pub fn substitute<'db>(
    state: &mut FullState<'_, 'db>,
    root: Value,
    map: &BTreeMap<Value, Value>,
    skip: &std::collections::HashSet<String>,
) -> Result<Value, Error> {
    if let Some(target) = map.get(&root) {
        return Ok(*target);
    }
    if map.is_empty() {
        return Ok(root);
    }

    let ctors = constructors(state, skip);
    let mut by_output: HashMap<String, Vec<usize>> = HashMap::new();
    for (index, ctor) in ctors.iter().enumerate() {
        by_output
            .entry(ctor.output.name().to_owned())
            .or_default()
            .push(index);
    }

    let mut walk = Walk {
        ctors,
        by_output,
        map,
        snapshot: Snapshot::default(),
        affected: HashSet::new(),
        images: HashMap::new(),
        container_images: HashMap::new(),
        container_leaves: HashMap::new(),
    };
    walk.collect(state, root)?;
    walk.mark();
    if !walk.affected.contains(&root) {
        return Ok(root);
    }
    walk.build(state, root)
}

/// A pending e-class visit in the collect pass. The sort is unknown for the
/// root only.
struct Visit(Value, Option<ArcSort>);

impl Walk<'_> {
    /// Gather the reachable e-nodes, container contents, and e-class
    /// dependency edges, starting from `root`.
    fn collect(&mut self, state: &FullState<'_, '_>, root: Value) -> Result<(), Error> {
        let mut stack = vec![Visit(root, None)];
        while let Some(Visit(value, sort)) = stack.pop() {
            if self.snapshot.nodes.contains_key(&value) {
                continue;
            }
            // E-class ids all come from one counter, so probing a constructor
            // of the wrong sort just finds nothing. That is what makes the
            // root's unknown sort affordable: it costs one probe per eq-sorted
            // constructor, once.
            let candidates: Vec<usize> = match &sort {
                Some(sort) => self.by_output.get(sort.name()).cloned().unwrap_or_default(),
                None => (0..self.ctors.len()).collect(),
            };
            let mut nodes = Vec::new();
            for index in candidates {
                let mut rows = Vec::new();
                state.enodes_for_eclass(&self.ctors[index].name, value, |enode| {
                    rows.push((enode.children.to_vec(), enode.subsumed));
                })?;
                nodes.extend(rows.into_iter().map(|(children, subsumed)| ENode {
                    ctor: index,
                    children,
                    subsumed,
                }));
            }

            let mut deps = Vec::new();
            for node in &nodes {
                let ctor = self.ctors[node.ctor];
                for (child, child_sort) in node.children.iter().zip(&ctor.input) {
                    match kind_of(child_sort) {
                        Kind::Eclass => {
                            deps.push(*child);
                            stack.push(Visit(*child, Some(child_sort.clone())));
                        }
                        Kind::Container(type_id) => {
                            for (leaf_sort, leaf) in
                                self.container_leaves(state, *child, type_id, child_sort)
                            {
                                deps.push(leaf);
                                stack.push(Visit(leaf, Some(leaf_sort)));
                            }
                        }
                        Kind::Opaque => {}
                    }
                }
            }
            // Recorded after the children are read, but before they are
            // visited, so a cycle back to `value` terminates.
            self.snapshot.nodes.insert(value, nodes);
            self.snapshot.deps.insert(value, deps);
        }
        Ok(())
    }

    /// The e-class leaves of a container value, flattened through nested
    /// containers, recording its contents on the way. Recursion is bounded by
    /// how deeply the program's sorts nest containers.
    fn container_leaves(
        &mut self,
        state: &FullState<'_, '_>,
        value: Value,
        type_id: TypeId,
        sort: &ArcSort,
    ) -> Vec<(ArcSort, Value)> {
        if let Some(leaves) = self.container_leaves.get(&value) {
            return leaves.clone();
        }
        let contents = sort.inner_values(state.container_values(), value);
        let mut leaves = Vec::new();
        for (inner_sort, inner) in &contents {
            match kind_of(inner_sort) {
                Kind::Eclass => leaves.push((inner_sort.clone(), *inner)),
                Kind::Container(inner_type_id) => {
                    leaves.extend(self.container_leaves(state, *inner, inner_type_id, inner_sort))
                }
                Kind::Opaque => {}
            }
        }
        self.snapshot.containers.insert(value, (type_id, contents));
        self.container_leaves.insert(value, leaves.clone());
        leaves
    }

    /// Least fixpoint of "references something the substitution changes",
    /// seeded with the map keys the walk actually reached.
    fn mark(&mut self) {
        let mut users: HashMap<Value, Vec<Value>> = HashMap::new();
        for (owner, deps) in &self.snapshot.deps {
            for dep in deps {
                users.entry(*dep).or_default().push(*owner);
            }
        }

        let mut queue: VecDeque<Value> = self
            .map
            .keys()
            .copied()
            .filter(|key| self.snapshot.nodes.contains_key(key))
            .collect();
        self.affected.extend(queue.iter().copied());
        while let Some(value) = queue.pop_front() {
            for user in users.get(&value).into_iter().flatten() {
                if self.affected.insert(*user) {
                    queue.push_back(*user);
                }
            }
        }
    }

    /// Copy the affected e-classes and return the root's image.
    ///
    /// Every copied e-node goes in through `lookup_or_insert`, so egglog names
    /// the copy's e-class — nothing here invents an id. That is why this is a
    /// sweep rather than a single postorder pass: an e-node can only be copied
    /// once its children have copies, and a cycle in the copied region needs
    /// one e-node whose children all lie outside it to get started. A cycle
    /// with no such e-node is reported instead.
    fn build(&mut self, state: &mut FullState<'_, '_>, root: Value) -> Result<Value, Error> {
        // Children before parents, so the acyclic case finishes in one sweep.
        let order = self.postorder(root);
        let mut pending: Vec<(Value, Vec<ENode>)> = order
            .into_iter()
            .map(|eclass| {
                let nodes = self.snapshot.nodes.remove(&eclass).unwrap_or_default();
                (eclass, nodes)
            })
            .collect();

        loop {
            let mut progress = false;
            for (eclass, nodes) in pending.iter_mut() {
                let mut blocked = Vec::new();
                for node in std::mem::take(nodes) {
                    let Some(args) = self.copied_args(state, &node) else {
                        blocked.push(node);
                        continue;
                    };
                    let name = self.ctors[node.ctor].name.clone();
                    // A subsumed source row's copy is born retired
                    // (add_subsumed, local engine patch): a live window —
                    // even one round — re-arms the orbit our :subsume
                    // termination discipline exists to prevent. A COMMITTED
                    // row is never force-retired (independent live material
                    // keeps its flag; an already-subsumed original merges
                    // sticky either way).
                    let copy =
                        if node.subsumed && !state.contains(&name, RawValues(args.clone()))? {
                            state.add_subsumed(&name, RawValues(args))?
                        } else {
                            state.add(&name, RawValues(args))?
                        };
                    match self.images.get(eclass) {
                        // The first e-node copied names the class; the rest are
                        // further ways to say the same one.
                        None => {
                            self.images.insert(*eclass, copy);
                        }
                        Some(image) if *image != copy => state.union(copy, *image)?,
                        Some(_) => {}
                    }
                    progress = true;
                }
                *nodes = blocked;
            }
            if !progress {
                break;
            }
        }

        if let Some((eclass, blocked)) = pending.iter().find(|(_, nodes)| !nodes.is_empty()) {
            // DIAGNOSTIC (vendoring delta): name the cycle's anatomy — every
            // row of the blocked class including the subsumed rows the walk
            // skipped, and which children are still imageless.
            for ctor in &self.ctors {
                let mut rows = Vec::new();
                let _ = state.enodes_for_eclass(&ctor.name, *eclass, |enode| {
                    rows.push((enode.children.to_vec(), enode.subsumed));
                });
                for (children, subsumed) in rows {
                    let status: Vec<String> = children
                        .iter()
                        .map(|child| {
                            if self.map.contains_key(child) {
                                format!("{child:?}=KEY")
                            } else if !self.affected.contains(child) {
                                format!("{child:?}=shared")
                            } else if self.images.contains_key(child) {
                                format!("{child:?}=copied")
                            } else {
                                format!("{child:?}=PENDING")
                            }
                        })
                        .collect();
                    eprintln!(
                        "[subst-cycle] {eclass:?} row {}({}){}",
                        ctor.name,
                        status.join(", "),
                        if subsumed { " [SUBSUMED->skipped]" } else { "" }
                    );
                }
            }
            let mut ctors: Vec<&str> = blocked
                .iter()
                .map(|node| self.ctors[node.ctor].name.as_str())
                .collect();
            ctors.sort_unstable();
            ctors.dedup();
            return Err(error(format!(
                "no order copies e-class {eclass:?}: its remaining e-nodes ({}) all refer to \
                 copies that do not exist yet. Every cycle in the substituted region needs at \
                 least one e-node whose children all lie outside it.",
                ctors.join(", "),
            )));
        }

        self.images
            .get(&root)
            .copied()
            .ok_or_else(|| error(format!("the root e-class {root:?} was not copied")))
    }

    /// The affected e-classes that need copying, children before parents.
    fn postorder(&self, root: Value) -> Vec<Value> {
        enum Frame {
            Enter(Value),
            Exit(Value),
        }

        let mut order = Vec::new();
        let mut seen = HashSet::new();
        let mut stack = vec![Frame::Enter(root)];
        while let Some(frame) = stack.pop() {
            match frame {
                Frame::Enter(eclass) => {
                    if !seen.insert(eclass) {
                        continue;
                    }
                    stack.push(Frame::Exit(eclass));
                    for dep in self.snapshot.deps.get(&eclass).into_iter().flatten() {
                        if self.needs_copy(*dep) && !seen.contains(dep) {
                            stack.push(Frame::Enter(*dep));
                        }
                    }
                }
                Frame::Exit(eclass) => order.push(eclass),
            }
        }
        order
    }

    /// Whether this e-class gets a copy: affected, and not a key (a key is
    /// replaced outright rather than copied).
    fn needs_copy(&self, eclass: Value) -> bool {
        self.affected.contains(&eclass) && !self.map.contains_key(&eclass)
    }

    /// The substituted children of `node`, or `None` if some child's copy does
    /// not exist yet.
    fn copied_args(&mut self, state: &mut FullState<'_, '_>, node: &ENode) -> Option<Vec<Value>> {
        let ctor = self.ctors[node.ctor];
        let mut args = Vec::with_capacity(node.children.len());
        for (child, child_sort) in node.children.iter().zip(&ctor.input) {
            let image = match kind_of(child_sort) {
                Kind::Eclass => self.eclass_image(*child)?,
                Kind::Container(type_id) => self.container_image(state, *child, type_id)?,
                Kind::Opaque => *child,
            };
            args.push(image);
        }
        Some(args)
    }

    fn eclass_image(&self, eclass: Value) -> Option<Value> {
        if let Some(target) = self.map.get(&eclass) {
            return Some(*target);
        }
        if !self.affected.contains(&eclass) {
            return Some(eclass);
        }
        self.images.get(&eclass).copied()
    }

    /// The interned value of `container` with its contents substituted, or
    /// `container` itself if nothing inside it changed. `None` while any
    /// e-class inside it is still waiting for its copy.
    fn container_image(
        &mut self,
        state: &mut FullState<'_, '_>,
        container: Value,
        type_id: TypeId,
    ) -> Option<Value> {
        if let Some(image) = self.container_images.get(&container) {
            return Some(*image);
        }
        let Some((_, contents)) = self.snapshot.containers.get(&container).cloned() else {
            return Some(container);
        };

        // Nothing is interned until every value inside resolves, so a blocked
        // container leaves no half-substituted copy behind.
        let mut remap: HashMap<Value, Value> = HashMap::new();
        for (inner_sort, inner) in &contents {
            let image = match kind_of(inner_sort) {
                Kind::Eclass => self.eclass_image(*inner)?,
                Kind::Container(inner_type_id) => {
                    self.container_image(state, *inner, inner_type_id)?
                }
                Kind::Opaque => continue,
            };
            if image != *inner {
                remap.insert(*inner, image);
            }
        }

        let image = if remap.is_empty() {
            container
        } else {
            let mapped = state.map_container(type_id, container, &|value| {
                remap.get(&value).copied().unwrap_or(value)
            });
            // `collect` already read this value's contents through the same
            // sort, so it is a container of this type.
            debug_assert!(mapped.is_some(), "{container:?} is not a {type_id:?}");
            mapped.unwrap_or(container)
        };
        self.container_images.insert(container, image);
        Some(image)
    }
}

fn error(message: String) -> Error {
    Error::BackendError(format!("{SUBST}: {message}"))
}

/// Substitute through the sub-e-graph reachable from `root`, returning the
/// e-class of the copy. The top-level form of the [`SUBST`] primitive.
///
/// `map` must be a `Map` container value whose key and value sorts are the same
/// eq-sort; `root` may be of any eq-sort. Constructor rows reachable from
/// `root` are copied with each key e-class replaced by its mapped value;
/// e-classes the substitution does not affect are shared with the original
/// rather than copied.
///
/// `root` must be an e-class that already has rows — see the module docs.
///
/// Errors if the substituted region contains a cycle in which every e-node
/// refers back into the cycle, since naming that copy would require an e-class
/// id no row produces.
pub fn subst(egraph: &mut EGraph, root: Value, map: Value) -> Result<Value, Error> {
    egraph.update(|mut state| {
        let entries = match state.value_to_container::<MapContainer>(map) {
            Some(entries) => entries.data.clone(),
            None => return Err(error(format!("{map:?} is not a Map container value"))),
        };
        substitute(&mut state, root, &entries, &Default::default())
    })
}

/// The `unstable-subst` primitive.
///
/// `skip` (vendoring delta): constructor tables whose rows are MEMO
/// METADATA, not term structure — e.g. our `int-subst-of (IntExpr
/// IndexMap) -> IntExpr`, the union-on-set memo idiom. Substitution
/// distributes over syntax, never over memo tables keyed by syntax:
/// walking a memo row and substituting inside its KEY mints
/// memo rows about the WRONG key (subst_aliased_min: the sigma-1 walk
/// copied `int-subst-of(c_row, row0_map)` out of the zero class as
/// `int-subst-of(1, row0_map)` and welded 0 = 1). Upstream design gap
/// to raise with #60: "constructor rows = the term structure" is
/// false under the memo-constructor idiom.
#[derive(Clone)]
pub struct Subst {
    pub skip: std::collections::HashSet<String>,
}

impl Primitive for Subst {
    fn name(&self) -> &str {
        SUBST
    }

    fn get_type_constraints(&self, span: &Span) -> Box<dyn TypeConstraint> {
        Box::new(SubstTypeConstraint { span: span.clone() })
    }
}

impl FullPrim for Subst {
    fn apply<'a, 'db>(&self, mut state: FullState<'a, 'db>, args: &[Value]) -> Option<Value> {
        let [root, map] = args else { return None };
        // Cloned out so the container registry is not still borrowed when the
        // walk starts interning new containers.
        let entries = state.value_to_container::<MapContainer>(*map)?.data.clone();
        match substitute(&mut state, *root, &entries, &self.skip) {
            Ok(image) => Some(image),
            Err(err) => {
                // A primitive cannot return an `Error`, and registering a
                // custom panic message needs the backend, which egglog does not
                // expose. So the reason prints to stderr (vendoring delta:
                // upstream uses log::error!, but no logger is configured here
                // and a silent reason would violate house doctrine) and the
                // program sees a generic primitive panic.
                eprintln!("[unstable-subst] {err}");
                state.panic();
                None
            }
        }
    }
}

/// `(unstable-subst root map) : (R, Map<K, K>) -> R` for any eq-sort `R`.
///
/// `R` is free rather than pinned to `K` because a substitution reaches through
/// every sort in the term structure, so a root of one sort can perfectly well
/// be rewritten by a map over another. `K` must be an eq-sort mapping to
/// itself: replacing a `K`-sorted child with a value of another sort would
/// produce an ill-typed row.
struct SubstTypeConstraint {
    span: Span,
}

impl TypeConstraint for SubstTypeConstraint {
    fn get(
        &self,
        arguments: &[AtomTerm],
        typeinfo: &TypeInfo,
    ) -> Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> {
        let [root, map, out] = arguments else {
            return vec![constraint::impossible(
                ImpossibleConstraint::ArityMismatch {
                    atom: Atom {
                        span: self.span.clone(),
                        head: SUBST.to_owned(),
                        args: arguments.to_vec(),
                    },
                    expected: 3,
                },
            )];
        };

        let mut cs: Vec<Box<dyn Constraint<AtomTerm, ArcSort>>> =
            vec![constraint::eq(root.clone(), out.clone())];

        // One instantiation per declared sort that could stand in each
        // position; `xor` defers until the surrounding program pins it down.
        //
        // A `Map` sort is identified by the Rust type its values intern
        // under, since the `ContainerSort` impl behind an `ArcSort` is
        // wrapped in a private type that out-of-tree code cannot downcast to.
        let mut map_sorts: Vec<ArcSort> = typeinfo.get_arcsorts_by(|sort| {
            sort.value_type() == Some(TypeId::of::<MapContainer>())
                && match sort.inner_sorts().as_slice() {
                    [key, value] => key.is_eq_sort() && key.name() == value.name(),
                    _ => false,
                }
        });
        map_sorts.sort_by_key(|sort| sort.name().to_owned());
        cs.push(constraint::xor(
            map_sorts
                .into_iter()
                .map(|sort| constraint::assign(map.clone(), sort))
                .collect(),
        ));

        let mut eq_sorts = typeinfo.get_arcsorts_by(|sort| sort.is_eq_sort());
        eq_sorts.sort_by_key(|sort| sort.name().to_owned());
        cs.push(constraint::xor(
            eq_sorts
                .into_iter()
                .map(|sort| {
                    constraint::and(vec![
                        constraint::assign(root.clone(), sort.clone()),
                        constraint::assign(out.clone(), sort),
                    ])
                })
                .collect(),
        ));

        cs
    }
}

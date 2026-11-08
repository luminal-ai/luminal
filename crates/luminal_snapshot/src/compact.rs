//! Compiler snapshot: numeric identities and contiguous, ordered storage.
//!
//! Text identities are accepted for hand-built graphs and synthetic extraction
//! values. Production exporters assign dense IDs in legacy lexical order; map
//! iteration separately preserves database order. Neither hash iteration nor
//! dense allocation order is allowed to change extraction tie breaking.
pub use egraph_serialize::Cost;
use indexmap::IndexMap;
use once_cell::sync::OnceCell;
use rustc_hash::FxBuildHasher;
use std::{collections::HashMap, sync::Arc};

macro_rules! identity {
    ($name:ident, $prefix:literal) => {
        #[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
        #[cfg_attr(feature = "serde", serde(untagged))]
        pub enum $name {
            Dense(u32),
            Text(Box<Arc<str>>),
        }
        impl $name {
            pub fn dense(index: usize) -> Self {
                Self::Dense(u32::try_from(index).expect("snapshot ID overflow"))
            }
            pub fn index(&self) -> Option<usize> {
                if let Self::Dense(i) = self {
                    Some(*i as usize)
                } else {
                    None
                }
            }
        }
        impl<S: Into<String>> From<S> for $name {
            fn from(s: S) -> Self {
                Self::Text(Box::new(s.into().into()))
            }
        }
        impl std::fmt::Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Dense(i) => write!(f, concat!($prefix, "{}"), i),
                    Self::Text(s) => f.write_str(s),
                }
            }
        }
        impl DenseKey for $name {
            fn dense_index(&self) -> Option<usize> {
                self.index()
            }
        }
    };
}
identity!(NodeId, "n");
identity!(ClassId, "c");

fn next_identity() -> u64 {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

pub trait DenseKey: Clone + Eq + std::hash::Hash {
    fn dense_index(&self) -> Option<usize>;
}

/// Packed rows in insertion order, with O(1) numeric lookup. Text lookup is
/// only used by fixtures and synthetic values. Removing rows rebuilds offsets.
#[derive(Clone, Debug)]
pub struct DenseMap<K: DenseKey, V: Clone> {
    rows: Arc<Vec<(K, V)>>,
    revision: u64,
    identity: u64,
    offsets: Arc<Vec<usize>>,
    text: Arc<IndexMap<K, usize, FxBuildHasher>>,
}
impl<K: DenseKey, V: Clone + PartialEq> PartialEq for DenseMap<K, V> {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows
    }
}
impl<K: DenseKey, V: Clone + Eq> Eq for DenseMap<K, V> {}
impl<K: DenseKey, V: Clone> Default for DenseMap<K, V> {
    fn default() -> Self {
        Self {
            identity: next_identity(),
            revision: 0,
            rows: Arc::default(),
            offsets: Arc::default(),
            text: Arc::default(),
        }
    }
}
impl<K: DenseKey, V: Clone> DenseMap<K, V> {
    pub fn revision(&self) -> (u64, u64) {
        (self.identity, self.revision)
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
    pub fn reserve(&mut self, n: usize) {
        Arc::make_mut(&mut self.rows).reserve(n);
        Arc::make_mut(&mut self.offsets).reserve(n)
    }
    pub fn insertion_index(&self, key: &K) -> Option<usize> {
        self.position(key)
    }
    fn position(&self, key: &K) -> Option<usize> {
        if let Some(i) = key.dense_index() {
            self.offsets.get(i).copied().filter(|i| *i != usize::MAX)
        } else {
            self.text.get(key).copied()
        }
    }
    pub fn get(&self, key: &K) -> Option<&V> {
        self.position(key).map(|i| &self.rows[i].1)
    }
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.revision += 1;
        self.position(key)
            .map(|i| &mut Arc::make_mut(&mut self.rows)[i].1)
    }
    pub fn contains_key(&self, key: &K) -> bool {
        self.position(key).is_some()
    }
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        self.revision += 1;
        if let Some(i) = self.position(&key) {
            return Some(std::mem::replace(
                &mut Arc::make_mut(&mut self.rows)[i].1,
                value,
            ));
        }
        self.record(&key, self.rows.len());
        Arc::make_mut(&mut self.rows).push((key, value));
        None
    }
    fn record(&mut self, key: &K, position: usize) {
        if let Some(i) = key.dense_index() {
            if i >= self.offsets.len() {
                Arc::make_mut(&mut self.offsets).resize(i + 1, usize::MAX)
            }
            Arc::make_mut(&mut self.offsets)[i] = position;
        } else {
            Arc::make_mut(&mut self.text).insert(key.clone(), position);
        }
    }
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &mut V) -> bool) {
        self.revision += 1;
        Arc::make_mut(&mut self.rows).retain_mut(|(k, v)| keep(k, v));
        Arc::make_mut(&mut self.offsets).fill(usize::MAX);
        Arc::make_mut(&mut self.text).clear();
        for i in 0..self.rows.len() {
            self.record(&self.rows[i].0.clone(), i)
        }
    }
    /// Rewrite identities in place without allocating another array of rows.
    /// The caller must produce unique keys, just as for an ordinary map.
    pub fn remap_entries(&mut self, mut remap: impl FnMut(&mut K, &mut V)) {
        self.revision += 1;
        for (k, v) in Arc::make_mut(&mut self.rows) {
            remap(k, v);
        }
        Arc::make_mut(&mut self.offsets).fill(usize::MAX);
        Arc::make_mut(&mut self.text).clear();
        for i in 0..self.rows.len() {
            self.record(&self.rows[i].0.clone(), i);
        }
    }
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = (&K, &V)> + ExactSizeIterator {
        self.rows.iter().map(|(k, v)| (k, v))
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (&K, &mut V)> {
        self.revision += 1;
        Arc::make_mut(&mut self.rows)
            .iter_mut()
            .map(|(k, v)| (&*k, v))
    }
    pub fn keys(&self) -> impl DoubleEndedIterator<Item = &K> + ExactSizeIterator {
        self.rows.iter().map(|(k, _)| k)
    }
    pub fn values(&self) -> impl DoubleEndedIterator<Item = &V> + ExactSizeIterator {
        self.rows.iter().map(|(_, v)| v)
    }
    pub fn values_mut(&mut self) -> impl Iterator<Item = &mut V> {
        self.revision += 1;
        Arc::make_mut(&mut self.rows).iter_mut().map(|(_, v)| v)
    }
    pub fn into_values(self) -> impl Iterator<Item = V> {
        Arc::unwrap_or_clone(self.rows).into_iter().map(|(_, v)| v)
    }
    pub fn entry(&mut self, key: K) -> Entry<'_, K, V> {
        Entry { map: self, key }
    }
}
pub struct Entry<'a, K: DenseKey, V: Clone> {
    map: &'a mut DenseMap<K, V>,
    key: K,
}
impl<'a, K: DenseKey, V: Clone> Entry<'a, K, V> {
    pub fn or_insert_with(self, create: impl FnOnce() -> V) -> &'a mut V {
        self.map.revision += 1;
        let index = if let Some(i) = self.map.position(&self.key) {
            i
        } else {
            let i = self.map.rows.len();
            self.map.record(&self.key, i);
            Arc::make_mut(&mut self.map.rows).push((self.key, create()));
            i
        };
        &mut Arc::make_mut(&mut self.map.rows)[index].1
    }
    pub fn or_default(self) -> &'a mut V
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
}
impl<K: DenseKey, V: Clone> std::ops::Index<&K> for DenseMap<K, V> {
    type Output = V;
    fn index(&self, k: &K) -> &V {
        self.get(k).expect("snapshot key missing")
    }
}
impl<K: DenseKey, V: Clone> std::ops::IndexMut<&K> for DenseMap<K, V> {
    fn index_mut(&mut self, k: &K) -> &mut V {
        self.get_mut(k).expect("snapshot key missing")
    }
}
impl<K: DenseKey, V: Clone> FromIterator<(K, V)> for DenseMap<K, V> {
    fn from_iter<T: IntoIterator<Item = (K, V)>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let mut m = Self::default();
        m.reserve(iter.size_hint().0);
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}
impl<K: DenseKey, V: Clone> IntoIterator for DenseMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<(K, V)>;
    fn into_iter(self) -> Self::IntoIter {
        Arc::unwrap_or_clone(self.rows).into_iter()
    }
}
impl<'a, K: DenseKey, V: Clone> IntoIterator for &'a DenseMap<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = std::iter::Map<std::slice::Iter<'a, (K, V)>, fn(&'a (K, V)) -> (&'a K, &'a V)>;
    fn into_iter(self) -> Self::IntoIter {
        self.rows.iter().map(|(k, v)| (k, v))
    }
}
#[cfg(feature = "serde")]
impl<K: DenseKey + serde::Serialize, V: Clone + serde::Serialize> serde::Serialize
    for DenseMap<K, V>
{
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.rows.serialize(s)
    }
}
#[cfg(feature = "serde")]
impl<'de, K: DenseKey + serde::Deserialize<'de>, V: Clone + serde::Deserialize<'de>>
    serde::Deserialize<'de> for DenseMap<K, V>
{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Vec::<(K, V)>::deserialize(d)?.into_iter().collect())
    }
}

/// A constructor name shared by every row of its table.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(transparent))]
pub struct Op(Arc<str>);
impl Op {
    pub fn as_str(&self) -> &str {
        &self.0
    }
    pub fn from_debug(value: impl std::fmt::Debug) -> Self {
        Self(format!("{value:?}").into())
    }
}
impl std::ops::Deref for Op {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}
impl std::borrow::Borrow<str> for Op {
    fn borrow(&self) -> &str {
        &self.0
    }
}
impl<S: Into<String>> From<S> for Op {
    fn from(s: S) -> Self {
        Self(s.into().into())
    }
}
impl std::fmt::Display for Op {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl PartialEq<str> for Op {
    fn eq(&self, s: &str) -> bool {
        self.as_str() == s
    }
}
impl PartialEq<&str> for Op {
    fn eq(&self, s: &&str) -> bool {
        self.as_str() == *s
    }
}
impl PartialEq<String> for Op {
    fn eq(&self, s: &String) -> bool {
        self.as_str() == s
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Node {
    pub op: Op,
    pub children: Children,
    pub eclass: ClassId,
    pub cost: Cost,
    pub subsumed: bool,
}
impl Node {
    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// Production operands share one contiguous allocation. An editing postpass
/// copies only the edited node's slice, never the entire operand arena.
#[derive(Clone)]
pub enum Children {
    Owned(Vec<NodeId>),
    Shared {
        data: Arc<Vec<NodeId>>,
        start: u32,
        end: u32,
    },
}
impl Children {
    pub fn shared(data: Arc<Vec<NodeId>>, range: std::ops::Range<u32>) -> Self {
        assert!(range.start <= range.end && range.end as usize <= data.len());
        Self::Shared {
            data,
            start: range.start,
            end: range.end,
        }
    }
}
impl Default for Children {
    fn default() -> Self {
        Self::Owned(Vec::new())
    }
}
impl std::ops::Deref for Children {
    type Target = [NodeId];
    fn deref(&self) -> &[NodeId] {
        match self {
            Self::Owned(v) => v,
            Self::Shared { data, start, end } => &data[*start as usize..*end as usize],
        }
    }
}
impl std::ops::DerefMut for Children {
    fn deref_mut(&mut self) -> &mut [NodeId] {
        if matches!(self, Self::Shared { .. }) {
            *self = Self::Owned(self.to_vec())
        }
        match self {
            Self::Owned(v) => v,
            _ => unreachable!(),
        }
    }
}
impl std::fmt::Debug for Children {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}
impl PartialEq for Children {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for Children {}
impl PartialEq<Vec<NodeId>> for Children {
    fn eq(&self, other: &Vec<NodeId>) -> bool {
        &**self == other.as_slice()
    }
}
impl From<Vec<NodeId>> for Children {
    fn from(v: Vec<NodeId>) -> Self {
        Self::Owned(v)
    }
}
impl FromIterator<NodeId> for Children {
    fn from_iter<T: IntoIterator<Item = NodeId>>(i: T) -> Self {
        Self::Owned(i.into_iter().collect())
    }
}
impl<'a> IntoIterator for &'a Children {
    type Item = &'a NodeId;
    type IntoIter = std::slice::Iter<'a, NodeId>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<'a> IntoIterator for &'a mut Children {
    type Item = &'a mut NodeId;
    type IntoIter = std::slice::IterMut<'a, NodeId>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}
impl IntoIterator for Children {
    type Item = NodeId;
    type IntoIter = std::vec::IntoIter<NodeId>;
    // The consuming iterator must own the selected range after `self` is dropped.
    #[allow(clippy::unnecessary_to_owned)]
    fn into_iter(self) -> Self::IntoIter {
        match self {
            Self::Owned(v) => v.into_iter(),
            other => other.to_vec().into_iter(),
        }
    }
}
#[cfg(feature = "serde")]
impl serde::Serialize for Children {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self[..].serialize(s)
    }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Children {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(Self::Owned(Vec::deserialize(d)?))
    }
}
/// Most classes have no annotations. Keep that case one word, and share the
/// uncommon maps across immutable graph clones.
#[derive(Clone, Debug, Default)]
pub struct Annotations(Option<Arc<HashMap<String, String>>>);
impl std::ops::Deref for Annotations {
    type Target = HashMap<String, String>;
    fn deref(&self) -> &Self::Target {
        static EMPTY: std::sync::OnceLock<HashMap<String, String>> = std::sync::OnceLock::new();
        self.0
            .as_deref()
            .unwrap_or_else(|| EMPTY.get_or_init(HashMap::new))
    }
}
impl std::ops::DerefMut for Annotations {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(self.0.get_or_insert_with(Arc::default))
    }
}
impl From<HashMap<String, String>> for Annotations {
    fn from(m: HashMap<String, String>) -> Self {
        Self((!m.is_empty()).then(|| Arc::new(m)))
    }
}
impl PartialEq for Annotations {
    fn eq(&self, other: &Self) -> bool {
        **self == **other
    }
}
impl Eq for Annotations {}
impl PartialEq<HashMap<String, String>> for Annotations {
    fn eq(&self, other: &HashMap<String, String>) -> bool {
        &**self == other
    }
}
#[cfg(feature = "serde")]
impl serde::Serialize for Annotations {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (**self).serialize(s)
    }
}
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Annotations {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(HashMap::deserialize(d)?.into())
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ClassData {
    pub typ: Option<Op>,
    pub extra: Annotations,
    /// Canonical Egglog value. Separate from the dense snapshot identity.
    pub value: Option<u64>,
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Class {
    pub id: ClassId,
    pub nodes: Children,
}
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct GraphIndexes {
    pub classes: DenseMap<ClassId, Class>,
    pub operators: IndexMap<Op, Vec<NodeId>, FxBuildHasher>,
}
/// In-memory proof that a snapshot still describes its source database.
/// Never persisted: archives must be validated against live rows.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct SourceStamp {
    pub identity: (u64, usize),
    pub tables: IndexMap<String, (usize, usize, usize, usize), FxBuildHasher>,
    pub node_revision: (u64, u64),
    pub class_revision: (u64, u64),
    pub function_nodes: Option<std::ops::Range<u32>>,
}
#[derive(Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EGraph {
    pub nodes: DenseMap<NodeId, Node>,
    pub class_data: DenseMap<ClassId, ClassData>,
    pub root_eclasses: Vec<ClassId>,
    pub function_nodes: Option<std::ops::Range<u32>>,
    pub function_offsets: Option<Arc<Vec<u32>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub source: Option<SourceStamp>,
    #[cfg_attr(feature = "serde", serde(skip))]
    indexes: OnceCell<Arc<GraphIndexes>>,
}
impl PartialEq for EGraph {
    fn eq(&self, other: &Self) -> bool {
        self.nodes == other.nodes
            && self.class_data == other.class_data
            && self.root_eclasses == other.root_eclasses
            && self.function_nodes == other.function_nodes
            && self.function_offsets == other.function_offsets
    }
}
impl Eq for EGraph {}
impl EGraph {
    /// Diagnostic labels are materialized only when requested. Renderers use
    /// these for cyclic/depth-limited terms, preserving legacy tie-break text.
    pub fn class_label(&self, id: &ClassId) -> String {
        if let Some(data) = self.class_data.get(id)
            && let (Some(sort), Some(value)) = (&data.typ, data.value)
        {
            return format!("{sort}-{value}");
        }
        id.to_string()
    }
    pub fn node_label(&self, id: &NodeId) -> String {
        let Some(index) = id.index() else {
            return id.to_string();
        };
        let Some(node) = self.nodes.get(id) else {
            return id.to_string();
        };
        if let (Some(range), Some(offsets)) = (&self.function_nodes, &self.function_offsets)
            && range.contains(&(index as u32))
        {
            return format!(
                "function-{}-{}",
                offsets[index - range.start as usize],
                node.op
            );
        }
        format!(
            "{}-{}",
            if node.op == "[...]" {
                "dummy"
            } else {
                "primitive"
            },
            self.class_label(&node.eclass)
        )
    }
    pub fn add_node(&mut self, id: impl Into<NodeId>, node: Node) {
        assert!(self.nodes.insert(id.into(), node).is_none());
        self.indexes.take();
    }
    pub fn nid_to_cid(&self, id: &NodeId) -> &ClassId {
        &self.nodes[id].eclass
    }
    pub fn nid_to_class(&self, id: &NodeId) -> &Class {
        &self.classes()[self.nid_to_cid(id)]
    }
    pub fn indexes(&self) -> &Arc<GraphIndexes> {
        self.indexes.get_or_init(|| {
            let mut offsets: DenseMap<ClassId, (usize, usize)> = DenseMap::default();
            offsets.reserve(self.class_data.len());
            let mut operators: IndexMap<Op, Vec<NodeId>, FxBuildHasher> = IndexMap::default();
            for (id, node) in &self.nodes {
                offsets.entry(node.eclass.clone()).or_default().1 += 1;
                operators
                    .entry(node.op.clone())
                    .or_default()
                    .push(id.clone());
            }
            let mut total = 0;
            for (_, (start, count)) in offsets.iter_mut() {
                *start = total;
                total += *count;
                *count = 0;
            }
            let mut members = vec![NodeId::dense(0); total];
            for (id, node) in &self.nodes {
                let (start, count) = offsets.get_mut(&node.eclass).unwrap();
                members[*start + *count] = id.clone();
                *count += 1;
            }
            let members = Arc::new(members);
            let classes = offsets
                .into_iter()
                .map(|(id, (start, count))| {
                    let nodes = Children::shared(
                        members.clone(),
                        u32::try_from(start).unwrap()..u32::try_from(start + count).unwrap(),
                    );
                    (id.clone(), Class { id, nodes })
                })
                .collect();
            Arc::new(GraphIndexes { classes, operators })
        })
    }
    pub fn operator_nodes<'a>(
        &'a self,
        op: &str,
    ) -> impl Iterator<Item = (&'a NodeId, &'a Node)> + 'a {
        self.indexes()
            .operators
            .get(op)
            .into_iter()
            .flatten()
            .map(|id| (id, &self.nodes[id]))
    }
    pub fn classes(&self) -> &DenseMap<ClassId, Class> {
        &self.indexes().classes
    }
    pub fn invalidate_indexes(&mut self) {
        self.indexes.take();
    }
}
impl std::ops::Index<&NodeId> for EGraph {
    type Output = Node;
    fn index(&self, id: &NodeId) -> &Node {
        &self.nodes[id]
    }
}
impl std::ops::Index<&ClassId> for EGraph {
    type Output = Class;
    fn index(&self, id: &ClassId) -> &Class {
        &self.classes()[id]
    }
}

/// Compatibility conversion for fixtures and external diagnostic archives.
/// Production compilation uses Egglog's direct numeric exporter.
impl From<&egraph_serialize::EGraph> for EGraph {
    fn from(source: &egraph_serialize::EGraph) -> Self {
        Self {
            nodes: source
                .nodes
                .iter()
                .map(|(id, n)| {
                    (
                        id.to_string().into(),
                        Node {
                            op: n.op.clone().into(),
                            children: n.children.iter().map(|id| id.to_string().into()).collect(),
                            eclass: n.eclass.to_string().into(),
                            cost: n.cost,
                            subsumed: n.subsumed,
                        },
                    )
                })
                .collect(),
            class_data: source
                .class_data
                .iter()
                .map(|(id, d)| {
                    (
                        id.to_string().into(),
                        ClassData {
                            typ: d.typ.clone().map(Into::into),
                            extra: d.extra.clone().into(),
                            value: None,
                        },
                    )
                })
                .collect(),
            root_eclasses: source
                .root_eclasses
                .iter()
                .map(|id| id.to_string().into())
                .collect(),
            function_nodes: None,
            function_offsets: None,
            source: None,
            indexes: OnceCell::new(),
        }
    }
}
impl From<egraph_serialize::EGraph> for EGraph {
    fn from(source: egraph_serialize::EGraph) -> Self {
        Self::from(&source)
    }
}

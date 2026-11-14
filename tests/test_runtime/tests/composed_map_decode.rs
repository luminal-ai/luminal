//! A view of a view composes in the e-graph: the window through the permute
//! is unioned with one view of the input through the composed map, a literal
//! the shared decoder `parse_index_map_entries` reads directly. The fixture
//! is a permute under a shifted slice, so the composite is arithmetic.

use luminal::index_expr::parse_index_map_entries;
use luminal::layout_ir::{ExtractionSite, SerializedIndex};

const SCHEDULE: &str = "(run-schedule (saturate (run prop)) (saturate (saturate (run main_ruleset) (run prop)) (run subst-walk)))";

/// x : [2,3,4]. `permuted = permute(x, [1,2,0])` : [3,4,2] (entries over
/// the out box, highest parent axis first: parent axis 2 reads out axis 0,
/// parent axis 1 reads out axis 2, parent axis 0 reads out axis 1).
/// `window = permuted[:, 1:3, :]` : [3,2,2], its middle entry `v + 1`
/// (a window of extent 1 would let the extent-1 collapse fold the shift
/// to a literal, which is right but would not exercise the substitution).
const FIXTURE: &str = r#"
(let s3 (ShapeLit (IntExprCons (IntLit 2) (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprNil))))))
(let s_perm (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 4) (IntExprCons (IntLit 2) (IntExprNil))))))
(let s_win (ShapeLit (IntExprCons (IntLit 3) (IntExprCons (IntLit 2) (IntExprCons (IntLit 2) (IntExprNil))))))
(let x (LogicalTensorInputLit (LogicalIdLit "x") s3 (F32)))
(let m_perm (IndexMapLit (IntExprCons (CoordVar s_perm 0) (IntExprCons (CoordVar s_perm 2) (IntExprCons (CoordVar s_perm 1) (IntExprNil)))) s3))
(let permuted (LogicalIndexMapApply x m_perm s_perm))
(let m_win (IndexMapLit (IntExprCons (CoordVar s_win 2) (IntExprCons (IntAdd (CoordVar s_win 1) (IntLit 1)) (IntExprCons (CoordVar s_win 0) (IntExprNil)))) s_perm))
(let window (LogicalIndexMapApply permuted m_win s_win))
"#;

#[test]
fn a_composed_view_decodes_to_the_composed_entries() {
    let graph = test_runtime::serialize_fixture(&format!("{FIXTURE}\n{SCHEDULE}\n"));
    let index = SerializedIndex::new(&graph);
    let has = |class: &luminal::prelude::egraph_serialize::ClassId, op: &str| {
        index
            .nodes_of(class)
            .iter()
            .filter_map(|id| graph.nodes.get(id))
            .any(|node| node.op == op)
    };
    let operand_is_input = |node: &luminal::prelude::egraph_serialize::Node| {
        node.children
            .first()
            .is_some_and(|child| has(&graph.nodes[child].eclass, "LogicalTensorInputLit"))
    };
    // The composed spelling of `window`: a view of the INPUT in the class
    // that also holds the nested spelling (a view of the permute).
    let (node_id, node) = graph
        .nodes
        .iter()
        .find(|(_, node)| {
            node.op == "LogicalIndexMapApply"
                && operand_is_input(node)
                && index
                    .nodes_of(&node.eclass)
                    .iter()
                    .filter_map(|id| graph.nodes.get(id))
                    .any(|other| other.op == "LogicalIndexMapApply" && !operand_is_input(other))
        })
        .expect("the window through the permute has a composed spelling over the input");
    let site = ExtractionSite {
        egraph: &graph,
        node_id,
        node,
        index: &index,
    };
    let entries =
        parse_index_map_entries(&site, 1, 2).expect("the composed map is a literal the decoder reads");
    assert_eq!(entries.len(), 3, "one entry per parent axis: {entries:?}");
    // Through the permute, x[a][b][c] = permuted[b][c][a] = window[b][c-1][a]:
    // at window coordinates (u, v, w), the parent coordinates are
    // (w, u, v + 1). Compared by value, since a saturated class spells the
    // shift in whichever order the integer rules left first.
    for u in 0..3usize {
        for v in 0..2usize {
            for w in 0..2usize {
                let coords = [u, v, w];
                let read: Vec<i64> = entries.iter().map(|e| e.eval(&coords)).collect();
                assert_eq!(read, vec![w as i64, u as i64, v as i64 + 1], "at {coords:?}: {entries:?}");
            }
        }
    }
}

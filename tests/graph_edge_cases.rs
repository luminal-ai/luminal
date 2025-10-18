#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_empty_graph() {
        let g = Graph::new();
        assert_eq!(g.nodes().len(), 0);
    }
    #[test]
    fn test_single_node_graph() {
        let mut g = Graph::new();
        let n = g.add_node("input", NodeType::Input);
        assert!(g.nodes().contains(&n));
    }
    #[test]
    fn test_cycle_detection() {
        let mut g = Graph::new();
        let a = g.add_node("A", NodeType::Input);
        let b = g.add_node("B", NodeType::Op);
        g.add_edge(a, b);
        g.add_edge(b, a);
        assert!(g.has_cycle());
    }
}

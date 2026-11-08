//! Least fixed point of realizable producer outputs, in dependency order.
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

pub(super) fn transitively_viable<'a, T: Clone + Eq + Hash + 'a>(
    terminals: impl IntoIterator<Item = &'a T>,
    rules: impl IntoIterator<Item = (&'a [T], &'a [T])>,
) -> HashSet<T> {
    let mut viable: HashSet<T> = terminals.into_iter().cloned().collect();
    let mut ready: Vec<T> = viable.iter().cloned().collect();
    let mut users: HashMap<&T, Vec<usize>> = HashMap::new();
    let mut remaining = Vec::new();
    let mut outputs = Vec::new();
    for (inputs, produced) in rules {
        let index = remaining.len();
        remaining.push(inputs.len());
        outputs.push(produced);
        // Count occurrences, including repeated operands. A class becomes
        // viable once and then discharges every occurrence that depends on it.
        for input in inputs {
            users.entry(input).or_default().push(index);
        }
        if inputs.is_empty() {
            for output in produced {
                if viable.insert(output.clone()) {
                    ready.push(output.clone());
                }
            }
        }
    }
    while let Some(class) = ready.pop() {
        for &index in users.get(&class).into_iter().flatten() {
            remaining[index] -= 1;
            if remaining[index] == 0 {
                for output in outputs[index] {
                    if viable.insert(output.clone()) {
                        ready.push(output.clone());
                    }
                }
            }
        }
    }
    viable
}

#[cfg(test)]
mod tests {
    use super::*;

    fn old_fixpoint(terminals: &[usize], rules: &[(Vec<usize>, Vec<usize>)]) -> HashSet<usize> {
        let mut viable: HashSet<_> = terminals.iter().copied().collect();
        loop {
            let before = viable.len();
            for (inputs, outputs) in rules {
                if inputs.iter().all(|input| viable.contains(input)) {
                    viable.extend(outputs);
                }
            }
            if viable.len() == before {
                return viable;
            }
        }
    }

    fn worklist(terminals: &[usize], rules: &[(Vec<usize>, Vec<usize>)]) -> HashSet<usize> {
        transitively_viable(
            terminals,
            rules.iter().map(|(i, o)| (i.as_slice(), o.as_slice())),
        )
    }

    #[test]
    fn matches_every_two_class_hypergraph_and_terminal_set() {
        let subset = |bits| {
            (0..2)
                .filter(|i| bits & (1 << i) != 0)
                .collect::<Vec<usize>>()
        };
        let possibilities: Vec<_> = (0..16).map(|n| (subset(n & 3), subset(n >> 2))).collect();
        for mask in 0..(1u32 << possibilities.len()) {
            let rules: Vec<_> = possibilities
                .iter()
                .enumerate()
                .filter(|(i, _)| mask & (1 << i) != 0)
                .map(|(_, rule)| rule.clone())
                .collect();
            for terminals in 0..4 {
                let terminals = subset(terminals);
                assert_eq!(
                    worklist(&terminals, &rules),
                    old_fixpoint(&terminals, &rules)
                );
            }
        }
    }

    #[test]
    fn duplicate_operands_multi_outputs_and_unseeded_cycles() {
        let rules = vec![
            (vec![0, 0], vec![1, 2, 1]),
            (vec![1, 2, 2], vec![3]),
            (vec![3], vec![0]),
            (vec![4], vec![5]),
            (vec![5], vec![4]),
            (vec![6, 3], vec![7]),
        ];
        assert_eq!(worklist(&[0, 0], &rules), HashSet::from([0, 1, 2, 3]));
        assert_eq!(worklist(&[0, 6], &rules), old_fixpoint(&[0, 6], &rules));
        assert!(worklist(&[], &rules).is_empty());
    }

    #[test]
    fn deep_reverse_order_chain_reaches_every_output() {
        let rules: Vec<_> = (0..100_000).rev().map(|i| (vec![i], vec![i + 1])).collect();
        let start = std::time::Instant::now();
        let actual = worklist(&[0], &rules);
        assert_eq!(actual.len(), 100_001);
        assert!((0..=100_000).all(|i| actual.contains(&i)));
        eprintln!("100,000-edge viability chain: {:?}", start.elapsed());
    }
}

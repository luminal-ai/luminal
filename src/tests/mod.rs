use std::fmt::Debug;

use crate::egglog_utils::{
    extract_generation, hash_choice_set, random_initial_choice, validate_choice_set,
};
use crate::prelude::*;
use candle_core::{Device, Tensor};
use proptest::prelude::*;
use rand::{Rng, rng};

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]
    #[test]
    fn simple(vals in proptest::collection::vec(-2.0f32..2.0, 3)) {
        prop_assume!(vals.iter().all(|v| v.abs() > 1e-3));
        let mut cx = Graph::new();
        let b = cx.tensor(3);
        let c = cx.tensor(3);
        let g = cx.tensor(3);
        let e = cx.tensor(3);

        let a = (b * c + g).output();
        let d = (b * c / e).sin().output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);

        rt.set_data(b.id, vals.clone());
        rt.set_data(c.id, vals.clone());
        rt.set_data(g.id, vals.clone());
        rt.set_data(e.id, vals.clone());

        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = Tensor::new(vals.clone(), &device).unwrap();
        let ref_c = Tensor::new(vals.clone(), &device).unwrap();
        let ref_g = Tensor::new(vals.clone(), &device).unwrap();
        let ref_e = Tensor::new(vals, &device).unwrap();

        let ref_a = (ref_b.clone() * ref_c.clone() + ref_g).unwrap();
        let ref_d = (ref_b * ref_c / ref_e).unwrap().sin().unwrap();

        assert_close(rt.get_f32(a.id), &ref_a.to_vec1::<f32>().unwrap());
        assert_close(rt.get_f32(d.id), &ref_d.to_vec1::<f32>().unwrap());
    }

    #[test]
    fn test_matmul(m in 1usize..6, k in 1usize..6, n in 1usize..6, lhs in proptest::collection::vec(-2.0f32..2.0, 1..100), rhs in proptest::collection::vec(-2.0f32..2.0, 1..100)) {
        prop_assume!(lhs.len() >= m * k);
        prop_assume!(rhs.len() >= k * n);
        let mut cx = Graph::new();
        let b = cx.tensor((m, k));
        let c = cx.tensor((k, n));

        let a = b.matmul(c).output();

        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        let lhs = lhs.into_iter().take(m * k).collect::<Vec<f32>>();
        let rhs = rhs.into_iter().take(k * n).collect::<Vec<f32>>();
        rt.set_data(b.id, lhs.clone());
        rt.set_data(c.id, rhs.clone());
        rt.execute(&cx.dyn_map);

        // Reference
        let device = Device::Cpu;
        let ref_b = Tensor::new(lhs, &device).unwrap().reshape((m, k)).unwrap();
        let ref_c = Tensor::new(rhs, &device).unwrap().reshape((k, n)).unwrap();
        let ref_a = ref_b.matmul(&ref_c).unwrap();
        assert_close(
            rt.get_f32(a.id),
            &ref_a.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
        );
    }

    #[test]
    fn test_shapes(values in proptest::collection::vec(-2.0f32..2.0, 4)) {
        let mut cx = Graph::new();
        let a = cx.tensor((2, 2));
        let b = (a.permute((1, 0)) * 1.0).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        rt.set_data(a.id, values.clone());
        rt.execute(&cx.dyn_map);

        assert_exact(rt.get_f32(b.id), &[values[0], values[2], values[1], values[3]]);
    }

    #[test]
    fn test_top_k_filter(rows in 1usize..6, cols in 3usize..10, k in 1usize..5, values in proptest::collection::vec(-2.0f32..2.0, 1..200)) {
        prop_assume!(k <= cols);
        prop_assume!(values.len() >= rows * cols);
        let mut cx = Graph::new();
        let a = cx.tensor((rows, cols));
        let kth_largest = a.gather(a.topk_indexes(k, 1).slice((.., (k - 1)..k)).squeeze(1));
        let mask = a.ge(kth_largest.expand_dim(1, cols));
        let filtered = (a * mask).output();
        cx.build_search_space::<NativeRuntime>();
        let mut rt = cx.search(NativeRuntime::default(), 1);
        let values = values.into_iter().take(rows * cols).collect::<Vec<f32>>();
        rt.set_data(a.id, values.clone());
        rt.execute(&cx.dyn_map);

        let mut expected = Vec::with_capacity(values.len());
        for row in values.chunks_exact(cols) {
            let mut indices = (0..cols).collect::<Vec<usize>>();
            indices.sort_by(|&i, &j| row[j].partial_cmp(&row[i]).unwrap());
            let kth_index = indices[k - 1];
            let threshold = values[kth_index];
            expected.extend(row.iter().map(|v| if *v >= threshold { *v } else { 0.0 }));
        }
        assert_close(rt.get_f32(filtered.id), &expected);
    }
}

/// Ensure two arrays are nearly equal
pub fn assert_close(a_vec: &[f32], b_vec: &[f32]) {
    assert_close_precision(a_vec, b_vec, 1e-3);
}

/// Ensure two arrays are nearly equal to a decimal place
pub fn assert_close_precision(a_vec: &[f32], b_vec: &[f32], threshold: f32) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if (a - b).abs() > threshold {
            panic!(
                "{a} is not close to {b}, index {i}, avg distance: {}",
                a_vec
                    .iter()
                    .zip(b_vec.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f32>()
                    / a_vec.len() as f32
            );
        }
    }
}

/// Ensure two arrays are exactly equal
pub fn assert_exact<T: PartialEq + Debug>(a_vec: &[T], b_vec: &[T]) {
    assert_eq!(a_vec.len(), b_vec.len(), "Number of elements doesn't match");
    for (i, (a, b)) in a_vec.iter().zip(b_vec.iter()).enumerate() {
        if a != b {
            panic!("{a:?} is not equal to {b:?}, index {i}");
        }
    }
}

pub fn random_array<const N: usize>() -> [f32; N] {
    let mut rng = rng();
    random_array_rng(&mut rng)
}

pub fn random_array_rng<const N: usize, R: Rng>(rng: &mut R) -> [f32; N] {
    let mut arr = [0.; N];
    for i in &mut arr {
        *i = rng.random_range(-0.5..0.5);
    }
    arr
}

pub fn random_vec(n: usize) -> Vec<f32> {
    let mut rng = rng();
    random_vec_rng(n, &mut rng)
}

pub fn random_vec_rng<R: Rng>(n: usize, rng: &mut R) -> Vec<f32> {
    (0..n).map(|_| rng.random_range(-0.5..0.5)).collect()
}

/// Fuzz test to verify all genomes in the search space produce valid graphs.
/// This tests the genetic algorithm by generating many random genomes and mutations,
/// validating each one, and checking that extraction doesn't panic.
#[test]
fn fuzz_test_genome_validity() {
    use crate::egglog_utils::egglog_to_llir;

    // Build a moderately complex graph to test
    let mut cx = Graph::new();
    let a = cx.tensor((4, 8));
    let b = cx.tensor((8, 4));
    let c = cx.tensor((4, 4));

    // Create a graph with multiple operations that can have equivalent rewrites
    let d = a.matmul(b);
    let e = (d + c).relu();
    let f = e.softmax(1);
    let _out = f.output();

    // Build search space
    cx.build_search_space::<NativeRuntime>();
    let egraph = cx.egraph().unwrap();
    let ops = cx.egglog_ops().unwrap();

    // Debug: count eclasses with choices
    let mutable_eclasses: usize = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .count();
    println!(
        "Search space: {} total eclasses, {} mutable (have >1 enode)",
        egraph.eclasses.len(),
        mutable_eclasses
    );

    let mut rng = rand::rng();
    let mut prev_selected: FxHashSet<u64> = FxHashSet::default();
    let mut list_cache = FxHashMap::default();
    let mut expr_cache = FxHashMap::default();

    // Test initial random choice
    let initial = random_initial_choice(egraph, &mut rng);
    println!("Initial choice has {} entries", initial.len());
    prev_selected.insert(hash_choice_set(&initial));

    // Validate initial choice
    if let Err(e) = validate_choice_set(egraph, &initial, ops) {
        panic!("Initial choice is invalid: {}", e);
    }
    println!("Initial choice validated successfully");

    // Test extraction doesn't panic
    let _graph = egglog_to_llir(
        egraph,
        initial.clone(),
        ops,
        &cx.custom_ops,
        &mut list_cache,
        &mut expr_cache,
        None,
    );
    println!(
        "Initial extraction successful, graph has {} nodes",
        _graph.node_count()
    );

    // Generate many mutations and validate each
    let mut base = initial;
    let mut valid_count = 0;
    let mut total_count = 0;
    let target_count = 100;

    for generation in 0..20 {
        let offspring = extract_generation(
            egraph,
            &base,
            10, // generation size
            2,  // mutations per offspring
            &mut prev_selected,
            &mut rng,
        );
        println!(
            "Generation {}: {} offspring, prev_selected has {} entries",
            generation,
            offspring.len(),
            prev_selected.len()
        );

        if offspring.is_empty() {
            // Search space exhausted
            println!("Search space exhausted at generation {}", generation);
            break;
        }

        for genome in offspring {
            total_count += 1;

            // Validate the choice set
            if let Err(e) = validate_choice_set(egraph, &genome, ops) {
                panic!("Generation {} produced invalid genome: {}", generation, e);
            }

            // Test extraction doesn't panic
            let graph = egglog_to_llir(
                egraph,
                genome.clone(),
                ops,
                &cx.custom_ops,
                &mut list_cache,
                &mut expr_cache,
                None,
            );

            // Basic sanity check on extracted graph
            assert!(graph.node_count() > 0, "Extracted graph has no nodes");

            valid_count += 1;

            // Use the first valid offspring as base for next generation
            if valid_count == 1 {
                base = genome;
            }

            if valid_count >= target_count {
                break;
            }
        }

        if valid_count >= target_count {
            break;
        }
    }

    println!(
        "Fuzz test: validated {}/{} genomes successfully",
        valid_count, total_count
    );
    // If no mutable eclasses, only the initial genome exists, which is valid
    if mutable_eclasses == 0 {
        println!("Search space has only one valid graph (no mutable eclasses)");
    } else {
        assert!(
            valid_count > 0,
            "No valid genomes were generated despite having {} mutable eclasses",
            mutable_eclasses
        );
    }
}

/// More extensive fuzz test with execution validation
#[test]
fn fuzz_test_genome_execution() {
    use crate::egglog_utils::egglog_to_llir;

    // Build a simple graph where we can verify correctness
    let mut cx = Graph::new();
    let a = cx.tensor((2, 3));
    let b = cx.tensor((2, 3));
    let c = (a + b).relu().output();

    cx.build_search_space::<NativeRuntime>();
    let egraph = cx.egraph().unwrap();
    let ops = cx.egglog_ops().unwrap();

    let mut rng = rand::rng();
    let mut prev_selected: FxHashSet<u64> = FxHashSet::default();

    // Generate and test multiple genomes
    let initial = random_initial_choice(egraph, &mut rng);
    prev_selected.insert(hash_choice_set(&initial));

    let test_input_a = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0];
    let test_input_b = vec![0.5f32, 0.5, 0.5, 0.5, 0.5, 0.5];
    let expected: Vec<f32> = test_input_a
        .iter()
        .zip(&test_input_b)
        .map(|(x, y)| (x + y).max(0.0))
        .collect();

    let mut base = initial;
    let mut tested = 0;

    for _generation in 0..10 {
        let offspring = extract_generation(egraph, &base, 5, 2, &mut prev_selected, &mut rng);

        if offspring.is_empty() {
            break;
        }

        for genome in offspring {
            // Validate
            if let Err(e) = validate_choice_set(egraph, &genome, ops) {
                panic!("Invalid genome: {}", e);
            }

            // Extract and execute
            let mut list_cache = FxHashMap::default();
            let mut expr_cache = FxHashMap::default();
            let llir_graph = egglog_to_llir(
                egraph,
                genome.clone(),
                ops,
                &cx.custom_ops,
                &mut list_cache,
                &mut expr_cache,
                None,
            );

            let mut rt = NativeRuntime::default();
            rt.load_llir(&llir_graph);
            rt.set_data(a.id, test_input_a.clone());
            rt.set_data(b.id, test_input_b.clone());
            rt.execute(&cx.dyn_map);

            let result = rt.get_f32(c.id);
            assert_close(result, &expected);

            tested += 1;
            base = genome;

            if tested >= 20 {
                break;
            }
        }

        if tested >= 20 {
            break;
        }
    }

    // Count mutable eclasses
    let mutable_eclasses: usize = egraph
        .eclasses
        .iter()
        .filter(|(_, (label, enodes))| {
            (label.contains("IR") || label.contains("IList")) && enodes.len() > 1
        })
        .count();

    println!(
        "Execution test: verified {} genomes produce correct results",
        tested
    );
    if mutable_eclasses == 0 {
        println!("Search space has only one valid graph (no mutable eclasses)");
    } else {
        assert!(
            tested > 0,
            "No genomes were tested for execution despite having {} mutable eclasses",
            mutable_eclasses
        );
    }
}

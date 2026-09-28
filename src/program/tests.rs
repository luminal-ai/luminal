use std::path::PathBuf;

use rand::SeedableRng;

use super::*;
use crate::egglog_utils::LlirExtractor;
use crate::graph::LLIRGraph;
use crate::hlir::ReferenceRuntime;
use crate::prelude::*;

fn tmp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("luminal-program-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A chain of three identical bodies so the loop-rolling prepass has
/// something to roll, over a bucketed dim `s`.
fn build_graph(width: usize) -> (Graph, GraphTensor, GraphTensor) {
    let mut cx = Graph::new();
    let x = cx.tensor(('s', width));
    let out = x.exp2().sin().exp2().sin().exp2().sin().output();
    (cx, x, out)
}

fn options() -> CompileOptions {
    CompileOptions::default()
        .dim_buckets(
            's',
            &[DimBucket::new(1, 1), DimBucket::new(2, 4).representative(3)],
        )
        .search_log(false)
}

/// What a search would leave behind: one random genome per bucket, as the
/// reference runtime's `compile` picks, plus the LLIR it extracts to.
fn select_random(cx: &Graph, seed: u64) -> (ProgramSelection, Vec<BucketLLIR>) {
    let space = cx.search_space().expect("search space built");
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut selection = Vec::new();
    let mut llirs = Vec::new();
    for ctx in space.bucket_contexts(&cx.dyn_map) {
        let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
        let genome = extractor.random_indexed_choice(&mut rng);
        let packed = extractor.extract_indexed_packed(&genome, &space.custom_ops);
        let fingerprint = packed.fingerprint();
        selection.push(BucketSelection {
            space_bucket_index: ctx.index,
            bucket_indices: ctx.bucket_indices().clone(),
            representative_dyn_map: ctx.representative_dyn_map.clone(),
            intervals: ctx.bucket().intervals.clone(),
            choices: extractor.choice_entries(&genome),
            fingerprint,
        });
        llirs.push((
            ctx.bucket_indices().clone(),
            ctx.representative_dyn_map.clone(),
            unroll_packed_llir(packed),
        ));
    }
    (ProgramSelection { buckets: selection }, llirs)
}

fn run_reference(
    cx: &mut Graph,
    llir: &LLIRGraph,
    x: GraphTensor,
    out: GraphTensor,
    s: usize,
    width: usize,
) -> Vec<f32> {
    let mut rt = ReferenceRuntime::default();
    rt.load_llir(llir);
    cx.set_dim('s', s);
    let vals: Vec<f32> = (0..s * width).map(|i| (i as f32 * 0.37).sin()).collect();
    rt.set_data(x.id, vals);
    rt.execute(&cx.dyn_map);
    rt.get_f32(out.id).clone()
}

#[test]
fn save_and_load_reproduces_the_selection() {
    let dir = tmp_dir("roundtrip");
    let width = 4;
    let (mut a, xa, outa) = build_graph(width);
    a.build_search_space::<ReferenceRuntime>(options());
    let (selection, llirs_a) = select_random(&a, 7);
    let manifest = save_program(
        &dir,
        &a,
        &selection,
        &options(),
        serde_json::json!({ "device": "cpu" }),
        serde_json::json!({ "model": "toy" }),
    )
    .unwrap();
    assert_eq!(manifest.buckets.len(), 2);

    let (mut b, xb, outb) = build_graph(width);
    let loaded = b
        .load_program::<ReferenceRuntime>(&dir, &options())
        .unwrap();
    assert_eq!(loaded.manifest.app["model"], "toy");
    assert_eq!(loaded.manifest.backend["device"], "cpu");
    assert_eq!(loaded.bucket_llirs.len(), 2);
    for (la, lb) in llirs_a.iter().zip(&loaded.bucket_llirs) {
        assert_eq!(la.0, lb.0, "bucket indices");
        assert_eq!(la.1, lb.1, "representative dyn map");
        assert_eq!(la.2.node_count(), lb.2.node_count());
        assert_eq!(la.2.edge_count(), lb.2.edge_count());
    }
    // Same program, same numbers, in a graph the search never touched.
    let want = run_reference(&mut a, &llirs_a[1].2, xa, outa, 3, width);
    let got = run_reference(&mut b, &loaded.bucket_llirs[1].2, xb, outb, 3, width);
    assert_eq!(want, got);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn choice_entries_and_index_owned_choices_are_inverses() {
    let (mut cx, _, _) = build_graph(3);
    cx.build_search_space::<ReferenceRuntime>(options());
    let space = cx.search_space().unwrap();
    let ctx = &space.bucket_contexts(&cx.dyn_map)[1];
    let mut extractor = LlirExtractor::new(ctx.egraph(), &space.ops);
    let mut rng = rand::rngs::StdRng::seed_from_u64(3);
    let genome = extractor.random_indexed_choice(&mut rng);
    let entries = extractor.choice_entries(&genome);
    assert!(!entries.is_empty());
    let again = extractor.index_owned_choices(&entries).unwrap();
    assert_eq!(extractor.choice_entries(&again), entries);
    let fp_a = extractor
        .extract_indexed_packed(&genome, &space.custom_ops)
        .fingerprint();
    let fp_b = extractor
        .extract_indexed_packed(&again, &space.custom_ops)
        .fingerprint();
    assert_eq!(fp_a, fp_b);
}

#[test]
fn egraph_json_round_trip_keeps_every_eclass_and_its_node_order() {
    let (mut cx, _, _) = build_graph(3);
    cx.build_search_space::<ReferenceRuntime>(options());
    let egraph = &cx.search_space().unwrap().buckets[0].egraph;
    let json = serde_json::to_vec(egraph).unwrap();
    let back: SerializedEGraph = serde_json::from_slice(&json).unwrap();
    assert_eq!(back.roots, egraph.roots);
    assert_eq!(back.eclasses, egraph.eclasses);
    assert_eq!(back.enodes, egraph.enodes);
    assert_eq!(back.node_to_class, egraph.node_to_class);
}

#[test]
fn rolled_graph_hashes_identically_across_builds() {
    let (mut a, _, _) = build_graph(4);
    a.build_search_space::<ReferenceRuntime>(options());
    let (mut b, _, _) = build_graph(4);
    b.run_auto_loop_rolling_prepass(&options());
    assert_eq!(hlir_identity(&a), hlir_identity(&b));
    let (mut c, _, _) = build_graph(5);
    c.run_auto_loop_rolling_prepass(&options());
    assert_ne!(hlir_identity(&a).0, hlir_identity(&c).0);
}

#[test]
fn load_refuses_a_different_graph() {
    let dir = tmp_dir("othergraph");
    let (mut a, _, _) = build_graph(4);
    a.build_search_space::<ReferenceRuntime>(options());
    let (selection, _) = select_random(&a, 1);
    save_program(&dir, &a, &selection, &options(), Default::default(), Default::default()).unwrap();
    let (mut b, _, _) = build_graph(5);
    let err = b.load_program::<ReferenceRuntime>(&dir, &options()).err().unwrap();
    assert!(matches!(err, ProgramError::HlirMismatch { .. }), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_refuses_different_buckets_and_foreign_choices() {
    let dir = tmp_dir("buckets");
    let (mut a, _, _) = build_graph(4);
    a.build_search_space::<ReferenceRuntime>(options());
    let (mut selection, _) = select_random(&a, 2);
    save_program(&dir, &a, &selection, &options(), Default::default(), Default::default()).unwrap();

    let (mut b, _, _) = build_graph(4);
    let other = CompileOptions::default().dim_buckets('s', &[DimBucket::new(1, 8)]);
    let err = b.load_program::<ReferenceRuntime>(&dir, &other).err().unwrap();
    assert!(matches!(err, ProgramError::OptionsMismatch { .. }), "{err}");

    // A choice naming an e-node the e-graph does not have.
    let (class, _) = selection.buckets[0].choices[0].clone();
    selection.buckets[0].choices[0] = (class, serde_json::from_str::<NodeId>("\"no-such-node\"").unwrap());
    save_program(&dir, &a, &selection, &options(), Default::default(), Default::default()).unwrap();
    let (mut c, _, _) = build_graph(4);
    let err = c.load_program::<ReferenceRuntime>(&dir, &options()).err().unwrap();
    assert!(matches!(err, ProgramError::ChoiceMismatch { bucket: 0, .. }), "{err}");

    // A fingerprint that the stored choices do not reproduce.
    let (selection, _) = select_random(&a, 2);
    let mut tampered = selection.clone();
    tampered.buckets[1].fingerprint = selection.buckets[0].fingerprint;
    save_program(&dir, &a, &tampered, &options(), Default::default(), Default::default()).unwrap();
    let (mut d, _, _) = build_graph(4);
    let err = d.load_program::<ReferenceRuntime>(&dir, &options()).err().unwrap();
    assert!(matches!(err, ProgramError::FingerprintMismatch { bucket: 1, .. }), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn manifest_format_version_is_checked() {
    let dir = tmp_dir("format");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join(MANIFEST_FILE),
        serde_json::to_vec(&serde_json::json!({
            "format_version": PROGRAM_FORMAT_VERSION + 1,
            "luminal_version": "x", "hlir_hash": 0, "hlir_node_count": 0,
            "custom_op_count": 0, "op_names": [], "dim_buckets": {}, "search_dims": {},
            "buckets": []
        }))
        .unwrap(),
    )
    .unwrap();
    let err = read_manifest(&dir).err().unwrap();
    assert!(matches!(err, ProgramError::FormatVersion { .. }), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

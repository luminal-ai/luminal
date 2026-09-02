use super::{DimBucket, Graph, LlirFingerprint, fingerprint_llir, unroll_packed_llir};
use crate::{
    dtype::DType,
    egglog_utils::{LlirExtractor, SerializedEGraph},
    hlir::HLIROps,
    op::{IntoEgglogOp, Runtime},
    shape::{DynMap, Symbol},
};
use petgraph::stable_graph::NodeIndex;
use rustc_hash::FxHashMap;

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct ScheduleBucket {
    pub(super) egraph: SerializedEGraph,
    pub(super) choices: Vec<(String, String)>,
    pub(super) bucket_indices: DynMap,
    pub(super) representative_dyn_map: DynMap,
    pub(super) unrolled_llir_fingerprint: LlirFingerprint,
}

#[derive(Debug, Clone, serde::Deserialize, serde::Serialize)]
pub struct SelectedSchedule {
    pub(super) dim_buckets: FxHashMap<Symbol, Vec<DimBucket>>,
    pub(super) buckets: Vec<ScheduleBucket>,
}

impl Graph {
    pub fn selected_schedule(&self) -> Option<&SelectedSchedule> {
        self.selected_schedule.as_ref()
    }

    pub fn from_selected_schedule(
        dyn_map: DynMap,
        input_meta: FxHashMap<NodeIndex, (String, DType)>,
        schedule: SelectedSchedule,
    ) -> Self {
        Self {
            dyn_map,
            input_meta,
            selected_schedule: Some(schedule),
            ..Self::default()
        }
    }

    pub fn load_selected_schedule<R: Runtime + 'static>(
        &mut self,
        runtime: &mut R,
    ) -> Result<(), String> {
        let schedule = self
            .selected_schedule
            .as_ref()
            .ok_or_else(|| "graph has no selected schedule".to_string())?;
        if !self.custom_ops.is_empty() {
            return Err("selected schedules with custom ops are not serializable".to_string());
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut ops = R::Ops::into_vec();
            ops.extend(<HLIROps as IntoEgglogOp>::into_vec());
            let bucket_llirs = schedule
                .buckets
                .iter()
                .enumerate()
                .map(|(bucket_idx, bucket)| {
                    let mut extractor = LlirExtractor::new(&bucket.egraph, &ops);
                    let choices = extractor.index_named_choices(&bucket.choices);
                    let packed = extractor.extract_indexed_packed(&choices, &[], None);
                    let llir = unroll_packed_llir(packed);
                    let fingerprint = fingerprint_llir(&llir);
                    assert_eq!(
                        fingerprint, bucket.unrolled_llir_fingerprint,
                        "selected schedule bucket {bucket_idx} unrolled LLIR fingerprint mismatch: expected {:?}, got {:?}",
                        bucket.unrolled_llir_fingerprint, fingerprint,
                    );
                    (
                        bucket.bucket_indices.clone(),
                        bucket.representative_dyn_map.clone(),
                        llir,
                    )
                })
                .collect::<Vec<_>>();
            runtime.load_llir_buckets(&schedule.dim_buckets, &bucket_llirs);
        }));
        result.map_err(|payload| {
            let detail = payload
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| payload.downcast_ref::<&str>().copied())
                .unwrap_or("non-string panic");
            format!("selected schedule could not be loaded: {detail}")
        })
    }
}

use luminal::buffer_tensor_ir::{BufferTensorIrOp, OpSlotNames};
use luminal::layout_ir::{
    AliasInfo, Bufferizable, ExtractionSite, LayoutIrOp, OpMatcher, Sharing, ToDps,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexMapApplyView {
    pub entries: Option<Vec<luminal::index_expr::IotaExpr>>,
}

impl OpSlotNames for IndexMapApplyView {
    fn operand_name(&self, operand: usize) -> String {
        match operand {
            0 => "input".to_string(),
            _ => format!("in{operand}"),
        }
    }
}

impl BufferTensorIrOp for IndexMapApplyView {
    fn label(&self) -> &str {
        "IndexMapApplyViewGeneric"
    }

    fn operand_reads_memory(&self, _operand: usize) -> bool {
        false // metadata op: no bytes observed
    }
    fn result_writes_memory(&self, _result: usize) -> bool {
        false // metadata op: no bytes produced
    }

    fn view_index_map(&self, _result: usize) -> Option<Vec<luminal::index_expr::IotaExpr>> {
        self.entries.clone()
    }
}

impl Bufferizable for IndexMapApplyView {
    fn alias_info(&self) -> Vec<AliasInfo> {
        vec![AliasInfo {
            operand: 0,
            result: 0,
            sharing: Sharing::Must,
        }]
    }
}

impl ToDps for IndexMapApplyView {
    fn to_dps(&self) -> Option<Box<dyn LayoutIrOp>> {
        None // nothing is written: there is no destination to pass
    }
}

impl LayoutIrOp for IndexMapApplyView {}

#[derive(Debug, Clone, Copy, Default)]
pub struct IndexMapApplyViewMatcher;

impl OpMatcher for IndexMapApplyViewMatcher {
    fn egglog_constructor(&self) -> &'static str {
        "LayoutTensorOpIndexMapApplyViewGeneric"
    }

    fn snippets(&self) -> Vec<luminal::egglog_snippet::EgglogSnippet> {
        vec![
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::LayoutOpConstructors,
                text: include_str!("match_functional_constructor.egg"),
            },
            luminal::egglog_snippet::EgglogSnippet {
                category: luminal::egglog_snippet::SpliceCategory::Match,
                text: include_str!("match_functional.egg"),
            },
        ]
    }

    fn metadata_slots(&self) -> &'static [(&'static str, usize)] {
        &[("index_map", 1), ("shape", 2), ("out_layout", 3)]
    }

    fn extract(&self, site: &ExtractionSite<'_>) -> Box<dyn LayoutIrOp> {
        Box::new(IndexMapApplyView {
            entries: luminal::index_expr::parse_index_map_entries(site, 1, 2),
        })
    }
}

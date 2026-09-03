use luminal::prelude::*;

fn census(name: &str, build: impl Fn(&mut Graph)) {
    let mut cx = Graph::new();
    build(&mut cx);
    match cx.logical.model_text() {
        Ok(model) => {
            let rows = model.lines().filter(|l| !l.trim().is_empty()).count();
            let applies = model.matches("(LogicalIndexMapApply").count();
            // longest consecutive apply chain: walk let lines, track operand->depth
            let mut depth = std::collections::HashMap::new();
            let mut max_chain = 0usize;
            for line in model.lines() {
                if let Some(rest) = line.strip_prefix("(let v") {
                    let id: usize = rest.split_whitespace().next().unwrap().parse().unwrap();
                    let d = if rest.contains("(LogicalIndexMapApply v") {
                        let op: usize = rest
                            .split("(LogicalIndexMapApply v")
                            .nth(1)
                            .unwrap()
                            .split_whitespace()
                            .next()
                            .unwrap()
                            .parse()
                            .unwrap();
                        depth.get(&op).copied().unwrap_or(0) + 1
                    } else {
                        0
                    };
                    max_chain = max_chain.max(d);
                    depth.insert(id, d);
                }
            }
            println!("{name}: rows={rows} applies={applies} max_apply_chain={max_chain}");
        }
        Err(e) => println!("{name}: POISONED: {e}"),
    }
}

fn main() {
    census("scalar_broadcast_rank4", |cx| {
        let x = cx.tensor((2, 3, 4, 5), DType::F32);
        let _ = (x * 2.0f32).output();
    });
    census("stable_argsort_rank2", |cx| {
        let x = cx.tensor((4, 8), DType::F32);
        let _ = x.stable_argsort(1, false).output();
    });
    census("topk_rank3", |cx| {
        let x = cx.tensor((2, 4, 8), DType::F32);
        let _ = x.topk_indexes(2, 2).output();
    });
    census("gather_elements_rank3", |cx| {
        let x = cx.tensor((2, 4, 8), DType::F32);
        let idx = cx.tensor((2, 4, 3), DType::Int);
        let _ = x.gather_elements(idx, 2).output();
    });
    census("concat_rank3", |cx| {
        let a = cx.tensor((2, 4, 8), DType::F32);
        let b = cx.tensor((2, 4, 8), DType::F32);
        let _ = a.concat_along(b, 2).output();
    });
    census("cumsum_rank4", |cx| {
        let x = cx.tensor((2, 3, 4, 5), DType::F32);
        let _ = x.cumsum(3).output();
    });
    census("expand_pytorch_rank4", |cx| {
        let x = cx.tensor((2, 1, 4, 1), DType::F32);
        let _ = x.expand((2, 3, 4, 5)).output();
    });
}

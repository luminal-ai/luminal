use std::{f32, path::PathBuf};

use colored::Colorize;
use itertools::Itertools;

use crate::{
    op::{self, Constant, ConstantValue},
    prelude::*,
};

impl GraphTensor {
    /// Cumulative sum last dimension
    pub fn cumsum_last_dim(mut self) -> Self {
        let axis = self.shape.len() - 1;
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        // Pad out length
        let orig_length = self.dims()[axis];
        self = self.pad_along(orig_length - 1, 0, axis).contiguous();

        // Pool
        self = self.pool_last_dim(orig_length, 1, 1);

        // Sum Reduce along new dimension
        self.sum(axis + 1)
    }

    /// Cumulative max last dimension
    pub fn cummax_last_dim(mut self) -> Self {
        let axis = self.shape.len() - 1;
        if !self.shape.is_contiguous() {
            self = self.contiguous();
        }
        // Pad out length
        let orig_length = self.dims()[axis];
        self.shape.padding[self.shape.indexes[axis]].0 = orig_length - 1;
        self = self.contiguous();

        // Pool
        self = self.pool_last_dim(orig_length, 1, 1);
        // Max Reduce along new dimension
        self.max(axis + 1)
    }

    /// Cumulative product last dimension
    pub fn cumprod_last_dim(self) -> Self {
        self.log().cumsum_last_dim().exp()
    }
}

impl Graph {
    /// A scalar constant
    pub fn constant(&mut self, i: impl Into<ConstantValue>) -> GraphTensor {
        GraphTensor::from_id(
            self.add_op(Constant(i.into(), &self.dyn_map)).finish(),
            ShapeTracker::new(()),
            self,
        )
    }

    /// ARange from 0 to N
    pub fn arange(&mut self, to: impl Into<Expression>) -> GraphTensor {
        let to = to.into();
        if to.to_usize().map(|i| i == 1).unwrap_or_default() {
            // Single number ARange is just 0
            self.constant(0.).expand_dim(0, to)
        } else {
            self.constant(1.).expand_dim(0, to).cumsum_last_dim() - 1.
        }
    }

    /// ARange from beg to end
    pub fn arange_in_range(&mut self, beg: usize, end: usize) -> GraphTensor {
        self.arange(end - beg) + beg
    }

    /// ARange from beg to end
    pub fn arange_step(&mut self, beg: f32, end: f32, step: f32) -> GraphTensor {
        assert!(step > 0.0, "step must be positive");

        let num_steps = ((end - beg) / step).ceil() as usize;

        let mut tensor = self.arange(num_steps);
        tensor = tensor * step + beg;
        tensor
    }

    /// Lower left-hand triangle of 1s. Currently required to be square
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.tril
    pub fn tril(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size);
        let vertical = self.arange(size).expand_dim(1, size);

        (horizontal - (diagonal as f32 + 1.)).lt(vertical)
    }

    /// Upper right-hand triangle of 1s
    ///
    /// Same API as https://pytorch.org/docs/stable/generated/torch.triu
    pub fn triu(&mut self, size: impl Into<Expression>, diagonal: i32) -> GraphTensor {
        let size = size.into();
        let horizontal = self.arange(size).expand_dim(0, size).contiguous();
        let vertical = self.arange(size).expand_dim(1, size).contiguous();

        (horizontal - (diagonal as f32 - 1.)).gt(vertical)
    }
}

impl GraphTensor {
    /// Gather a batch of vectors from a matrix
    pub fn gather(self, indexes: GraphTensor) -> GraphTensor {
        let (vocab, dim) = self.dims2();
        let batch = indexes.dims1();
        let one_hot = indexes
            .graph()
            .arange(vocab)
            .expand_dim(0, batch)
            .eq(indexes.expand_dim(1, vocab));
        (one_hot.expand_dim(2, dim) * self.expand_dim(0, batch)).sum(1)
    }

    /// Print the value of this tensor when the graph is ran
    pub fn print<T: ToString>(&self, message: T) -> Self {
        let message = message.to_string();
        let id = self
            .graph()
            .add_op(op::Function(
                "Print".to_string(),
                Box::new(move |inp| {
                    for (i, (tensor, tracker)) in inp.iter().enumerate() {
                        println!("{message} ({})", i + 1);
                        let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                        println!(
                            "Elements: {} Start: {:?} Mid: {:?} End: {:?}",
                            d.len(),
                            &d[..d.len().min(5)],
                            &d[d.len().saturating_div(2)
                                ..(d.len().saturating_div(2) + 5).min(d.len())],
                            &d[d.len().saturating_sub(5)..]
                        );
                        println!("Shape: {tracker:?}");
                    }
                    vec![]
                }),
            ))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
        *self
    }

    /// Check the tensor value against a binary file
    pub fn diff(&self, file: impl Into<PathBuf>, atol: f32, rtol: f32) -> Self {
        let path = file.into();
        let id = self
            .graph()
            .add_op(op::Function(
                format!("Diff {path:?}"),
                Box::new(move |mut inp| {
                    // Get tensor data and file data
                    let (tensor, shape) = inp.pop().unwrap();
                    let d = tensor.borrowed().downcast_ref::<Vec<f32>>().unwrap();
                    let mut data = vec![0.; d.len()];
                    let (ind, val) = (shape.index_expression(), shape.valid_expression());
                    let mut stack = vec![];
                    #[allow(unused_mut)]
                    for (i, mut r) in data.iter_mut().enumerate() {
                        if val.exec_single_var_stack(i, &mut stack) != 0 {
                            *r = d[ind.exec_single_var_stack(i, &mut stack)];
                        }
                    }
                    let Ok(bytes) = std::fs::read(&path) else {
                        return vec![];
                    };
                    let bin_data = bytes
                        .chunks(4)
                        .map(|i| {
                            f32::from_ne_bytes([i[0], i[1], i[2], i[3]]).clamp(f32::MIN, f32::MAX)
                        })
                        .collect::<Vec<_>>();
                    if data.len() != bin_data.len() {
                        println!(
                            "{}",
                            format!(
                                "{} | Length mismatch! Data: {}, File: {}",
                                path.as_os_str().to_str().unwrap(),
                                data.len(),
                                bin_data.len()
                            )
                            .bold()
                            .red()
                        );
                        println!("Data Shape: {shape:?}");
                        return vec![];
                    }
                    let data_nan = data.iter().any(|i| i.is_nan());
                    let file_nan = bin_data.iter().any(|i| i.is_nan());
                    if data_nan {
                        println!(
                            "{}",
                            format!("{} | Data contains nan!", path.to_str().unwrap())
                                .bold()
                                .red()
                        );
                    }
                    if file_nan {
                        println!(
                            "{}",
                            format!("{} | File contains nan!", path.to_str().unwrap())
                                .bold()
                                .red()
                        );
                    }
                    if data_nan || file_nan {
                        return vec![];
                    }
                    let mut matched = true;
                    for (i, (a, b)) in data.iter().zip(bin_data.iter()).enumerate() {
                        let tolerance = atol + rtol * a.abs().max(b.abs());
                        if (a - b).abs() > tolerance {
                            println!(
                                "{}",
                                format!("{} | Value Mismatch!", path.to_str().unwrap())
                                    .bold()
                                    .red()
                            );
                            if let Some((i, _)) = data.iter().enumerate().find(|(_, i)| i.is_nan())
                            {
                                println!("Index {} is nan!", i.to_string().bold());
                            }
                            println!("{a} is not equal to {b}, index {i}");

                            let mut diffs: Vec<f32> = data
                                .iter()
                                .zip(bin_data.iter())
                                .map(|(a, b)| (a - b).abs())
                                .collect();
                            diffs.sort_by(|x, y| {
                                x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            let len = diffs.len();
                            // percentile indices (clamp to len-1)
                            let p50_idx = ((len as f32) * 0.50).round() as usize;
                            let p95_idx = ((len as f32) * 0.95).round() as usize;
                            let p99_idx = ((len as f32) * 0.99).round() as usize;
                            let p50 = diffs[p50_idx.min(len - 1)];
                            let p95 = diffs[p95_idx.min(len - 1)];
                            let p99 = diffs[p99_idx.min(len - 1)];

                            // summary stats
                            let avg_dist = diffs.iter().sum::<f32>() / len as f32;
                            let max_dist = *diffs.last().unwrap();
                            let sum_dist = data
                                .iter()
                                .zip(bin_data.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f32>();

                            println!(
                                "Avg dist: {}, Max dist: {} Sum dist: {}",
                                avg_dist.to_string().bold().red(),
                                max_dist.to_string().bold().red(),
                                sum_dist.to_string().bold().red(),
                            );
                            println!(
                                "p50: {}  p95: {}  p99: {}",
                                p50.to_string().bold().red(),
                                p95.to_string().bold().red(),
                                p99.to_string().bold().red(),
                            );

                            println!("Data Shape: {shape:?}");
                            println!("{}: {:?}", "This".bold(), &data[..10]);
                            println!("{}: {:?}", "File".bold(), &bin_data[..10]);
                            println!(
                                "Largest Mismatches: {:?}",
                                data.iter()
                                    .zip(bin_data.iter())
                                    .filter(|(a, b)| (**a - **b).abs() > 0.01)
                                    .sorted_by(|(a, b), (c, d)| (**c - **d)
                                        .abs()
                                        .partial_cmp(&(**a - **b).abs())
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .take(10)
                                    .collect::<Vec<_>>()
                            );
                            println!(
                                "A avg: {} B avg: {}",
                                data.iter().sum::<f32>() / data.len() as f32,
                                bin_data.iter().sum::<f32>() / bin_data.len() as f32
                            );
                            println!(
                                "A max: {} B max: {}",
                                data.iter()
                                    .max_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap(),
                                bin_data
                                    .iter()
                                    .max_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap()
                            );
                            println!(
                                "A min: {} B min: {}",
                                data.iter()
                                    .min_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap(),
                                bin_data
                                    .iter()
                                    .min_by(|a, b| a
                                        .partial_cmp(b)
                                        .unwrap_or(std::cmp::Ordering::Equal))
                                    .unwrap()
                            );
                            matched = false;
                            break;
                        }
                    }
                    if matched {
                        println!(
                            "{}",
                            format!("{} matched", path.to_str().unwrap())
                                .bold()
                                .bright_green()
                        );
                    }
                    vec![]
                }),
            ))
            .input(self.id, 0, self.shape)
            .finish();
        self.graph().no_delete.insert(id);
        *self
    }
}

#[cfg(test)]
mod tests {
    crate::test_imports!();
    #[test]
    fn test_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange(10).retrieve();
        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]);
    }

    #[test]
    fn test_arange_from_zero() {
        let mut cx = Graph::new();

        let tensor = cx.arange(5).retrieve();
        cx.execute();

        assert_eq!(tensor.data(), vec![0., 1., 2., 3., 4.]);
    }

    #[test]
    fn test_arange_in_range() {
        let mut cx = Graph::new();

        let tensor = cx.arange_in_range(3, 8).retrieve();
        cx.execute();

        assert_eq!(tensor.data(), vec![3., 4., 5., 6., 7.]);
    }

    #[test]
    fn test_arange_step_simple() {
        let mut cx = Graph::new();

        let tensor = cx.arange_step(1.0, 5.0, 1.0).retrieve();
        cx.execute();

        assert_eq!(tensor.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_arange_step_fractional() {
        let mut cx = Graph::new();

        let tensor = cx.arange_step(0.0, 1.0, 0.3).retrieve();
        cx.execute();

        // Should produce [0.0, 0.3, 0.6, 0.9] — note that 1.2 would be >= 1.0 so we stop before that.
        let expected = &[0.0, 0.3, 0.6, 0.9];

        // Floating point comparison with tolerance:
        assert_eq!(tensor.data().len(), expected.len());
        for (v, e) in tensor.data().iter().zip(expected.iter()) {
            assert!((v - e).abs() < 1e-5, "Expected {e}, got {v}");
        }
    }

    #[test]
    #[should_panic(expected = "step must be positive")]
    fn test_arange_step_zero_step_panics() {
        let mut cx = Graph::new();

        // Should panic because step is zero
        cx.arange_step(0.0, 5.0, 0.0);
    }

    #[test]
    fn test_cumprod() {
        let mut cx = Graph::new();

        let a = cx.tensor(3).set(vec![3., 2., 5.]);
        let b = a.cumprod_last_dim().retrieve();
        cx.execute();

        assert_close(&b.data(), &[3., 6., 30.]);
    }

    #[test]
    fn test_gather() {
        let mut cx = Graph::new();

        let matrix = cx.tensor((3, 2)).set(vec![1., 2., 3., 4., 5., 6.]);
        let indexes = cx.tensor(2).set(vec![2., 0.]);
        let result = matrix.gather(indexes).retrieve();

        cx.execute();

        assert_exact(&result.data(), &[5., 6., 1., 2.]);
    }

    #[test]
    fn test_dyn_arange() {
        let mut cx = Graph::new();

        let arange = cx.arange('a').retrieve();
        cx.set_dyn_dim('a', 6);

        cx.execute();

        assert_exact(&arange.data(), &[0., 1., 2., 3., 4., 5.]);
    }

    #[test]
    fn test_tril() {
        let mut cx = Graph::new();

        let triangle = cx.tril(5, 1).retrieve();

        cx.execute();

        assert_exact(
            &triangle.data(),
            &[
                [1.00, 1.00, 0.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 0.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 0.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
                [1.00, 1.00, 1.00, 1.00, 1.00],
            ]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_triu() {
        let mut cx = Graph::new();

        let a = cx.triu(3, -1).retrieve();
        let b = cx.triu(3, 0).retrieve();
        let c = cx.triu(3, 1).retrieve();

        cx.execute();

        assert_exact(
            &a.data(),
            &[[1.00, 1.00, 1.00], [1.00, 1.00, 1.00], [0.00, 1.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &b.data(),
            &[[1.00, 1.00, 1.00], [0.00, 1.00, 1.00], [0.00, 0.00, 1.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
        assert_exact(
            &c.data(),
            &[[0.00, 1.00, 1.00], [0.00, 0.00, 1.00], [0.00, 0.00, 0.00]]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>(),
        );
    }
}

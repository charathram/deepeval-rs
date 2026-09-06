//! Evaluation metrics for deepeval-rs.
//!
//! This module defines the [`Metric`] trait that all metrics implement, plus
//! shared configuration and result types. Concrete metrics land in Phases 3-5.

mod base;

pub use base::{MetricConfig, MetricResult};

use async_trait::async_trait;

/// The base metric trait.
///
/// A metric measures a single [`LLMTestCase`](crate::test_case::LLMTestCase)
/// and records its score, reason, and pass/fail status on itself. Metrics can
/// be cloned via [`Metric::clone_box`] so an evaluation can measure each
/// `(test_case, metric)` pair independently.
///
/// This trait is object-safe (`async_trait` + `Send + Sync`) so a
/// heterogeneous set of metrics can be held behind `Box<dyn Metric>`.
#[async_trait]
pub trait Metric: Send + Sync {
    /// The name of this metric.
    fn name(&self) -> &str;

    /// The pass/fail threshold, if any.
    fn threshold(&self) -> Option<f32>;

    /// Measure the given test case, recording the result on `self`.
    ///
    /// Implementations set `score`, `reason`, and `success` on `self`. They
    /// return `Err` only on a hard failure (e.g. a missing required field or
    /// an LLM error); a below-threshold score is not an error.
    async fn measure(
        &mut self,
        test_case: &crate::test_case::LLMTestCase,
    ) -> Result<(), crate::error::MetricError>;

    /// Whether the metric passed, based on the recorded score and threshold.
    ///
    /// Returns `None` when no threshold is configured or no score was recorded.
    fn is_successful(&self) -> Option<bool> {
        match (self.threshold(), self.score()) {
            (Some(threshold), Some(score)) => Some(score >= threshold),
            _ => None,
        }
    }

    /// The recorded score, if any.
    fn score(&self) -> Option<f32>;

    /// The recorded reason/explanation, if any.
    fn reason(&self) -> Option<&str>;

    /// Clone this metric as a boxed trait object.
    fn clone_box(&self) -> Box<dyn Metric>;
}

impl Clone for Box<dyn Metric> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

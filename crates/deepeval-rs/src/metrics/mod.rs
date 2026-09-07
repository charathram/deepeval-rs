//! Evaluation metrics for deepeval-rs.
//!
//! This module defines the [`Metric`] trait that all metrics implement, plus
//! shared configuration and result types, and the concrete metric set.

mod base;
mod llm_judge;

mod answer_relevancy;
mod contextual_precision;
mod contextual_recall;
mod contextual_relevancy;
mod exact_match;
mod faithfulness;
mod geval;
mod hallucination;
mod json_correctness;
mod pattern_match;
mod prompt_alignment;
mod ragas;

pub use answer_relevancy::AnswerRelevancyMetric;
pub(crate) use base::MetricState;
pub use base::{MetricConfig, MetricResult};
pub use contextual_precision::ContextualPrecisionMetric;
pub use contextual_recall::ContextualRecallMetric;
pub use contextual_relevancy::ContextualRelevancyMetric;
pub use exact_match::ExactMatchMetric;
pub use faithfulness::FaithfulnessMetric;
pub use geval::GEval;
pub use hallucination::HallucinationMetric;
pub use json_correctness::JsonCorrectnessMetric;
pub use pattern_match::PatternMatchMetric;
pub use prompt_alignment::PromptAlignmentMetric;
pub use ragas::RagasMetric;

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
    /// return `Err` only on a hard failure (e.g. an LLM error); a
    /// below-threshold score is not an error.
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

    /// Whether this metric was skipped (e.g. a required field was missing).
    ///
    /// Skipped metrics have no score and do not count as failures.
    fn skipped(&self) -> bool {
        false
    }

    /// Clone this metric as a boxed trait object.
    fn clone_box(&self) -> Box<dyn Metric>;
}

impl Clone for Box<dyn Metric> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Generate the boilerplate [`Metric`] implementation for a metric type.
///
/// The target type must expose:
/// - a `state: MetricState` field (for `threshold`, `score`, `reason`, `skipped`),
/// - a `Clone` implementation (for `clone_box`),
/// - an `async fn measure_impl(&mut self, &LLMTestCase) -> Result<(), MetricError>`
///   method (for `measure`).
macro_rules! impl_metric {
    ($ty:ty, $name:expr) => {
        #[async_trait::async_trait]
        impl $crate::metrics::Metric for $ty {
            fn name(&self) -> &str {
                $name
            }

            fn threshold(&self) -> Option<f32> {
                self.state.config.threshold
            }

            fn score(&self) -> Option<f32> {
                self.state.score
            }

            fn reason(&self) -> Option<&str> {
                if self.state.config.include_reason {
                    self.state.reason.as_deref()
                } else {
                    None
                }
            }

            fn skipped(&self) -> bool {
                self.state.skipped
            }

            fn clone_box(&self) -> Box<dyn $crate::metrics::Metric> {
                Box::new(self.clone())
            }

            async fn measure(
                &mut self,
                test_case: &$crate::test_case::LLMTestCase,
            ) -> Result<(), $crate::error::MetricError> {
                self.measure_impl(test_case).await
            }
        }
    };
}

pub(crate) use impl_metric;

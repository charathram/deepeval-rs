//! Evaluation metrics for deepeval-rs.
//!
//! This module defines the [`Metric`] trait that all metrics implement, plus
//! shared configuration and result types, and the concrete metric set.

mod answer_relevancy;
mod argument_correctness;
mod base;
mod contextual_precision;
mod contextual_recall;
mod contextual_relevancy;
mod conversation_completeness;
mod conversation_summary;
mod conversational_llm_judge;
mod exact_match;
mod faithfulness;
mod geval;
mod goal_accuracy;
mod hallucination;
mod json_correctness;
mod knowledge_retention;
mod llm_judge;
mod pattern_match;
mod plan_adherence;
mod plan_quality;
mod prompt_alignment;
mod ragas;
mod role_adherence;
mod step_efficiency;
mod task_completion;
mod tool_correctness;
mod tool_use;
mod turn_faithfulness;
mod turn_relevancy;

pub use answer_relevancy::{AnswerRelevancyMetric, AnswerRelevancyMetricBuilder};
pub use argument_correctness::ArgumentCorrectnessMetric;
pub(crate) use base::MetricState;
pub use base::{MetricConfig, MetricResult};
pub use contextual_precision::{ContextualPrecisionMetric, ContextualPrecisionMetricBuilder};
pub use contextual_recall::{ContextualRecallMetric, ContextualRecallMetricBuilder};
pub use contextual_relevancy::{ContextualRelevancyMetric, ContextualRelevancyMetricBuilder};
pub use conversation_completeness::ConversationCompletenessMetric;
pub use conversation_summary::ConversationSummaryMetric;
pub use exact_match::{ExactMatchMetric, ExactMatchMetricBuilder};
pub use faithfulness::{FaithfulnessMetric, FaithfulnessMetricBuilder};
pub use geval::{GEval, GEvalBuilder};
pub use goal_accuracy::GoalAccuracyMetric;
pub use hallucination::{HallucinationMetric, HallucinationMetricBuilder};
pub use json_correctness::{JsonCorrectnessMetric, JsonCorrectnessMetricBuilder};
pub use knowledge_retention::KnowledgeRetentionMetric;
pub use pattern_match::{PatternMatchMetric, PatternMatchMetricBuilder};
pub use plan_adherence::PlanAdherenceMetric;
pub use plan_quality::PlanQualityMetric;
pub use prompt_alignment::{PromptAlignmentMetric, PromptAlignmentMetricBuilder};
pub use ragas::{RagasMetric, RagasMetricBuilder};
pub use role_adherence::RoleAdherenceMetric;
pub use step_efficiency::StepEfficiencyMetric;
pub use task_completion::TaskCompletionMetric;
pub use tool_correctness::ToolCorrectnessMetric;
pub use tool_use::ToolUseMetric;
pub use turn_faithfulness::TurnFaithfulnessMetric;
pub use turn_relevancy::TurnRelevancyMetric;

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

    /// The estimated cost of the LLM calls made while measuring, if known.
    fn cost(&self) -> Option<f64> {
        None
    }

    /// The number of input tokens used while measuring.
    fn input_tokens(&self) -> u32 {
        0
    }

    /// The number of output tokens used while measuring.
    fn output_tokens(&self) -> u32 {
        0
    }

    /// Clone this metric as a boxed trait object.
    fn clone_box(&self) -> Box<dyn Metric>;
}

impl Clone for Box<dyn Metric> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// The base trait for multi-turn (conversational) metrics.
///
/// A conversational metric measures a single
/// [`ConversationalTestCase`](crate::test_case::ConversationalTestCase) — a
/// sequence of [`Turn`](crate::test_case::Turn)s — and records its score,
/// reason, and pass/fail status on itself. It mirrors [`Metric`] for the
/// multi-turn case.
///
/// This trait is object-safe (`async_trait` + `Send + Sync`) so a
/// heterogeneous set of metrics can be held behind `Box<dyn ConversationalMetric>`.
#[async_trait]
pub trait ConversationalMetric: Send + Sync {
    /// The name of this metric.
    fn name(&self) -> &str;

    /// The pass/fail threshold, if any.
    fn threshold(&self) -> Option<f32>;

    /// Measure the given conversational test case, recording the result on `self`.
    async fn measure(
        &mut self,
        test_case: &crate::test_case::ConversationalTestCase,
    ) -> Result<(), crate::error::MetricError>;

    /// Whether the metric passed, based on the recorded score and threshold.
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
    fn skipped(&self) -> bool {
        false
    }

    /// The estimated cost of the LLM calls made while measuring, if known.
    fn cost(&self) -> Option<f64> {
        None
    }

    /// The number of input tokens used while measuring.
    fn input_tokens(&self) -> u32 {
        0
    }

    /// The number of output tokens used while measuring.
    fn output_tokens(&self) -> u32 {
        0
    }

    /// Clone this metric as a boxed trait object.
    fn clone_box(&self) -> Box<dyn ConversationalMetric>;
}

impl Clone for Box<dyn ConversationalMetric> {
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

            fn cost(&self) -> Option<f64> {
                self.state.cost
            }

            fn input_tokens(&self) -> u32 {
                self.state.input_tokens
            }

            fn output_tokens(&self) -> u32 {
                self.state.output_tokens
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

/// Generate the boilerplate [`ConversationalMetric`] implementation for a
/// multi-turn metric type.
///
/// The target type must expose:
/// - a `state: MetricState` field (for `threshold`, `score`, `reason`, `skipped`),
/// - a `Clone` implementation (for `clone_box`),
/// - an `async fn measure_impl(&mut self, &ConversationalTestCase)
///   -> Result<(), MetricError>` method (for `measure`).
macro_rules! impl_conversational_metric {
    ($ty:ty, $name:expr) => {
        #[async_trait::async_trait]
        impl $crate::metrics::ConversationalMetric for $ty {
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

            fn cost(&self) -> Option<f64> {
                self.state.cost
            }

            fn input_tokens(&self) -> u32 {
                self.state.input_tokens
            }

            fn output_tokens(&self) -> u32 {
                self.state.output_tokens
            }

            fn clone_box(&self) -> Box<dyn $crate::metrics::ConversationalMetric> {
                Box::new(self.clone())
            }

            async fn measure(
                &mut self,
                test_case: &$crate::test_case::ConversationalTestCase,
            ) -> Result<(), $crate::error::MetricError> {
                self.measure_impl(test_case).await
            }
        }
    };
}

pub(crate) use impl_conversational_metric;

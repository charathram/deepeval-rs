//! RAGAS: a composite metric averaging answer relevancy, faithfulness,
//! contextual precision, and contextual recall.

use super::answer_relevancy::AnswerRelevancyMetric;
use super::contextual_precision::ContextualPrecisionMetric;
use super::contextual_recall::ContextualRecallMetric;
use super::faithfulness::FaithfulnessMetric;
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::metrics::Metric;
use crate::test_case::LLMTestCase;

/// RagasMetric: the average of four RAG metrics — answer relevancy,
/// faithfulness, contextual precision, and contextual recall.
///
/// Requires the union of the sub-metrics' required fields: `input`,
/// `actual_output`, `expected_output`, and `retrieval_context`. If any
/// sub-metric is skipped (a required field is missing), the whole metric is
/// skipped.
#[derive(Debug, Clone)]
pub struct RagasMetric {
    state: MetricState,
    answer_relevancy: AnswerRelevancyMetric,
    faithfulness: FaithfulnessMetric,
    contextual_precision: ContextualPrecisionMetric,
    contextual_recall: ContextualRecallMetric,
}

impl RagasMetric {
    /// Start building a [`RagasMetric`].
    pub fn builder() -> RagasMetricBuilder {
        RagasMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        self.answer_relevancy.measure(test_case).await?;
        self.faithfulness.measure(test_case).await?;
        self.contextual_precision.measure(test_case).await?;
        self.contextual_recall.measure(test_case).await?;

        let sub_metrics: [&dyn Metric; 4] = [
            &self.answer_relevancy,
            &self.faithfulness,
            &self.contextual_precision,
            &self.contextual_recall,
        ];

        // If any sub-metric was skipped, the average is incomplete.
        if sub_metrics.iter().any(|m| m.skipped()) {
            self.state.skipped = true;
            return Ok(());
        }

        let scores: Vec<f32> = sub_metrics.iter().filter_map(|m| m.score()).collect();
        if scores.len() != sub_metrics.len() {
            self.state.skipped = true;
            return Ok(());
        }

        let average = scores.iter().sum::<f32>() / scores.len() as f32;
        self.state.score = Some(average);
        if self.state.config.include_reason {
            self.state.reason = Some(format!(
                "answer_relevancy={:.2}, faithfulness={:.2}, contextual_precision={:.2}, contextual_recall={:.2}",
                scores[0], scores[1], scores[2], scores[3]
            ));
        }
        Ok(())
    }
}

impl_metric!(RagasMetric, "RagasMetric");

/// Builder for [`RagasMetric`].
#[derive(Debug, Default)]
pub struct RagasMetricBuilder {
    config: MetricConfig,
    provider: Option<super::llm_judge::Provider>,
}

impl RagasMetricBuilder {
    /// Set the LLM provider used by all four sub-metrics.
    pub fn provider<P>(mut self, provider: P) -> Self
    where
        P: crate::llm::LlmProvider + 'static,
    {
        self.provider = Some(super::llm_judge::Provider::new(provider));
        self
    }

    /// Set the pass/fail threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = Some(threshold);
        self
    }

    /// Set whether to include a reason.
    pub fn include_reason(mut self, include_reason: bool) -> Self {
        self.config.include_reason = include_reason;
        self
    }

    /// Set strict mode.
    pub fn strict_mode(mut self, strict_mode: bool) -> Self {
        self.config.strict_mode = strict_mode;
        self
    }

    /// Build the metric.
    ///
    /// # Panics
    ///
    /// Panics if no provider was set.
    pub fn build(self) -> RagasMetric {
        let provider = self.provider.expect("RagasMetric requires an LLM provider");
        RagasMetric {
            state: MetricState::new(self.config),
            answer_relevancy: AnswerRelevancyMetric::builder()
                .provider(provider.clone())
                .build(),
            faithfulness: FaithfulnessMetric::builder()
                .provider(provider.clone())
                .build(),
            contextual_precision: ContextualPrecisionMetric::builder()
                .provider(provider.clone())
                .build(),
            contextual_recall: ContextualRecallMetric::builder().provider(provider).build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::MockLlmProvider;
    use crate::metrics::Metric;

    fn case() -> LLMTestCase {
        LLMTestCase::builder()
            .input("What is the refund policy?")
            .actual_output("We offer a 30-day full refund.")
            .expected_output("We offer a 30-day full refund at no extra costs.")
            .retrieval_context(vec![
                "All customers are eligible for a 30-day full refund.".to_string()
            ])
            .build()
    }

    #[tokio::test]
    async fn averages_sub_metric_scores() {
        // One verdict per sub-metric, in order: answer relevancy, faithfulness,
        // contextual precision, contextual recall.
        let provider = MockLlmProvider::responses([
            crate::llm::LlmResponse::new(r#"{"score": 0.8}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.9}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.7}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.6}"#),
        ]);
        let mut metric = RagasMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        // (0.8 + 0.9 + 0.7 + 0.6) / 4 = 0.75
        assert_eq!(metric.score(), Some(0.75));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn fails_when_average_below_threshold() {
        let provider = MockLlmProvider::responses([
            crate::llm::LlmResponse::new(r#"{"score": 0.4}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.5}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.3}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.2}"#),
        ]);
        let mut metric = RagasMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        // (0.4 + 0.5 + 0.3 + 0.2) / 4 = 0.35
        let score = metric.score().unwrap();
        assert!((score - 0.35).abs() < 1e-6, "expected ~0.35, got {score}");
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_expected_output_missing() {
        let provider = MockLlmProvider::responses([
            crate::llm::LlmResponse::new(r#"{"score": 0.8}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.9}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.7}"#),
            crate::llm::LlmResponse::new(r#"{"score": 0.6}"#),
        ]);
        let mut metric = RagasMetric::builder().provider(provider).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .retrieval_context(vec!["context".to_string()])
            .build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

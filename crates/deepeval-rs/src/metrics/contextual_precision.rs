//! Contextual precision: an LLM-judge metric measuring how well the retrieval
//! context ranks the relevant information needed to answer the input.

use std::collections::HashMap;

use super::llm_judge::{measure_llm_judge, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "ContextualPrecisionMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/contextual_precision/generate_verdict.txt");

/// ContextualPrecisionMetric: scores how precisely the retrieval context ranks
/// the relevant information needed to answer the input (relevant chunks ranked
/// early score higher).
///
/// Requires `input`, `actual_output`, and `retrieval_context` on the test case.
#[derive(Debug, Clone)]
pub struct ContextualPrecisionMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl ContextualPrecisionMetric {
    /// Start building a [`ContextualPrecisionMetric`].
    pub fn builder() -> ContextualPrecisionMetricBuilder {
        ContextualPrecisionMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        match measure_llm_judge(
            &[
                SingleTurnParams::Input,
                SingleTurnParams::ActualOutput,
                SingleTurnParams::RetrievalContext,
            ],
            self.provider.as_provider(),
            &self.registry,
            CLASS_NAME,
            METHOD,
            &HashMap::new(),
            test_case,
        )
        .await?
        {
            MeasureOutcome::Skipped => self.state.skipped = true,
            MeasureOutcome::Scored { score, reason } => {
                self.state.score = Some(score);
                self.state.reason = reason;
            }
        }
        Ok(())
    }
}

impl_metric!(ContextualPrecisionMetric, "ContextualPrecisionMetric");

/// Builder for [`ContextualPrecisionMetric`].
#[derive(Debug, Default)]
pub struct ContextualPrecisionMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl ContextualPrecisionMetricBuilder {
    /// Set the LLM provider used to judge the output.
    pub fn provider<P>(mut self, provider: P) -> Self
    where
        P: crate::llm::LlmProvider + 'static,
    {
        self.provider = Some(Provider::new(provider));
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
    pub fn build(self) -> ContextualPrecisionMetric {
        let provider = self
            .provider
            .expect("ContextualPrecisionMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        ContextualPrecisionMetric {
            state: MetricState::new(self.config),
            provider,
            registry: std::sync::Arc::new(registry),
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
            .retrieval_context(vec![
                "All customers are eligible for a 30-day full refund.".to_string(),
                "Shipping is free over $50.".to_string(),
            ])
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_relevant_ranked_first() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "relevant first"}"#);
        let mut metric = ContextualPrecisionMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_relevant_ranked_last() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "relevant last"}"#);
        let mut metric = ContextualPrecisionMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.2));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_retrieval_context_missing() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9}"#);
        let mut metric = ContextualPrecisionMetric::builder()
            .provider(provider)
            .build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

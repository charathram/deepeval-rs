//! Answer relevancy: an LLM-judge metric measuring how relevant the actual
//! output is to the input.

use std::collections::HashMap;

use super::llm_judge::{measure_llm_judge, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "AnswerRelevancyMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/answer_relevancy/generate_verdict.txt");

/// AnswerRelevancyMetric: scores how relevant the actual output is to the input.
///
/// Requires `input` and `actual_output` on the test case.
#[derive(Debug, Clone)]
pub struct AnswerRelevancyMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl AnswerRelevancyMetric {
    /// Start building an [`AnswerRelevancyMetric`].
    pub fn builder() -> AnswerRelevancyMetricBuilder {
        AnswerRelevancyMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        match measure_llm_judge(
            &[SingleTurnParams::Input, SingleTurnParams::ActualOutput],
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
            MeasureOutcome::Scored {
                score,
                reason,
                usage,
            } => {
                self.state.score = Some(score);
                self.state.reason = reason;
                self.state.accrue_usage(&usage);
            }
        }
        Ok(())
    }
}

impl_metric!(AnswerRelevancyMetric, "AnswerRelevancyMetric");

/// Builder for [`AnswerRelevancyMetric`].
#[derive(Debug, Default)]
pub struct AnswerRelevancyMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl AnswerRelevancyMetricBuilder {
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
    pub fn build(self) -> AnswerRelevancyMetric {
        let provider = self
            .provider
            .expect("AnswerRelevancyMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        AnswerRelevancyMetric {
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
            .input("What if these shoes don't fit?")
            .actual_output("We offer a 30-day full refund at no extra costs.")
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_relevant() {
        let provider = MockLlmProvider::text(r#"{"score": 0.85, "reason": "on-topic"}"#);
        let mut metric = AnswerRelevancyMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.85));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_irrelevant() {
        let provider = MockLlmProvider::text(r#"{"score": 0.1, "reason": "off-topic"}"#);
        let mut metric = AnswerRelevancyMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.1));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_input_missing() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9}"#);
        let mut metric = AnswerRelevancyMetric::builder().provider(provider).build();
        let tc = LLMTestCase::builder().actual_output("hi").build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

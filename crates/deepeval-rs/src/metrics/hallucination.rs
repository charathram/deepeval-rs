//! Hallucination: an LLM-judge metric measuring whether the actual output is
//! grounded in the retrieval context (higher score = less hallucination).

use std::collections::HashMap;

use super::llm_judge::{measure_llm_judge, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "HallucinationMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/hallucination/generate_verdict.txt");

/// HallucinationMetric: scores how grounded the actual output is in the
/// retrieval context. A higher score means less hallucination.
///
/// Requires `input`, `actual_output`, and `retrieval_context` on the test case.
#[derive(Debug, Clone)]
pub struct HallucinationMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl HallucinationMetric {
    /// Start building a [`HallucinationMetric`].
    pub fn builder() -> HallucinationMetricBuilder {
        HallucinationMetricBuilder::default()
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

impl_metric!(HallucinationMetric, "HallucinationMetric");

/// Builder for [`HallucinationMetric`].
#[derive(Debug, Default)]
pub struct HallucinationMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl HallucinationMetricBuilder {
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
    pub fn build(self) -> HallucinationMetric {
        let provider = self
            .provider
            .expect("HallucinationMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        HallucinationMetric {
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
                "All customers are eligible for a 30-day full refund.".to_string()
            ])
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_grounded() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "grounded"}"#);
        let mut metric = HallucinationMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_hallucinated() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "hallucinated"}"#);
        let mut metric = HallucinationMetric::builder()
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
        let mut metric = HallucinationMetric::builder().provider(provider).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

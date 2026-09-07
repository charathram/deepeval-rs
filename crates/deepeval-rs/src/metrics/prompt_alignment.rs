//! Prompt alignment: an LLM-judge metric measuring how well the actual output
//! aligns with the prompt and any stated criteria.

use std::collections::HashMap;

use super::llm_judge::{measure_llm_judge, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "PromptAlignmentMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/prompt_alignment/generate_verdict.txt");

/// PromptAlignmentMetric: scores how well the actual output aligns with the
/// prompt and any expected criteria.
///
/// Requires `input` and `actual_output` on the test case. `expected_criteria`
/// is optional and included in the prompt when present.
#[derive(Debug, Clone)]
pub struct PromptAlignmentMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl PromptAlignmentMetric {
    /// Start building a [`PromptAlignmentMetric`].
    pub fn builder() -> PromptAlignmentMetricBuilder {
        PromptAlignmentMetricBuilder::default()
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
            MeasureOutcome::Scored { score, reason } => {
                self.state.score = Some(score);
                self.state.reason = reason;
            }
        }
        Ok(())
    }
}

impl_metric!(PromptAlignmentMetric, "PromptAlignmentMetric");

/// Builder for [`PromptAlignmentMetric`].
#[derive(Debug, Default)]
pub struct PromptAlignmentMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl PromptAlignmentMetricBuilder {
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
    pub fn build(self) -> PromptAlignmentMetric {
        let provider = self
            .provider
            .expect("PromptAlignmentMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        PromptAlignmentMetric {
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
            .input("Summarize the refund policy in one sentence.")
            .actual_output("Customers get a 30-day full refund.")
            .expected_criteria(vec!["One sentence.".to_string()])
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_aligned() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "aligned"}"#);
        let mut metric = PromptAlignmentMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_misaligned() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "misaligned"}"#);
        let mut metric = PromptAlignmentMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.2));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_actual_output_missing() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9}"#);
        let mut metric = PromptAlignmentMetric::builder().provider(provider).build();
        let tc = LLMTestCase::builder().input("hi").build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

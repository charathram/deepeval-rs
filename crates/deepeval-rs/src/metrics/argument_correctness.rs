//! Argument correctness: an LLM-judge metric measuring the logical correctness
//! of the assistant's reasoning and arguments.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "ArgumentCorrectnessMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/argument_correctness/generate_verdict.txt");

/// ArgumentCorrectnessMetric: scores the logical correctness of the
/// assistant's reasoning and arguments.
///
/// Requires `input` and `actual_output` on at least one turn.
#[derive(Debug, Clone)]
pub struct ArgumentCorrectnessMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl ArgumentCorrectnessMetric {
    /// Start building a [`ArgumentCorrectnessMetric`].
    pub fn builder() -> ArgumentCorrectnessMetricBuilder {
        ArgumentCorrectnessMetricBuilder::default()
    }

    async fn measure_impl(
        &mut self,
        test_case: &ConversationalTestCase,
    ) -> Result<(), MetricError> {
        match measure_conversation_llm_judge(
            &[MultiTurnParams::Input, MultiTurnParams::ActualOutput],
            &self.provider,
            &self.registry,
            CLASS_NAME,
            METHOD,
            &HashMap::new(),
            test_case,
        )
        .await?
        {
            None => self.state.skipped = true,
            Some((verdict, response)) => {
                self.state.score = Some(verdict.score.clamp(0.0, 1.0));
                self.state.reason = verdict.reason;
                self.state.accrue_usage(&response);
            }
        }
        Ok(())
    }
}

impl_conversational_metric!(ArgumentCorrectnessMetric, "ArgumentCorrectnessMetric");

/// Builder for [`ArgumentCorrectnessMetric`].
#[derive(Debug, Default)]
pub struct ArgumentCorrectnessMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl ArgumentCorrectnessMetricBuilder {
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
    pub fn build(self) -> ArgumentCorrectnessMetric {
        let provider = self
            .provider
            .expect("ArgumentCorrectnessMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        ArgumentCorrectnessMetric {
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
    use crate::metrics::ConversationalMetric;
    use crate::test_case::Turn;

    fn case() -> ConversationalTestCase {
        ConversationalTestCase::builder()
            .turns(vec![
                Turn::builder()
                    .input("Why is caching effective here?")
                    .actual_output("Caching reduces repeated computation, so latency drops.")
                    .build(),
                Turn::builder()
                    .input("Any trade-offs?")
                    .actual_output("Staleness is the main trade-off.")
                    .build(),
            ])
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_argument_sound() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#);
        let mut metric = ArgumentCorrectnessMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_argument_wrong() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2}"#);
        let mut metric = ArgumentCorrectnessMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.2));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_no_required_fields() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9}"#);
        let mut metric = ArgumentCorrectnessMetric::builder()
            .provider(provider)
            .build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

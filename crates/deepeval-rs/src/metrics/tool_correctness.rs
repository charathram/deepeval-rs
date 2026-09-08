//! Tool correctness: an LLM-judge metric measuring whether the assistant called
//! the correct tools with the correct arguments across the multi-turn
//! conversation.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "ToolCorrectnessMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/tool_correctness/generate_verdict.txt");

/// ToolCorrectnessMetric: scores whether the assistant called the expected
/// tools with the correct arguments across the conversation.
///
/// Requires at least one turn with `input`, `actual_output`, and
/// `expected_tools`.
#[derive(Debug, Clone)]
pub struct ToolCorrectnessMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl ToolCorrectnessMetric {
    /// Start building a [`ToolCorrectnessMetric`].
    pub fn builder() -> ToolCorrectnessMetricBuilder {
        ToolCorrectnessMetricBuilder::default()
    }

    async fn measure_impl(
        &mut self,
        test_case: &ConversationalTestCase,
    ) -> Result<(), MetricError> {
        match measure_conversation_llm_judge(
            &[
                MultiTurnParams::Input,
                MultiTurnParams::ActualOutput,
                MultiTurnParams::ExpectedTools,
            ],
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

impl_conversational_metric!(ToolCorrectnessMetric, "ToolCorrectnessMetric");

/// Builder for [`ToolCorrectnessMetric`].
#[derive(Debug, Default)]
pub struct ToolCorrectnessMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl ToolCorrectnessMetricBuilder {
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
    pub fn build(self) -> ToolCorrectnessMetric {
        let provider = self
            .provider
            .expect("ToolCorrectnessMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        ToolCorrectnessMetric {
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
    use crate::test_case::{ConversationalTestCase, ToolCall, Turn};

    fn case() -> ConversationalTestCase {
        ConversationalTestCase::builder()
            .turn(
                Turn::builder()
                    .input("Find me the latest news on Rust.")
                    .actual_output("I searched for the latest Rust news.")
                    .expected_tools(vec![ToolCall::new("search", serde_json::json!({}))])
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Sum up the top stories.")
                    .actual_output("Here are the top stories.")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_tools_correct() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#);
        let mut metric = ToolCorrectnessMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_tools_wrong() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "bad"}"#);
        let mut metric = ToolCorrectnessMetric::builder()
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
        let mut metric = ToolCorrectnessMetric::builder().provider(provider).build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

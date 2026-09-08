//! Task completion: an LLM-judge metric measuring whether the assistant solved
//! the user's task across the multi-turn conversation.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "TaskCompletionMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/task_completion/generate_verdict.txt");

/// TaskCompletionMetric: scores how fully the assistant solved the user's task
/// across the conversation.
///
/// Requires at least one turn with `input` and `actual_output`.
#[derive(Debug, Clone)]
pub struct TaskCompletionMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl TaskCompletionMetric {
    /// Start building a [`TaskCompletionMetric`].
    pub fn builder() -> TaskCompletionMetricBuilder {
        TaskCompletionMetricBuilder::default()
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
            Some(verdict) => {
                self.state.score = Some(verdict.score.clamp(0.0, 1.0));
                self.state.reason = verdict.reason;
            }
        }
        Ok(())
    }
}

impl_conversational_metric!(TaskCompletionMetric, "TaskCompletionMetric");

/// Builder for [`TaskCompletionMetric`].
#[derive(Debug, Default)]
pub struct TaskCompletionMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl TaskCompletionMetricBuilder {
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
    pub fn build(self) -> TaskCompletionMetric {
        let provider = self
            .provider
            .expect("TaskCompletionMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        TaskCompletionMetric {
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
    use crate::test_case::{ConversationalTestCase, Turn};

    fn case() -> ConversationalTestCase {
        ConversationalTestCase::builder()
            .turn(
                Turn::builder()
                    .input("Book me a flight to Paris.")
                    .actual_output("I booked your flight to Paris.")
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Great, thanks!")
                    .actual_output("You're welcome!")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_task_completed() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#);
        let mut metric = TaskCompletionMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_task_incomplete() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "bad"}"#);
        let mut metric = TaskCompletionMetric::builder()
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
        let mut metric = TaskCompletionMetric::builder().provider(provider).build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

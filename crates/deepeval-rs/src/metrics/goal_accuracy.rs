//! Goal accuracy: an LLM-judge metric measuring how accurately the assistant's
//! output achieves the stated goal across the multi-turn conversation.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "GoalAccuracyMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/goal_accuracy/generate_verdict.txt");

/// GoalAccuracyMetric: scores how accurately the assistant's output achieves
/// the expected goal across the conversation.
///
/// Requires at least one turn with `input`, `actual_output`, and
/// `expected_output`.
#[derive(Debug, Clone)]
pub struct GoalAccuracyMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl GoalAccuracyMetric {
    /// Start building a [`GoalAccuracyMetric`].
    pub fn builder() -> GoalAccuracyMetricBuilder {
        GoalAccuracyMetricBuilder::default()
    }

    async fn measure_impl(
        &mut self,
        test_case: &ConversationalTestCase,
    ) -> Result<(), MetricError> {
        match measure_conversation_llm_judge(
            &[
                MultiTurnParams::Input,
                MultiTurnParams::ActualOutput,
                MultiTurnParams::ExpectedOutput,
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
            Some(score) => self.state.score = Some(score),
        }
        Ok(())
    }
}

impl_conversational_metric!(GoalAccuracyMetric, "GoalAccuracyMetric");

/// Builder for [`GoalAccuracyMetric`].
#[derive(Debug, Default)]
pub struct GoalAccuracyMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl GoalAccuracyMetricBuilder {
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
    pub fn build(self) -> GoalAccuracyMetric {
        let provider = self
            .provider
            .expect("GoalAccuracyMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        GoalAccuracyMetric {
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
                    .input("Help me plan a trip to Kyoto.")
                    .actual_output("Here is your Kyoto itinerary.")
                    .expected_output("Provide a detailed Kyoto itinerary.")
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Any hotel suggestions?")
                    .actual_output("I suggest hotels near Gion.")
                    .expected_output("Suggest hotels near Gion.")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_goal_met() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#);
        let mut metric = GoalAccuracyMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_goal_missed() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "bad"}"#);
        let mut metric = GoalAccuracyMetric::builder()
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
        let mut metric = GoalAccuracyMetric::builder().provider(provider).build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

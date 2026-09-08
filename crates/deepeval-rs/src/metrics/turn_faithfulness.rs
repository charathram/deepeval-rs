//! Turn faithfulness: an LLM-judge metric measuring whether each assistant
//! response stays grounded in its retrieval context.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "TurnFaithfulnessMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/turn_faithfulness/generate_verdict.txt");

/// TurnFaithfulnessMetric: scores how faithful each assistant response is to
/// its turn's retrieval context across the conversation.
///
/// Requires `input`, `actual_output`, and `retrieval_context` on at least one
/// turn.
#[derive(Debug, Clone)]
pub struct TurnFaithfulnessMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl TurnFaithfulnessMetric {
    /// Start building a [`TurnFaithfulnessMetric`].
    pub fn builder() -> TurnFaithfulnessMetricBuilder {
        TurnFaithfulnessMetricBuilder::default()
    }

    async fn measure_impl(
        &mut self,
        test_case: &ConversationalTestCase,
    ) -> Result<(), MetricError> {
        match measure_conversation_llm_judge(
            &[
                MultiTurnParams::Input,
                MultiTurnParams::ActualOutput,
                MultiTurnParams::RetrievalContext,
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
            Some((verdict, response)) => {
                self.state.score = Some(verdict.score.clamp(0.0, 1.0));
                self.state.reason = verdict.reason;
                self.state.accrue_usage(&response);
            }
            None => self.state.skipped = true,
        }
        Ok(())
    }
}

impl_conversational_metric!(TurnFaithfulnessMetric, "TurnFaithfulnessMetric");

/// Builder for [`TurnFaithfulnessMetric`].
#[derive(Debug, Default)]
pub struct TurnFaithfulnessMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl TurnFaithfulnessMetricBuilder {
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
    pub fn build(self) -> TurnFaithfulnessMetric {
        let provider = self
            .provider
            .expect("TurnFaithfulnessMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        TurnFaithfulnessMetric {
            state: MetricState::new(self.config),
            provider,
            registry: std::sync::Arc::new(registry),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmResponse, MockLlmProvider};
    use crate::metrics::ConversationalMetric;
    use crate::test_case::Turn;

    fn case() -> ConversationalTestCase {
        ConversationalTestCase::builder()
            .turn(
                Turn::builder()
                    .input("What is the refund policy?")
                    .actual_output("We offer a 30-day full refund.")
                    .retrieval_context(vec!["All customers get a 30-day refund.".to_string()])
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Is shipping free?")
                    .actual_output("Shipping is free over $50.")
                    .retrieval_context(vec!["Shipping is free over $50.".to_string()])
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_faithful() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "all claims grounded"}"#);
        let mut metric = TurnFaithfulnessMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .include_reason(true)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.reason(), Some("all claims grounded"));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_not_faithful() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2}"#);
        let mut metric = TurnFaithfulnessMetric::builder()
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
        let mut metric = TurnFaithfulnessMetric::builder().provider(provider).build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }

    #[tokio::test]
    async fn accrues_usage_from_response() {
        let response = LlmResponse::new(r#"{"score": 0.8, "reason": "grounded"}"#)
            .with_usage(120, 30)
            .with_cost(0.0015);
        let provider = MockLlmProvider::responses([response]);
        let mut metric = TurnFaithfulnessMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.8));
        assert_eq!(metric.input_tokens(), 120);
        assert_eq!(metric.output_tokens(), 30);
        assert_eq!(metric.cost(), Some(0.0015));
    }
}

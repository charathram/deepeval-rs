//! Role adherence: an LLM-judge metric measuring how well the assistant stays
//! in its assigned role across the conversation.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "RoleAdherenceMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/role_adherence/generate_verdict.txt");

/// RoleAdherenceMetric: scores how well the assistant stays in its assigned
/// role across the conversation.
///
/// Requires `input` and `actual_output` on at least one turn.
#[derive(Debug, Clone)]
pub struct RoleAdherenceMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl RoleAdherenceMetric {
    /// Start building a [`RoleAdherenceMetric`].
    pub fn builder() -> RoleAdherenceMetricBuilder {
        RoleAdherenceMetricBuilder::default()
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
            Some(verdict) => {
                self.state.score = Some(verdict.score.clamp(0.0, 1.0));
                self.state.reason = verdict.reason;
            }
            None => self.state.skipped = true,
        }
        Ok(())
    }
}

impl_conversational_metric!(RoleAdherenceMetric, "RoleAdherenceMetric");

/// Builder for [`RoleAdherenceMetric`].
#[derive(Debug, Default)]
pub struct RoleAdherenceMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl RoleAdherenceMetricBuilder {
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
    pub fn build(self) -> RoleAdherenceMetric {
        let provider = self
            .provider
            .expect("RoleAdherenceMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        RoleAdherenceMetric {
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
            .turn(
                Turn::builder()
                    .input("You are my personal shopper.")
                    .actual_output("As your personal shopper, I'll help you find great items.")
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Recommend a jacket.")
                    .actual_output("For you, I'd suggest this waterproof field jacket.")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_adhering_to_role() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "consistently in role"}"#);
        let mut metric = RoleAdherenceMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_straying_from_role() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2}"#);
        let mut metric = RoleAdherenceMetric::builder()
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
        let mut metric = RoleAdherenceMetric::builder().provider(provider).build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

//! Knowledge retention: an LLM-judge metric measuring how well the assistant
//! retains and consistently applies information introduced earlier in the
//! conversation.

use std::collections::HashMap;

use super::conversational_llm_judge::measure_conversation_llm_judge;
use super::llm_judge::Provider;
use super::{impl_conversational_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams};

const CLASS_NAME: &str = "KnowledgeRetentionMetric";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/knowledge_retention/generate_verdict.txt");

/// KnowledgeRetentionMetric: scores how consistently the assistant retains and
/// applies information introduced earlier in the conversation.
///
/// Requires `input` and `actual_output` on at least one turn.
#[derive(Debug, Clone)]
pub struct KnowledgeRetentionMetric {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
}

impl KnowledgeRetentionMetric {
    /// Start building a [`KnowledgeRetentionMetric`].
    pub fn builder() -> KnowledgeRetentionMetricBuilder {
        KnowledgeRetentionMetricBuilder::default()
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

impl_conversational_metric!(KnowledgeRetentionMetric, "KnowledgeRetentionMetric");

/// Builder for [`KnowledgeRetentionMetric`].
#[derive(Debug, Default)]
pub struct KnowledgeRetentionMetricBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
}

impl KnowledgeRetentionMetricBuilder {
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
    pub fn build(self) -> KnowledgeRetentionMetric {
        let provider = self
            .provider
            .expect("KnowledgeRetentionMetric requires an LLM provider");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        KnowledgeRetentionMetric {
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
                    .input("Remember I'm a vegetarian.")
                    .actual_output("Got it, I'll recommend vegetarian options.")
                    .build(),
            )
            .turn(
                Turn::builder()
                    .input("Which dish would you recommend?")
                    .actual_output("The mushroom risotto is a great vegetarian choice.")
                    .build(),
            )
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_knowledge_retained() {
        let provider =
            MockLlmProvider::text(r#"{"score": 0.9, "reason": "preference consistently applied"}"#);
        let mut metric = KnowledgeRetentionMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_knowledge_lost() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2}"#);
        let mut metric = KnowledgeRetentionMetric::builder()
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
        let mut metric = KnowledgeRetentionMetric::builder()
            .provider(provider)
            .build();
        let tc = ConversationalTestCase::builder().build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

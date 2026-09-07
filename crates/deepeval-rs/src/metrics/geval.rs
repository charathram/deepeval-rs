//! GEval: an LLM-judge metric that scores an output against custom criteria.
//!
//! This is a simplified chain-of-thought version. Logprob-based scoring is
//! deferred to Phase 6.

use std::collections::HashMap;

use minijinja::Value;

use super::llm_judge::{measure_llm_judge, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "GEval";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/geval/generate_verdict.txt");

/// GEval: scores an output against user-supplied criteria using an LLM judge.
///
/// Requires `input` and `actual_output` on the test case, plus `criteria` on
/// the metric. Optional `evaluation_steps` guide the judge's reasoning.
#[derive(Debug, Clone)]
pub struct GEval {
    state: MetricState,
    provider: Provider,
    registry: std::sync::Arc<TemplateRegistry>,
    criteria: String,
    evaluation_steps: Vec<String>,
}

impl GEval {
    /// Start building a [`GEval`] metric.
    pub fn builder() -> GEvalBuilder {
        GEvalBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        let mut extra = HashMap::new();
        extra.insert("criteria".to_string(), Value::from(self.criteria.clone()));
        extra.insert(
            "evaluation_steps".to_string(),
            Value::from(self.evaluation_steps.join("\n")),
        );

        match measure_llm_judge(
            &[SingleTurnParams::Input, SingleTurnParams::ActualOutput],
            self.provider.as_provider(),
            &self.registry,
            CLASS_NAME,
            METHOD,
            &extra,
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

impl_metric!(GEval, "GEval");

/// Builder for [`GEval`].
#[derive(Debug, Default)]
pub struct GEvalBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
    criteria: Option<String>,
    evaluation_steps: Vec<String>,
}

impl GEvalBuilder {
    /// Set the LLM provider used to judge the output.
    pub fn provider<P>(mut self, provider: P) -> Self
    where
        P: crate::llm::LlmProvider + 'static,
    {
        self.provider = Some(Provider::new(provider));
        self
    }

    /// Set the evaluation criteria (required).
    pub fn criteria(mut self, criteria: impl Into<String>) -> Self {
        self.criteria = Some(criteria.into());
        self
    }

    /// Set optional evaluation steps that guide the judge.
    pub fn evaluation_steps(mut self, steps: Vec<String>) -> Self {
        self.evaluation_steps = steps;
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
    /// Panics if no provider or no criteria was set.
    pub fn build(self) -> GEval {
        let provider = self.provider.expect("GEval requires an LLM provider");
        let criteria = self.criteria.expect("GEval requires criteria");
        let mut registry = TemplateRegistry::new();
        registry.register(CLASS_NAME, METHOD, TEMPLATE);
        GEval {
            state: MetricState::new(self.config),
            provider,
            registry: std::sync::Arc::new(registry),
            criteria,
            evaluation_steps: self.evaluation_steps,
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
            .input("What is the capital of France?")
            .actual_output("The capital of France is Paris.")
            .build()
    }

    #[tokio::test]
    async fn scores_high_when_llm_returns_high_score() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "correct"}"#);
        let mut metric = GEval::builder()
            .provider(provider)
            .criteria("The answer must be factually correct.")
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.reason(), Some("correct"));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_llm_returns_low_score() {
        let provider = MockLlmProvider::text(r#"{"score": 0.2, "reason": "wrong"}"#);
        let mut metric = GEval::builder()
            .provider(provider)
            .criteria("The answer must be factually correct.")
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.2));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_actual_output_missing() {
        let provider = MockLlmProvider::text(r#"{"score": 0.9}"#);
        let mut metric = GEval::builder()
            .provider(provider)
            .criteria("factual")
            .build();
        let tc = LLMTestCase::builder().input("hi").build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

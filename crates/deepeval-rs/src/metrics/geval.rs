//! GEval: an LLM-judge metric that scores an output against custom criteria.
//!
//! GEval asks the judge for an integer score in a range (default 0-10) and
//! requests token log probabilities. The raw score is then confidence-weighted
//! against the log probabilities (mirroring deepeval's g-eval fix) and
//! normalized to `[0, 1]`.

use std::collections::HashMap;

use minijinja::Value;

use super::llm_judge::{measure_geval, MeasureOutcome, Provider};
use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

const CLASS_NAME: &str = "GEval";
const METHOD: &str = "generate_verdict";
const TEMPLATE: &str = include_str!("../../templates/geval/generate_verdict.txt");
const DEFAULT_MIN_SCORE: u32 = 0;
const DEFAULT_MAX_SCORE: u32 = 10;

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
    min_score: u32,
    max_score: u32,
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

        match measure_geval(
            &[SingleTurnParams::Input, SingleTurnParams::ActualOutput],
            self.provider.as_provider(),
            &self.registry,
            CLASS_NAME,
            METHOD,
            &extra,
            test_case,
            self.min_score,
            self.max_score,
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

impl_metric!(GEval, "GEval");

/// Builder for [`GEval`].
#[derive(Debug)]
pub struct GEvalBuilder {
    config: MetricConfig,
    provider: Option<Provider>,
    criteria: Option<String>,
    evaluation_steps: Vec<String>,
    min_score: u32,
    max_score: u32,
}

impl Default for GEvalBuilder {
    fn default() -> Self {
        Self {
            config: MetricConfig::default(),
            provider: None,
            criteria: None,
            evaluation_steps: Vec::new(),
            min_score: DEFAULT_MIN_SCORE,
            max_score: DEFAULT_MAX_SCORE,
        }
    }
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

    /// Set the integer score range the judge is asked to use (default 0-10).
    pub fn score_range(mut self, min: u32, max: u32) -> Self {
        self.min_score = min;
        self.max_score = max;
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
            min_score: self.min_score,
            max_score: self.max_score,
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
        let provider = MockLlmProvider::text(r#"{"score": 9, "reason": "correct"}"#);
        let mut metric = GEval::builder()
            .provider(provider)
            .criteria("The answer must be factually correct.")
            .threshold(0.7)
            .include_reason(true)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.reason(), Some("correct"));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_low_when_llm_returns_low_score() {
        let provider = MockLlmProvider::text(r#"{"score": 2, "reason": "wrong"}"#);
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
    async fn suppresses_reason_when_include_reason_is_false() {
        let provider = MockLlmProvider::text(r#"{"score": 9, "reason": "correct"}"#);
        let mut metric = GEval::builder()
            .provider(provider)
            .criteria("The answer must be factually correct.")
            .threshold(0.7)
            .build();

        metric.measure(&case()).await.unwrap();
        assert_eq!(metric.score(), Some(0.9));
        assert_eq!(metric.reason(), None);
    }

    #[tokio::test]
    async fn skips_when_actual_output_missing() {
        let provider = MockLlmProvider::text(r#"{"score": 9}"#);
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

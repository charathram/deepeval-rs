//! YAML test-suite loading and metric/test-case construction for `test run`.

use anyhow::{bail, Context, Result};
use deepeval_rs::llm::{AnthropicProvider, LlmProvider, OpenAIProvider};
use deepeval_rs::metrics::{
    AnswerRelevancyMetric, AnswerRelevancyMetricBuilder, ContextualPrecisionMetric,
    ContextualPrecisionMetricBuilder, ContextualRecallMetric, ContextualRecallMetricBuilder,
    ContextualRelevancyMetric, ContextualRelevancyMetricBuilder, ExactMatchMetric,
    ExactMatchMetricBuilder, FaithfulnessMetric, FaithfulnessMetricBuilder, GEval, GEvalBuilder,
    HallucinationMetric, HallucinationMetricBuilder, JsonCorrectnessMetric,
    JsonCorrectnessMetricBuilder, Metric, PatternMatchMetric, PatternMatchMetricBuilder,
    PromptAlignmentMetric, PromptAlignmentMetricBuilder, RagasMetric, RagasMetricBuilder,
};
use deepeval_rs::test_case::LLMTestCase;
use serde::Deserialize;

/// A full test suite as defined in a YAML file.
#[derive(Debug, Deserialize)]
pub struct Suite {
    /// A human-readable name for the suite.
    #[serde(default)]
    pub name: String,
    /// The metrics to run against every test case.
    pub metrics: Vec<MetricConfig>,
    /// The test cases to evaluate.
    pub test_cases: Vec<TestCaseConfig>,
}

/// A single test case.
#[derive(Debug, Deserialize)]
pub struct TestCaseConfig {
    /// The prompt/input given to the model.
    #[serde(default)]
    pub input: String,
    /// The model's output to evaluate.
    #[serde(default)]
    pub actual_output: String,
    /// The expected output (used by exact-match and recall metrics).
    #[serde(default)]
    pub expected_output: String,
    /// Retrieval context (used by RAG metrics).
    #[serde(default)]
    pub retrieval_context: Vec<String>,
}

/// A metric configuration, tagged by `type`.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum MetricConfig {
    /// 1.0 when actual output exactly equals expected output.
    ExactMatch {
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// 1.0 when actual output matches a regex pattern.
    PatternMatch {
        pattern: String,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// 1.0 when actual output parses as valid JSON.
    JsonCorrectness {
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: how relevant the output is to the input.
    AnswerRelevancy {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the output is faithful to the retrieval context.
    Faithfulness {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the output hallucinates.
    Hallucination {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the output aligns with the expected criteria.
    PromptAlignment {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the retrieval context contains the relevant info first.
    ContextualPrecision {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the retrieval context supports the expected output.
    ContextualRecall {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: whether the retrieval context is relevant to the input.
    ContextualRelevancy {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// Composite of four RAG metrics.
    Ragas {
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
    /// LLM-judge: score an output against custom criteria.
    #[serde(rename = "geval")]
    GEval {
        criteria: String,
        provider: ProviderConfig,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        threshold: Option<f32>,
    },
}

/// Which provider an LLM-judge metric should use.
#[derive(Debug, Deserialize)]
pub enum ProviderConfig {
    /// OpenAI-compatible chat completions, from `OPENAI_API_KEY`.
    #[serde(rename = "openai")]
    OpenAI,
    /// Anthropic completions, from `ANTHROPIC_API_KEY`.
    #[serde(rename = "anthropic")]
    Anthropic,
}

impl Suite {
    /// Load and parse a suite from a YAML file.
    pub fn from_file(path: &str) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read suite file `{path}`"))?;
        serde_yaml::from_str(&text).with_context(|| format!("failed to parse `{path}` as YAML"))
    }

    /// Build the configured metrics.
    pub fn build_metrics(&self) -> Result<Vec<Box<dyn Metric>>> {
        self.metrics.iter().map(build_metric).collect()
    }

    /// Build the configured test cases.
    pub fn build_test_cases(&self) -> Vec<LLMTestCase> {
        self.test_cases
            .iter()
            .map(|tc| {
                let mut b = LLMTestCase::builder()
                    .input(tc.input.clone())
                    .actual_output(tc.actual_output.clone());
                if !tc.expected_output.is_empty() {
                    b = b.expected_output(tc.expected_output.clone());
                }
                if !tc.retrieval_context.is_empty() {
                    b = b.retrieval_context(tc.retrieval_context.clone());
                }
                b.build()
            })
            .collect()
    }
}

fn build_metric(config: &MetricConfig) -> Result<Box<dyn Metric>> {
    let metric: Box<dyn Metric> = match config {
        MetricConfig::ExactMatch { threshold } => Box::new(
            ExactMatchMetric::builder()
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::PatternMatch { pattern, threshold } => Box::new(
            PatternMatchMetric::builder()
                .pattern(pattern.clone())
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::JsonCorrectness { threshold } => Box::new(
            JsonCorrectnessMetric::builder()
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::AnswerRelevancy {
            provider,
            model,
            threshold,
        } => Box::new(
            AnswerRelevancyMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::Faithfulness {
            provider,
            model,
            threshold,
        } => Box::new(
            FaithfulnessMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::Hallucination {
            provider,
            model,
            threshold,
        } => Box::new(
            HallucinationMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::PromptAlignment {
            provider,
            model,
            threshold,
        } => Box::new(
            PromptAlignmentMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::ContextualPrecision {
            provider,
            model,
            threshold,
        } => Box::new(
            ContextualPrecisionMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::ContextualRecall {
            provider,
            model,
            threshold,
        } => Box::new(
            ContextualRecallMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::ContextualRelevancy {
            provider,
            model,
            threshold,
        } => Box::new(
            ContextualRelevancyMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::Ragas {
            provider,
            model,
            threshold,
        } => Box::new(
            RagasMetric::builder()
                .provider(build_provider(provider, model)?)
                .maybe_threshold(*threshold)
                .build(),
        ),
        MetricConfig::GEval {
            criteria,
            provider,
            model,
            threshold,
        } => Box::new(
            GEval::builder()
                .provider(build_provider(provider, model)?)
                .criteria(criteria.clone())
                .maybe_threshold(*threshold)
                .build(),
        ),
    };
    Ok(metric)
}

fn build_provider(config: &ProviderConfig, model: &Option<String>) -> Result<Box<dyn LlmProvider>> {
    let provider: Box<dyn LlmProvider> = match config {
        ProviderConfig::OpenAI => match model {
            Some(m) => Box::new(OpenAIProvider::from_env_with_model(m.clone())?),
            None => Box::new(OpenAIProvider::from_env()?),
        },
        ProviderConfig::Anthropic => match model {
            Some(m) => Box::new(AnthropicProvider::from_env_with_model(m.clone())?),
            None => Box::new(AnthropicProvider::from_env()?),
        },
    };
    Ok(provider)
}

/// Apply a threshold only when present.
trait MaybeThreshold {
    fn maybe_threshold(self, threshold: Option<f32>) -> Self;
}

impl MaybeThreshold for ExactMatchMetricBuilder {
    fn maybe_threshold(self, threshold: Option<f32>) -> Self {
        match threshold {
            Some(t) => self.threshold(t),
            None => self,
        }
    }
}

macro_rules! impl_maybe_threshold {
    ($($ty:ty),* $(,)?) => {
        $(
            impl MaybeThreshold for $ty {
                fn maybe_threshold(self, threshold: Option<f32>) -> Self {
                    match threshold {
                        Some(t) => self.threshold(t),
                        None => self,
                    }
                }
            }
        )*
    };
}

impl_maybe_threshold!(
    PatternMatchMetricBuilder,
    JsonCorrectnessMetricBuilder,
    AnswerRelevancyMetricBuilder,
    FaithfulnessMetricBuilder,
    HallucinationMetricBuilder,
    PromptAlignmentMetricBuilder,
    ContextualPrecisionMetricBuilder,
    ContextualRecallMetricBuilder,
    ContextualRelevancyMetricBuilder,
    RagasMetricBuilder,
    GEvalBuilder,
);

/// Validate that a suite is well-formed before running.
pub fn validate(suite: &Suite) -> Result<()> {
    if suite.metrics.is_empty() {
        bail!("suite must define at least one metric");
    }
    if suite.test_cases.is_empty() {
        bail!("suite must define at least one test case");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUITE_YAML: &str = r#"
name: example
metrics:
  - type: exact_match
    threshold: 0.5
  - type: pattern_match
    pattern: "refund"
  - type: answer_relevancy
    provider: openai
    model: gpt-4o-mini
    threshold: 0.7
  - type: geval
    criteria: "The output is concise."
    provider: anthropic
test_cases:
  - input: "What is a refund?"
    actual_output: "A refund returns your money."
    expected_output: "A refund returns your money."
    retrieval_context:
      - "A refund returns your money to you."
"#;

    #[test]
    fn parses_suite() {
        let suite: Suite = serde_yaml::from_str(SUITE_YAML).expect("suite should parse");
        assert_eq!(suite.name, "example");
        assert_eq!(suite.metrics.len(), 4);
        assert_eq!(suite.test_cases.len(), 1);
    }

    #[test]
    fn builds_deterministic_metrics() {
        let yaml = "name: det\nmetrics:\n  - type: exact_match\n  - type: pattern_match\n    pattern: x\n  - type: json_correctness\ntest_cases:\n  - input: a\n    actual_output: b\n";
        let suite: Suite = serde_yaml::from_str(yaml).expect("suite should parse");
        let metrics = suite.build_metrics().expect("metrics should build");
        assert_eq!(metrics.len(), 3);
    }

    #[test]
    fn llm_judge_metric_requires_env() {
        // Building an LLM-judge metric needs a real API key; without one it
        // must fail cleanly rather than panic.
        let suite: Suite = serde_yaml::from_str(SUITE_YAML).expect("suite should parse");
        let result = suite.build_metrics();
        assert!(result.is_err());
    }

    #[test]
    fn builds_test_cases() {
        let suite: Suite = serde_yaml::from_str(SUITE_YAML).expect("suite should parse");
        let cases = suite.build_test_cases();
        assert_eq!(cases.len(), 1);
        assert_eq!(cases[0].input, "What is a refund?");
        assert_eq!(
            cases[0].actual_output.as_deref(),
            Some("A refund returns your money.")
        );
    }

    #[test]
    fn validate_rejects_empty_suite() {
        let suite: Suite = serde_yaml::from_str("name: empty\nmetrics: []\ntest_cases: []\n")
            .expect("should parse");
        assert!(validate(&suite).is_err());
    }

    #[test]
    fn unknown_metric_type_is_rejected() {
        let yaml = "name: bad\nmetrics:\n  - type: nope\ntest_cases:\n  - input: x\n    actual_output: y\n";
        assert!(serde_yaml::from_str::<Suite>(yaml).is_err());
    }
}

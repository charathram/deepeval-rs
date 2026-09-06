//! Error types for the deepeval-rs framework.
//!
//! All public errors are defined here using [`thiserror`]. They are re-exported
//! from the crate root.

/// Errors that can occur while running an evaluation.
#[derive(Debug, thiserror::Error)]
pub enum EvalError {
    /// A metric failed to measure a test case.
    #[error("metric `{metric}` failed: {source}")]
    Metric {
        /// The name of the metric that failed.
        metric: String,
        /// The underlying metric error.
        #[source]
        source: MetricError,
    },

    /// A required field was missing from a test case.
    #[error("missing required field `{field}` for metric `{metric}`")]
    MissingField {
        /// The name of the metric that required the field.
        metric: String,
        /// The name of the missing field.
        field: &'static str,
    },

    /// An assertion failed because one or more metrics did not pass.
    #[error("assertion failed: {0}")]
    AssertionFailed(String),
}

/// Errors that can occur while interacting with an LLM provider.
#[derive(Debug, thiserror::Error)]
pub enum LlmError {
    /// The provider returned an error response.
    #[error("LLM provider error: {0}")]
    Provider(String),

    /// The provider response could not be parsed.
    #[error("failed to parse LLM response: {0}")]
    Parse(String),

    /// The provider response did not contain the expected structured output.
    #[error("LLM response did not contain expected structured output: {0}")]
    MissingStructuredOutput(String),

    /// A transport-level error occurred (network, timeout, etc.).
    #[error("LLM transport error: {0}")]
    Transport(String),
}

/// Errors that can occur while measuring a metric.
#[derive(Debug, thiserror::Error)]
pub enum MetricError {
    /// A required field was missing from the test case.
    #[error("missing required field `{0}`")]
    MissingField(&'static str),

    /// The metric's evaluation model was not configured.
    #[error("no evaluation model configured for metric `{0}`")]
    NoModel(String),

    /// An LLM call failed while measuring the metric.
    #[error("LLM call failed: {0}")]
    Llm(#[from] LlmError),

    /// A prompt template could not be rendered.
    #[error("template error: {0}")]
    Template(#[from] TemplateError),

    /// The metric produced an invalid score.
    #[error("invalid score `{0}` produced by metric `{1}`")]
    InvalidScore(f32, String),
}

/// Errors that can occur while rendering a prompt template.
#[derive(Debug, thiserror::Error)]
pub enum TemplateError {
    /// The requested template was not found.
    #[error("template not found for class `{class}` method `{method}`")]
    NotFound {
        /// The metric class name.
        class: String,
        /// The template method name.
        method: String,
    },

    /// The template could not be rendered.
    #[error("failed to render template `{0}`: {1}")]
    Render(String, String),
}

impl EvalError {
    /// Build an [`EvalError::AssertionFailed`] from a list of failing metric names.
    pub fn assertion_failed(failures: &[String]) -> Self {
        let detail = failures
            .iter()
            .map(|name| format!("  - {name}"))
            .collect::<Vec<_>>()
            .join("\n");
        EvalError::AssertionFailed(format!("the following metrics did not pass:\n{detail}"))
    }
}

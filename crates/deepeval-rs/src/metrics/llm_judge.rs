//! Shared helpers for LLM-judge metrics.
//!
//! LLM-judge metrics follow a common flow: validate required fields, render a
//! prompt, ask the LLM for a JSON verdict, and record a 0-1 score. This module
//! centralizes that flow so each metric is a thin wrapper.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use minijinja::Value;
use serde::Deserialize;

use crate::error::MetricError;
use crate::llm::{ChatMessage, LlmProvider, LlmRequest};
use crate::template::TemplateRegistry;
use crate::test_case::{LLMTestCase, SingleTurnParams};

/// A parsed LLM verdict: a 0-1 score and an optional reason.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Verdict {
    /// The score in `[0, 1]`.
    pub score: f32,
    /// An optional natural-language reason.
    #[serde(default)]
    pub reason: Option<String>,
}

/// The outcome of measuring an LLM-judge metric.
#[derive(Debug, Clone)]
pub(crate) enum MeasureOutcome {
    /// A required field was missing; the metric is skipped.
    Skipped,
    /// The metric produced a score.
    Scored {
        /// The score in `[0, 1]`.
        score: f32,
        /// An optional reason.
        reason: Option<String>,
    },
}

/// A shared, cloneable, debuggable handle to an LLM provider.
///
/// Wraps [`Arc<dyn LlmProvider>`] so metric structs can derive `Debug` and
/// `Clone` without requiring the provider itself to be `Debug` or `Clone`.
#[derive(Clone)]
pub(crate) struct Provider(Arc<dyn LlmProvider>);

impl Provider {
    /// Wrap a concrete provider.
    pub fn new<P: LlmProvider + 'static>(provider: P) -> Self {
        Self(Arc::new(provider))
    }

    /// Borrow the underlying provider.
    pub fn as_provider(&self) -> &dyn LlmProvider {
        self.0.as_ref()
    }
}

impl std::fmt::Debug for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Provider")
            .field(&self.0.model_name())
            .finish()
    }
}

#[async_trait]
impl LlmProvider for Provider {
    fn model_name(&self) -> &str {
        self.0.model_name()
    }

    async fn complete(
        &self,
        request: LlmRequest,
    ) -> Result<crate::llm::LlmResponse, crate::error::LlmError> {
        self.0.complete(request).await
    }
}

/// Build a minijinja context from a test case, filling missing fields with
/// empty defaults.
pub(crate) fn case_context(test_case: &LLMTestCase) -> HashMap<String, Value> {
    let mut ctx = HashMap::new();
    ctx.insert("input".to_string(), Value::from(test_case.input.clone()));
    ctx.insert(
        "actual_output".to_string(),
        Value::from(test_case.actual_output.clone().unwrap_or_default()),
    );
    ctx.insert(
        "expected_output".to_string(),
        Value::from(test_case.expected_output.clone().unwrap_or_default()),
    );
    ctx.insert(
        "retrieval_context".to_string(),
        Value::from(join_lines(test_case.retrieval_context.as_deref())),
    );
    ctx.insert(
        "expected_criteria".to_string(),
        Value::from(join_lines(test_case.expected_criteria.as_deref())),
    );
    ctx.insert(
        "context".to_string(),
        Value::from(join_lines(test_case.context.as_deref())),
    );
    ctx
}

fn join_lines(items: Option<&[String]>) -> String {
    items.map(|v| v.join("\n")).unwrap_or_default()
}

/// Whether a required field is present on the test case.
pub(crate) fn field_present(field: SingleTurnParams, test_case: &LLMTestCase) -> bool {
    match field {
        SingleTurnParams::Input => !test_case.input.is_empty(),
        SingleTurnParams::ActualOutput => test_case.actual_output.is_some(),
        SingleTurnParams::ExpectedOutput => test_case.expected_output.is_some(),
        SingleTurnParams::RetrievalContext => test_case.retrieval_context.is_some(),
        SingleTurnParams::ExpectedTools => test_case.expected_tools.is_some(),
        SingleTurnParams::ExpectedCriteria => test_case.expected_criteria.is_some(),
        SingleTurnParams::Context => test_case.context.is_some(),
        SingleTurnParams::Feedbacks => test_case.feedbacks.is_some(),
    }
}

/// Ask the LLM for a score verdict given a rendered prompt.
pub(crate) async fn score_via_llm(
    provider: &dyn LlmProvider,
    prompt: String,
) -> Result<Verdict, MetricError> {
    let request = LlmRequest::new(vec![ChatMessage::user(prompt)]);
    let value = provider.complete_structured(request).await?;
    let verdict: Verdict = serde_json::from_value(value)
        .map_err(|e| crate::error::LlmError::Parse(format!("failed to parse verdict: {e}")))?;
    Ok(verdict)
}

/// Run the common LLM-judge flow.
///
/// Validates `required` fields (returning [`MeasureOutcome::Skipped`] if any
/// is missing), renders the `(class_name, method)` template with the test-case
/// context plus `extra_context`, calls the LLM, and parses the verdict.
pub(crate) async fn measure_llm_judge(
    required: &[SingleTurnParams],
    provider: &dyn LlmProvider,
    registry: &TemplateRegistry,
    class_name: &str,
    method: &str,
    extra_context: &HashMap<String, Value>,
    test_case: &LLMTestCase,
) -> Result<MeasureOutcome, MetricError> {
    for field in required {
        if !field_present(*field, test_case) {
            return Ok(MeasureOutcome::Skipped);
        }
    }

    let mut ctx = case_context(test_case);
    ctx.extend(extra_context.clone());

    let prompt = registry.resolve(class_name, method, &ctx)?;
    let verdict = score_via_llm(provider, prompt).await?;
    let score = verdict.score.clamp(0.0, 1.0);

    Ok(MeasureOutcome::Scored {
        score,
        reason: verdict.reason,
    })
}

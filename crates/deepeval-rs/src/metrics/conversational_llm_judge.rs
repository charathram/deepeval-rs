//! Shared helpers for multi-turn (conversational) LLM-judge metrics.
//!
//! Conversational LLM-judge metrics follow a common flow: validate that the
//! conversation has turns with the required fields, render a prompt over the
//! full conversation (or a selected turn), ask the LLM for a JSON verdict, and
//! record a 0-1 score. This module centralizes that flow so each metric is a
//! thin wrapper.

use std::collections::HashMap;

use minijinja::Value;

use crate::error::MetricError;
use crate::llm::{ChatMessage, LlmProvider, LlmRequest};
use crate::metrics::llm_judge::{Provider, Verdict};
use crate::template::TemplateRegistry;
use crate::test_case::{ConversationalTestCase, MultiTurnParams, Turn};

/// Whether a required field is present on a turn.
pub(crate) fn turn_field_present(field: MultiTurnParams, turn: &Turn) -> bool {
    match field {
        MultiTurnParams::Input => !turn.input.is_empty(),
        MultiTurnParams::ActualOutput => turn.actual_output.is_some(),
        MultiTurnParams::ExpectedOutput => turn.expected_output.is_some(),
        MultiTurnParams::RetrievalContext => turn.retrieval_context.is_some(),
        MultiTurnParams::ExpectedTools => turn.expected_tools.is_some(),
        MultiTurnParams::ExpectedCriteria => turn.expected_criteria.is_some(),
    }
}

/// Whether *any* turn satisfies every entry in `required`.
///
/// Conversational metrics operate over the whole conversation. If no turn has
/// all required fields present, the metric is skipped.
pub(crate) fn any_turn_has(fields: &[MultiTurnParams], turns: &[Turn]) -> bool {
    turns
        .iter()
        .any(|turn| fields.iter().all(|f| turn_field_present(*f, turn)))
}

/// Render a single turn into a small prompt fragment.
fn turn_lines(turn: &Turn) -> Vec<(String, String)> {
    let mut lines = Vec::new();
    lines.push(("input".to_string(), turn.input.clone()));
    if let Some(v) = &turn.actual_output {
        lines.push(("actual_output".to_string(), v.clone()));
    }
    if let Some(v) = &turn.expected_output {
        lines.push(("expected_output".to_string(), v.clone()));
    }
    if let Some(v) = &turn.retrieval_context {
        lines.push(("retrieval_context".to_string(), v.join("\n")));
    }
    if let Some(v) = &turn.expected_criteria {
        lines.push(("expected_criteria".to_string(), v.join("\n")));
    }
    if let Some(v) = &turn.expected_tools {
        let tools: Vec<String> = v.iter().map(|t| t.name.clone()).collect();
        lines.push(("expected_tools".to_string(), tools.join(", ")));
    }
    lines
}

/// Build a minijinja context from a conversational test case.
pub(crate) fn conversation_context(input: &ConversationalTestCase) -> HashMap<String, Value> {
    let turns: Vec<String> = input
        .turns
        .iter()
        .map(|turn| {
            let fields = turn_lines(turn);
            if fields.is_empty() {
                "(empty turn)".to_string()
            } else {
                fields
                    .into_iter()
                    .map(|(k, v)| format!("\"{k}\": {v}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        })
        .collect();

    let mut ctx = HashMap::new();
    ctx.insert("turns".to_string(), Value::from(turns));
    ctx.insert(
        "turn_count".to_string(),
        Value::from(input.turns.len() as i64),
    );
    ctx
}

/// Render a flat list of turns as numbered dialogs (input/output pairs).
fn dialog_turns(turns: &[Turn]) -> String {
    turns
        .iter()
        .enumerate()
        .map(|(i, turn)| {
            let out = turn
                .actual_output
                .clone()
                .unwrap_or_else(|| "(none)".to_string());
            format!("Turn {i}:\nInput: {}\nOutput: {}", turn.input, out)
        })
        .collect::<Vec<_>>()
        .join("\n")
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

/// Run the common conversational LLM-judge flow over a whole conversation.
///
/// Validates that at least one turn satisfies every `required` field (returning
/// a skip marker if not), renders the `(class_name, method)` template with a
/// serialized conversation plus `extra_context`, calls the LLM, and parses the
/// verdict.
///
/// Returns `Ok(None)` when the metric should be skipped (no turn satisfies all
/// required fields), otherwise `Ok(Some(verdict))` with the parsed score and
/// reason.
pub(crate) async fn measure_conversation_llm_judge(
    required: &[MultiTurnParams],
    provider: &Provider,
    registry: &TemplateRegistry,
    class_name: &str,
    method: &str,
    extra_context: &HashMap<String, Value>,
    test_case: &ConversationalTestCase,
) -> Result<Option<Verdict>, MetricError> {
    if !any_turn_has(required, &test_case.turns) {
        return Ok(None);
    }

    let mut ctx = conversation_context(test_case);
    ctx.insert(
        "dialog".to_string(),
        Value::from(dialog_turns(&test_case.turns)),
    );
    ctx.extend(extra_context.clone());

    let prompt = registry.resolve(class_name, method, &ctx)?;
    let verdict = score_via_llm(provider.as_provider(), prompt).await?;
    Ok(Some(verdict))
}

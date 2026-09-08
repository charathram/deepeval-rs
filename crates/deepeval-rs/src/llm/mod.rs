//! LLM provider abstraction built on [rig](https://github.com/0xplaygrounds/rig).
//!
//! This module defines a thin, object-safe [`LlmProvider`] trait that wraps
//! rig's [`CompletionModel`] trait. Metrics depend on [`LlmProvider`] rather
//! than on rig types directly, which keeps the metric layer provider-agnostic
//! and testable with a mock.

mod logprobs;
mod mock;
mod providers;
mod retry;
mod types;

pub use logprobs::{calculate_weighted_summed_score, extract_logprobs};
pub use mock::MockLlmProvider;
pub use providers::{
    AnthropicProvider, OpenAIProvider, RigProvider, DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL,
};
pub use retry::{RetryPolicy, RetryProvider};
pub use types::{ChatMessage, LlmRequest, LlmResponse, Role, TokenLogprob, TokenLogprobs};

use crate::error::LlmError;
use async_trait::async_trait;

/// A provider that can generate LLM completions.
///
/// This is the seam between metrics and rig. It is object-safe (`async_trait`
/// + `Send + Sync`) so a heterogeneous set of providers can be held behind
/// `Box<dyn LlmProvider>`.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// The name of the model this provider uses.
    fn model_name(&self) -> &str;

    /// Generate a completion for the given request.
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError>;

    /// Generate a completion and parse the response as structured JSON.
    ///
    /// The default implementation requests plain text and parses the first
    /// JSON value in the response. Providers that support native structured
    /// output may override this to use rig's `output_schema`.
    async fn complete_structured(
        &self,
        request: LlmRequest,
    ) -> Result<serde_json::Value, LlmError> {
        let response = self.complete(request).await?;
        let json = extract_json(&response.content)
            .ok_or_else(|| LlmError::MissingStructuredOutput(response.content.clone()))?;
        Ok(json)
    }
}

#[async_trait]
impl LlmProvider for Box<dyn LlmProvider> {
    fn model_name(&self) -> &str {
        (**self).model_name()
    }

    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        (**self).complete(request).await
    }

    async fn complete_structured(
        &self,
        request: LlmRequest,
    ) -> Result<serde_json::Value, LlmError> {
        (**self).complete_structured(request).await
    }
}

/// Extract the first JSON value from a string that may contain surrounding
/// prose or code fences.
pub(crate) fn extract_json(content: &str) -> Option<serde_json::Value> {
    // Strip a surrounding ```json ... ``` fence if present.
    let trimmed = content.trim();
    let inner = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .map(str::trim)
        .unwrap_or(trimmed);

    // Try parsing the whole (possibly fenced) string first.
    if let Ok(value) = serde_json::from_str(inner) {
        return Some(value);
    }

    // Try repairing common LLM JSON mistakes (trailing commas, unquoted
    // keys, single-quoted strings) before falling back to scanning.
    if let Some(repaired) = repair_json(inner) {
        if let Ok(value) = serde_json::from_str(&repaired) {
            return Some(value);
        }
    }

    // Otherwise scan for the first balanced JSON object or array.
    let bytes = inner.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i];
        if c == b'{' || c == b'[' {
            if let Some(value) = parse_balanced(inner, i) {
                return Some(value);
            }
        }
        i += 1;
    }
    None
}

/// Best-effort repair of common LLM JSON mistakes: trailing commas, unquoted
/// keys, and single-quoted strings. Returns `None` if nothing needed fixing.
fn repair_json(input: &str) -> Option<String> {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut i = 0;
    let mut changed = false;
    let mut in_string = false;
    let mut escaped = false;
    // The quote character that opened the current string (b'"' or b'\'').
    let mut open_quote = b'"';

    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if escaped {
                out.push(b as char);
                escaped = false;
            } else if b == b'\\' {
                out.push(b as char);
                escaped = true;
            } else if b == open_quote {
                out.push('"');
                in_string = false;
            } else {
                out.push(b as char);
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                open_quote = b'"';
                out.push('"');
                i += 1;
            }
            b'\'' => {
                // Convert a single-quoted string to a double-quoted one.
                in_string = true;
                open_quote = b'\'';
                out.push('"');
                changed = true;
                i += 1;
            }
            b',' => {
                // Drop a trailing comma before `}` or `]`.
                let next = bytes.get(i + 1).copied();
                if next == Some(b'}') || next == Some(b']') {
                    changed = true;
                } else {
                    out.push(',');
                }
                i += 1;
            }
            b'{' | b'[' | b':' => {
                out.push(b as char);
                i += 1;
            }
            _ if b.is_ascii_whitespace() => {
                out.push(b as char);
                i += 1;
            }
            _ => {
                // Unquoted key: scan ahead to the colon and quote it.
                if let Some(colon) = input[i..].find(':') {
                    let key = &input[i..i + colon];
                    let key = key.trim();
                    if !key.is_empty()
                        && key.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
                    {
                        out.push('"');
                        out.push_str(key);
                        out.push('"');
                        out.push(':');
                        i += colon + 1;
                        changed = true;
                        continue;
                    }
                }
                out.push(b as char);
                i += 1;
            }
        }
    }

    if changed {
        Some(out)
    } else {
        None
    }
}

/// Parse a balanced JSON value starting at `start` (which must point at `{`
/// or `[`).
fn parse_balanced(input: &str, start: usize) -> Option<serde_json::Value> {
    let bytes = input.as_bytes();
    let open = bytes[start];
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (offset, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        if b == b'"' {
            in_string = true;
        } else if b == open {
            depth += 1;
        } else if b == close {
            depth -= 1;
            if depth == 0 {
                let end = offset + 1;
                return serde_json::from_str(&input[start..end]).ok();
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_plain_json() {
        let content = r#"{"score": 0.8, "reason": "good"}"#;
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.8);
    }

    #[test]
    fn extracts_json_from_fence() {
        let content = "Here is the result:\n```json\n{\"score\": 0.5}\n```";
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.5);
    }

    #[test]
    fn extracts_json_from_prose() {
        let content = "The verdict is: {\"verdict\": \"yes\"} as expected.";
        let value = extract_json(content).unwrap();
        assert_eq!(value["verdict"], "yes");
    }

    #[test]
    fn extracts_nested_json() {
        let content = "Result: {\"a\": {\"b\": [1, 2, 3]}} done";
        let value = extract_json(content).unwrap();
        assert_eq!(value["a"]["b"][1], 2);
    }

    #[test]
    fn returns_none_for_no_json() {
        assert!(extract_json("no json here").is_none());
    }

    #[test]
    fn repairs_trailing_comma() {
        let content = r#"{"score": 0.8, "reason": "good",}"#;
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.8);
        assert_eq!(value["reason"], "good");
    }

    #[test]
    fn repairs_unquoted_keys() {
        let content = r#"{score: 0.8, reason: "good"}"#;
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.8);
        assert_eq!(value["reason"], "good");
    }

    #[test]
    fn repairs_single_quoted_strings() {
        let content = r#"{'score': 0.8, 'reason': 'good'}"#;
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.8);
        assert_eq!(value["reason"], "good");
    }

    #[test]
    fn repairs_combined_mistakes() {
        let content = r#"{score: 0.8, reason: 'good',}"#;
        let value = extract_json(content).unwrap();
        assert_eq!(value["score"], 0.8);
        assert_eq!(value["reason"], "good");
    }
}

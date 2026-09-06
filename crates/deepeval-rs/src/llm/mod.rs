//! LLM provider abstraction built on [rig](https://github.com/0xplaygrounds/rig).
//!
//! This module defines a thin, object-safe [`LlmProvider`] trait that wraps
//! rig's [`CompletionModel`] trait. Metrics depend on [`LlmProvider`] rather
//! than on rig types directly, which keeps the metric layer provider-agnostic
//! and testable with a mock.

mod mock;
mod providers;
mod types;

pub use mock::MockLlmProvider;
pub use providers::RigProvider;
pub use types::{ChatMessage, LlmRequest, LlmResponse, Role};

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
    /// JSON object in the response. Providers that support native structured
    /// output may override this to use rig's `output_schema`.
    async fn complete_structured<T>(&self, request: LlmRequest) -> Result<T, LlmError>
    where
        T: serde::de::DeserializeOwned + Send,
    {
        let response = self.complete(request).await?;
        let json = extract_json(&response.content)
            .ok_or_else(|| LlmError::MissingStructuredOutput(response.content.clone()))?;
        serde_json::from_value(json)
            .map_err(|e| LlmError::Parse(format!("failed to parse structured output: {e}")))
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
}

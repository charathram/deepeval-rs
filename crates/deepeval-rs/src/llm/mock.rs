//! A mock [`LlmProvider`] for keyless unit tests.
//!
//! [`MockLlmProvider`] replays a scripted queue of responses, one per
//! [`complete`](super::LlmProvider::complete) call. It is backed by rig's
//! [`MockCompletionModel`] so it exercises the same conversion path as a real
//! provider.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use super::{LlmProvider, LlmRequest, LlmResponse};
use crate::error::LlmError;

/// A scripted [`LlmProvider`] for tests.
///
/// Each call to [`complete`](super::LlmProvider::complete) consumes the next
/// scripted response. If the queue is exhausted, the provider returns an error.
#[derive(Clone, Default)]
pub struct MockLlmProvider {
    responses: Arc<Mutex<std::collections::VecDeque<Result<LlmResponse, LlmError>>>>,
    model_name: String,
}

impl MockLlmProvider {
    /// Create a mock provider that replays the given responses in order.
    pub fn new(responses: impl IntoIterator<Item = Result<LlmResponse, LlmError>>) -> Self {
        Self {
            responses: Arc::new(Mutex::new(responses.into_iter().collect())),
            model_name: "mock".to_string(),
        }
    }

    /// Create a mock provider that always returns the given text.
    pub fn text(text: impl Into<String>) -> Self {
        Self::new([Ok(LlmResponse::new(text))])
    }

    /// Create a mock provider that returns the given responses in order.
    pub fn responses(responses: impl IntoIterator<Item = LlmResponse>) -> Self {
        Self::new(responses.into_iter().map(Ok))
    }

    /// Set the model name reported by this provider.
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    /// The number of responses remaining in the queue.
    pub fn remaining(&self) -> usize {
        self.responses.lock().unwrap().len()
    }
}

#[async_trait]
impl LlmProvider for MockLlmProvider {
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn complete(&self, _request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let mut queue = self.responses.lock().unwrap();
        match queue.pop_front() {
            Some(response) => response,
            None => Err(LlmError::Provider(
                "mock provider exhausted its scripted responses".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;

    #[tokio::test]
    async fn replays_scripted_responses_in_order() {
        let provider = MockLlmProvider::responses([
            LlmResponse::new("first").with_usage(1, 2),
            LlmResponse::new("second"),
        ]);

        let req = LlmRequest::new(vec![ChatMessage::user("hi")]);
        let r1 = provider.complete(req.clone()).await.unwrap();
        let r2 = provider.complete(req).await.unwrap();

        assert_eq!(r1.content, "first");
        assert_eq!(r1.input_tokens, 1);
        assert_eq!(r1.output_tokens, 2);
        assert_eq!(r2.content, "second");
        assert_eq!(provider.remaining(), 0);
    }

    #[tokio::test]
    async fn errors_when_exhausted() {
        let provider = MockLlmProvider::text("only one");
        let req = LlmRequest::new(vec![ChatMessage::user("hi")]);
        provider.complete(req.clone()).await.unwrap();
        let err = provider.complete(req).await.unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)));
    }

    #[tokio::test]
    async fn reports_model_name() {
        let provider = MockLlmProvider::text("x").with_model_name("mock-model");
        assert_eq!(provider.model_name(), "mock-model");
    }
}

//! Retry policy and a retrying [`LlmProvider`] wrapper.
//!
//! [`RetryProvider`] wraps any [`LlmProvider`] and retries transient failures
//! (transport and provider errors) with exponential backoff, mirroring
//! deepeval's `retry_policy`. Non-transient errors (parse failures, missing
//! structured output) are returned immediately.

use std::time::Duration;

use async_trait::async_trait;

use super::{LlmProvider, LlmRequest, LlmResponse};
use crate::error::LlmError;

/// A retry policy: how many times to retry and how to back off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetryPolicy {
    /// The maximum number of retry attempts after the initial call.
    pub max_retries: u32,
    /// The base delay before the first retry.
    pub base_delay: Duration,
    /// The maximum delay between retries (caps exponential growth).
    pub max_delay: Duration,
    /// The multiplier applied to the delay after each retry.
    pub backoff_factor: f64,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            base_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(8),
            backoff_factor: 2.0,
        }
    }
}

impl RetryPolicy {
    /// A policy that never retries.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// The delay to wait before retry attempt `attempt` (0-based).
    fn delay_for(&self, attempt: u32) -> Duration {
        let factor = self.backoff_factor.max(1.0).powi(attempt as i32);
        let millis = (self.base_delay.as_millis() as f64 * factor) as u64;
        Duration::from_millis(millis.min(self.max_delay.as_millis() as u64))
    }
}

/// Whether an [`LlmError`] is transient (worth retrying).
fn is_transient(error: &LlmError) -> bool {
    matches!(error, LlmError::Transport(_) | LlmError::Provider(_))
}

/// A [`LlmProvider`] that retries transient failures with backoff.
///
/// `P` is the inner provider type. The wrapper is `Clone` when the inner
/// provider is `Clone`.
#[derive(Clone)]
pub struct RetryProvider<P> {
    inner: P,
    policy: RetryPolicy,
}

impl<P> RetryProvider<P> {
    /// Wrap a provider with a retry policy.
    pub fn new(inner: P, policy: RetryPolicy) -> Self {
        Self { inner, policy }
    }

    /// Wrap a provider with the default retry policy.
    pub fn with_default_policy(inner: P) -> Self {
        Self::new(inner, RetryPolicy::default())
    }
}

#[async_trait]
impl<P> LlmProvider for RetryProvider<P>
where
    P: LlmProvider + Send + Sync,
{
    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let mut attempt = 0u32;
        loop {
            match self.inner.complete(request.clone()).await {
                Ok(response) => return Ok(response),
                Err(error) if is_transient(&error) && attempt < self.policy.max_retries => {
                    tokio::time::sleep(self.policy.delay_for(attempt)).await;
                    attempt += 1;
                }
                Err(error) => return Err(error),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{ChatMessage, MockLlmProvider};

    #[test]
    fn policy_backoff_grows_and_caps() {
        let policy = RetryPolicy {
            base_delay: Duration::from_millis(100),
            max_delay: Duration::from_millis(500),
            backoff_factor: 2.0,
            ..RetryPolicy::default()
        };
        assert_eq!(policy.delay_for(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for(2), Duration::from_millis(400));
        // Capped at max_delay.
        assert_eq!(policy.delay_for(5), Duration::from_millis(500));
    }

    #[tokio::test]
    async fn retries_transient_errors_then_succeeds() {
        // A provider that fails twice with a transport error, then succeeds.
        let inner = MockLlmProvider::new(vec![
            Err(LlmError::Transport("timeout".to_string())),
            Err(LlmError::Transport("timeout".to_string())),
            Ok(crate::llm::LlmResponse::new("ok")),
        ]);
        let provider = RetryProvider::new(inner, RetryPolicy::none().with_max_retries(3));

        let request = LlmRequest::new(vec![ChatMessage::user("hi")]);
        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.content, "ok");
    }

    #[tokio::test]
    async fn gives_up_after_max_retries() {
        let inner = MockLlmProvider::new(vec![
            Err(LlmError::Transport("timeout".to_string())),
            Err(LlmError::Transport("timeout".to_string())),
            Err(LlmError::Transport("timeout".to_string())),
        ]);
        let provider = RetryProvider::new(inner, RetryPolicy::none().with_max_retries(2));

        let request = LlmRequest::new(vec![ChatMessage::user("hi")]);
        let err = provider.complete(request).await.unwrap_err();
        assert!(matches!(err, LlmError::Transport(_)));
    }

    #[tokio::test]
    async fn does_not_retry_non_transient_errors() {
        let inner = MockLlmProvider::new(vec![Err(LlmError::Parse("bad".to_string()))]);
        let provider = RetryProvider::new(inner, RetryPolicy::none().with_max_retries(3));

        let request = LlmRequest::new(vec![ChatMessage::user("hi")]);
        let err = provider.complete(request).await.unwrap_err();
        assert!(matches!(err, LlmError::Parse(_)));
    }
}

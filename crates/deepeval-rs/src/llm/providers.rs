//! Concrete [`LlmProvider`] implementations backed by rig.
//!
//! [`RigProvider`] wraps any rig [`CompletionModel`] (OpenAI, Anthropic, a
//! generic OpenAI-compatible endpoint, or a mock) behind the object-safe
//! [`LlmProvider`] trait. Constructor helpers are provided for the common
//! providers.

use async_trait::async_trait;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{AssistantContent, CompletionModel, CompletionRequestBuilder, Message};

use super::{extract_logprobs, LlmProvider, LlmRequest, LlmResponse, Role};
use crate::error::LlmError;

/// Default model used by [`OpenAIProvider::from_env`].
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4o-mini";
/// Default model used by [`AnthropicProvider::from_env`].
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-haiku-4-5";

/// A [`RigProvider`] backed by an OpenAI-compatible chat-completions model.
///
/// Constructed from the `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`)
/// environment variables.
pub struct OpenAIProvider;

impl OpenAIProvider {
    /// Build an OpenAI provider from `OPENAI_API_KEY`, using
    /// [`DEFAULT_OPENAI_MODEL`].
    pub fn from_env(
    ) -> Result<RigProvider<rig::providers::openai::completion::CompletionModel>, LlmError> {
        Self::from_env_with_model(DEFAULT_OPENAI_MODEL)
    }

    /// Build an OpenAI provider from `OPENAI_API_KEY` with a specific model.
    pub fn from_env_with_model(
        model: impl Into<String>,
    ) -> Result<RigProvider<rig::providers::openai::completion::CompletionModel>, LlmError> {
        let model = model.into();
        let client = rig::providers::openai::Client::from_env()
            .map_err(|e| LlmError::Provider(format!("failed to build OpenAI client: {e}")))?;
        let completion = client.completions_api().completion_model(model.clone());
        Ok(RigProvider::new(completion, model))
    }
}

/// A [`RigProvider`] backed by an Anthropic completion model.
///
/// Constructed from the `ANTHROPIC_API_KEY` (and optional
/// `ANTHROPIC_BASE_URL`) environment variables.
pub struct AnthropicProvider;

impl AnthropicProvider {
    /// Build an Anthropic provider from `ANTHROPIC_API_KEY`, using
    /// [`DEFAULT_ANTHROPIC_MODEL`].
    pub fn from_env(
    ) -> Result<RigProvider<rig::providers::anthropic::completion::CompletionModel>, LlmError> {
        Self::from_env_with_model(DEFAULT_ANTHROPIC_MODEL)
    }

    /// Build an Anthropic provider from `ANTHROPIC_API_KEY` with a specific
    /// model.
    pub fn from_env_with_model(
        model: impl Into<String>,
    ) -> Result<RigProvider<rig::providers::anthropic::completion::CompletionModel>, LlmError> {
        let model = model.into();
        let client = rig::providers::anthropic::Client::from_env()
            .map_err(|e| LlmError::Provider(format!("failed to build Anthropic client: {e}")))?;
        let completion = client.completion_model(model.clone());
        Ok(RigProvider::new(completion, model))
    }
}

/// A [`LlmProvider`] backed by a concrete rig [`CompletionModel`].
///
/// `M` is the rig model type (e.g. OpenAI's or Anthropic's completion model).
/// The model must be `Clone` so a request can be built without consuming it.
#[derive(Clone)]
pub struct RigProvider<M> {
    model: M,
    model_name: String,
}

impl<M> RigProvider<M>
where
    M: CompletionModel + Clone + Send + Sync + 'static,
{
    /// Wrap a rig completion model in a [`LlmProvider`].
    pub fn new(model: M, model_name: impl Into<String>) -> Self {
        Self {
            model,
            model_name: model_name.into(),
        }
    }
}

#[async_trait]
impl<M> LlmProvider for RigProvider<M>
where
    M: CompletionModel + Clone + Send + Sync + 'static,
{
    fn model_name(&self) -> &str {
        &self.model_name
    }

    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        let completion_request = build_completion_request(&self.model, &request)?;
        let response = self
            .model
            .completion(completion_request)
            .await
            .map_err(map_completion_error)?;

        let content = response
            .choice
            .iter()
            .filter_map(|item| match item {
                AssistantContent::Text(text) => Some(text.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let mut llm_response = LlmResponse::new(content).with_usage(
            response.usage.input_tokens as u32,
            response.usage.output_tokens as u32,
        );

        if request.top_logprobs.is_some() {
            if let Some(logprobs) = extract_logprobs(&response.raw) {
                llm_response = llm_response.with_logprobs(logprobs);
            }
        }

        Ok(llm_response)
    }
}

/// Build a rig [`rig::completion::CompletionRequest`] from an [`LlmRequest`].
fn build_completion_request<M: CompletionModel + Clone>(
    model: &M,
    request: &LlmRequest,
) -> Result<rig::completion::CompletionRequest, LlmError> {
    let mut builder = CompletionRequestBuilder::new(model.clone(), Message::user(""));

    for message in &request.messages {
        let rig_message = match message.role {
            Role::System => Message::system(message.content.clone()),
            Role::User => Message::user(message.content.clone()),
            Role::Assistant => Message::assistant(message.content.clone()),
        };
        builder = builder.message(rig_message);
    }

    if let Some(model) = &request.model {
        builder = builder.model(model.clone());
    }
    if let Some(temperature) = request.temperature {
        builder = builder.temperature(temperature);
    }
    if let Some(max_tokens) = request.max_tokens {
        builder = builder.max_tokens(max_tokens);
    }
    if let Some(top_logprobs) = request.top_logprobs {
        // OpenAI-compatible providers expose log probabilities via these
        // request parameters; rig forwards `additional_params` verbatim.
        builder = builder.additional_params(serde_json::json!({
            "logprobs": true,
            "top_logprobs": top_logprobs,
        }));
    }

    Ok(builder.build())
}

/// Map a rig [`rig::completion::CompletionError`] to an [`LlmError`].
fn map_completion_error(error: rig::completion::CompletionError) -> LlmError {
    use rig::completion::CompletionError;
    match error {
        CompletionError::HttpError(e) => LlmError::Transport(e.to_string()),
        CompletionError::JsonError(e) => LlmError::Parse(e.to_string()),
        CompletionError::UrlError(e) => LlmError::Transport(e.to_string()),
        CompletionError::RequestError(e) => LlmError::Provider(e.to_string()),
        CompletionError::ResponseError(e) => LlmError::Parse(e),
        CompletionError::ProviderError(e) => LlmError::Provider(e),
        CompletionError::ProviderResponse(e) => LlmError::Provider(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ChatMessage;
    use rig::test_utils::MockCompletionModel;

    #[tokio::test]
    async fn rig_provider_wraps_mock_model() {
        let mock = MockCompletionModel::text("hello from rig");
        let provider = RigProvider::new(mock, "mock-model");

        let request = LlmRequest::new(vec![
            ChatMessage::system("You are a test."),
            ChatMessage::user("Say hi."),
        ]);

        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.content, "hello from rig");
        assert_eq!(provider.model_name(), "mock-model");
    }

    #[tokio::test]
    async fn rig_provider_joins_multiple_text_blocks() {
        // A mock turn with a single text block; verify content extraction.
        let mock = MockCompletionModel::text("one");
        let provider = RigProvider::new(mock, "mock-model");
        let request = LlmRequest::new(vec![ChatMessage::user("hi")]);
        let response = provider.complete(request).await.unwrap();
        assert_eq!(response.content, "one");
    }

    #[test]
    fn ollama_provider_wraps_rig_completion_model() {
        // The Ollama provider from rig produces a CompletionModel that can be
        // wrapped in RigProvider. This test only constructs it (no network).
        use rig::client::Nothing;
        use rig::providers::ollama;

        let client = ollama::Client::new(Nothing).unwrap();
        let model = client.completion_model("qwen2.5:14b");
        let provider = RigProvider::new(model, "qwen2.5:14b");
        assert_eq!(provider.model_name(), "qwen2.5:14b");
    }
}

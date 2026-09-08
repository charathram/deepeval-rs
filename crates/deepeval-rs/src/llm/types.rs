//! Request and response types for the [`LlmProvider`] trait.
//!
//! [`super::LlmProvider`]

use serde::{Deserialize, Serialize};

/// The role of a chat message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// A system instruction.
    System,
    /// A user message.
    User,
    /// An assistant message.
    Assistant,
}

/// A single chat message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// The role of the message.
    pub role: Role,
    /// The message content.
    pub content: String,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A request to an LLM provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// The messages to send, in order.
    pub messages: Vec<ChatMessage>,
    /// The model to use. If `None`, the provider's default model is used.
    pub model: Option<String>,
    /// The sampling temperature.
    pub temperature: Option<f64>,
    /// The maximum number of tokens to generate.
    pub max_tokens: Option<u64>,
    /// The number of top alternative tokens to return log probabilities for.
    ///
    /// When set, the provider is asked to return per-token log probabilities
    /// (and the top alternatives) on the response. Used by logprob-based
    /// scoring such as GEval.
    pub top_logprobs: Option<u32>,
}

impl LlmRequest {
    /// Create a request from a single user message.
    pub fn new(messages: Vec<ChatMessage>) -> Self {
        Self {
            messages,
            model: None,
            temperature: None,
            max_tokens: None,
            top_logprobs: None,
        }
    }

    /// Set the model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Set the temperature.
    pub fn with_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens.
    pub fn with_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Request log probabilities for the top `n` alternative tokens.
    pub fn with_top_logprobs(mut self, top_logprobs: u32) -> Self {
        self.top_logprobs = Some(top_logprobs);
        self
    }
}

/// The log probability of a single token.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprob {
    /// The token text.
    pub token: String,
    /// The log probability of the token.
    pub logprob: f64,
}

/// Log probabilities for one generated token position.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TokenLogprobs {
    /// The token text.
    pub token: String,
    /// The log probability of the token.
    pub logprob: f64,
    /// The top alternative tokens and their log probabilities.
    pub top_logprobs: Vec<TokenLogprob>,
}

/// A response from an LLM provider.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmResponse {
    /// The generated text content.
    pub content: String,
    /// The number of input (prompt) tokens used.
    pub input_tokens: u32,
    /// The number of output (completion) tokens used.
    pub output_tokens: u32,
    /// The estimated cost of the call, if known.
    pub cost: Option<f64>,
    /// Per-token log probabilities, if the request asked for them.
    pub logprobs: Option<Vec<TokenLogprobs>>,
}

impl LlmResponse {
    /// Create a response with the given content and no usage metadata.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            input_tokens: 0,
            output_tokens: 0,
            cost: None,
            logprobs: None,
        }
    }

    /// Set the token usage.
    pub fn with_usage(mut self, input_tokens: u32, output_tokens: u32) -> Self {
        self.input_tokens = input_tokens;
        self.output_tokens = output_tokens;
        self
    }

    /// Set the estimated cost.
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Set the per-token log probabilities.
    pub fn with_logprobs(mut self, logprobs: Vec<TokenLogprobs>) -> Self {
        self.logprobs = Some(logprobs);
        self
    }
}

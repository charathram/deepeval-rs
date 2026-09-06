//! Single-turn test-case types.
//!
//! [`LLMTestCase`] is the primary input to single-turn metrics. It mirrors
//! deepeval's `LLMTestCase` and the `SingleTurnParams` enum that metrics use
//! to declare which fields they require.

use serde::{Deserialize, Serialize};

/// The fields of an [`LLMTestCase`] that a metric may require.
///
/// Mirrors deepeval's `SingleTurnParams`. Metrics declare which fields they
/// need via [`LLMTestCase::evaluation_params`]-style configuration; the
/// [`Display`] impl yields the snake_case name used in prompts and errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SingleTurnParams {
    /// The user input to the LLM application.
    Input,
    /// The actual output produced by the LLM application.
    ActualOutput,
    /// The expected (ideal) output.
    ExpectedOutput,
    /// The retrieval context used by a RAG pipeline.
    RetrievalContext,
    /// The tools the application was expected to call.
    ExpectedTools,
    /// The criteria the output was expected to satisfy.
    ExpectedCriteria,
    /// Additional context provided to the application.
    Context,
    /// User feedback on the output.
    Feedbacks,
}

impl std::fmt::Display for SingleTurnParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Input => "input",
            Self::ActualOutput => "actual_output",
            Self::ExpectedOutput => "expected_output",
            Self::RetrievalContext => "retrieval_context",
            Self::ExpectedTools => "expected_tools",
            Self::ExpectedCriteria => "expected_criteria",
            Self::Context => "context",
            Self::Feedbacks => "feedbacks",
        };
        f.write_str(name)
    }
}

/// The type of a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    /// A function call.
    Function,
    /// A custom tool call.
    Custom,
}

/// A tool call made by an LLM application, and the tool call it was expected
/// to make.
///
/// Mirrors deepeval's `ToolCall`. `name`/`args` describe the actual call;
/// `expected_name`/`expected_args` describe the expected call (used by
/// tool-correctness metrics).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// The name of the tool that was called.
    pub name: String,
    /// The arguments passed to the tool.
    pub args: serde_json::Value,
    /// The name of the tool that was expected to be called.
    pub expected_name: Option<String>,
    /// The arguments that were expected to be passed.
    pub expected_args: Option<serde_json::Value>,
    /// The type of the tool call.
    pub kind: ToolCallType,
    /// Whether the tool call succeeded.
    pub success: Option<bool>,
}

impl ToolCall {
    /// Create a tool call with the given name and arguments.
    pub fn new(name: impl Into<String>, args: serde_json::Value) -> Self {
        Self {
            name: name.into(),
            args,
            expected_name: None,
            expected_args: None,
            kind: ToolCallType::Function,
            success: None,
        }
    }

    /// Set the expected tool name.
    pub fn with_expected_name(mut self, name: impl Into<String>) -> Self {
        self.expected_name = Some(name.into());
        self
    }

    /// Set the expected arguments.
    pub fn with_expected_args(mut self, args: serde_json::Value) -> Self {
        self.expected_args = Some(args);
        self
    }

    /// Set the tool call type.
    pub fn with_kind(mut self, kind: ToolCallType) -> Self {
        self.kind = kind;
        self
    }

    /// Set whether the tool call succeeded.
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self
    }
}

/// User feedback on an output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Feedback {
    /// The feedback text.
    pub text: String,
    /// The score associated with the feedback, if any.
    pub score: Option<f32>,
}

impl Feedback {
    /// Create feedback with the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            score: None,
        }
    }

    /// Set the feedback score.
    pub fn with_score(mut self, score: f32) -> Self {
        self.score = Some(score);
        self
    }
}

/// A single-turn LLM test case.
///
/// Mirrors deepeval's `LLMTestCase`. All fields are optional except `input`,
/// which is required. Metrics validate that the fields they need are present
/// and mark the metric as skipped otherwise.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LLMTestCase {
    /// The user input to the LLM application.
    pub input: String,
    /// The actual output produced by the LLM application.
    pub actual_output: Option<String>,
    /// The expected (ideal) output.
    pub expected_output: Option<String>,
    /// The retrieval context used by a RAG pipeline.
    pub retrieval_context: Option<Vec<String>>,
    /// The tools the application was expected to call.
    pub expected_tools: Option<Vec<ToolCall>>,
    /// The criteria the output was expected to satisfy.
    pub expected_criteria: Option<Vec<String>>,
    /// Additional context provided to the application.
    pub context: Option<Vec<String>>,
    /// User feedback on the output.
    pub feedbacks: Option<Vec<Feedback>>,
}

impl LLMTestCase {
    /// Start building a test case.
    pub fn builder() -> LLMTestCaseBuilder {
        LLMTestCaseBuilder::default()
    }

    /// The retrieval context, if any.
    pub fn retrieval_context(&self) -> Option<&[String]> {
        self.retrieval_context.as_deref()
    }
}

/// Builder for [`LLMTestCase`].
#[derive(Debug, Clone, Default)]
pub struct LLMTestCaseBuilder {
    input: String,
    actual_output: Option<String>,
    expected_output: Option<String>,
    retrieval_context: Option<Vec<String>>,
    expected_tools: Option<Vec<ToolCall>>,
    expected_criteria: Option<Vec<String>>,
    context: Option<Vec<String>>,
    feedbacks: Option<Vec<Feedback>>,
}

impl LLMTestCaseBuilder {
    /// Set the user input.
    pub fn input(mut self, input: impl Into<String>) -> Self {
        self.input = input.into();
        self
    }

    /// Set the actual output.
    pub fn actual_output(mut self, output: impl Into<String>) -> Self {
        self.actual_output = Some(output.into());
        self
    }

    /// Set the expected output.
    pub fn expected_output(mut self, output: impl Into<String>) -> Self {
        self.expected_output = Some(output.into());
        self
    }

    /// Set the retrieval context.
    pub fn retrieval_context(mut self, context: Vec<String>) -> Self {
        self.retrieval_context = Some(context);
        self
    }

    /// Set the expected tools.
    pub fn expected_tools(mut self, tools: Vec<ToolCall>) -> Self {
        self.expected_tools = Some(tools);
        self
    }

    /// Set the expected criteria.
    pub fn expected_criteria(mut self, criteria: Vec<String>) -> Self {
        self.expected_criteria = Some(criteria);
        self
    }

    /// Set the additional context.
    pub fn context(mut self, context: Vec<String>) -> Self {
        self.context = Some(context);
        self
    }

    /// Set the feedbacks.
    pub fn feedbacks(mut self, feedbacks: Vec<Feedback>) -> Self {
        self.feedbacks = Some(feedbacks);
        self
    }

    /// Build the test case.
    pub fn build(self) -> LLMTestCase {
        LLMTestCase {
            input: self.input,
            actual_output: self.actual_output,
            expected_output: self.expected_output,
            retrieval_context: self.retrieval_context,
            expected_tools: self.expected_tools,
            expected_criteria: self.expected_criteria,
            context: self.context,
            feedbacks: self.feedbacks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_sets_all_fields() {
        let tc = LLMTestCase::builder()
            .input("What if these shoes don't fit?")
            .actual_output("We offer a 30-day full refund.")
            .expected_output("We offer a 30-day full refund at no extra costs.")
            .retrieval_context(vec!["All customers get a 30 day refund.".to_string()])
            .expected_criteria(vec!["Mentions the refund policy.".to_string()])
            .build();

        assert_eq!(tc.input, "What if these shoes don't fit?");
        assert_eq!(
            tc.actual_output.as_deref(),
            Some("We offer a 30-day full refund.")
        );
        assert_eq!(
            tc.expected_output.as_deref(),
            Some("We offer a 30-day full refund at no extra costs.")
        );
        assert_eq!(
            tc.retrieval_context.as_deref(),
            Some(&["All customers get a 30 day refund.".to_string()][..])
        );
        assert_eq!(
            tc.expected_criteria.as_deref(),
            Some(&["Mentions the refund policy.".to_string()][..])
        );
    }

    #[test]
    fn builder_defaults_to_empty_input() {
        let tc = LLMTestCase::builder().build();
        assert_eq!(tc.input, "");
        assert!(tc.actual_output.is_none());
        assert!(tc.expected_output.is_none());
        assert!(tc.retrieval_context.is_none());
    }

    #[test]
    fn single_turn_params_display() {
        assert_eq!(SingleTurnParams::Input.to_string(), "input");
        assert_eq!(SingleTurnParams::ActualOutput.to_string(), "actual_output");
        assert_eq!(
            SingleTurnParams::RetrievalContext.to_string(),
            "retrieval_context"
        );
    }

    #[test]
    fn tool_call_builder() {
        let call = ToolCall::new("search", serde_json::json!({"q": "shoes"}))
            .with_expected_name("search")
            .with_expected_args(serde_json::json!({"q": "shoes"}))
            .with_success(true);

        assert_eq!(call.name, "search");
        assert_eq!(call.expected_name.as_deref(), Some("search"));
        assert_eq!(call.success, Some(true));
        assert_eq!(call.kind, ToolCallType::Function);
    }

    #[test]
    fn serde_round_trip() {
        let tc = LLMTestCase::builder()
            .input("hello")
            .actual_output("hi")
            .retrieval_context(vec!["ctx".to_string()])
            .build();
        let json = serde_json::to_string(&tc).unwrap();
        let back: LLMTestCase = serde_json::from_str(&json).unwrap();
        assert_eq!(tc, back);
    }
}

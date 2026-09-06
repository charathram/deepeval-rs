//! Multi-turn (conversational) test-case types.
//!
//! [`ConversationalTestCase`] models a multi-turn conversation as a sequence
//! of [`Turn`]s, mirroring deepeval's `ConversationalTestCase` and `Turn`.

use serde::{Deserialize, Serialize};

use super::single_turn::{Feedback, ToolCall};

/// The fields of a [`Turn`] that a metric may require.
///
/// Mirrors deepeval's `MultiTurnParams`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MultiTurnParams {
    /// The user input for a turn.
    Input,
    /// The actual output for a turn.
    ActualOutput,
    /// The expected output for a turn.
    ExpectedOutput,
    /// The retrieval context for a turn.
    RetrievalContext,
    /// The tools expected to be called in a turn.
    ExpectedTools,
    /// The criteria the turn output was expected to satisfy.
    ExpectedCriteria,
}

impl std::fmt::Display for MultiTurnParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Self::Input => "input",
            Self::ActualOutput => "actual_output",
            Self::ExpectedOutput => "expected_output",
            Self::RetrievalContext => "retrieval_context",
            Self::ExpectedTools => "expected_tools",
            Self::ExpectedCriteria => "expected_criteria",
        };
        f.write_str(name)
    }
}

/// A single turn in a multi-turn conversation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Turn {
    /// The user input for this turn.
    pub input: String,
    /// The actual output produced for this turn.
    pub actual_output: Option<String>,
    /// The expected output for this turn.
    pub expected_output: Option<String>,
    /// The retrieval context for this turn.
    pub retrieval_context: Option<Vec<String>>,
    /// The tools expected to be called in this turn.
    pub expected_tools: Option<Vec<ToolCall>>,
    /// The criteria the turn output was expected to satisfy.
    pub expected_criteria: Option<Vec<String>>,
    /// User feedback on this turn.
    pub feedbacks: Option<Vec<Feedback>>,
}

impl Turn {
    /// Start building a turn.
    pub fn builder() -> TurnBuilder {
        TurnBuilder::default()
    }
}

/// Builder for [`Turn`].
#[derive(Debug, Clone, Default)]
pub struct TurnBuilder {
    input: String,
    actual_output: Option<String>,
    expected_output: Option<String>,
    retrieval_context: Option<Vec<String>>,
    expected_tools: Option<Vec<ToolCall>>,
    expected_criteria: Option<Vec<String>>,
    feedbacks: Option<Vec<Feedback>>,
}

impl TurnBuilder {
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

    /// Set the feedbacks.
    pub fn feedbacks(mut self, feedbacks: Vec<Feedback>) -> Self {
        self.feedbacks = Some(feedbacks);
        self
    }

    /// Build the turn.
    pub fn build(self) -> Turn {
        Turn {
            input: self.input,
            actual_output: self.actual_output,
            expected_output: self.expected_output,
            retrieval_context: self.retrieval_context,
            expected_tools: self.expected_tools,
            expected_criteria: self.expected_criteria,
            feedbacks: self.feedbacks,
        }
    }
}

/// A multi-turn conversational test case.
///
/// Mirrors deepeval's `ConversationalTestCase`. Metrics that operate over a
/// conversation consume the full sequence of [`Turn`]s.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ConversationalTestCase {
    /// The ordered turns of the conversation.
    pub turns: Vec<Turn>,
}

impl ConversationalTestCase {
    /// Start building a conversational test case.
    pub fn builder() -> ConversationalTestCaseBuilder {
        ConversationalTestCaseBuilder::default()
    }
}

/// Builder for [`ConversationalTestCase`].
#[derive(Debug, Clone, Default)]
pub struct ConversationalTestCaseBuilder {
    turns: Vec<Turn>,
}

impl ConversationalTestCaseBuilder {
    /// Add a turn.
    pub fn turn(mut self, turn: Turn) -> Self {
        self.turns.push(turn);
        self
    }

    /// Add multiple turns.
    pub fn turns(mut self, turns: Vec<Turn>) -> Self {
        self.turns.extend(turns);
        self
    }

    /// Build the conversational test case.
    pub fn build(self) -> ConversationalTestCase {
        ConversationalTestCase { turns: self.turns }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turn_builder_sets_fields() {
        let turn = Turn::builder()
            .input("What if these shoes don't fit?")
            .actual_output("We offer a 30-day refund.")
            .expected_output("We offer a 30-day full refund.")
            .retrieval_context(vec!["Refund policy.".to_string()])
            .build();

        assert_eq!(turn.input, "What if these shoes don't fit?");
        assert_eq!(
            turn.actual_output.as_deref(),
            Some("We offer a 30-day refund.")
        );
        assert_eq!(
            turn.retrieval_context.as_deref(),
            Some(&["Refund policy.".to_string()][..])
        );
    }

    #[test]
    fn conversational_builder_accumulates_turns() {
        let tc = ConversationalTestCase::builder()
            .turn(Turn::builder().input("hi").actual_output("hello").build())
            .turn(
                Turn::builder()
                    .input("bye")
                    .actual_output("goodbye")
                    .build(),
            )
            .build();

        assert_eq!(tc.turns.len(), 2);
        assert_eq!(tc.turns[0].input, "hi");
        assert_eq!(tc.turns[1].input, "bye");
    }

    #[test]
    fn multi_turn_params_display() {
        assert_eq!(MultiTurnParams::Input.to_string(), "input");
        assert_eq!(MultiTurnParams::ActualOutput.to_string(), "actual_output");
    }

    #[test]
    fn serde_round_trip() {
        let tc = ConversationalTestCase::builder()
            .turn(Turn::builder().input("hi").actual_output("hello").build())
            .build();
        let json = serde_json::to_string(&tc).unwrap();
        let back: ConversationalTestCase = serde_json::from_str(&json).unwrap();
        assert_eq!(tc, back);
    }
}

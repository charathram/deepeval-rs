//! Test-case data models for deepeval-rs.
//!
//! These types mirror deepeval's `deepeval/test_case/` module. A test case
//! captures the inputs and expected outputs of a single LLM application turn
//! (or a multi-turn conversation), which metrics then evaluate.

mod conversational;
mod single_turn;

pub use conversational::{ConversationalTestCase, MultiTurnParams, Turn};
pub use single_turn::{Feedback, LLMTestCase, SingleTurnParams, ToolCall, ToolCallType};

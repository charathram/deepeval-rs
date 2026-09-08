//! GEval — score an output against custom criteria.
//!
//! Run with: `cargo run --example geval`
//!
//! GEval is an LLM-judge metric that evaluates an output against
//! user-supplied `criteria`. Optional `evaluation_steps` guide the judge's
//! reasoning. This example uses a [`MockLlmProvider`] so it runs keyless.

use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{GEval, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_case = LLMTestCase::builder()
        .input("Explain the return policy in one sentence.")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .build();

    // The mock replays a canned verdict. GEval asks the judge for an integer
    // score in a range (default 0-10) and confidence-weights it against token
    // log probabilities before normalizing to 0-1. A real provider would judge
    // the output against the criteria below.
    let provider = MockLlmProvider::text(r#"{"score": 9, "reason": "concise and accurate"}"#);

    let mut metric = GEval::builder()
        .provider(provider)
        .criteria("The answer is concise, accurate, and directly addresses the question.")
        .evaluation_steps(vec![
            "Check that the answer is a single sentence.".to_string(),
            "Check that the answer states the refund terms.".to_string(),
        ])
        .threshold(0.7)
        .include_reason(true)
        .build();

    metric.measure(&test_case).await?;

    println!("GEval score: {:.2}", metric.score().unwrap_or(0.0));
    println!("Passed:      {}", metric.is_successful() == Some(true));
    if let Some(reason) = metric.reason() {
        println!("Reason:      {reason}");
    }

    Ok(())
}

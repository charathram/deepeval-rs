//! A full evaluation run — combine metrics over multiple test cases.
//!
//! Run with: `cargo run --example evaluate_run`
//!
//! [`evaluate`] runs a set of metrics over a set of test cases concurrently
//! and returns a serializable [`EvalReport`]. This example mixes a
//! deterministic metric (no LLM) with an LLM-judge metric (mock provider) and
//! prints a summary.

use deepeval_rs::eval::evaluate;
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, ExactMatchMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Two test cases with different inputs.
    let test_cases = vec![
        LLMTestCase::builder()
            .input("What is the return policy?")
            .actual_output("We offer a 30-day full refund.")
            .expected_output("We offer a 30-day full refund.")
            .build(),
        LLMTestCase::builder()
            .input("What is the shipping cost?")
            .actual_output("Shipping is free over $50.")
            .expected_output("Shipping is free over $50.")
            .build(),
    ];

    // A deterministic metric and an LLM-judge metric. The mock replays one
    // verdict per call; `evaluate` measures the LLM-judge metric once per test
    // case, so provide one response per test case.
    let provider = MockLlmProvider::responses([
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9}"#),
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9}"#),
    ]);
    let metrics: Vec<Box<dyn deepeval_rs::metrics::Metric>> = vec![
        Box::new(ExactMatchMetric::builder().threshold(0.5).build()),
        Box::new(
            AnswerRelevancyMetric::builder()
                .provider(provider)
                .threshold(0.7)
                .build(),
        ),
    ];

    let report = evaluate(&test_cases, &metrics).await;

    println!("Measurements: {}", report.total_measurements());
    println!("Passed:       {}", report.passed());
    println!("Failed:       {}", report.failed());
    println!("Skipped:      {}", report.skipped());
    println!("Errored:      {}", report.errored());

    // The report is serializable, so you can write it to JSON.
    println!("\n{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}

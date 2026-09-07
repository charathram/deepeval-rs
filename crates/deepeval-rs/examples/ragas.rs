//! RAGAS composite metric — average of four RAG sub-metrics.
//!
//! Run with: `cargo run --example ragas`
//!
//! [`RagasMetric`] measures answer relevancy, faithfulness, contextual
//! precision, and contextual recall, then averages them into a single 0-1
//! score. This example uses [`MockLlmProvider`] so it runs with no API key. To
//! use a real model, swap the mock for a [`RigProvider`] wrapping a rig
//! completion model — see `real_provider_openai` and `real_provider_anthropic`
//! for complete wiring examples.

use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{Metric, RagasMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A grounded, relevant answer to a question, with a retrieval context.
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .expected_output("Customers get a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

    // RagasMetric measures four sub-metrics, so the provider must supply one
    // response per sub-metric (in order: answer relevancy, faithfulness,
    // contextual precision, contextual recall).
    let mut ragas = RagasMetric::builder()
        .provider(MockLlmProvider::responses([
            deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9, "reason": "relevant"}"#),
            deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.8, "reason": "grounded"}"#),
            deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.7, "reason": "precise"}"#),
            deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.6, "reason": "recalled"}"#),
        ]))
        .threshold(0.5)
        .build();

    ragas.measure(&test_case).await?;

    let status = match ragas.is_successful() {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "SKIPPED",
    };
    println!(
        "RagasMetric score={:.2} {status}",
        ragas.score().unwrap_or(0.0)
    );
    if let Some(reason) = ragas.reason() {
        println!("reason: {reason}");
    }

    Ok(())
}

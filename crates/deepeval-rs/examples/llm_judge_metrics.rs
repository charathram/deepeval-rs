//! LLM-judge metrics — score an output with an LLM.
//!
//! Run with: `cargo run --example llm_judge_metrics`
//!
//! These metrics ask an LLM to judge the output and return a 0-1 score. This
//! example uses [`MockLlmProvider`] so it runs with no API key. To use a real
//! model, swap the mock for a [`RigProvider`] wrapping a rig completion model —
//! see `real_provider_openai` and `real_provider_anthropic` for complete
//! wiring examples.

use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, Metric, PromptAlignmentMetric,
};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A grounded, relevant answer to a question.
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

    // Each metric needs its own provider (the mock replays a canned verdict
    // once per call, so a fresh provider per metric avoids exhausting a shared
    // queue).
    let mut relevancy = AnswerRelevancyMetric::builder()
        .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
        .threshold(0.7)
        .build();
    relevancy.measure(&test_case).await?;
    print_result("AnswerRelevancyMetric", &relevancy);

    let mut faithfulness = FaithfulnessMetric::builder()
        .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
        .threshold(0.7)
        .build();
    faithfulness.measure(&test_case).await?;
    print_result("FaithfulnessMetric", &faithfulness);

    let mut hallucination = HallucinationMetric::builder()
        .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
        .threshold(0.7)
        .build();
    hallucination.measure(&test_case).await?;
    print_result("HallucinationMetric", &hallucination);

    let mut alignment = PromptAlignmentMetric::builder()
        .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
        .threshold(0.7)
        .build();
    alignment.measure(&test_case).await?;
    print_result("PromptAlignmentMetric", &alignment);

    Ok(())
}

/// Print a metric's name, score, and pass/fail status.
fn print_result(name: &str, metric: &dyn Metric) {
    let status = match metric.is_successful() {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "SKIPPED",
    };
    println!(
        "{name:<22} score={:<4} {status}",
        metric
            .score()
            .map_or_else(|| "-".to_string(), |s| format!("{s:.2}"))
    );
}

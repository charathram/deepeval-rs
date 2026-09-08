//! Real provider: OpenAI via [`OpenAIProvider`].
//!
//! Run with: `OPENAI_API_KEY=... cargo run --example real_provider_openai`
//!
//! This shows how to replace the [`MockLlmProvider`] used in the other
//! examples with a real model. [`OpenAIProvider::from_env`] reads
//! `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) and wraps a rig OpenAI
//! completion model behind the [`LlmProvider`] seam, so it can be handed
//! straight to any metric builder.

use deepeval_rs::llm::OpenAIProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build an OpenAI provider from the environment. Use
    //    `from_env_with_model("gpt-4o")` to pick a specific model instead of
    //    the default (`gpt-4o-mini`).
    let provider = OpenAIProvider::from_env()?;

    // 2. Use it like any other provider.
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

    let mut metric = AnswerRelevancyMetric::builder()
        .provider(provider)
        .threshold(0.7)
        .include_reason(true)
        .build();

    metric.measure(&test_case).await?;

    println!("score:  {:.2}", metric.score().unwrap_or(0.0));
    println!("passed: {}", metric.is_successful() == Some(true));
    if let Some(reason) = metric.reason() {
        println!("reason: {reason}");
    }

    Ok(())
}

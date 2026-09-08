//! Real provider: Anthropic via [`AnthropicProvider`].
//!
//! Run with: `ANTHROPIC_API_KEY=... cargo run --example real_provider_anthropic`
//!
//! The same pattern as `real_provider_openai`, but with Anthropic.
//! [`AnthropicProvider::from_env`] reads `ANTHROPIC_API_KEY` (and optional
//! `ANTHROPIC_BASE_URL`) and wraps a rig Anthropic completion model behind the
//! [`LlmProvider`] seam.

use deepeval_rs::llm::AnthropicProvider;
use deepeval_rs::metrics::{FaithfulnessMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Build an Anthropic provider from the environment. Use
    //    `from_env_with_model("claude-sonnet-4-6")` to pick a specific model
    //    instead of the default (`claude-haiku-4-5`).
    let provider = AnthropicProvider::from_env()?;

    // 2. Use it like any other provider.
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

    let mut metric = FaithfulnessMetric::builder()
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

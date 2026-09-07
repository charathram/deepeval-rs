//! Real provider: Anthropic via [`RigProvider`].
//!
//! Run with: `ANTHROPIC_API_KEY=... cargo run --example real_provider_anthropic`
//!
//! The same pattern as `real_provider_openai`, but with Anthropic. Build a rig
//! Anthropic client, get a completion model, wrap it in a [`RigProvider`], and
//! hand that to a metric.

use deepeval_rs::llm::RigProvider;
use deepeval_rs::metrics::{FaithfulnessMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;
use rig::client::CompletionClient;
use rig::providers::anthropic;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read the API key from the environment.
    let api_key = match std::env::var("ANTHROPIC_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("ANTHROPIC_API_KEY is not set. Set it and re-run.");
            std::process::exit(1);
        }
    };

    // 2. Build a rig Anthropic client and a completion model.
    let client = anthropic::Client::new(api_key)?;
    let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);

    // 3. Wrap the rig model in a RigProvider (the LlmProvider seam).
    let provider = RigProvider::new(model, anthropic::completion::CLAUDE_SONNET_4_6);

    // 4. Use it like any other provider.
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

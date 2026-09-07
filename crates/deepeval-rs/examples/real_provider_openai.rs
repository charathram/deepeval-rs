//! Real provider: OpenAI via [`RigProvider`].
//!
//! Run with: `OPENAI_API_KEY=... cargo run --example real_provider_openai`
//!
//! This shows how to replace the [`MockLlmProvider`] used in the other
//! examples with a real model. We build a rig OpenAI client, get a completion
//! model, wrap it in a [`RigProvider`], and hand that to a metric.

use deepeval_rs::llm::RigProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;
use rig::client::CompletionClient;
use rig::providers::openai;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Read the API key from the environment.
    let api_key = match std::env::var("OPENAI_API_KEY") {
        Ok(key) => key,
        Err(_) => {
            eprintln!("OPENAI_API_KEY is not set. Set it and re-run.");
            std::process::exit(1);
        }
    };

    // 2. Build a rig OpenAI client and a completion model.
    let client = openai::Client::new(api_key)?;
    let model = client.completion_model(openai::GPT_5_2);

    // 3. Wrap the rig model in a RigProvider (the LlmProvider seam).
    let provider = RigProvider::new(model, openai::GPT_5_2);

    // 4. Use it like any other provider.
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

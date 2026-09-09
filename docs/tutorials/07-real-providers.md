# Tutorial 7: Real providers

Every tutorial so far used a [`MockLlmProvider`](crate::llm::MockLlmProvider)
so the examples run with no API key. In production you'll want a real model.
This tutorial shows how to swap the mock for a real provider, using OpenAI and
Anthropic. It combines the `real_provider_openai` and `real_provider_anthropic`
examples, which share the same pattern.

## What you'll build

Two programs — one for OpenAI, one for Anthropic — that build a provider from
the environment and hand it to a metric builder exactly like a mock:

```rust
// real_provider_openai.rs
use deepeval_rs::llm::OpenAIProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = OpenAIProvider::from_env()?;

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
```

```rust
// real_provider_anthropic.rs
use deepeval_rs::llm::AnthropicProvider;
use deepeval_rs::metrics::{FaithfulnessMetric, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let provider = AnthropicProvider::from_env()?;

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
```

## Prerequisites

- [Tutorial 2](02-llm-judge-metrics.md) — providers and the `LlmProvider` seam.
- Any metric tutorial — the pattern is identical regardless of metric.

## Step-by-step build

### 1. Build a provider from the environment

Both providers read their API key from the environment. `from_env` uses a
sensible default model; `from_env_with_model` lets you pick one:

```rust
// OpenAI — reads OPENAI_API_KEY (and optional OPENAI_BASE_URL)
let provider = OpenAIProvider::from_env()?;
// or: OpenAIProvider::from_env_with_model("gpt-4o")?

// Anthropic — reads ANTHROPIC_API_KEY (and optional ANTHROPIC_BASE_URL)
let provider = AnthropicProvider::from_env()?;
// or: AnthropicProvider::from_env_with_model("claude-sonnet-4-6")?
```

### 2. Use it like any other provider

The provider implements the same [`LlmProvider`](crate::llm::LlmProvider) seam
as the mock, so you hand it straight to a metric builder:

```rust
let mut metric = AnswerRelevancyMetric::builder()
    .provider(provider)
    .threshold(0.7)
    .include_reason(true)
    .build();
```

### 3. Measure and read the result

Identical to the mock-based tutorials:

```rust
metric.measure(&test_case).await?;

println!("score:  {:.2}", metric.score().unwrap_or(0.0));
println!("passed: {}", metric.is_successful() == Some(true));
if let Some(reason) = metric.reason() {
    println!("reason: {reason}");
}
```

## Key concepts

- **`OpenAIProvider` / `AnthropicProvider`** — real providers that wrap rig
  completion models behind the `LlmProvider` seam.
- **`from_env` / `from_env_with_model`** — constructors that read API keys from
  the environment.
- **Drop-in replacement** — a real provider is used exactly like a mock, so
  swapping is a one-line change.

## Run it

```bash
OPENAI_API_KEY=... cargo run --example real_provider_openai
ANTHROPIC_API_KEY=... cargo run --example real_provider_anthropic
```

The output format matches the mock examples, but the score and reason now come
from a real model judging your output.

## Exercise

Swap the mock in the `geval` example ([Tutorial 3](03-geval.md)) for an
`OpenAIProvider` and run it with a real key. Then try `from_env_with_model` to
compare how a larger model scores the same output.

This is the final tutorial in the series. Revisit
[the index](README.md) to review the full curriculum, or run the complete
`evaluate_run` example to see everything combined.

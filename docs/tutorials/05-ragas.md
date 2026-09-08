# Tutorial 5: RAGAS

RAGAS is a composite metric for evaluating **retrieval-augmented generation**
(RAG) pipelines. [`RagasMetric`](crate::metrics::RagasMetric) measures four
sub-metrics — answer relevancy, faithfulness, contextual precision, and
contextual recall — and averages them into a single 0–1 score. This tutorial
builds the `ragas` example.

## What you'll build

A program that scores a grounded RAG answer against a retrieval context:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{Metric, RagasMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .expected_output("Customers get a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

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
```

## Prerequisites

- [Tutorial 2](02-llm-judge-metrics.md) — providers and LLM-judge metrics.
- [Tutorial 4](04-evaluate-run.md) — multiple metrics over multiple cases.

## Step-by-step build

### 1. Provide a retrieval context

RAG metrics need to know what the system retrieved. Set `retrieval_context` on
the test case:

```rust
let test_case = LLMTestCase::builder()
    .input("What is the return policy?")
    .actual_output("We offer a 30-day full refund at no extra costs.")
    .expected_output("Customers get a 30-day full refund at no extra costs.")
    .retrieval_context(vec![
        "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
    ])
    .build();
```

### 2. Supply one response per sub-metric

`RagasMetric` measures **four** sub-metrics, so the provider must supply one
response per sub-metric, in order: answer relevancy, faithfulness, contextual
precision, contextual recall:

```rust
let mut ragas = RagasMetric::builder()
    .provider(MockLlmProvider::responses([
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9, "reason": "relevant"}"#),
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.8, "reason": "grounded"}"#),
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.7, "reason": "precise"}"#),
        deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.6, "reason": "recalled"}"#),
    ]))
    .threshold(0.5)
    .build();
```

### 3. Measure and read the result

`score()` returns the average of the four sub-metrics; `reason()` returns the
last sub-metric's reason:

```rust
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
```

## Key concepts

- **Composite metric** — `RagasMetric` averages four sub-metrics into one score.
- **`retrieval_context`** — the documents the RAG system retrieved; required
  for RAG metrics.
- **Sub-metric order** — answer relevancy, faithfulness, contextual precision,
  contextual recall.
- **One response per sub-metric** — with a mock, supply four responses.

## Run it

```bash
cargo run --example ragas
```

Expected output:

```
RagasMetric score=0.75 PASS
reason: recalled
```

## Exercise

Drop one of the four responses from the mock's list and re-run. The metric
should error or skip because the provider ran out of responses — a useful
reminder that the response count must match the number of sub-metrics.

Next: [Tutorial 6 — Multi-turn conversations](06-multiturn.md), where you'll
evaluate conversational and agentic metrics.

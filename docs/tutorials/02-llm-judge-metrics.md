# Tutorial 2: LLM-judge metrics

LLM-judge metrics ask an **LLM to judge the output** and return a 0–1 score.
Unlike the deterministic metrics from [Tutorial 1](01-deterministic-metrics.md),
they need a **provider** — the thing that talks to the model. This tutorial
builds the `llm_judge_metrics` example, using `MockLlmProvider` so it runs
keyless.

## What you'll build

A program that measures four LLM-judge metrics — answer relevancy, faithfulness,
hallucination, and prompt alignment — against a single test case:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{
    AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric, Metric, PromptAlignmentMetric,
};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_case = LLMTestCase::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .retrieval_context(vec![
            "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
        ])
        .build();

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
```

## Prerequisites

- [Tutorial 1](01-deterministic-metrics.md) — test cases, builders, `measure`,
  score & threshold.

## Step-by-step build

### 1. Understand the provider

An LLM-judge metric needs a [`LlmProvider`](crate::llm::LlmProvider) — the seam
that generates completions. `deepeval-rs` ships a
[`MockLlmProvider`](crate::llm::MockLlmProvider) for tests and demos. It replays
a canned response instead of calling a real model, so it runs with no API key.

```rust
use deepeval_rs::llm::MockLlmProvider;

let provider = MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#);
```

The mock returns a JSON verdict with a `score` (and optionally a `reason`).
A real provider would return the same shape from the model.

### 2. Build a metric with a provider

Pass the provider to the metric's builder, just like you passed a threshold in
Tutorial 1:

```rust
let mut relevancy = AnswerRelevancyMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
    .threshold(0.7)
    .build();
relevancy.measure(&test_case).await?;
```

### 3. Give each metric its own provider

The mock replays a canned verdict **once per call**. If you share one provider
across several metrics, the response queue can be exhausted. Give each metric a
fresh provider:

```rust
let mut faithfulness = FaithfulnessMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "good"}"#))
    .threshold(0.7)
    .build();
faithfulness.measure(&test_case).await?;
```

Repeat for `HallucinationMetric` and `PromptAlignmentMetric`.

### 4. What each metric judges

All four judge the same test case but ask different questions:

- **`AnswerRelevancyMetric`** — is the output relevant to the input?
- **`FaithfulnessMetric`** — is the output grounded in the retrieval context?
- **`HallucinationMetric`** — does the output invent facts not in the context?
- **`PromptAlignmentMetric`** — does the output follow the expected criteria?

The test case includes a `retrieval_context` because faithfulness and
hallucination need it to judge grounding:

```rust
let test_case = LLMTestCase::builder()
    .input("What is the return policy?")
    .actual_output("We offer a 30-day full refund at no extra costs.")
    .retrieval_context(vec![
        "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
    ])
    .build();
```

## Key concepts

- **Provider** — the `LlmProvider` seam that generates completions.
- **`MockLlmProvider`** — replays canned verdicts; runs keyless.
- **LLM-as-a-judge** — the metric prompts the model to score the output and
  parses the JSON verdict.
- **One provider per metric** — with a mock, avoid sharing a response queue.

## Run it

```bash
cargo run --example llm_judge_metrics
```

Expected output:

```
AnswerRelevancyMetric  score=0.90 PASS
FaithfulnessMetric     score=0.90 PASS
HallucinationMetric    score=0.90 PASS
PromptAlignmentMetric  score=0.90 PASS
```

## Exercise

Change the mock's canned score to `0.5` and re-run. All four metrics should now
report `FAIL` (below the `0.7` threshold). This shows how the threshold gates
pass/fail.

Next: [Tutorial 3 — GEval](03-geval.md), where you'll score an output against
custom criteria.

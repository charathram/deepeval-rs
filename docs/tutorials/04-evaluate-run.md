# Tutorial 4: A full evaluation run

So far you've measured one metric against one test case at a time. Real
evaluations combine **many metrics over many test cases**. The
[`evaluate`](crate::eval::evaluate) function does this concurrently and returns
a serializable [`EvalReport`](crate::eval::EvalReport). This tutorial builds
the `evaluate_run` example.

## What you'll build

A program that runs a deterministic metric and an LLM-judge metric over two test
cases, then prints a summary and the full report as JSON:

```rust
use deepeval_rs::eval::evaluate;
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, ExactMatchMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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

    println!("\n{}", serde_json::to_string_pretty(&report)?);

    Ok(())
}
```

## Prerequisites

- [Tutorial 1](01-deterministic-metrics.md) — deterministic metrics.
- [Tutorial 2](02-llm-judge-metrics.md) — providers and LLM-judge metrics.

## Step-by-step build

### 1. Build multiple test cases

`evaluate` takes a slice of test cases. Build a `Vec`:

```rust
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
```

### 2. Build a heterogeneous metric list

`evaluate` takes `&[Box<dyn Metric>]`, so you can mix different metric types.
Box each metric:

```rust
let metrics: Vec<Box<dyn deepeval_rs::metrics::Metric>> = vec![
    Box::new(ExactMatchMetric::builder().threshold(0.5).build()),
    Box::new(
        AnswerRelevancyMetric::builder()
            .provider(provider)
            .threshold(0.7)
            .build(),
    ),
];
```

### 3. Provide one response per LLM-judge call

`evaluate` measures the LLM-judge metric **once per test case**. With a mock,
you must supply one response per call. Use `MockLlmProvider::responses` with a
list — here, two test cases means two responses:

```rust
let provider = MockLlmProvider::responses([
    deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9}"#),
    deepeval_rs::llm::LlmResponse::new(r#"{"score": 0.9}"#),
]);
```

### 4. Run the evaluation

```rust
let report = evaluate(&test_cases, &metrics).await;
```

`evaluate` runs the metrics over the test cases concurrently and returns an
`EvalReport`.

### 5. Read the summary

`EvalReport` exposes aggregate counts:

```rust
println!("Measurements: {}", report.total_measurements());
println!("Passed:       {}", report.passed());
println!("Failed:       {}", report.failed());
println!("Skipped:      {}", report.skipped());
println!("Errored:      {}", report.errored());
```

### 6. Serialize the report

`EvalReport` is `Serialize`, so you can write it to JSON for storage or
dashboards:

```rust
println!("\n{}", serde_json::to_string_pretty(&report)?);
```

## Key concepts

- **`evaluate`** — runs many metrics over many test cases concurrently.
- **`EvalReport`** — aggregate counts (`passed`, `failed`, `skipped`,
  `errored`, `total_measurements`) plus per-case results.
- **`Box<dyn Metric>`** — heterogeneous metrics in one list.
- **Response count** — with a mock, supply one response per LLM-judge call.

## Run it

```bash
cargo run --example evaluate_run
```

Expected output (abridged):

```
Measurements: 4
Passed:       4
Failed:       0
Skipped:      0
Errored:      0

{
  "per_case": [ ... ],
  ...
}
```

## Exercise

Add a third test case and a third response to the mock's list, then re-run.
`Measurements` should increase by 2 (one per metric). If you forget the extra
response, the mock's queue runs out — observe what happens.

Next: [Tutorial 5 — RAGAS](05-ragas.md), where you'll evaluate a RAG pipeline.

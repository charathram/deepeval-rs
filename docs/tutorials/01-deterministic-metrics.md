# Tutorial 1: Deterministic metrics

Deterministic metrics score an output **without calling an LLM**. They're the
simplest metrics in `deepeval-rs` and the best place to start: they need no API
key, no provider, and no network. This tutorial builds the
`deterministic_metrics` example from scratch.

## What you'll build

A small program that measures three deterministic metrics against a test case
and prints each metric's score and pass/fail status:

```rust
use deepeval_rs::metrics::{ExactMatchMetric, JsonCorrectnessMetric, Metric, PatternMatchMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let phone_case = LLMTestCase::builder()
        .input("What is the support phone number?")
        .actual_output("Call us at 555-1234.")
        .expected_output("Call us at 555-1234.")
        .build();

    let mut exact = ExactMatchMetric::builder().threshold(0.5).build();
    exact.measure(&phone_case).await?;
    print_result("ExactMatchMetric", &exact);

    let mut pattern = PatternMatchMetric::builder()
        .pattern(r"\d{3}-\d{4}")
        .threshold(0.5)
        .build();
    pattern.measure(&phone_case).await?;
    print_result("PatternMatchMetric", &pattern);

    let json_case = LLMTestCase::builder()
        .input("Return the capital of France as JSON.")
        .actual_output(r#"{"capital": "Paris"}"#)
        .build();
    let mut json = JsonCorrectnessMetric::builder().threshold(0.5).build();
    json.measure(&json_case).await?;
    print_result("JsonCorrectnessMetric", &json);

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

- A project with `deepeval-rs` and `tokio` as dependencies (see
  [Getting Started](../getting-started.md)).
- No prior tutorials required — this is the foundation.

## Step-by-step build

### 1. Create a test case

Every metric measures a [`LLMTestCase`](crate::test_case::LLMTestCase) — the
input and output of a single LLM application turn. Build one with the builder:

```rust
let phone_case = LLMTestCase::builder()
    .input("What is the support phone number?")
    .actual_output("Call us at 555-1234.")
    .expected_output("Call us at 555-1234.")
    .build();
```

- `input` — the prompt given to the application.
- `actual_output` — what the application produced.
- `expected_output` — the ideal output (used by exact-match).

### 2. Measure a metric

A metric is built with a builder and **measured** against a test case. The
`measure` call is `async`, so the function is `#[tokio::main]`:

```rust
let mut exact = ExactMatchMetric::builder().threshold(0.5).build();
exact.measure(&phone_case).await?;
```

`ExactMatchMetric` scores `1.0` when `actual_output` equals `expected_output`,
and `0.0` otherwise. The `threshold(0.5)` sets the pass/fail boundary.

### 3. Add a pattern metric

`PatternMatchMetric` scores `1.0` when `actual_output` matches a regex:

```rust
let mut pattern = PatternMatchMetric::builder()
    .pattern(r"\d{3}-\d{4}")
    .threshold(0.5)
    .build();
pattern.measure(&phone_case).await?;
```

The pattern `\d{3}-\d{4}` matches a phone number like `555-1234`.

### 4. Add a JSON metric

`JsonCorrectnessMetric` scores `1.0` when `actual_output` parses as valid JSON.
It needs a test case whose output is JSON:

```rust
let json_case = LLMTestCase::builder()
    .input("Return the capital of France as JSON.")
    .actual_output(r#"{"capital": "Paris"}"#)
    .build();
let mut json = JsonCorrectnessMetric::builder().threshold(0.5).build();
json.measure(&json_case).await?;
```

### 5. Read the result

After measuring, a metric exposes its score and pass/fail status through the
[`Metric`](crate::metrics::Metric) trait:

- `metric.score()` → `Option<f32>` (the 0–1 score).
- `metric.is_successful()` → `Option<bool>` (`Some(true)` if `score >=
  threshold`, `None` if no threshold was set or no score was recorded).

The `print_result` helper formats these:

```rust
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

## Key concepts

- **Test case** — the unit of evaluation: input + actual output (+ optional
  expected output, retrieval context, etc.).
- **Builder pattern** — every metric is configured with a builder
  (`.builder()...build()`).
- **`measure`** — the async call that runs a metric against a test case.
- **Score & threshold** — a metric reports a 0–1 score; `is_successful()`
  compares it against the configured threshold.
- **Skipped** — a metric is skipped (not a failure) when a required field is
  missing. For example, `ExactMatchMetric` without an `expected_output` is
  skipped.

## Run it

```bash
cargo run --example deterministic_metrics
```

Expected output:

```
ExactMatchMetric       score=1.00 PASS
PatternMatchMetric     score=1.00 PASS
JsonCorrectnessMetric  score=1.00 PASS
```

## Exercise

Try changing the `actual_output` of `phone_case` so it no longer matches the
expected output, and re-run. `ExactMatchMetric` should now report `FAIL` while
`PatternMatchMetric` still passes (the phone number is still present).

Next: [Tutorial 2 — LLM-judge metrics](02-llm-judge-metrics.md), where you'll
introduce a provider and let an LLM judge the output.

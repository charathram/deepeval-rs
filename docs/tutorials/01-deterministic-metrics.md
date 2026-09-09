# Tutorial 1: Deterministic metrics

Deterministic metrics score an output **without calling an LLM**. They're the
simplest metrics in `deepeval-rs` and the best place to start: they need no API
key, no provider, and no network. This tutorial builds the
`deterministic_metrics` example from scratch.

## Learning objectives

By the end of this tutorial you'll be able to:

- Build an [`LLMTestCase`](crate::test_case::LLMTestCase) with the builder
  pattern.
- Configure and `measure` a deterministic metric against a test case.
- Read a metric's score, threshold, and pass/fail status.
- Summarize several metrics with a single shared helper.

## Key concepts

Before we write code, let's pin down the ideas this tutorial relies on. Later
tutorials build on these, so it's worth getting them right.

### What is an `LLMTestCase`?

An `LLMTestCase` is **plain data** that describes a single turn of an LLM
application. It holds the `input` (the prompt given to the app), the
`actual_output` (what the app produced), the `expected_output` (the ideal
answer), and a few optional fields you'll meet in later tutorials (like
`retrieval_context`).

The "LLM" in the name refers to the **application under evaluation**, not to
anything the test case itself does. A test case never calls a model — it's just
a struct that carries inputs and outputs so a metric can score them.

### Why does `LLMTestCase` need no API key?

Because it never calls an LLM. A key is only needed when a *metric* fires a
judge model to score a case. The deterministic metrics in this tutorial score
by **plain comparison** — string equality, a regex match, or JSON parsing — so
they need no provider and no key. (In [Tutorial 2](02-llm-judge-metrics.md) the
metrics *do* call an LLM, and that's exactly when you'll introduce a provider.)

### The `Metric` trait

Every metric implements the [`Metric`](crate::metrics::Metric) trait. After
`measure` runs, a metric exposes:

- `score()` → `Option<f32>` — the 0–1 score, or `None` if none was recorded.
- `reason()` → `Option<&str>` — a short explanation, when available.
- `is_successful()` → `Option<bool>` — `Some(true)` if `score >= threshold`,
  `Some(false)` if below, `None` if there's no threshold or no score.
- `skipped()` → `bool` — whether the metric was skipped (e.g. a required field
  was missing). Skipped metrics don't count as failures.
- `name()`, `cost()`, `input_tokens()`, `output_tokens()` — metadata.

### Score vs threshold

A metric reports a score; the **threshold** decides pass/fail. `is_successful()`
is just `score >= threshold`. If you don't set a threshold, or no score was
recorded, it returns `None` — the metric is neither passing nor failing.

### The builder pattern

Every metric (and test case) is configured with a builder: `.builder()`, chain
the options you want, then `.build()`. This keeps construction explicit and
readable.

### `measure` is async

`measure` is an `async` method, so the example runs inside a `#[tokio::main]`
async function. Even though deterministic metrics do no I/O, the API is uniform
across all metrics — including the LLM-judge ones that do await a model.

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

Notice the shape: build a test case, build a metric, `measure` it, then read
the result. Every metric in the library follows this same rhythm.

## Prerequisites

- A project with `deepeval-rs` and `tokio` as dependencies (see
  [Getting Started](../getting-started.md)).
- No prior tutorials required — this is the foundation.

## Step-by-step build

### 1. Create a test case

Every metric measures an [`LLMTestCase`](crate::test_case::LLMTestCase) — the
input and output of a single LLM application turn. Build one with the builder:

```rust
let phone_case = LLMTestCase::builder()
    .input("What is the support phone number?")
    .actual_output("Call us at 555-1234.")
    .expected_output("Call us at 555-1234.")
    .build();
```

The three fields we set are the core of any test case:

- `input` — the prompt given to the application.
- `actual_output` — what the application produced (this is what gets scored).
- `expected_output` — the ideal output, used by metrics that compare against a
  reference (like exact match).

There are more optional fields (`retrieval_context`, `expected_tools`, etc.)
that you'll use in later tutorials. For now, these three are enough.

### 2. Measure a metric

A metric is built with a builder and **measured** against a test case. The
`measure` call is `async`, which is why the function is `#[tokio::main]`:

```rust
let mut exact = ExactMatchMetric::builder().threshold(0.5).build();
exact.measure(&phone_case).await?;
```

`ExactMatchMetric` scores `1.0` when `actual_output` equals `expected_output`,
and `0.0` otherwise. The `threshold(0.5)` sets the pass/fail boundary: any score
`>= 0.5` passes, anything below fails.

Two details worth noting:

- `measure` takes `&mut self` and records the result on the metric, so the
  metric must be `mut`.
- If `expected_output` is missing, `ExactMatchMetric` **skips** the case rather
  than failing it — it sets `skipped = true` and records no score. That's why
  the test case above always sets `expected_output`.

### 3. Add a pattern metric

`PatternMatchMetric` scores `1.0` when `actual_output` matches a regex:

```rust
let mut pattern = PatternMatchMetric::builder()
    .pattern(r"\d{3}-\d{4}")
    .threshold(0.5)
    .build();
pattern.measure(&phone_case).await?;
```

The pattern `\d{3}-\d{4}` matches a phone number like `555-1234`: three digits,
a hyphen, then four digits. Unlike exact match, this metric doesn't need an
`expected_output` — it only checks that the output *contains* a match. If the
output had no phone number, the score would be `0.0` and the metric would fail.

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

Note the `r#"..."#` raw string: it lets us write the JSON with literal double
quotes without escaping them. This metric is useful for validating that an app
returns well-formed structured output.

### 5. Read the result

After measuring, a metric exposes its score and pass/fail status through the
[`Metric`](crate::metrics::Metric) trait:

- `metric.score()` → `Option<f32>` (the 0–1 score).
- `metric.is_successful()` → `Option<bool>` (`Some(true)` if `score >=
  threshold`, `None` if no threshold was set or no score was recorded).

The `print_result` helper formats these. It takes a `&dyn Metric` — a trait
object — so it can accept any metric type:

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

The `match` on `is_successful()` maps the three possible states to `PASS`,
`FAIL`, or `SKIPPED`. The `score()` formatting prints `-` when there's no score
and a two-decimal number otherwise.

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

## Key concepts recap

- **Test case** — the unit of evaluation: input + actual output (+ optional
  expected output, retrieval context, etc.). It's plain data and never calls an
  LLM, so it needs no API key.
- **Builder pattern** — every metric and test case is configured with a builder
  (`.builder()...build()`).
- **`measure`** — the async call that runs a metric against a test case and
  records the result on the metric.
- **Score & threshold** — a metric reports a 0–1 score; `is_successful()`
  compares it against the configured threshold (`score >= threshold`).
- **Skipped** — a metric is skipped (not a failure) when a required field is
  missing. For example, `ExactMatchMetric` without an `expected_output` is
  skipped.

## Exercise

Try changing the `actual_output` of `phone_case` so it no longer matches the
expected output, and re-run. `ExactMatchMetric` should now report `FAIL` while
`PatternMatchMetric` still passes (the phone number is still present). Then
remove `expected_output` from `phone_case` and re-run — `ExactMatchMetric`
should report `SKIPPED`.

Next: [Tutorial 2 — LLM-judge metrics](02-llm-judge-metrics.md), where you'll
introduce a provider and let an LLM judge the output.

# Tutorial 3: GEval

[`GEval`](crate::metrics::GEval) is an LLM-judge metric that scores an output
against **user-supplied criteria**. It's the most flexible metric in the
library: you describe what "good" means, and the judge scores the output
against it. This tutorial builds the `geval` example.

## What you'll build

A program that scores an output against custom criteria, with optional
`evaluation_steps` to guide the judge's reasoning:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{GEval, Metric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_case = LLMTestCase::builder()
        .input("Explain the return policy in one sentence.")
        .actual_output("We offer a 30-day full refund at no extra costs.")
        .build();

    let provider = MockLlmProvider::text(r#"{"score": 9, "reason": "concise and accurate"}"#);

    let mut metric = GEval::builder()
        .provider(provider)
        .criteria("The answer is concise, accurate, and directly addresses the question.")
        .evaluation_steps(vec![
            "Check that the answer is a single sentence.".to_string(),
            "Check that the answer states the refund terms.".to_string(),
        ])
        .threshold(0.7)
        .include_reason(true)
        .build();

    metric.measure(&test_case).await?;

    println!("GEval score: {:.2}", metric.score().unwrap_or(0.0));
    println!("Passed:      {}", metric.is_successful() == Some(true));
    if let Some(reason) = metric.reason() {
        println!("Reason:      {reason}");
    }

    Ok(())
}
```

## Prerequisites

- [Tutorial 2](02-llm-judge-metrics.md) — providers and LLM-judge metrics.

## Step-by-step build

### 1. Write the criteria

The heart of GEval is the `criteria` string — a plain-English description of
what a good output looks like:

```rust
.criteria("The answer is concise, accurate, and directly addresses the question.")
```

The judge scores the output against this criteria on a scale (default 0–10).

### 2. Add evaluation steps (optional)

`evaluation_steps` break the criteria into concrete checks that guide the
judge's reasoning. They're optional but improve consistency:

```rust
.evaluation_steps(vec![
    "Check that the answer is a single sentence.".to_string(),
    "Check that the answer states the refund terms.".to_string(),
])
```

### 3. Configure the rest

Like other metrics, GEval takes a provider and a threshold. `include_reason`
asks the judge to also return a textual explanation:

```rust
let mut metric = GEval::builder()
    .provider(provider)
    .criteria("The answer is concise, accurate, and directly addresses the question.")
    .evaluation_steps(vec![
        "Check that the answer is a single sentence.".to_string(),
        "Check that the answer states the refund terms.".to_string(),
    ])
    .threshold(0.7)
    .include_reason(true)
    .build();
```

### 4. Understand the scoring

GEval asks the judge for an **integer score** in a range (default 0–10). It then
**confidence-weights** that score against the token log probabilities of the
judge's response before normalizing to 0–1. This mirrors deepeval's g-eval fix
and makes the score more robust than a raw integer.

The mock replays a canned verdict:

```rust
let provider = MockLlmProvider::text(r#"{"score": 9, "reason": "concise and accurate"}"#);
```

### 5. Read the result

`score()` returns the normalized 0–1 value, and `reason()` returns the judge's
explanation when `include_reason` is set:

```rust
println!("GEval score: {:.2}", metric.score().unwrap_or(0.0));
println!("Passed:      {}", metric.is_successful() == Some(true));
if let Some(reason) = metric.reason() {
    println!("Reason:      {reason}");
}
```

## Key concepts

- **Criteria** — the plain-English definition of a good output.
- **Evaluation steps** — optional concrete checks that guide the judge.
- **Integer score + logprob weighting** — the judge returns an integer; GEval
  confidence-weights it against token log probabilities before normalizing.
- **`include_reason`** — capture the judge's textual explanation.

## Run it

```bash
cargo run --example geval
```

Expected output:

```
GEval score: 0.90
Passed:      true
Reason:      concise and accurate
```

## Exercise

Change the criteria to something the output clearly violates (e.g. "The answer
must be at least three sentences.") and set the mock's score to `2`. Re-run and
observe the lower normalized score and `FAIL` status.

Next: [Tutorial 4 — A full evaluation run](04-evaluate-run.md), where you'll
combine multiple metrics over multiple test cases.

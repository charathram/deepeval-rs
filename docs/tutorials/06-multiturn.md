# Tutorial 6: Multi-turn conversations

So far every metric scored a single input/output pair. **Conversational
metrics** evaluate a [`ConversationalTestCase`](crate::test_case::ConversationalTestCase) —
an ordered sequence of [`Turn`](crate::test_case::Turn)s — which is how you
score agentic, multi-turn conversations. This tutorial builds the `multiturn`
example.

## What you'll build

A program that measures three conversational metrics over a short
customer-support conversation:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{
    ConversationCompletenessMetric, ConversationalMetric, TurnFaithfulnessMetric,
    TurnRelevancyMetric,
};
use deepeval_rs::test_case::{ConversationalTestCase, Turn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_case = ConversationalTestCase::builder()
        .turn(
            Turn::builder()
                .input("What is the return policy?")
                .actual_output("We offer a 30-day full refund at no extra costs.")
                .retrieval_context(vec![
                    "All customers are eligible for a 30 day full refund.".to_string()
                ])
                .build(),
        )
        .turn(
            Turn::builder()
                .input("What if I need to return a gift?")
                .actual_output("Gift returns work the same way — 30 days, full refund.")
                .retrieval_context(vec![
                    "Gift recipients can return items within 30 days.".to_string()
                ])
                .build(),
        )
        .build();

    let metrics: Vec<Box<dyn ConversationalMetric>> = vec![
        Box::new(
            ConversationCompletenessMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.9, "reason": "complete"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
        Box::new(
            TurnRelevancyMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.8, "reason": "relevant"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
        Box::new(
            TurnFaithfulnessMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.85, "reason": "grounded"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
    ];

    let report =
        deepeval_rs::eval::evaluate_conversational(std::slice::from_ref(&test_case), &metrics)
            .await;

    let case = &report.per_case[0];
    for result in &case.results {
        let status = match result.success {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "SKIPPED",
        };
        println!("{:<28} {status}", result.name);
        if let Some(score) = result.score {
            println!("  score={score:.2}");
        }
        if let Some(reason) = &result.reason {
            println!("  reason: {reason}");
        }
    }

    Ok(())
}
```

## Prerequisites

- [Tutorial 2](02-llm-judge-metrics.md) — providers and LLM-judge metrics.
- [Tutorial 4](04-evaluate-run.md) — evaluation reports.

## Step-by-step build

### 1. Build a conversation from turns

A `ConversationalTestCase` is a sequence of `Turn`s. Each turn has its own
input, output, and (optionally) retrieval context:

```rust
let test_case = ConversationalTestCase::builder()
    .turn(
        Turn::builder()
            .input("What is the return policy?")
            .actual_output("We offer a 30-day full refund at no extra costs.")
            .retrieval_context(vec![
                "All customers are eligible for a 30 day full refund.".to_string()
            ])
            .build(),
    )
    .turn(
        Turn::builder()
            .input("What if I need to return a gift?")
            .actual_output("Gift returns work the same way — 30 days, full refund.")
            .retrieval_context(vec![
                "Gift recipients can return items within 30 days.".to_string()
            ])
            .build(),
    )
    .build();
```

### 2. Give each metric its own provider

Conversational metrics are measured independently, so each needs a **fresh**
provider. With a mock, that means one `MockLlmProvider` per metric — otherwise
the response queues would interfere:

```rust
let metrics: Vec<Box<dyn ConversationalMetric>> = vec![
    Box::new(
        ConversationCompletenessMetric::builder()
            .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "complete"}"#))
            .threshold(0.7)
            .include_reason(true)
            .build(),
    ),
    Box::new(
        TurnRelevancyMetric::builder()
            .provider(MockLlmProvider::text(r#"{"score": 0.8, "reason": "relevant"}"#))
            .threshold(0.7)
            .include_reason(true)
            .build(),
    ),
    Box::new(
        TurnFaithfulnessMetric::builder()
            .provider(MockLlmProvider::text(r#"{"score": 0.85, "reason": "grounded"}"#))
            .threshold(0.7)
            .include_reason(true)
            .build(),
    ),
];
```

### 3. Run the conversational evaluation

`evaluate_conversational` takes a slice of conversational test cases and a
slice of `Box<dyn ConversationalMetric>`:

```rust
let report =
    deepeval_rs::eval::evaluate_conversational(std::slice::from_ref(&test_case), &metrics)
        .await;
```

### 4. Read the per-case results

The report's `per_case` holds one entry per test case, each with a `results`
list. Each result has a `name`, `score`, `reason`, and `success`:

```rust
let case = &report.per_case[0];
for result in &case.results {
    let status = match result.success {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "SKIPPED",
    };
    println!("{:<28} {status}", result.name);
    if let Some(score) = result.score {
        println!("  score={score:.2}");
    }
    if let Some(reason) = &result.reason {
        println!("  reason: {reason}");
    }
}
```

## Key concepts

- **`ConversationalTestCase`** — an ordered sequence of `Turn`s.
- **`Turn`** — one exchange in the conversation, with its own input, output,
  and retrieval context.
- **`ConversationalMetric`** — the trait for metrics that score conversations.
- **One provider per metric** — each metric is measured independently, so give
  each its own provider.
- **`evaluate_conversational`** — runs conversational metrics and returns a
  report with per-case results.

## Run it

```bash
cargo run --example multiturn
```

Expected output:

```
ConversationCompletenessMetric PASS
  score=0.90
  reason: complete
TurnRelevancyMetric          PASS
  score=0.80
  reason: relevant
TurnFaithfulnessMetric       PASS
  score=0.85
  reason: grounded
```

## Exercise

Add a third turn to the conversation and re-run. The completeness metric
should still pass, but notice how the turn-level metrics now have more turns to
score. Try lowering one metric's threshold below its mock score to see a `FAIL`.

Next: [Tutorial 7 — Real providers](07-real-providers.md), where you'll swap
the mocks for real OpenAI and Anthropic models.

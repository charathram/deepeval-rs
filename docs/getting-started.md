# Getting Started

This guide shows how to use the `deepeval-rs` library to evaluate LLM
applications. It assumes you have a Rust project with `tokio` available.

## Add the dependency

```toml
[dependencies]
deepeval-rs = "0.1"
tokio = { version = "1", features = ["full"] }
```

## Core concepts

- **Test case** — the input and expected output of a single LLM application
  turn ([`LLMTestCase`](crate::test_case::LLMTestCase)), or a multi-turn
  conversation ([`ConversationalTestCase`](crate::test_case::ConversationalTestCase)).
- **Metric** — a single evaluation, e.g. answer relevancy or faithfulness.
  Metrics implement the [`Metric`](crate::metrics::Metric) trait.
- **Evaluation** — running one or more metrics over one or more test cases,
  either via [`evaluate`](crate::eval::evaluate) (returns a report) or
  [`assert_test`](crate::eval::assert_test) (fails if any metric fails).

## Build a test case

```rust
use deepeval_rs::test_case::LLMTestCase;

let test_case = LLMTestCase::builder()
    .input("What if these shoes don't fit?")
    .actual_output("We offer a 30-day full refund at no extra costs.")
    .expected_output("We offer a 30-day full refund at no extra costs.")
    .retrieval_context(vec![
        "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
    ])
    .build();
```

## Configure an LLM provider

Metrics that use an LLM-as-a-judge need a provider. The library ships a
[`MockLlmProvider`](crate::llm::MockLlmProvider) for tests and a
[`RigProvider`](crate::llm::RigProvider) that wraps any
[rig](https://github.com/0xplaygrounds/rig) completion model.

```rust
use deepeval_rs::llm::{ChatMessage, LlmProvider, LlmRequest, MockLlmProvider};

let provider = MockLlmProvider::text("hello");
let request = LlmRequest::new(vec![ChatMessage::user("Say hi.")]);
let response = provider.complete(request).await?;
```

> **Note:** concrete provider constructors (e.g. `OpenAIProvider::from_env()`)
> land in a later phase. For now, wrap a rig model with `RigProvider::new`.

## Build a metric

Metrics come in two flavors: **LLM-judge** metrics (which need a provider) and
**deterministic** metrics (which don't). Both are built with a builder and
scored 0–1 against a `threshold`.

### LLM-judge metrics

`GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`, `HallucinationMetric`,
`PromptAlignmentMetric`, and the RAG metrics (`ContextualPrecisionMetric`,
`ContextualRecallMetric`, `ContextualRelevancyMetric`, `RagasMetric`) ask an LLM
to judge the output. They need a provider and a threshold:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::AnswerRelevancyMetric;

let metric = AnswerRelevancyMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9}"#))
    .threshold(0.7)
    .build();
```

`GEval` additionally requires `criteria` describing what to evaluate:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::GEval;

let metric = GEval::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.8}"#))
    .criteria("The answer is concise and accurate.")
    .threshold(0.7)
    .build();
```

### RAG metrics

The RAG metrics judge a retrieval pipeline. They need a provider, a threshold,
and a test case with a `retrieval_context` (and, for `ContextualRecallMetric`,
an `expected_output`). `RagasMetric` composes four metrics into one score:

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::RagasMetric;

let metric = RagasMetric::builder()
    .provider(MockLlmProvider::responses([
        r#"{"score": 0.9}"#,
        r#"{"score": 0.8}"#,
        r#"{"score": 0.7}"#,
        r#"{"score": 0.6}"#,
    ]))
    .threshold(0.5)
    .build();
```

> **Note:** `RagasMetric` measures four sub-metrics, so its provider must supply
> one response per sub-metric (see the `ragas` example).

### Deterministic metrics

`ExactMatchMetric`, `PatternMatchMetric`, and `JsonCorrectnessMetric` need no
provider:

```rust
use deepeval_rs::metrics::{ExactMatchMetric, PatternMatchMetric};

let exact = ExactMatchMetric::builder().threshold(0.5).build();
let pattern = PatternMatchMetric::builder()
    .pattern(r"^\d{3}-\d{4}$")
    .threshold(0.5)
    .build();
```

A metric is **skipped** (not a failure) when a required field is missing — for
example, `ExactMatchMetric` without an `expected_output` on the test case.

### Multi-turn & agentic metrics

These implement the [`ConversationalMetric`](crate::metrics::ConversationalMetric)
trait and score a whole [`ConversationalTestCase`](crate::test_case::ConversationalTestCase)
— an ordered sequence of [`Turn`](crate::test_case::Turn)s — instead of a single
input/output pair. Each turn can carry an input, actual/expected output,
retrieval context, expected tools, expected criteria, and feedback.

Multi-turn metrics:

- `ConversationCompletenessMetric` — how well the conversation covers the request.
- `TurnRelevancyMetric` — how relevant each turn's output is to its input.
- `TurnFaithfulnessMetric` — whether each turn's output stays grounded in its
  retrieval context.
- `KnowledgeRetentionMetric` — whether the agent retains information across turns.
- `RoleAdherenceMetric` — whether the agent stays in role.

Agentic metrics (also over a `ConversationalTestCase`):

- `TaskCompletionMetric`, `GoalAccuracyMetric`, `StepEfficiencyMetric`,
  `ToolCorrectnessMetric`, `ToolUseMetric`, `PlanAdherenceMetric`,
  `PlanQualityMetric`, `ArgumentCorrectnessMetric`, `ConversationSummaryMetric`.

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::TurnFaithfulnessMetric;
use deepeval_rs::test_case::{ConversationalTestCase, Turn};

let test_case = ConversationalTestCase::builder()
    .turn(
        Turn::builder()
            .input("What is the return policy?")
            .actual_output("We offer a 30-day full refund.")
            .retrieval_context(vec!["30 day full refund.".to_string()])
            .build(),
    )
    .build();

let mut metric = TurnFaithfulnessMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "grounded"}"#))
    .threshold(0.7)
    .build();
metric.measure(&test_case).await?;
```

A conversational metric is **skipped** when no turn satisfies its required
fields (e.g. a faithfulness-style metric needs a turn with both an actual output
and a retrieval context).

## Run an evaluation

### `evaluate` — get a report

```rust
use deepeval_rs::eval::evaluate;
use deepeval_rs::test_case::LLMTestCase;

let test_cases = vec![LLMTestCase::builder().input("hi").build()];
// metrics: Vec<Box<dyn Metric>>
let report = evaluate(&test_cases, &metrics).await;

println!("passed: {}", report.passed());
println!("failed: {}", report.failed());
```

### `assert_test` — fail fast

```rust
use deepeval_rs::eval::assert_test;
use deepeval_rs::test_case::LLMTestCase;

let test_case = LLMTestCase::builder().input("hi").build();
// metrics: Vec<Box<dyn Metric>>
assert_test(&test_case, &metrics).await?;
```

`assert_test` returns `Ok(())` when every metric passes (or has no threshold),
and an error naming the failing metrics otherwise.

### `evaluate_conversational` — multi-turn reports

```rust
use deepeval_rs::eval::evaluate_conversational;
use deepeval_rs::test_case::ConversationalTestCase;

let test_cases = vec![ConversationalTestCase::builder().build()];
// metrics: Vec<Box<dyn ConversationalMetric>>
let report = evaluate_conversational(&test_cases, &metrics).await;

println!("passed: {}", report.passed());
```

## Prompt templating

Metrics render prompts from Jinja-compatible templates. You can build your own
registry with [`TemplateRegistry`](crate::template::TemplateRegistry).

```rust
use deepeval_rs::template::TemplateRegistry;
use minijinja::Value;
use std::collections::HashMap;

let mut registry = TemplateRegistry::new();
registry.register("MyMetric", "greet", "Hello, {{ name }}!");
let ctx = HashMap::from([("name".to_string(), Value::from("world"))]);
let out = registry.resolve("MyMetric", "greet", &ctx)?;
```

## Next steps

- See the [CLI guide](cli.md) for the `deepeval` command-line tool.
- See the [README](../README.md) for the project status and roadmap.

## Examples

Runnable demos live in `crates/deepeval-rs/examples/`. They run keyless with
[`MockLlmProvider`](crate::llm::MockLlmProvider):

```bash
cargo run --example deterministic_metrics   # exact match, pattern match, JSON correctness
cargo run --example llm_judge_metrics        # answer relevancy, faithfulness, hallucination, prompt alignment
cargo run --example geval                    # GEval with custom criteria
cargo run --example evaluate_run             # combine metrics over multiple test cases
cargo run --example ragas                     # RAGAS composite metric over a RAG pipeline
cargo run --example multiturn                 # multi-turn & agentic metrics over a conversation
```

Real-provider examples (need an API key):

```bash
OPENAI_API_KEY=... cargo run --example real_provider_openai
ANTHROPIC_API_KEY=... cargo run --example real_provider_anthropic
```

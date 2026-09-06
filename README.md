# deepeval-rs

The LLM evaluation framework for Rust — a port of
[deepeval](https://github.com/confident-ai/deepeval).

deepeval-rs lets you evaluate LLM applications (agents, RAG pipelines, chatbots)
with research-backed metrics such as G-Eval, answer relevancy, faithfulness, and
hallucination. Metrics run asynchronously and can be composed into an
`evaluate` run over many test cases.

## Documentation

- [Getting Started](docs/getting-started.md) — how to use the library.
- [The `deepeval` CLI](docs/cli.md) — the command-line tool.
- [Docs index](docs/README.md).

## Status

This project is under active development, phase by phase. Each phase lands as a
pull request.

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 0 | Workspace scaffold, error types, CI | ✅ merged |
| 1 | Core test-case types + rig-backed LLM layer | ✅ merged |
| 2 | Core engine: `Metric` trait, templating, `evaluate`/`assert_test` | ✅ merged |
| 3 | Core LLM-judge + deterministic metrics | 🔜 next |
| 4 | RAG metrics | ⏳ planned |
| 5 | Multi-turn + agentic metrics | ⏳ planned |
| 6 | Hardening, GEval logprobs, CLI, examples, docs | ⏳ planned |

## Workspace layout

```
crates/
  deepeval-rs/   # the library
  deepeval/      # the `deepeval` CLI (stub; `test run` lands in Phase 6)
```

## What's implemented so far

### `test_case` — test-case data model

Mirrors deepeval's `deepeval/test_case/` module.

- **`LLMTestCase`** + builder — the primary input to single-turn metrics. Fields:
  `input`, `actual_output`, `expected_output`, `retrieval_context`,
  `expected_tools`, `expected_criteria`, `context`, `feedbacks`. Serde-serializable.
- **`SingleTurnParams`** — the fields a metric may require.
- **`ToolCall`**, **`ToolCallType`**, **`Feedback`** — supporting types.
- **`ConversationalTestCase`**, **`Turn`**, **`MultiTurnParams`** — multi-turn model.

```rust
use deepeval_rs::test_case::LLMTestCase;

let test_case = LLMTestCase::builder()
    .input("What if these shoes don't fit?")
    .actual_output("We offer a 30-day full refund at no extra costs.")
    .retrieval_context(vec![
        "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
    ])
    .build();
```

### `llm` — LLM provider layer (built on [rig](https://github.com/0xplaygrounds/rig))

- **`LlmProvider`** — an object-safe trait (`async_trait` + `Send + Sync`) that
  is the seam between metrics and rig. Methods: `model_name`, `complete`,
  `complete_structured`.
- **`LlmRequest`** / **`LlmResponse`** / **`ChatMessage`** / **`Role`** —
  provider-agnostic request/response types.
- **`RigProvider<M>`** — wraps any rig `CompletionModel` (OpenAI, Anthropic, a
  generic OpenAI-compatible endpoint, or a mock) behind `LlmProvider`.
- **`MockLlmProvider`** — a scripted provider for keyless unit tests.
- **JSON extraction** — parses the first balanced JSON value from prose or code
  fences, for structured output.

```rust
use deepeval_rs::llm::{ChatMessage, LlmProvider, LlmRequest, MockLlmProvider};

let provider = MockLlmProvider::text("hello");
let request = LlmRequest::new(vec![ChatMessage::user("Say hi.")]);
let response = provider.complete(request).await?;
```

### `metrics` — the `Metric` trait

- **`Metric`** — the object-safe trait all metrics implement. Methods: `name`,
  `threshold`, `measure` (async, records score/reason/success on `self`),
  `is_successful`, `score`, `reason`, `clone_box`.
- **`MetricConfig`** — shared configuration (threshold, include_reason,
  strict_mode, async_mode, verbose_mode).
- **`MetricResult`** — the serializable output of a single measurement.

### `template` — Jinja-compatible prompt templating

- **`TemplateRegistry`** — registers and renders prompt templates keyed by
  `(metric_class_name, method)` using [`minijinja`](https://github.com/mitsuhiko/minijinja).
- **`resolve_template`** — convenience wrapper.

```rust
use deepeval_rs::template::TemplateRegistry;
use minijinja::Value;
use std::collections::HashMap;

let mut registry = TemplateRegistry::new();
registry.register("MyMetric", "greet", "Hello, {{ name }}!");
let ctx = HashMap::from([("name".to_string(), Value::from("world"))]);
let out = registry.resolve("MyMetric", "greet", &ctx)?;
```

### `eval` — evaluation orchestration

- **`evaluate`** — runs a set of metrics over a set of test cases concurrently
  (bounded by a semaphore) and returns an [`EvalReport`].
- **`assert_test`** — runs metrics over a single test case and returns an error
  if any metric fails (the plain-library analogue of deepeval's `assert_test`).
- **`EvalReport`** / **`CaseReport`** — serializable results with pass/fail/
  skipped/errored counts.

```rust
use deepeval_rs::eval::assert_test;
use deepeval_rs::test_case::LLMTestCase;

let test_case = LLMTestCase::builder().input("hi").build();
// metrics: Vec<Box<dyn Metric>>
assert_test(&test_case, &metrics).await?;
```

## Roadmap

- **Phase 3** — core LLM-judge metrics (G-Eval, answer relevancy, faithfulness,
  hallucination, prompt alignment) and deterministic metrics (exact match,
  pattern match, JSON correctness).
- **Phase 4** — RAG metrics (contextual precision/recall/relevancy, RAGAS).
- **Phase 5** — multi-turn and agentic metrics.
- **Phase 6** — hardening, GEval logprob scoring, the `deepeval` CLI, examples,
  and full docs.

## Development

```bash
cargo build --workspace
cargo test --all-features
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## License

Apache-2.0

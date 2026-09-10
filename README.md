# deepeval-rs

The LLM evaluation framework for Rust — a ground-up rewrite of
[deepeval](https://github.com/confident-ai/deepeval).

deepeval-rs lets you evaluate LLM applications (agents, RAG pipelines, chatbots)
with research-backed metrics such as G-Eval, answer relevancy, faithfulness, and
hallucination. Metrics run asynchronously and can be composed into an
`evaluate` run over many test cases.

## Documentation

- [Getting Started](https://github.com/charathram/deepeval-rs/blob/main/docs/getting-started.md) — how to use the library.
- [Tutorials](https://github.com/charathram/deepeval-rs/blob/main/docs/tutorials/README.md) — step-by-step guides that build each
  of the examples.
- [The `deepeval` CLI](https://github.com/charathram/deepeval-rs/blob/main/docs/cli.md) — the command-line tool.
- [Docs index](https://github.com/charathram/deepeval-rs/blob/main/docs/README.md).
- [Changelog](https://github.com/charathram/deepeval-rs/blob/main/CHANGELOG.md) — release history.

## Status

This project is under active development, phase by phase. Each phase lands as a
pull request.

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 0 | Workspace scaffold, error types, CI | ✅ merged |
| 1 | Core test-case types + rig-backed LLM layer | ✅ merged |
| 2 | Core engine: `Metric` trait, templating, `evaluate`/`assert_test` | ✅ merged |
| 3 | Core LLM-judge + deterministic metrics | ✅ merged |
| 4 | RAG metrics | ✅ merged |
| 5 | Multi-turn + agentic metrics | ✅ merged |
| 6 | Hardening, GEval logprobs, CLI, full examples/docs coverage | ✅ merged |

## Workspace layout

```
crates/
  deepeval-rs/   # the library + the `deepeval` CLI binary (feature-gated)
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

### `metrics` — the `Metric` trait and the concrete metric set

- **`Metric`** — the object-safe trait all metrics implement. Methods: `name`,
  `threshold`, `measure` (async, records score/reason/success on `self`),
  `is_successful`, `score`, `reason`, `skipped`, `clone_box`.
- **`ConversationalMetric`** — the twin trait for multi-turn metrics. Same shape
  as `Metric`, but `measure` takes a [`ConversationalTestCase`]. Used by the
  multi-turn and agentic metrics below.
- **`MetricConfig`** — shared configuration (threshold, include_reason,
  strict_mode, async_mode, verbose_mode).
- **`MetricResult`** — the serializable output of a single measurement,
  including `cost`, `input_tokens`, and `output_tokens` when the metric used an
  LLM.
- **`RetryPolicy` / `RetryProvider`** — wrap any provider with exponential
  backoff retries on transient errors (default 3 retries, 200ms base, 8s max,
  2.0 factor).

**LLM-judge metrics** (need a provider; score 0–1 from an LLM verdict):

- **`GEval`** — scores an output against custom `criteria` (simplified
  chain-of-thought; logprob scoring lands in Phase 6).
- **`AnswerRelevancyMetric`** — how relevant the actual output is to the input.
- **`FaithfulnessMetric`** — whether the actual output stays grounded in the
  retrieval context.
- **`HallucinationMetric`** — whether the actual output is supported by the
  context.
- **`PromptAlignmentMetric`** — whether the actual output follows the expected
  criteria.

**RAG metrics** (LLM-judge; need a provider and a retrieval context):

- **`ContextualPrecisionMetric`** — whether the retrieval context ranks the
  relevant chunks above the irrelevant ones.
- **`ContextualRecallMetric`** — whether the retrieval context contains the
  information needed to answer the input.
- **`ContextualRelevancyMetric`** — whether the retrieval context is relevant to
  the input.
- **`RagasMetric`** — a composite score averaging answer relevancy, faithfulness,
  contextual precision, and contextual recall.

**Deterministic metrics** (no LLM required):

- **`ExactMatchMetric`** — 1.0 when actual output exactly equals expected output.
- **`PatternMatchMetric`** — 1.0 when actual output matches a regex `pattern`.
- **`JsonCorrectnessMetric`** — 1.0 when actual output parses as valid JSON.

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{AnswerRelevancyMetric, ExactMatchMetric, GEval};

// LLM-judge metric: needs a provider.
let relevancy = AnswerRelevancyMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9}"#))
    .threshold(0.7)
    .build();

// Deterministic metric: no provider.
let exact = ExactMatchMetric::builder().threshold(0.5).build();

// GEval: needs criteria too. It asks the judge for an integer score in a
// range (default 0-10) and confidence-weights it against token log
// probabilities before normalizing to 0-1.
let geval = GEval::builder()
    .provider(MockLlmProvider::text(r#"{"score": 8}"#))
    .criteria("The answer is concise and accurate.")
    .threshold(0.7)
    .build();
```

**Multi-turn & agentic metrics** (LLM-judge over a [`ConversationalTestCase`];
each turn can carry an input, output, retrieval context, expected tools, etc.):

- **`ConversationCompletenessMetric`** — how well the conversation covers what
  the user asked for.
- **`TurnRelevancyMetric`** — how relevant each turn's output is to the input.
- **`TurnFaithfulnessMetric`** — whether each turn's output stays grounded in its
  retrieval context.
- **`KnowledgeRetentionMetric`** — whether the agent retains information across
  turns.
- **`RoleAdherenceMetric`** — whether the agent stays in character/role.
- **`TaskCompletionMetric`** — whether the agent accomplishes the requested task.
- **`GoalAccuracyMetric`** — whether the outcome matches the expected goal.
- **`StepEfficiencyMetric`** — whether the agent took an efficient number of steps.
- **`ToolCorrectnessMetric`** — whether the agent's tool calls match the expected
  tools and arguments.
- **`ToolUseMetric`** — whether the agent uses the right tools at the right times.
- **`PlanAdherenceMetric`** — whether the agent follows its stated plan.
- **`PlanQualityMetric`** — whether the plan is sound.
- **`ArgumentCorrectnessMetric`** — whether the agent's reasoning is correct.
- **`ConversationSummaryMetric`** — how well the output summarizes the
  conversation.

```rust
use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{ConversationalMetric, TurnFaithfulnessMetric};
use deepeval_rs::test_case::{ConversationalTestCase, Turn};

let test_case = ConversationalTestCase::builder()
    .turns(vec![Turn::builder()
        .input("What is the return policy?")
        .actual_output("We offer a 30-day full refund.")
        .retrieval_context(vec!["30 day full refund.".to_string()])
        .build()])
    .build();

let mut metric = TurnFaithfulnessMetric::builder()
    .provider(MockLlmProvider::text(r#"{"score": 0.9, "reason": "grounded"}"#))
    .threshold(0.7)
    .build();
metric.measure(&test_case).await?; // implements ConversationalMetric
```

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

- **`evaluate`** — runs a set of single-turn metrics over a set of test cases
  concurrently (bounded by a semaphore) and returns an [`EvalReport`].
- **`evaluate_conversational`** — the multi-turn analogue: runs
  [`ConversationalMetric`]s over [`ConversationalTestCase`]s, grouped by the
  first turn's input.
- **`assert_test`** — runs metrics over a single test case and returns an error
  if any metric fails (the plain-library analogue of deepeval's `assert_test`).
- **`EvalReport`** / **`CaseReport`** — serializable results with pass/fail/
  skipped/errored counts, plus `total_cost()`, `total_input_tokens()`, and
  `total_output_tokens()` aggregates.

```rust
use deepeval_rs::eval::assert_test;
use deepeval_rs::test_case::LLMTestCase;

let test_case = LLMTestCase::builder().input("hi").build();
// metrics: Vec<Box<dyn Metric>>
assert_test(&test_case, &metrics).await?;
```

## Examples

Runnable demos live in `crates/deepeval-rs/examples/`. Each runs keyless with
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

## Roadmap

- **Phase 6** — hardening (retry/backoff, cost/token accounting, JSON repair ✅),
  GEval logprob scoring ✅, the `deepeval` CLI `test run` ✅, full examples/docs
  coverage ✅.

## Development

```bash
cargo build --workspace
cargo test --all-features
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## Releasing

Releases are driven by a custom `cargo release` command (an in-repo xtask
binary in `crates/release/`, registered as a cargo alias in `.cargo/config.toml`).
It bumps the workspace version, updates the CHANGELOG, refreshes `Cargo.lock`,
commits, tags `vX.Y.Z`, and pushes the tag — which auto-triggers the release
workflow in GitHub Actions.

```bash
# Bump the patch version (0.1.0 -> 0.1.1), tag, and push.
cargo release patch

# Bump minor (0.1.0 -> 0.2.0) or major (0.1.0 -> 1.0.0).
cargo release minor
cargo release major

# Or specify an exact semver version (optionally a prerelease).
cargo release 0.3.0
cargo release 0.1.1-alpha

# Preview what a release would do without committing or pushing.
cargo release patch --dry-run
```

The release workflow (`.github/workflows/release.yml`) runs on every `v*` tag
push and does three things:

1. **Publish to crates.io** — runs fmt/clippy/test, then `cargo publish` for
   `deepeval-rs` (the single published crate), using the `CRATES_IO_KEY`
   repository secret as the registry token.
2. **Build downloadable binaries** — a matrix over macOS arm64, Linux x86_64,
   and Windows x86_64, building `deepeval-rs` with the `cli`
   feature and packaging the `deepeval` CLI binary and the compiled
   `libdeepeval_rs.rlib` library artifact into a per-platform archive.
3. **Create a GitHub Release** — attaches every platform archive to a release
   for the tag.

The `deepeval` CLI is a binary target inside the `deepeval-rs` crate, gated
behind the `cli` feature. Build it locally with:

```bash
cargo build -p deepeval-rs --features cli
```

## License

Apache-2.0

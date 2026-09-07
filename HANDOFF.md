# Handoff Document

> Last updated: 2026-09-06 · After Phase 4 (PR #9)

This document captures the current state of the `deepeval-rs` project so a new
developer (or a future agent session) can pick up where the work left off.

## What this project is

`deepeval-rs` is a Rust port of [deepeval](https://github.com/confident-ai/deepeval),
the LLM evaluation framework. It evaluates LLM applications (agents, RAG
pipelines, chatbots) with research-backed metrics (G-Eval, answer relevancy,
faithfulness, hallucination, etc.).

## Current status

| Phase | Description | Status |
| ----- | ----------- | ------ |
| 0 | Workspace scaffold, error types, CI | ✅ merged |
| 1 | Core test-case types + rig-backed LLM layer | ✅ merged |
| 2 | Core engine: `Metric` trait, templating, `evaluate`/`assert_test` | ✅ merged |
| 3 | Core LLM-judge + deterministic metrics | ✅ merged |
| 4 | RAG metrics | ✅ merged |
| 5 | Multi-turn + agentic metrics | 🔜 next |
| 6 | Hardening, GEval logprobs, CLI, examples, docs | ⏳ planned |

## Repository layout

```
crates/
  deepeval-rs/   # the library (crate name: deepeval-rs, import path: deepeval_rs)
  deepeval/      # the `deepeval` CLI (stub; `test run` lands in Phase 6)
docs/            # usage guides (getting-started.md, cli.md, README.md)
.github/
  prompts/plan-deepevalRs.prompt.md   # the original implementation plan
  workflows/ci.yml                   # fmt + clippy + test
```

## What's implemented (as of Phase 4)

### `test_case` module (`crates/deepeval-rs/src/test_case/`)
- `LLMTestCase` + builder — single-turn test case (input, actual_output,
  expected_output, retrieval_context, expected_tools, expected_criteria,
  context, feedbacks). Serde-serializable.
- `SingleTurnParams` enum, `ToolCall`, `ToolCallType`, `Feedback`.
- `ConversationalTestCase`, `Turn`, `MultiTurnParams` — multi-turn model.

### `llm` module (`crates/deepeval-rs/src/llm/`)
- `LlmProvider` trait — object-safe (`async_trait` + `Send + Sync`) seam
  between metrics and rig. Methods: `model_name`, `complete`,
  `complete_structured` (default = prompt-based JSON + extractor).
- `LlmRequest` / `LlmResponse` / `ChatMessage` / `Role` — provider-agnostic types.
- `RigProvider<M>` — wraps any rig `CompletionModel` behind `LlmProvider`.
- `MockLlmProvider` — scripted queue for keyless tests.
- `extract_json` — parses the first balanced JSON value from prose/code fences.

### `metrics` module (`crates/deepeval-rs/src/metrics/`)
- `Metric` trait — object-safe (`async_trait` + `Send + Sync`). Methods: `name`,
  `threshold`, `measure` (async), `is_successful`, `score`, `reason`, `skipped`,
  `clone_box`. `Box<dyn Metric>` is `Clone` via `clone_box`.
- `MetricConfig` — threshold, include_reason, strict_mode, async_mode, verbose_mode.
- `MetricResult` — serializable measurement output.
- `MetricState` — shared in-memory measurement state (config + score/reason/skipped);
  cloning resets measurement fields so a cloned metric starts fresh.
- `impl_metric!` macro — generates the boilerplate `Metric` impl for a metric
  type exposing a `state: MetricState` field and a `measure_impl` method.
- `llm_judge` module — shared LLM-judge flow: validate required fields → render
  prompt → `provider.complete_structured` → parse `{score, reason}` verdict →
  clamp score to 0–1.
- **LLM-judge metrics:** `GEval` (simplified CoT), `AnswerRelevancyMetric`,
  `FaithfulnessMetric`, `HallucinationMetric`, `PromptAlignmentMetric`.
- **RAG metrics (LLM-judge):** `ContextualPrecisionMetric`,
  `ContextualRecallMetric`, `ContextualRelevancyMetric`, and the composite
  `RagasMetric` (averages answer relevancy, faithfulness, contextual precision,
  contextual recall).
- **Deterministic metrics (no LLM):** `ExactMatchMetric`, `PatternMatchMetric`
  (`regex`), `JsonCorrectnessMetric`.
- Prompt templates live in `crates/deepeval-rs/templates/` (e.g.
  `geval/generate_verdict.txt`, `contextual_precision/generate_verdict.txt`).
- `Provider` (in `llm_judge`) — a `Debug`/`Clone` wrapper around
  `Arc<dyn LlmProvider>` that also implements `LlmProvider` (delegating to the
  inner provider), so it can be passed to sub-metric builders (used by
  `RagasMetric`).

### `template` module (`crates/deepeval-rs/src/template/`)
- `TemplateRegistry` — registers/renders Jinja-compatible templates keyed by
  `(metric_class_name, method)` using `minijinja`.
- `resolve_template` — convenience wrapper.

### `eval` module (`crates/deepeval-rs/src/eval/`)
- `evaluate` — runs metrics over test cases concurrently (semaphore-bounded,
  default 10), groups results by test case, returns `EvalReport`.
- `assert_test` — runs metrics over one test case; returns
  `EvalError::AssertionFailed` if any metric fails.
- `EvalReport` / `CaseReport` — serializable results with pass/fail/skipped/
  errored counts.

### `error` module (`crates/deepeval-rs/src/error.rs`)
- `EvalError`, `LlmError`, `MetricError`, `TemplateError` via `thiserror`.

### `embeddings` module (`crates/deepeval-rs/src/embeddings/`)
- `EmbeddingProvider` trait — a seam only; no concrete impls yet.

## Key decisions (locked)

1. **GEval:** ship a simplified CoT version now; add logprob-based scoring in Phase 6.
2. **Metric ergonomics:** faithful to deepeval — `measure(&mut self, ...)` stores
   results on the metric; metrics are cloned per `(test_case, metric)` pair.
3. **Providers:** OpenAI + Anthropic + a generic OpenAI-compatible base-URL client.
4. **LLM interactions:** ALWAYS use [rig](https://github.com/0xplaygrounds/rig).
   No hand-rolled `reqwest` LLM clients.
5. **Testing:** write unit tests first, iterate until green. Keyless tests via
   `MockLlmProvider`.
6. **Conventions:** stay faithful to Rust conventions (module layout, `thiserror`,
   builder patterns, `cargo fmt`/`clippy`).
7. **Docs:** always update the README and any other documentation as each PR lands.
8. **Workflow:** phase-by-phase; open a PR per phase and seek approval before
   proceeding to the next phase.

## Technical notes / gotchas

- **`async_trait` is required** for both `Metric` and `LlmProvider` because
  native `async fn` in traits is not dyn-compatible. `Box<dyn Metric>` needs
  object safety.
- **`Clone` on `Box<dyn Metric>`** is provided via a `clone_box` method on the
  trait, NOT a `Clone` supertrait (a `Clone` supertrait breaks dyn-compatibility).
- **rig 0.42 API** (researched from the cargo registry source):
  - `rig::completion::CompletionModel` trait: `completion(&self, CompletionRequest)
    -> impl Future<Output=Result<CompletionResponse, CompletionError>>`. NOT
    object-safe, hence our `LlmProvider` wrapper.
  - `CompletionRequestBuilder<M>`: `.preamble()`, `.message()`, `.temperature()`,
    `.max_tokens()`, `.output_schema()`, `.build()`, `.send()`.
  - Providers: `rig::providers::openai::Client` (Responses API) and
    `CompletionsClient` (Chat Completions); `rig::providers::anthropic::Client`.
    `Client::builder().api_key(k).base_url(url).build()`.
  - Mock: `rig::test_utils::{MockCompletionModel, MockTurn}` — gated behind the
    `test-utils` feature (already enabled in the workspace `Cargo.toml`).
- **`evaluate` groups results by test-case input string.** Two test cases with
  the same `input` collapse into one `CaseReport`. This is a known simplification;
  revisit if distinct-but-identical inputs need separate reports.
- **Feature flags** `openai`/`anthropic`/`openai-compatible` are placeholders
  (empty). Concrete provider constructors (e.g. `OpenAIProvider::from_env()`) are
  not yet written — `RigProvider` already supports any rig model, so wiring a
  specific provider is a thin constructor.

## What's next (Phase 5)

Multi-turn and agentic metrics. The multi-turn data model
(`ConversationalTestCase`, `Turn`, `MultiTurnParams`) already exists in
`test_case`; Phase 5 wires it up to metrics and `evaluate`.

## Verification commands

```bash
cargo build --workspace
cargo test --all-features
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
```

## References

- Original plan: `.github/prompts/plan-deepevalRs.prompt.md`
- deepeval (upstream): https://github.com/confident-ai/deepeval
- rig: https://github.com/0xplaygrounds/rig
- Docs: `docs/getting-started.md`, `docs/cli.md`

# Handoff Document

> Last updated: 2026-09-08 · After Phase 6 PR D (examples, docs, CHANGELOG) + example tutorials

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
| 5 | Multi-turn + agentic metrics | ✅ merged |
| 6 | Hardening, GEval logprobs, CLI, full examples/docs coverage | ✅ merged |

## Repository layout

```
crates/
  deepeval-rs/   # the library (crate name: deepeval-rs, import path: deepeval_rs)
  deepeval/      # the `deepeval` CLI (`test run` implemented in Phase 6)
docs/            # usage guides (getting-started.md, cli.md, README.md)
docs/tutorials/  # step-by-step tutorials that build each example (01-07)
.github/
  prompts/plan-deepevalRs.prompt.md   # the original implementation plan
  workflows/ci.yml                   # fmt + clippy + test
```

## What's implemented (as of Phase 5)

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
  `LlmRequest` has an optional `top_logprobs` (via `with_top_logprobs`) that
  asks the provider for per-token log probabilities; `LlmResponse` carries them
  in an optional `logprobs: Vec<TokenLogprobs>` (via `with_logprobs`).
- `TokenLogprob` / `TokenLogprobs` — normalized per-token log-probability types.
- `RigProvider<M>` — wraps any rig `CompletionModel` behind `LlmProvider`. When
  `top_logprobs` is set it forwards OpenAI-compatible `logprobs`/`top_logprobs`
  request params via rig's `additional_params` and extracts log probabilities
  from the raw response (`choices[0].logprobs.content`).
- `extract_logprobs` — parses the OpenAI wire-format log-probability payload
  into a `Vec<TokenLogprobs>`.
- `calculate_weighted_summed_score` — confidence-weights a raw integer score
  against token log probabilities (mirroring deepeval's g-eval fix).
- `MockLlmProvider` — scripted queue for keyless tests.
- `RetryPolicy` / `RetryProvider<P>` — exponential-backoff retry wrapper around
  any `LlmProvider`. `RetryPolicy` (max_retries, base_delay, max_delay,
  backoff_factor) defaults to 3 retries / 200ms base / 8s max / 2.0 factor;
  `RetryProvider` retries transient errors (`LlmError::Transport` |
  `LlmError::Provider`).
- `extract_json` — parses the first balanced JSON value from prose/code fences,
  with best-effort repair of common LLM mistakes (trailing commas, unquoted
  keys, single-quoted strings).

### `metrics` module (`crates/deepeval-rs/src/metrics/`)
- `Metric` trait — object-safe (`async_trait` + `Send + Sync`). Methods: `name`,
  `threshold`, `measure` (async), `is_successful`, `score`, `reason`, `skipped`,
  `clone_box`. `Box<dyn Metric>` is `Clone` via `clone_box`.
- `ConversationalMetric` trait — twin trait for multi-turn metrics (same shape as
  `Metric` but `measure` takes a `ConversationalTestCase`).
- `MetricConfig` — threshold, include_reason, strict_mode, async_mode, verbose_mode.
- `MetricResult` — serializable measurement output.
- `MetricState` — shared in-memory measurement state (config + score/reason/skipped
  + cost/input_tokens/output_tokens); cloning resets measurement fields so a
  cloned metric starts fresh. `accrue_usage(&LlmResponse)` accumulates token
  counts (saturating) and cost.
- `impl_metric!` macro — generates the boilerplate `Metric` impl for a metric
  type exposing a `state: MetricState` field and a `measure_impl` method. Also
  generates `cost()`, `input_tokens()`, `output_tokens()` accessors.
- `impl_conversational_metric!` macro — same, for `ConversationalMetric`.
- `llm_judge` module — shared LLM-judge flow: validate required fields → render
  prompt → `provider.complete` → parse `{score, reason}` verdict → clamp score
  to 0–1. `measure_llm_judge` returns a `MeasureOutcome::Scored { score, reason,
  usage }` carrying the raw `LlmResponse` so metrics can accrue cost/tokens.
  `measure_geval` is a GEval-specific variant that requests log probabilities,
  confidence-weights the raw integer score, and normalizes to 0–1.
- **`conversational_llm_judge` module — shared multi-turn LLM-judge flow.
  `measure_conversation_llm_judge(required, provider, registry, class_name,
  method, extra_context, test_case)` returns `Ok(None)` when no turn satisfies
  all required fields (metric marked skipped), otherwise `Ok(Some((verdict,
  response)))` with the parsed score/reason and the raw `LlmResponse` so metrics
  can accrue cost/tokens. It auto-injects a `dialog` variable and the
  `turns` field fragments.
- **LLM-judge metrics:** `GEval` (logprob-weighted integer score, default range
  0–10 via `score_range`), `AnswerRelevancyMetric`,
  `FaithfulnessMetric`, `HallucinationMetric`, `PromptAlignmentMetric`.
- **RAG metrics (LLM-judge):** `ContextualPrecisionMetric`,
  `ContextualRecallMetric`, `ContextualRelevancyMetric`, and the composite
  `RagasMetric` (averages answer relevancy, faithfulness, contextual precision,
  contextual recall).
- **Multi-turn metrics (LLM-judge):** `ConversationCompletenessMetric`,
  `TurnRelevancyMetric`, `TurnFaithfulnessMetric`, `KnowledgeRetentionMetric`,
  `RoleAdherenceMetric`.
- **Agentic metrics (LLM-judge, over a `ConversationalTestCase`):**
  `TaskCompletionMetric`, `GoalAccuracyMetric`, `StepEfficiencyMetric`,
  `ToolCorrectnessMetric`, `ToolUseMetric`, `PlanAdherenceMetric`,
  `PlanQualityMetric`, `ArgumentCorrectnessMetric`, `ConversationSummaryMetric`.
- **Deterministic metrics (no LLM):** `ExactMatchMetric`, `PatternMatchMetric`
  (`regex`), `JsonCorrectnessMetric`.
- Prompt templates live in `crates/deepeval-rs/templates/` (e.g.
  `geval/generate_verdict.txt`, `contextual_precision/generate_verdict.txt`,
  `turn_faithfulness/generate_verdict.txt`, `tool_correctness/generate_verdict.txt`).
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
- `evaluate_conversational` — multi-turn analogue; runs
  `ConversationalMetric`s over `ConversationalTestCase`s, grouped by the first
  turn's input.
- `assert_test` — runs metrics over one test case; returns
  `EvalError::AssertionFailed` if any metric fails.
- `EvalReport` / `CaseReport` — serializable results with pass/fail/skipped/
  errored counts, plus `total_cost()`, `total_input_tokens()`,
  `total_output_tokens()` aggregates over all measurements.

### `error` module (`crates/deepeval-rs/src/error.rs`)
- `EvalError`, `LlmError`, `MetricError`, `TemplateError` via `thiserror`.

### `embeddings` module (`crates/deepeval-rs/src/embeddings/`)
- `EmbeddingProvider` trait — a seam only; no concrete impls yet.

## Key decisions (locked)

1. **GEval:** logprob-based scoring (Phase 6 PR B). The judge returns an integer
   score in a range (default 0–10); the raw score is confidence-weighted against
   token log probabilities and normalized to 0–1.
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
  `evaluate_conversational` groups by the first turn's input (empty string when a
  case has no turns).
- **`conversational_llm_judge::measure_conversation_llm_judge` returns a
  `(Verdict, LlmResponse)` tuple.** Multi-turn metrics record both score and
  reason on their state and accrue cost/tokens from the response; `reason` is
  surfaced when `include_reason` is set.
- **Cost/token accounting:** every LLM-judge metric accrues usage from the raw
  `LlmResponse` into its `MetricState`. `MetricResult` carries `cost` /
  `input_tokens` / `output_tokens`, and `EvalReport` aggregates them. Metrics
  that don't call the LLM (deterministic) report `None`/`0`.
- **GEval logprob scoring:** the judge is asked for an integer score in a range
  (default 0–10, configurable via `score_range`). The request sets
  `top_logprobs`; `calculate_weighted_summed_score` finds the last generated
  token equal to `str(raw_score)`, filters its `top_logprobs` to decimal tokens
  with logprob ≥ `ln(0.01)`, and returns a probability-weighted sum (falling
  back to the raw score if nothing survives). The result is normalized to 0–1.
  Providers that don't return logprobs (e.g. `MockLlmProvider`) fall back to
  the raw score.
- **Feature flags** `openai`/`anthropic`/`openai-compatible` are placeholders
  (empty). Concrete provider constructors (e.g. `OpenAIProvider::from_env()`) are
  not yet written — `RigProvider` already supports any rig model, so wiring a
  specific provider is a thin constructor.

## What's next

Phase 6 is complete. The `deepeval` CLI `test run` subcommand (PR C) loads a
YAML suite file, builds deterministic and LLM-judge metrics, evaluates the
test cases, and prints a pass/fail report (see `docs/cli.md`). PR D added
runnable examples per metric family, real-provider examples using the new
`OpenAIProvider`/`AnthropicProvider` constructors, a `CHANGELOG.md`, and
updated docs. A `docs/tutorials/` series (01-07) now walks through building
each example as a progressive curriculum.

Possible follow-ups: expose the agentic metric required-field lists as
user-facing knobs (e.g. choosing which fields per turn) beyond what
`Turn`/`MultiTurnParams` provide, and add a `deepeval login` command for the
Confident AI platform.

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
- Tutorials: `docs/tutorials/README.md` (index) and `docs/tutorials/01-07`

# Plan: DeepEval-RS — LLM Evaluation Framework in Rust

## TL;DR
Build a Rust async port of [deepeval](https://github.com/confident-ai/deepeval) — an LLM-evaluation
framework — as a Cargo workspace. Core: a `Metric` trait (async, `Box<dyn>`-compatible), the
`LLMTestCase`/`ConversationalTestCase` data models, an LLM layer built on **rig**, Jinja-compatible
prompt templating, and `evaluate()`/`assert_test()` orchestration running many metrics × test cases
concurrently. Ship a **broad metric set** (LLM-judge + RAG + multi-turn + agentic + deterministic)
that needs **no local NLP models** — those deferred behind a `LocalModel`/`EmbeddingProvider` seam.
Async (tokio), plain library API (no custom test harness), OpenAI + Anthropic + generic
OpenAI-compatible base-URL providers via rig.

## Locked Decisions
1. **GEval:** ship a simplified CoT version now; add logprob-based scoring in Phase 6.
2. **Metric ergonomics:** faithful to deepeval — `measure(&mut self, ...)` stores results on the
   metric (`score`, `reason`, `success`, cost/token accrual); metrics are `Clone` so `evaluate` can
   measure each `(test_case, metric)` pair independently.
3. **Providers:** OpenAI + Anthropic + a generic OpenAI-compatible base-URL client (covers
   Ollama/vLLM/Together) — all via rig.
4. **LLM interactions:** ALWAYS use [rig](https://github.com/0xplaygrounds/rig) for any LLM
   interaction. No hand-rolled `reqwest` LLM clients. rig's `Completion`/`StreamingCompletion`/
   `Embedding` traits and provider clients are the transport layer.
5. **Testing:** always write unit tests first, then iterate on the implementation until the tests
   pass. Keyless tests via a mock LLM provider.
6. **Conventions:** stay faithful to Rust conventions, taxonomy, and idioms (module layout, error
   handling via `thiserror`, builder patterns, `cargo fmt`/`clippy`).

---

## Architecture (core design)

### Crate layout — Cargo workspace
- `crates/deepeval` — the library. Modules: `test_case`, `metrics`, `llm`, `embeddings`, `eval`,
  `template`, `error`.
- `crates/deepeval-cli` — binary `deepeval` (Phase 6, `clap`). Deferred; not in default build.
- `examples/` — runnable `cargo run --example` demos per metric family.
- Single library crate with clear module boundaries + feature flags (simpler for v1 than many crates;
  can split `metrics` into its own crate later).

### Feature flags
- `default = ["openai", "anthropic", "openai-compatible"]`
- `openai`, `anthropic`, `openai-compatible` — provider impls (feature-gated so users pull only what
  they use)
- `embeddings` — embedding provider (deferred; seam only in v1)
- `cli` — pulls `clap`

### LLM layer (`llm`) — built on rig
- rig is the single transport. We depend on `rig` (root facade) and use its provider clients:
  `rig::providers::openai`, `rig::providers::anthropic`, and a generic OpenAI-compatible client
  (rig supports custom base URLs / OpenAI-compatible endpoints).
- We define a thin **`LlmProvider`** wrapper trait over rig's `Completion`/`StreamingCompletion`
  traits so metrics depend on our abstraction, not rig types directly. This keeps the metric layer
  provider-agnostic and testable with a mock.
  - `async fn complete(&self, req: LlmRequest) -> Result<LlmResponse, LlmError>`
  - `async fn complete_structured<T: DeserializeOwned>(&self, req: LlmRequest) -> Result<T, LlmError>`
    (JSON extraction; uses rig's structured-output support where available, else prompt-based JSON +
    fence-strip / first-`{}` extractor).
  - `fn model_name(&self) -> &str`
- `LlmRequest { messages: Vec<ChatMessage>, model: String, temperature, response_format: Option<JsonSchema> }`
- `LlmResponse { content: String, input_tokens: u32, output_tokens: u32, cost: Option<f64>,
  logprobs: Option<...> }` (logprobs for GEval scoring in Phase 6).
- `ChatMessage { role: Role, content }`; `Role` enum (system/user/assistant/tool).
- **`MockLlmProvider`** — a rig-backed mock that replays a canned queue of responses. Enables
  keyless unit tests for every metric. (rig's own cassette/replay tooling may be reused.)
- **`LlmProvider` trait is `async_trait` + `Send + Sync`** for object safety (`Box<dyn LlmProvider>`).

> **Why `async_trait`:** native `async fn` in traits is not object-safe (future type leaks), so
> `Box<dyn Metric>` — needed for a heterogeneous `Vec` of metrics in `evaluate()` — requires boxing.
> `async-trait` (or manual `Pin<Box<dyn Future>>`) gives dyn-compatibility. Same for `LlmProvider`.

### Data model (`test_case`)
- `LLMTestCase { input: String, actual_output: Option<String>, expected_output: Option<String>,
  retrieval_context: Option<Vec<String>>, expected_tools: Option<Vec<ToolCall>>,
  expected_criteria: Option<Vec<String>>, context: Option<Vec<String>>, feedbacks: Option<Vec<Feedback>> }`
  — builder pattern + `Default` + `Serialize/Deserialize`.
- `ConversationalTestCase { turns: Vec<Turn>, ... }`; `Turn { input, actual_output, expected_output,
  retrieval_context, expected_tools, expected_criteria }`.
- `ToolCall { name, args: Value, expected_name: Option, expected_args: Option, kind, success }`.
- `SingleTurnParams` / `MultiTurnParams` — enums with `Display`/`AsRef<str>` (mirror deepeval's string
  constants). Used by `GEval::evaluation_params`.

### Metric trait (`metrics`)
- `Metric` (async, `async_trait`, `Send + Sync + Clone`):
  - `fn name(&self) -> &str`
  - `fn threshold(&self) -> Option<f32>`
  - `async fn measure(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError>` — stores
    `score`, `reason`, `success`, cost/token accrual on `&mut self` (faithful to deepeval).
  - `fn is_successful(&self) -> Option<bool>` — default: `score >= threshold` (None if no threshold).
  - `fn score(&self) -> Option<f32>`
  - `fn reason(&self) -> Option<&str>`
- `MetricConfig { threshold, include_reason, strict_mode, async_mode, verbose_mode }`.
- Cost/token accrual helpers (`_accrue_cost`/`_accrue_tokens` analogues) on a shared base.
- `EmbeddingProvider` — seam trait (deferred impls): `embed(&self, &str) -> Vec<f32>`.
- `LocalModel` — seam trait (deferred) for toxicity/bias/summarization.

### Prompt templates (`template`)
- `minijinja` (pure-Rust, Jinja-compatible) — matches deepeval's Jinja2 templates.
- Templates embedded via `include_str!` from `templates/<Metric>/<method>.txt`, resolved by
  `(metric_class_name, method)` with shared `fragments/` (faithfulness verdict blocks, multimodal rules).
- `resolve_template(feature, class, method, ctx) -> String`.

### Orchestration (`eval`)
- `async fn evaluate(test_cases: &[LLMTestCase], metrics: &[Box<dyn Metric>]) -> EvalReport`.
- `async fn assert_test(test_case: &LLMTestCase, metrics: &[Box<dyn Metric>]) -> Result<(), EvalError>`
  (plain library API; returns `Result` so callers `?` it in `#[tokio::test]`; optional panic mode).
- Concurrency: `tokio` + `FuturesUnordered`/`join_all`, `tokio::sync::Semaphore` to cap concurrent LLM
  calls (configurable, default ~10). Each `(test_case, metric)` pair gets a **cloned** metric (metrics
  are `Clone`), measured independently, then results collected into the report.
- `EvalReport { per_case: Vec<CaseReport>, ... }`; `MetricResult { name, score, reason, success,
  threshold, cost, input_tokens, output_tokens, skipped, error }`. JSON-serializable.

---

## Steps

### Phase 0 — Scaffold
1. `cargo new --workspace`; create `crates/deepeval` lib, `crates/deepeval-cli` (stub, gated), `examples/`.
2. `Cargo.toml` deps: `tokio` (full), `rig` (with `openai`, `anthropic` features), `serde`+`serde_json`,
   `async-trait`, `minijinja`, `thiserror`, `regex`, `futures`, `tracing`+`tracing-subscriber`;
   optional `clap` (cli feature).
3. `error.rs`: `EvalError`, `LlmError`, `MetricError`, `TemplateError` via `thiserror`; re-export
   public errors.
4. CI: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`.

### Phase 1 — Core types + LLM layer (parallel-friendly)
5. `test_case` module: `LLMTestCase`, `ConversationalTestCase`, `Turn`, `ToolCall`, `SingleTurnParams`,
   `MultiTurnParams`, `Feedback`. Builders + `Default` + serde. *(blocks most later work)*
6. `llm` module: `LlmProvider` trait + `LlmRequest/Response/ChatMessage/Role` + `MockLlmProvider`
   (canned queue of responses — enables keyless tests) *(blocks Phase 2, 3+)*.
7. rig-backed provider impls (feature-gated): OpenAI, Anthropic, generic OpenAI-compatible base-URL.
   Unit-tested against `MockLlmProvider` and rig's cassette/replay tooling. *(parallel with 6's
   consumer side)*
8. `embeddings` module: `EmbeddingProvider` trait + `LocalModel` seam (no impls yet). *(independent)*

### Phase 2 — Core engine (depends on 1)
9. `template` module: `resolve_template` over `minijinja`, embed a starter `templates/` dir.
10. `metrics` base: `Metric` trait (`async_trait`, `Clone`), `MetricConfig`, cost/token accrual helpers,
    `is_successful` default via `score >= threshold`.
11. `eval` module: `evaluate`, `assert_test`, `EvalReport`/`MetricResult`, semaphore-bounded concurrency.
12. A text report + JSON serialization. Console output of pass/fail + scores + reasons.
13. Verify: `cargo test` on Phase 2 with `MockLlmProvider` — a trivial no-op metric passes `evaluate`.

### Phase 3 — Core LLM-judge + deterministic metrics (depends on 2)
14. **Template every metric** with this pattern: resolve fields → validate required (else `skipped=true`)
    → render prompt(s) → `provider.complete` → parse JSON verdicts → compute `score` (0–1) → optional
    `reason` (2nd LLM call if `include_reason`) → `is_successful`.
15. Implement: `GEval` (simplified CoT; logprob scoring deferred to Phase 6), `AnswerRelevancyMetric`,
    `FaithfulnessMetric`, `HallucinationMetric`, `PromptAlignmentMetric`.
16. Deterministic (no LLM): `ExactMatchMetric`, `PatternMatchMetric` (`regex`),
    `JsonCorrectnessMetric` (`jsonschema` or manual schema check).
17. Each metric gets unit tests using `MockLlmProvider` with canned JSON responses + a threshold pass/fail
    assertion. Verify: `cargo test` per metric.

### Phase 4 — RAG metrics (depends on 3)
18. `ContextualPrecisionMetric`, `ContextualRecallMetric`, `ContextualRelevancyMetric`
    (all LLM-judge; `ContextualRecall` extracts ground-truth facts from `expected_output`).
19. `RagasMetric` = average of answer relevancy, faithfulness, contextual precision, contextual recall
    (compose existing metrics — reuse Phase 3 impls). Verify: `cargo test`.

### Phase 5 — Multi-turn + agentic metrics (depends on 3, uses `ConversationalTestCase` from 5)
20. Multi-turn: `ConversationCompletenessMetric`, `TurnRelevancyMetric`, `TurnFaithfulnessMetric`,
    `KnowledgeRetentionMetric`, `RoleAdherenceMetric` (operate over `Vec<Turn>`).
21. Agentic: `TaskCompletionMetric`, `ToolCorrectnessMetric`, `GoalAccuracyMetric`, `StepEfficiencyMetric`,
    `PlanAdherenceMetric`, `PlanQualityMetric`, `ToolUseMetric`, `ArgumentCorrectnessMetric`
    (use `ToolCall` + trajectory/`context` fields). Verify: `cargo test`.

### Phase 6 — Hardening, GEval logprobs, CLI, examples, docs
22. Retry policy + backoff (mirror deepeval `retry_policy`), robust JSON repair, cost accounting totals.
23. **GEval logprob-based scoring** (token-logprob selection, mirroring deepeval's g-eval fix).
24. `examples/` — one runnable demo per family (chatbot, RAG, agent) using `MockLlmProvider` + a
    real-provider example gated on env vars.
25. `deepeval-cli`: `clap` binary with `deepeval test run <file>` stub (optional; feature `cli`).
26. README (quickstart mirroring deepeval), `docs/`, `CHANGELOG`. Verify: `cargo test`, `cargo run --example *`.

---

## Relevant files / reuse
- `crates/deepeval/src/test_case/` — data model (mirror deepeval `deepeval/test_case/`).
- `crates/deepeval/src/metrics/base.rs` — `Metric` trait (mirror `deepeval/metrics/base_metric.py`:
  `measure`/`is_successful`/`_accrue_cost`/`_accrue_tokens`).
- `crates/deepeval/src/llm/` — rig-backed provider abstraction (mirror `deepeval/models/base_llm.py` +
  `llms/`).
- `crates/deepeval/src/template/` — prompt resolution (mirror `deepeval/metric_templates/resolver.py` +
  `templates/`).
- `crates/deepeval/src/eval/` — `evaluate`/`assert_test` (mirror `deepeval/evaluate.py`).
- Per-metric module under `src/metrics/<name>/` mirroring deepeval's one-dir-per-metric layout.

## Verification
1. `cargo test` (unit, keyless via `MockLlmProvider`) passes at end of every phase.
2. Per-metric: a passing-score case and a failing-score case (below threshold) each asserted.
3. `cargo clippy -- -D warnings`, `cargo fmt --check` in CI.
4. Real-provider integration tests gated behind `#[cfg(feature = "integration")]` + `OPENAI_API_KEY`/
   `ANTHROPIC_API_KEY` (not run in default CI).
5. `cargo run --example <name>` for each example compiles + runs with the mock provider.

## Decisions (assumptions — override if needed)
- **Scope:** broad LLM-judge + RAG + multi-turn + agentic + deterministic metric set.
- **Runtime:** async (tokio) throughout; metrics are `async`.
- **LLM layer:** rig for ALL LLM interactions; thin `LlmProvider` wrapper over rig's traits for
  testability + provider-agnostic metrics.
- **Providers v1:** OpenAI + Anthropic + generic OpenAI-compatible base-URL, feature-gated.
- **Testing model:** plain library `evaluate`/`assert_test` returning `Result` (no pytest-like harness).
- **Local NLP models:** deferred behind `EmbeddingProvider`/`LocalModel` seams — toxicity, bias,
  summarization (detoxify/unbias/summac) NOT in v1; Faithfulness/Answer-Relevancy use the LLM-judge path
  (deepeval's default), so no local model needed.
- **Crate shape:** one library crate (`deepeval`) + `deepeval-cli` + `examples`, feature flags for
  providers. Split `metrics` into its own crate only if it grows large.
- **Async trait:** `async-trait` for `Metric` + `LlmProvider` (object safety for `Box<dyn>`).
- **Structured output:** prompt-based JSON by default; rig's structured-output support where available.

## Explicitly excluded (scope boundaries)
- Local NLP/embedding model impls (detoxify, unbias, summac, answer-relevancy model).
- Tracing/spans (OpenTelemetry-style), Confident AI platform sync, red-teaming, synthetic dataset
  generation, benchmark harnesses (MMLU/GSM8K/etc.), multimodal metrics, DAG metric builder,
  Python/`maturin` interop, framework integrations (LangChain/etc.).
- Left as seams/TODOs so they slot in later without breaking the API.

# Changelog

All notable changes to `deepeval-rs` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Release pipeline** — a `cargo release` command (xtask in `crates/release/`)
  that bumps the workspace version, updates the CHANGELOG, refreshes
  `Cargo.lock`, commits, tags `vX.Y.Z`, and pushes the tag. A GitHub Actions
  workflow (`.github/workflows/release.yml`) triggered on `v*` tag pushes
  publishes both crates to crates.io (using the `CRATES_IO_KEY` secret) and
  attaches per-platform archives (CLI binary + library rlib) to a GitHub
  Release. See the "Releasing" section of the README.
- **`deepeval` CLI `test run`** — run a YAML test suite from the terminal and
  print a pass/fail report. Supports deterministic and single-turn LLM-judge
  metrics. See `docs/cli.md`.
- **Concrete provider constructors** — `OpenAIProvider::from_env()` /
  `from_env_with_model()` and `AnthropicProvider::from_env()` /
  `from_env_with_model()`, reading API keys from the environment.
- **`impl LlmProvider for Box<dyn LlmProvider>`** — lets a heterogeneous set of
  providers be passed to metric builders.
- **GEval logprob-based scoring** — the judge's integer score is
  confidence-weighted against token log probabilities before normalizing to
  0–1, mirroring deepeval's g-eval fix.
- **Retry policy with exponential backoff** — `RetryProvider` retries
  transient provider errors (transport failures, 5xx).
- **Cost and token accounting** — metrics accrue input/output tokens and
  estimated cost; `EvalReport` exposes totals.
- **Robust JSON repair** — LLM responses with trailing commas, unquoted keys,
  or single-quoted strings are repaired before parsing.
- **RAG metrics** — `ContextualPrecisionMetric`, `ContextualRecallMetric`,
  `ContextualRelevancyMetric`, and the composite `RagasMetric`.
- **Multi-turn metrics** — `ConversationCompletenessMetric`,
  `TurnRelevancyMetric`, `TurnFaithfulnessMetric`, `KnowledgeRetentionMetric`,
  `RoleAdherenceMetric`.
- **Agentic metrics** — `TaskCompletionMetric`, `ToolCorrectnessMetric`,
  `GoalAccuracyMetric`, `StepEfficiencyMetric`, `PlanAdherenceMetric`,
  `PlanQualityMetric`, `ToolUseMetric`, `ArgumentCorrectnessMetric`,
  `ConversationSummaryMetric`.
- **Core metrics** — `GEval`, `AnswerRelevancyMetric`, `FaithfulnessMetric`,
  `HallucinationMetric`, `PromptAlignmentMetric`, `ExactMatchMetric`,
  `PatternMatchMetric`, `JsonCorrectnessMetric`.
- **Core engine** — the `Metric` trait, Jinja-compatible prompt templating,
  `evaluate`/`evaluate_conversational` (report) and `assert_test` (fail-fast).
- **Test-case model** — `LLMTestCase`, `ConversationalTestCase`, `Turn`,
  `ToolCall`, `Feedback`, and supporting params types.
- **Runnable examples** — one demo per metric family (deterministic,
  LLM-judge, GEval, RAGAS, multi-turn) plus real-provider examples for OpenAI
  and Anthropic.

## [0.1.0] - 2026-09-06

Initial workspace scaffold: error types, CI, and the `deepeval-rs` library
crate with the `deepeval` CLI stub.

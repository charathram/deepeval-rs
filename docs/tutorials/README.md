# Tutorials

Step-by-step guides that teach the `deepeval-rs` library by building each of
the runnable examples in `crates/deepeval-rs/examples/`. The tutorials form a
**progressive curriculum**: each one builds on concepts introduced in the
previous ones, so it's best to follow them in order.

Each tutorial opens with a **"What you'll build"** walkthrough of the complete
example, then guides you through building it yourself, step by step.

## Reading order

| # | Tutorial | Example | Concepts introduced |
|---|----------|---------|---------------------|
| 1 | [Deterministic metrics](01-deterministic-metrics.md) | `deterministic_metrics` | `LLMTestCase`, builders, `measure`, score & threshold |
| 2 | [LLM-judge metrics](02-llm-judge-metrics.md) | `llm_judge_metrics` | Providers, `MockLlmProvider`, LLM-as-a-judge |
| 3 | [GEval](03-geval.md) | `geval` | `GEval`, custom `criteria`, `evaluation_steps`, logprob scoring |
| 4 | [A full evaluation run](04-evaluate-run.md) | `evaluate_run` | `evaluate`, `EvalReport`, combining metrics |
| 5 | [RAGAS](05-ragas.md) | `ragas` | RAG metrics, `retrieval_context`, composite metrics |
| 6 | [Multi-turn & agentic metrics](06-multiturn.md) | `multiturn` | `ConversationalTestCase`, `Turn`, conversational metrics |
| 7 | [Real providers](07-real-providers.md) | `real_provider_openai`, `real_provider_anthropic` | `OpenAIProvider`, `AnthropicProvider`, `RigProvider` |

## Prerequisites

- A Rust toolchain (edition 2021 or later).
- The `deepeval-rs` crate available in your project (see
  [Getting Started](../getting-started.md)).
- Tutorials 1–6 run **keyless** using `MockLlmProvider`. Only tutorial 7 needs
  an API key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`).

## Running the examples

Every tutorial ends with a runnable example. From the workspace root:

```bash
cargo run --example deterministic_metrics
```

See [Getting Started](../getting-started.md) for the full list of examples.

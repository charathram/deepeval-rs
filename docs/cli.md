# The `deepeval` CLI

The `deepeval` command-line tool is the companion to the `deepeval-rs` library.
It lets you run evaluation test suites from the terminal, similar to how
deepeval's Python CLI runs `deepeval test run`.

## Building the CLI

The CLI lives in the `crates/deepeval` crate. Build it from the workspace root:

```bash
cargo build -p deepeval
```

The binary is named `deepeval`.

## Usage

```
deepeval test run <file>
```

Run an evaluation test suite defined in the YAML file `<file>`. The CLI parses
the suite, builds the configured metrics and test cases, evaluates them, prints
a pass/fail report, and exits with a non-zero status if any test case failed.

## Suite file format

A suite file is YAML with three top-level keys: `name`, `metrics`, and
`test_cases`.

```yaml
name: my_suite
metrics:
  - type: exact_match
    threshold: 0.5
  - type: pattern_match
    pattern: "refund"
  - type: answer_relevancy
    provider: openai
    model: gpt-4o-mini
    threshold: 0.7
  - type: geval
    criteria: "The output is concise and accurate."
    provider: anthropic
test_cases:
  - input: "What is a refund?"
    actual_output: "A refund returns your money."
    expected_output: "A refund returns your money."
    retrieval_context:
      - "A refund returns your money to you."
```

### Metrics

Each metric is an object tagged by `type`. Deterministic metrics need no
provider:

| `type` | Extra fields |
|--------|--------------|
| `exact_match` | `threshold` |
| `pattern_match` | `pattern`, `threshold` |
| `json_correctness` | `threshold` |

LLM-judge metrics require a `provider` (`openai` or `anthropic`) and an optional
`model` (defaults to the provider's default model):

| `type` | Extra fields |
|--------|--------------|
| `answer_relevancy` | `provider`, `model`, `threshold` |
| `faithfulness` | `provider`, `model`, `threshold` |
| `hallucination` | `provider`, `model`, `threshold` |
| `prompt_alignment` | `provider`, `model`, `threshold` |
| `contextual_precision` | `provider`, `model`, `threshold` |
| `contextual_recall` | `provider`, `model`, `threshold` |
| `contextual_relevancy` | `provider`, `model`, `threshold` |
| `ragas` | `provider`, `model`, `threshold` |
| `geval` | `criteria`, `provider`, `model`, `threshold` |

`threshold` is optional; when omitted the metric reports its score without a
pass/fail verdict.

### Test cases

Each test case supports `input`, `actual_output`, `expected_output`, and
`retrieval_context` (a list of strings). `expected_output` and
`retrieval_context` are optional and only used by metrics that need them.

## Environment variables

The CLI reads the same environment variables as the library:

- `OPENAI_API_KEY` — for the OpenAI provider.
- `ANTHROPIC_API_KEY` — for the Anthropic provider.
- `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` — optional custom base URLs.

## Releasing

Releases are driven by a `cargo release` command (an in-repo xtask in
`crates/release/`, registered as a cargo alias). It bumps the workspace
version, updates the CHANGELOG, refreshes `Cargo.lock`, commits, tags
`vX.Y.Z`, and pushes the tag, which auto-triggers the release workflow in
GitHub Actions (publish to crates.io + downloadable binaries + GitHub Release).

```bash
cargo release patch   # 0.1.0 -> 0.1.1
cargo release minor   # 0.1.0 -> 0.2.0
cargo release major   # 0.1.0 -> 1.0.0
cargo release 0.3.0   # exact version
cargo release patch --dry-run   # preview without committing/pushing
```

See the "Releasing" section of the README for details.

## Roadmap

- `deepeval login` — authenticate with the Confident AI platform (future).

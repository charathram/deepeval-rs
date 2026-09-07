# The `deepeval` CLI

The `deepeval` command-line tool is the companion to the `deepeval-rs` library.
It lets you run evaluation test suites from the terminal, similar to how
deepeval's Python CLI runs `deepeval test run`.

> **Status:** the CLI is scaffolded but the `test run` subcommand is not yet
> implemented. It lands in Phase 6. This page documents the intended interface.

## Building the CLI

The CLI lives in the `crates/deepeval` crate. Build it from the workspace root:

```bash
cargo build -p deepeval
```

The binary is named `deepeval`.

## Intended usage

```
deepeval test run <file>
```

Run an evaluation test suite defined in `<file>`. The file is expected to
define test cases and metrics using the `deepeval-rs` library.

## Environment variables

The CLI reads the same environment variables as the library:

- `OPENAI_API_KEY` — for the OpenAI provider.
- `ANTHROPIC_API_KEY` — for the Anthropic provider.
- `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` — optional custom base URLs.

## Roadmap

- `deepeval test run <file>` — run a test suite and print a pass/fail report.
- `deepeval login` — authenticate with the Confident AI platform (future).

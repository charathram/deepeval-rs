//! The `deepeval` command-line interface.
//!
//! Scaffolded in Phase 0; the `test run` subcommand is implemented in Phase 6.

use clap::{Parser, Subcommand};

/// The deepeval-rs LLM evaluation framework CLI.
#[derive(Debug, Parser)]
#[command(name = "deepeval", version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run an evaluation test suite.
    Test {
        /// Path to the test file to run.
        file: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let _cli = Cli::parse();
    // The `test run` subcommand is implemented in Phase 6.
    Ok(())
}

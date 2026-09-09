//! The `deepeval` command-line interface.
//!
//! The `test run` subcommand loads a YAML test suite, evaluates it, and prints
//! a pass/fail report.

mod suite;

use clap::{Parser, Subcommand};
use deepeval_rs::eval::evaluate;
use suite::Suite;

/// The deepeval-rs LLM evaluation framework CLI.
#[derive(Debug, Parser)]
#[command(name = "deepeval", version, about)]
pub struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Run evaluation test suites.
    Test {
        #[command(subcommand)]
        command: TestCommand,
    },
}

#[derive(Debug, Subcommand)]
enum TestCommand {
    /// Run an evaluation test suite from a YAML file.
    Run {
        /// Path to the YAML test suite file to run.
        file: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Test {
            command: TestCommand::Run { file },
        } => run_test(&file).await,
    }
}

async fn run_test(file: &str) -> anyhow::Result<()> {
    let suite = Suite::from_file(file)?;
    suite::validate(&suite)?;

    let metrics = suite.build_metrics()?;
    let test_cases = suite.build_test_cases();

    let report = evaluate(&test_cases, &metrics).await;

    print_report(&suite, &report);

    if report.failed() > 0 {
        std::process::exit(1);
    }
    Ok(())
}

fn print_report(suite: &Suite, report: &deepeval_rs::eval::EvalReport) {
    let title = if suite.name.is_empty() {
        "Test suite".to_string()
    } else {
        format!("Test suite: {}", suite.name)
    };
    println!("{title}");
    println!("{}", "-".repeat(title.len()));

    for case in &report.per_case {
        println!("\nCase: {}", case.input);
        for result in &case.results {
            let status = if result.skipped {
                "SKIPPED"
            } else {
                match result.success {
                    Some(true) => "PASS",
                    Some(false) => "FAIL",
                    // No threshold configured: report the score without a verdict.
                    None => "INFO",
                }
            };
            let score = result
                .score
                .map(|s| format!("{s:.2}"))
                .unwrap_or_else(|| "-".into());
            let threshold = result
                .threshold
                .map(|t| format!("{t:.2}"))
                .unwrap_or_else(|| "-".into());
            println!(
                "  [{status}] {}: {score} (threshold {threshold})",
                result.name
            );
            if let Some(reason) = &result.reason {
                if !reason.is_empty() {
                    println!("    reason: {reason}");
                }
            }
            if let Some(error) = &result.error {
                println!("    error: {error}");
            }
        }
    }

    println!(
        "\n{} passed, {} failed, {} skipped, {} errored",
        report.passed(),
        report.failed(),
        report.skipped(),
        report.errored()
    );
    let cost = report
        .total_cost()
        .map(|c| format!("${c:.4}"))
        .unwrap_or_else(|| "n/a".into());
    println!(
        "tokens: {} in / {} out | cost: {cost}",
        report.total_input_tokens(),
        report.total_output_tokens()
    );
}

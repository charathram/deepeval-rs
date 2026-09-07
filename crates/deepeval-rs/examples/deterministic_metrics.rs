//! Deterministic metrics — no LLM required.
//!
//! Run with: `cargo run --example deterministic_metrics`
//!
//! These metrics score an output without calling an LLM, so they work with no
//! API key and no provider. Each one is built with a builder, measured against
//! a test case, and reports a 0-1 score plus whether it passed its threshold.

use deepeval_rs::metrics::{ExactMatchMetric, JsonCorrectnessMetric, Metric, PatternMatchMetric};
use deepeval_rs::test_case::LLMTestCase;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A test case whose actual output is a phone number.
    let phone_case = LLMTestCase::builder()
        .input("What is the support phone number?")
        .actual_output("Call us at 555-1234.")
        .expected_output("Call us at 555-1234.")
        .build();

    // ExactMatchMetric: 1.0 when actual == expected.
    let mut exact = ExactMatchMetric::builder().threshold(0.5).build();
    exact.measure(&phone_case).await?;
    print_result("ExactMatchMetric", &exact);

    // PatternMatchMetric: 1.0 when actual matches a regex.
    let mut pattern = PatternMatchMetric::builder()
        .pattern(r"\d{3}-\d{4}")
        .threshold(0.5)
        .build();
    pattern.measure(&phone_case).await?;
    print_result("PatternMatchMetric", &pattern);

    // JsonCorrectnessMetric: 1.0 when actual parses as valid JSON.
    let json_case = LLMTestCase::builder()
        .input("Return the capital of France as JSON.")
        .actual_output(r#"{"capital": "Paris"}"#)
        .build();
    let mut json = JsonCorrectnessMetric::builder().threshold(0.5).build();
    json.measure(&json_case).await?;
    print_result("JsonCorrectnessMetric", &json);

    Ok(())
}

/// Print a metric's name, score, and pass/fail status.
fn print_result(name: &str, metric: &dyn Metric) {
    let status = match metric.is_successful() {
        Some(true) => "PASS",
        Some(false) => "FAIL",
        None => "SKIPPED",
    };
    println!(
        "{name:<22} score={:<4} {status}",
        metric
            .score()
            .map_or_else(|| "-".to_string(), |s| format!("{s:.2}"))
    );
}

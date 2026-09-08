//! Evaluation orchestration: `evaluate` and `assert_test`.
//!
//! [`evaluate`] runs a set of metrics over a set of test cases concurrently and
//! returns a serializable [`EvalReport`]. [`assert_test`] runs metrics over a
//! single test case and returns an error if any metric fails.

mod report;

pub use report::{CaseReport, EvalReport};

use crate::error::EvalError;
use crate::metrics::{ConversationalMetric, Metric, MetricResult};
use crate::test_case::{ConversationalTestCase, LLMTestCase};
use futures::stream::{FuturesUnordered, StreamExt};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Run `metrics` over `test_cases` concurrently and return a report.
///
/// Each `(test_case, metric)` pair is measured independently (metrics are
/// cloned per pair). A [`Semaphore`] caps the number of concurrent LLM calls.
pub async fn evaluate(test_cases: &[LLMTestCase], metrics: &[Box<dyn Metric>]) -> EvalReport {
    evaluate_with_concurrency(test_cases, metrics, DEFAULT_CONCURRENCY).await
}

/// The default maximum number of concurrent metric measurements.
pub const DEFAULT_CONCURRENCY: usize = 10;

/// Run `metrics` over `test_cases` with a bounded concurrency limit.
pub async fn evaluate_with_concurrency(
    test_cases: &[LLMTestCase],
    metrics: &[Box<dyn Metric>],
    concurrency: usize,
) -> EvalReport {
    let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut tasks = FuturesUnordered::new();

    for test_case in test_cases {
        for metric in metrics {
            let semaphore = semaphore.clone();
            let test_case = test_case.clone();
            let mut metric = metric.clone();
            tasks.push(async move {
                let _permit = semaphore.acquire().await;
                let result = measure_one(&mut metric, &test_case).await;
                (test_case, result)
            });
        }
    }

    // Group results by test case input, preserving order of first appearance.
    let mut per_case: Vec<CaseReport> = Vec::new();
    let mut index_by_input: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    while let Some((test_case, result)) = tasks.next().await {
        let idx = match index_by_input.get(&test_case.input) {
            Some(&idx) => idx,
            None => {
                per_case.push(CaseReport {
                    input: test_case.input.clone(),
                    results: Vec::new(),
                });
                let idx = per_case.len() - 1;
                index_by_input.insert(test_case.input.clone(), idx);
                idx
            }
        };
        per_case[idx].results.push(result);
    }

    EvalReport { per_case }
}

/// Measure a single metric against a test case, capturing the result.
async fn measure_one(metric: &mut Box<dyn Metric>, test_case: &LLMTestCase) -> MetricResult {
    let name = metric.name().to_string();
    let threshold = metric.threshold();
    let result = metric.measure(test_case).await;

    match result {
        Ok(()) => MetricResult {
            name,
            score: metric.score(),
            reason: metric.reason().map(str::to_string),
            success: metric.is_successful(),
            threshold,
            cost: metric.cost(),
            input_tokens: metric.input_tokens(),
            output_tokens: metric.output_tokens(),
            skipped: metric.skipped(),
            error: None,
        },
        Err(e) => MetricResult {
            name,
            score: None,
            reason: None,
            success: Some(false),
            threshold,
            cost: metric.cost(),
            input_tokens: metric.input_tokens(),
            output_tokens: metric.output_tokens(),
            skipped: metric.skipped(),
            error: Some(e.to_string()),
        },
    }
}

/// Run `metrics` over a single `test_case` and return an error if any metric
/// fails.
///
/// This is the plain-library analogue of deepeval's `assert_test`. It returns
/// `Ok(())` when every metric passes (or has no threshold), and
/// [`EvalError::AssertionFailed`] otherwise.
pub async fn assert_test(
    test_case: &LLMTestCase,
    metrics: &[Box<dyn Metric>],
) -> Result<(), EvalError> {
    let report = evaluate(std::slice::from_ref(test_case), metrics).await;
    let failures = report
        .per_case
        .iter()
        .flat_map(|case| case.results.iter())
        .filter(|result| result.success == Some(false))
        .map(|result| result.name.clone())
        .collect::<Vec<_>>();

    if failures.is_empty() {
        Ok(())
    } else {
        Err(EvalError::assertion_failed(&failures))
    }
}

/// Run `metrics` over multi-turn `test_cases` concurrently and return a report.
///
/// Each `(test_case, metric)` pair is measured independently (metrics are
/// cloned per pair). A [`Semaphore`] caps the number of concurrent LLM calls.
/// Results are grouped per test case, keyed by the first turn's input (or an
/// empty string when the case has no turns).
pub async fn evaluate_conversational(
    test_cases: &[ConversationalTestCase],
    metrics: &[Box<dyn ConversationalMetric>],
) -> EvalReport {
    evaluate_conversational_with_concurrency(test_cases, metrics, DEFAULT_CONCURRENCY).await
}

/// Run `metrics` over multi-turn `test_cases` with a bounded concurrency limit.
pub async fn evaluate_conversational_with_concurrency(
    test_cases: &[ConversationalTestCase],
    metrics: &[Box<dyn ConversationalMetric>],
    concurrency: usize,
) -> EvalReport {
    let semaphore = Arc::new(Semaphore::new(concurrency.max(1)));
    let mut tasks = FuturesUnordered::new();

    for test_case in test_cases {
        for metric in metrics {
            let semaphore = semaphore.clone();
            let test_case = test_case.clone();
            let mut metric = metric.clone();
            tasks.push(async move {
                let _permit = semaphore.acquire().await;
                let result = measure_conversational_one(&mut metric, &test_case).await;
                (test_case, result)
            });
        }
    }

    // Group results by the first turn's input, preserving order of first
    // appearance.
    let mut per_case: Vec<CaseReport> = Vec::new();
    let mut index_by_input: std::collections::HashMap<String, usize> =
        std::collections::HashMap::new();
    while let Some((test_case, result)) = tasks.next().await {
        let key = case_key(&test_case);
        let idx = match index_by_input.get(&key) {
            Some(&idx) => idx,
            None => {
                per_case.push(CaseReport {
                    input: key.clone(),
                    results: Vec::new(),
                });
                let idx = per_case.len() - 1;
                index_by_input.insert(key, idx);
                idx
            }
        };
        per_case[idx].results.push(result);
    }

    EvalReport { per_case }
}

/// A stable key for grouping reports: the first turn's input (or empty).
fn case_key(test_case: &ConversationalTestCase) -> String {
    test_case
        .turns
        .first()
        .map(|turn| turn.input.clone())
        .unwrap_or_default()
}

/// Measure a single multi-turn metric against a test case, capturing the
/// result.
async fn measure_conversational_one(
    metric: &mut Box<dyn ConversationalMetric>,
    test_case: &ConversationalTestCase,
) -> MetricResult {
    let name = metric.name().to_string();
    let threshold = metric.threshold();
    let result = metric.measure(test_case).await;

    match result {
        Ok(()) => MetricResult {
            name,
            score: metric.score(),
            reason: metric.reason().map(str::to_string),
            success: metric.is_successful(),
            threshold,
            cost: metric.cost(),
            input_tokens: metric.input_tokens(),
            output_tokens: metric.output_tokens(),
            skipped: metric.skipped(),
            error: None,
        },
        Err(e) => MetricResult {
            name,
            score: None,
            reason: None,
            success: Some(false),
            threshold,
            cost: metric.cost(),
            input_tokens: metric.input_tokens(),
            output_tokens: metric.output_tokens(),
            skipped: metric.skipped(),
            error: Some(e.to_string()),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::MetricConfig;
    use crate::test_case::Turn;
    use async_trait::async_trait;

    /// A trivial metric that always scores a fixed value.
    #[derive(Clone)]
    struct FixedMetric {
        name: String,
        config: MetricConfig,
        score: Option<f32>,
        reason: Option<String>,
    }

    impl FixedMetric {
        fn new(name: &str, score: f32, threshold: Option<f32>) -> Self {
            Self {
                name: name.to_string(),
                config: MetricConfig {
                    threshold,
                    ..MetricConfig::default()
                },
                score: Some(score),
                reason: None,
            }
        }
    }

    #[async_trait]
    impl Metric for FixedMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn threshold(&self) -> Option<f32> {
            self.config.threshold
        }

        async fn measure(
            &mut self,
            _test_case: &LLMTestCase,
        ) -> Result<(), crate::error::MetricError> {
            Ok(())
        }

        fn score(&self) -> Option<f32> {
            self.score
        }

        fn reason(&self) -> Option<&str> {
            self.reason.as_deref()
        }

        fn clone_box(&self) -> Box<dyn Metric> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn evaluate_measures_all_pairs() {
        let tc1 = LLMTestCase::builder().input("first").build();
        let tc2 = LLMTestCase::builder().input("second").build();
        let metrics: Vec<Box<dyn Metric>> = vec![
            Box::new(FixedMetric::new("a", 0.9, Some(0.5))),
            Box::new(FixedMetric::new("b", 0.2, Some(0.5))),
        ];

        let report = evaluate(&[tc1, tc2], &metrics).await;
        assert_eq!(report.per_case.len(), 2);
        assert_eq!(report.per_case[0].results.len(), 2);
        assert_eq!(report.per_case[0].results[0].name, "a");
        assert_eq!(report.per_case[0].results[0].success, Some(true));
        assert_eq!(report.per_case[0].results[1].success, Some(false));
    }

    #[tokio::test]
    async fn assert_test_passes_when_all_pass() {
        let tc = LLMTestCase::builder().input("hi").build();
        let metrics: Vec<Box<dyn Metric>> = vec![Box::new(FixedMetric::new("a", 0.9, Some(0.5)))];
        assert_test(&tc, &metrics).await.unwrap();
    }

    #[tokio::test]
    async fn assert_test_fails_when_metric_fails() {
        let tc = LLMTestCase::builder().input("hi").build();
        let metrics: Vec<Box<dyn Metric>> = vec![Box::new(FixedMetric::new("a", 0.2, Some(0.5)))];
        let err = assert_test(&tc, &metrics).await.unwrap_err();
        assert!(matches!(err, EvalError::AssertionFailed(_)));
    }

    #[tokio::test]
    async fn assert_test_ignores_metrics_without_threshold() {
        let tc = LLMTestCase::builder().input("hi").build();
        let metrics: Vec<Box<dyn Metric>> = vec![Box::new(FixedMetric::new("a", 0.2, None))];
        assert_test(&tc, &metrics).await.unwrap();
    }

    /// A trivial multi-turn metric that always scores a fixed value.
    #[derive(Clone)]
    struct FixedConversationalMetric {
        name: String,
        config: MetricConfig,
        score: Option<f32>,
    }

    impl FixedConversationalMetric {
        fn new(name: &str, score: f32, threshold: Option<f32>) -> Self {
            Self {
                name: name.to_string(),
                config: MetricConfig {
                    threshold,
                    ..MetricConfig::default()
                },
                score: Some(score),
            }
        }
    }

    #[async_trait]
    impl ConversationalMetric for FixedConversationalMetric {
        fn name(&self) -> &str {
            &self.name
        }

        fn threshold(&self) -> Option<f32> {
            self.config.threshold
        }

        async fn measure(
            &mut self,
            _test_case: &ConversationalTestCase,
        ) -> Result<(), crate::error::MetricError> {
            Ok(())
        }

        fn score(&self) -> Option<f32> {
            self.score
        }

        fn reason(&self) -> Option<&str> {
            None
        }

        fn clone_box(&self) -> Box<dyn ConversationalMetric> {
            Box::new(self.clone())
        }
    }

    #[tokio::test]
    async fn evaluate_conversational_measures_all_pairs() {
        let tc1 = ConversationalTestCase::builder()
            .turns(vec![Turn::builder().input("first").build()])
            .build();
        let tc2 = ConversationalTestCase::builder()
            .turns(vec![Turn::builder().input("second").build()])
            .build();
        let metrics: Vec<Box<dyn ConversationalMetric>> = vec![
            Box::new(FixedConversationalMetric::new("a", 0.9, Some(0.5))),
            Box::new(FixedConversationalMetric::new("b", 0.2, Some(0.5))),
        ];

        let report = evaluate_conversational(&[tc1, tc2], &metrics).await;
        assert_eq!(report.per_case.len(), 2);
        assert_eq!(report.per_case[0].input, "first");
        assert_eq!(report.per_case[0].results.len(), 2);
        assert_eq!(report.per_case[0].results[0].name, "a");
        assert_eq!(report.per_case[0].results[0].success, Some(true));
        assert_eq!(report.per_case[0].results[1].success, Some(false));
    }

    #[tokio::test]
    async fn evaluate_conversational_empty_case_groups_by_empty_key() {
        let tc = ConversationalTestCase::builder().build();
        let metrics: Vec<Box<dyn ConversationalMetric>> =
            vec![Box::new(FixedConversationalMetric::new("a", 0.5, None))];

        let report = evaluate_conversational(&[tc.clone(), tc], &metrics).await;
        assert_eq!(report.per_case.len(), 1);
        assert_eq!(report.per_case[0].input, "");
        assert_eq!(report.per_case[0].results.len(), 2);
    }
}

//! Exact match: a deterministic metric scoring 1.0 when the actual output
//! exactly equals the expected output.

use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::test_case::LLMTestCase;

/// ExactMatchMetric: scores 1.0 when the actual output exactly equals the
/// expected output, else 0.0. No LLM required.
///
/// Requires `actual_output` and `expected_output` on the test case.
#[derive(Debug, Clone)]
pub struct ExactMatchMetric {
    state: MetricState,
}

impl ExactMatchMetric {
    /// Start building an [`ExactMatchMetric`].
    pub fn builder() -> ExactMatchMetricBuilder {
        ExactMatchMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        let (Some(actual), Some(expected)) = (&test_case.actual_output, &test_case.expected_output)
        else {
            self.state.skipped = true;
            return Ok(());
        };

        let matched = actual == expected;
        self.state.score = Some(if matched { 1.0 } else { 0.0 });
        self.state.reason = Some(if matched {
            "actual output exactly matches expected output".to_string()
        } else {
            "actual output does not match expected output".to_string()
        });
        Ok(())
    }
}

impl_metric!(ExactMatchMetric, "ExactMatchMetric");

/// Builder for [`ExactMatchMetric`].
#[derive(Debug, Default)]
pub struct ExactMatchMetricBuilder {
    config: MetricConfig,
}

impl ExactMatchMetricBuilder {
    /// Set the pass/fail threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.config.threshold = Some(threshold);
        self
    }

    /// Set whether to include a reason.
    pub fn include_reason(mut self, include_reason: bool) -> Self {
        self.config.include_reason = include_reason;
        self
    }

    /// Build the metric.
    pub fn build(self) -> ExactMatchMetric {
        ExactMatchMetric {
            state: MetricState::new(self.config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metric;

    #[tokio::test]
    async fn scores_one_on_exact_match() {
        let mut metric = ExactMatchMetric::builder().threshold(0.5).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .expected_output("hello")
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(1.0));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_zero_on_mismatch() {
        let mut metric = ExactMatchMetric::builder().threshold(0.5).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .expected_output("goodbye")
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(0.0));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_expected_output_missing() {
        let mut metric = ExactMatchMetric::builder().build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("hello")
            .build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

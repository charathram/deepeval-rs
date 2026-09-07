//! Pattern match: a deterministic metric scoring 1.0 when the actual output
//! matches a regular expression.

use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::test_case::LLMTestCase;

/// PatternMatchMetric: scores 1.0 when the actual output matches a regular
/// expression, else 0.0. No LLM required.
///
/// Requires `actual_output` on the test case and a `pattern` on the metric.
#[derive(Debug, Clone)]
pub struct PatternMatchMetric {
    state: MetricState,
    pattern: regex::Regex,
}

impl PatternMatchMetric {
    /// Start building a [`PatternMatchMetric`].
    pub fn builder() -> PatternMatchMetricBuilder {
        PatternMatchMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        let Some(actual) = &test_case.actual_output else {
            self.state.skipped = true;
            return Ok(());
        };

        let matched = self.pattern.is_match(actual);
        self.state.score = Some(if matched { 1.0 } else { 0.0 });
        self.state.reason = Some(if matched {
            "actual output matches the pattern".to_string()
        } else {
            "actual output does not match the pattern".to_string()
        });
        Ok(())
    }
}

impl_metric!(PatternMatchMetric, "PatternMatchMetric");

/// Builder for [`PatternMatchMetric`].
#[derive(Debug, Default)]
pub struct PatternMatchMetricBuilder {
    config: MetricConfig,
    pattern: Option<String>,
}

impl PatternMatchMetricBuilder {
    /// Set the regular expression pattern (required).
    pub fn pattern(mut self, pattern: impl Into<String>) -> Self {
        self.pattern = Some(pattern.into());
        self
    }

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
    ///
    /// # Panics
    ///
    /// Panics if no pattern was set or the pattern is not a valid regex.
    pub fn build(self) -> PatternMatchMetric {
        let pattern = self.pattern.expect("PatternMatchMetric requires a pattern");
        let pattern = regex::Regex::new(&pattern).expect("invalid regex pattern");
        PatternMatchMetric {
            state: MetricState::new(self.config),
            pattern,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metric;

    #[tokio::test]
    async fn scores_one_on_match() {
        let mut metric = PatternMatchMetric::builder()
            .pattern(r"^\d{3}-\d{4}$")
            .threshold(0.5)
            .build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("123-4567")
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(1.0));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_zero_on_no_match() {
        let mut metric = PatternMatchMetric::builder()
            .pattern(r"^\d{3}-\d{4}$")
            .threshold(0.5)
            .build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("not-a-phone")
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(0.0));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_actual_output_missing() {
        let mut metric = PatternMatchMetric::builder().pattern(".*").build();
        let tc = LLMTestCase::builder().input("hi").build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

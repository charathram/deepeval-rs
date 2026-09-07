//! JSON correctness: a deterministic metric scoring 1.0 when the actual output
//! is valid JSON.

use super::{impl_metric, MetricConfig, MetricState};
use crate::error::MetricError;
use crate::test_case::LLMTestCase;

/// JsonCorrectnessMetric: scores 1.0 when the actual output parses as valid
/// JSON, else 0.0. No LLM required.
///
/// Requires `actual_output` on the test case.
#[derive(Debug, Clone)]
pub struct JsonCorrectnessMetric {
    state: MetricState,
}

impl JsonCorrectnessMetric {
    /// Start building a [`JsonCorrectnessMetric`].
    pub fn builder() -> JsonCorrectnessMetricBuilder {
        JsonCorrectnessMetricBuilder::default()
    }

    async fn measure_impl(&mut self, test_case: &LLMTestCase) -> Result<(), MetricError> {
        let Some(actual) = &test_case.actual_output else {
            self.state.skipped = true;
            return Ok(());
        };

        let valid = serde_json::from_str::<serde_json::Value>(actual).is_ok();
        self.state.score = Some(if valid { 1.0 } else { 0.0 });
        self.state.reason = Some(if valid {
            "actual output is valid JSON".to_string()
        } else {
            "actual output is not valid JSON".to_string()
        });
        Ok(())
    }
}

impl_metric!(JsonCorrectnessMetric, "JsonCorrectnessMetric");

/// Builder for [`JsonCorrectnessMetric`].
#[derive(Debug, Default)]
pub struct JsonCorrectnessMetricBuilder {
    config: MetricConfig,
}

impl JsonCorrectnessMetricBuilder {
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
    pub fn build(self) -> JsonCorrectnessMetric {
        JsonCorrectnessMetric {
            state: MetricState::new(self.config),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metric;

    #[tokio::test]
    async fn scores_one_on_valid_json() {
        let mut metric = JsonCorrectnessMetric::builder().threshold(0.5).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output(r#"{"answer": "Paris"}"#)
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(1.0));
        assert_eq!(metric.is_successful(), Some(true));
    }

    #[tokio::test]
    async fn scores_zero_on_invalid_json() {
        let mut metric = JsonCorrectnessMetric::builder().threshold(0.5).build();
        let tc = LLMTestCase::builder()
            .input("hi")
            .actual_output("not json")
            .build();

        metric.measure(&tc).await.unwrap();
        assert_eq!(metric.score(), Some(0.0));
        assert_eq!(metric.is_successful(), Some(false));
    }

    #[tokio::test]
    async fn skips_when_actual_output_missing() {
        let mut metric = JsonCorrectnessMetric::builder().build();
        let tc = LLMTestCase::builder().input("hi").build();

        metric.measure(&tc).await.unwrap();
        assert!(metric.skipped());
        assert_eq!(metric.score(), None);
    }
}

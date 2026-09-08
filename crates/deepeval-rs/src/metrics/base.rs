//! Shared metric configuration and result types.

use serde::{Deserialize, Serialize};

/// Configuration shared by all metrics.
///
/// Mirrors deepeval's per-metric configuration (threshold, include_reason,
/// strict_mode, async_mode, verbose_mode).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricConfig {
    /// The pass/fail threshold. A metric passes when `score >= threshold`.
    pub threshold: Option<f32>,
    /// Whether to generate a natural-language reason for the score.
    pub include_reason: bool,
    /// Whether to use strict mode (affects how some metrics parse verdicts).
    pub strict_mode: bool,
    /// Whether to run asynchronously.
    pub async_mode: bool,
    /// Whether to emit verbose logs.
    pub verbose_mode: bool,
}

impl Default for MetricConfig {
    fn default() -> Self {
        Self {
            threshold: None,
            include_reason: false,
            strict_mode: false,
            async_mode: true,
            verbose_mode: true,
        }
    }
}

impl MetricConfig {
    /// Create a config with the given threshold.
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            threshold: Some(threshold),
            ..Self::default()
        }
    }

    /// Set the threshold.
    pub fn threshold(mut self, threshold: f32) -> Self {
        self.threshold = Some(threshold);
        self
    }

    /// Set whether to include a reason.
    pub fn include_reason(mut self, include_reason: bool) -> Self {
        self.include_reason = include_reason;
        self
    }

    /// Set strict mode.
    pub fn strict_mode(mut self, strict_mode: bool) -> Self {
        self.strict_mode = strict_mode;
        self
    }
}

/// The result of measuring a metric against a test case.
///
/// This is the serializable output of an evaluation run, distinct from the
/// in-memory state a [`Metric`](super::Metric) records on itself.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricResult {
    /// The name of the metric.
    pub name: String,
    /// The recorded score, if any.
    pub score: Option<f32>,
    /// The recorded reason, if any.
    pub reason: Option<String>,
    /// Whether the metric passed, if determinable.
    pub success: Option<bool>,
    /// The threshold used, if any.
    pub threshold: Option<f32>,
    /// The estimated cost of the evaluation, if known.
    pub cost: Option<f64>,
    /// The number of input tokens used.
    pub input_tokens: u32,
    /// The number of output tokens used.
    pub output_tokens: u32,
    /// Whether the metric was skipped (e.g. a required field was missing).
    pub skipped: bool,
    /// An error message if the metric failed to measure.
    pub error: Option<String>,
}

impl MetricResult {
    /// Create a result for a metric that was skipped.
    pub fn skipped(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            score: None,
            reason: None,
            success: None,
            threshold: None,
            cost: None,
            input_tokens: 0,
            output_tokens: 0,
            skipped: true,
            error: None,
        }
    }
}

/// Shared in-memory measurement state for a metric.
///
/// Holds the metric's configuration plus the score, reason, and skip flag that
/// [`Metric::measure`](super::Metric::measure) records on `self`. Cloning a
/// [`MetricState`] resets the measurement fields so a cloned metric starts
/// fresh (mirroring deepeval's per-`(test_case, metric)` cloning).
#[derive(Debug)]
pub(crate) struct MetricState {
    /// The metric configuration.
    pub config: MetricConfig,
    /// The recorded score, if any.
    pub score: Option<f32>,
    /// The recorded reason, if any.
    pub reason: Option<String>,
    /// Whether the metric was skipped (a required field was missing).
    pub skipped: bool,
    /// The estimated cost of the LLM calls, if known.
    pub cost: Option<f64>,
    /// The number of input tokens used.
    pub input_tokens: u32,
    /// The number of output tokens used.
    pub output_tokens: u32,
}

impl MetricState {
    /// Create fresh state for the given configuration.
    pub fn new(config: MetricConfig) -> Self {
        Self {
            config,
            score: None,
            reason: None,
            skipped: false,
            cost: None,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    /// A fresh copy with the same configuration but cleared measurement state.
    pub fn reset(&self) -> Self {
        Self::new(self.config.clone())
    }

    /// Accrue usage from an LLM response into this state.
    pub fn accrue_usage(&mut self, response: &crate::llm::LlmResponse) {
        self.input_tokens = self.input_tokens.saturating_add(response.input_tokens);
        self.output_tokens = self.output_tokens.saturating_add(response.output_tokens);
        if let Some(cost) = response.cost {
            self.cost = Some(self.cost.unwrap_or(0.0) + cost);
        }
    }
}

impl Clone for MetricState {
    fn clone(&self) -> Self {
        self.reset()
    }
}

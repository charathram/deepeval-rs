//! Evaluation report types.

use serde::{Deserialize, Serialize};

use crate::metrics::MetricResult;

/// The result of evaluating one test case against one or more metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaseReport {
    /// The input of the test case.
    pub input: String,
    /// The metric results for this test case.
    pub results: Vec<MetricResult>,
}

/// The full report of an evaluation run.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EvalReport {
    /// One report per test case.
    pub per_case: Vec<CaseReport>,
}

impl EvalReport {
    /// The total number of metric measurements in this report.
    pub fn total_measurements(&self) -> usize {
        self.per_case.iter().map(|c| c.results.len()).sum()
    }

    /// The number of metric measurements that passed.
    pub fn passed(&self) -> usize {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|r| r.success == Some(true))
            .count()
    }

    /// The number of metric measurements that failed.
    pub fn failed(&self) -> usize {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|r| r.success == Some(false))
            .count()
    }

    /// The number of metric measurements that were skipped.
    pub fn skipped(&self) -> usize {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|r| r.skipped)
            .count()
    }

    /// The number of metric measurements that errored.
    pub fn errored(&self) -> usize {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .filter(|r| r.error.is_some())
            .count()
    }

    /// The total number of input tokens used across all measurements.
    pub fn total_input_tokens(&self) -> u64 {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .map(|r| r.input_tokens as u64)
            .sum()
    }

    /// The total number of output tokens used across all measurements.
    pub fn total_output_tokens(&self) -> u64 {
        self.per_case
            .iter()
            .flat_map(|c| c.results.iter())
            .map(|r| r.output_tokens as u64)
            .sum()
    }

    /// The total estimated cost across all measurements, if any are known.
    pub fn total_cost(&self) -> Option<f64> {
        let mut total = 0.0;
        let mut any = false;
        for result in self.per_case.iter().flat_map(|c| c.results.iter()) {
            if let Some(cost) = result.cost {
                total += cost;
                any = true;
            }
        }
        if any {
            Some(total)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(name: &str, success: Option<bool>, skipped: bool) -> MetricResult {
        MetricResult {
            name: name.to_string(),
            score: None,
            reason: None,
            success,
            threshold: None,
            cost: None,
            input_tokens: 0,
            output_tokens: 0,
            skipped,
            error: None,
        }
    }

    #[test]
    fn report_counts() {
        let report = EvalReport {
            per_case: vec![CaseReport {
                input: "hi".to_string(),
                results: vec![
                    result("a", Some(true), false),
                    result("b", Some(false), false),
                    result("c", None, true),
                ],
            }],
        };

        assert_eq!(report.total_measurements(), 3);
        assert_eq!(report.passed(), 1);
        assert_eq!(report.failed(), 1);
        assert_eq!(report.skipped(), 1);
        assert_eq!(report.errored(), 0);
    }
}

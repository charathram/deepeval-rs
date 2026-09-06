//! # deepeval-rs
//!
//! The LLM evaluation framework for Rust — a port of
//! [deepeval](https://github.com/confident-ai/deepeval).
//!
//! deepeval-rs lets you evaluate LLM applications (agents, RAG pipelines, chatbots)
//! with research-backed metrics such as G-Eval, answer relevancy, faithfulness, and
//! hallucination. Metrics run asynchronously and can be composed into an
//! [`evaluate`] run over many test cases.
//!
//! ## Quickstart
//!
//! ```ignore
//! use deepeval::test_case::LLMTestCase;
//! use deepeval::metrics::AnswerRelevancyMetric;
//! use deepeval::eval::assert_test;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let metric = AnswerRelevancyMetric::builder().threshold(0.7).build();
//!     let test_case = LLMTestCase::builder()
//!         .input("What if these shoes don't fit?")
//!         .actual_output("We offer a 30-day full refund at no extra costs.")
//!         .retrieval_context(vec![
//!             "All customers are eligible for a 30 day full refund at no extra costs.".to_string(),
//!         ])
//!         .build();
//!
//!     assert_test(&test_case, &[Box::new(metric)]).await?;
//!     Ok(())
//! }
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod embeddings;
pub mod error;
pub mod eval;
pub mod llm;
pub mod metrics;
pub mod template;
pub mod test_case;

pub use error::{EvalError, LlmError, MetricError, TemplateError};

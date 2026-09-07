//! Multi-turn (conversational) metrics — scoring agentic conversations.
//!
//! Run with: `cargo run --example multiturn`
//!
//! Conversational metrics evaluate a [`ConversationalTestCase`] — an ordered
//! sequence of [`Turn`]s — rather than a single input/output pair. This
//! example shows three of them measured over a short customer-support
//! conversation.
//!
//! It uses [`MockLlmProvider`] so it runs with no API key. To use a real
//! model, swap the mock for a [`RigProvider`](crate::llm::RigProvider) wrapping
//! a rig completion model — see `real_provider_openai` and
//! `real_provider_anthropic` for complete wiring examples.

use deepeval_rs::llm::MockLlmProvider;
use deepeval_rs::metrics::{
    ConversationCompletenessMetric, ConversationalMetric, TurnFaithfulnessMetric,
    TurnRelevancyMetric,
};
use deepeval_rs::test_case::{ConversationalTestCase, Turn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // A short back-and-forth: the agent answers, then follows up and grounds
    // its answer in a retrieval context.
    let test_case = ConversationalTestCase::builder()
        .turn(
            Turn::builder()
                .input("What is the return policy?")
                .actual_output("We offer a 30-day full refund at no extra costs.")
                .retrieval_context(vec![
                    "All customers are eligible for a 30 day full refund.".to_string()
                ])
                .build(),
        )
        .turn(
            Turn::builder()
                .input("What if I need to return a gift?")
                .actual_output("Gift returns work the same way — 30 days, full refund.")
                .retrieval_context(vec![
                    "Gift recipients can return items within 30 days.".to_string()
                ])
                .build(),
        )
        .build();

    // Each metric needs its own provider so the mock response queue is fresh.
    let metrics: Vec<Box<dyn ConversationalMetric>> = vec![
        Box::new(
            ConversationCompletenessMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.9, "reason": "complete"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
        Box::new(
            TurnRelevancyMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.8, "reason": "relevant"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
        Box::new(
            TurnFaithfulnessMetric::builder()
                .provider(MockLlmProvider::text(
                    r#"{"score": 0.85, "reason": "grounded"}"#,
                ))
                .threshold(0.7)
                .include_reason(true)
                .build(),
        ),
    ];

    let report =
        deepeval_rs::eval::evaluate_conversational(std::slice::from_ref(&test_case), &metrics)
            .await;

    let case = &report.per_case[0];
    for result in &case.results {
        let status = match result.success {
            Some(true) => "PASS",
            Some(false) => "FAIL",
            None => "SKIPPED",
        };
        println!("{:<28} {status}", result.name);
        if let Some(score) = result.score {
            println!("  score={score:.2}");
        }
        if let Some(reason) = &result.reason {
            println!("  reason: {reason}");
        }
    }

    Ok(())
}

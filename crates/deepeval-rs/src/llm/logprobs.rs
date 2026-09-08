//! Log-probability extraction and weighted scoring.
//!
//! Some metrics (notably GEval) score a model's output by inspecting the log
//! probabilities the provider assigned to the tokens it generated. This module
//! provides two pieces:
//!
//! * [`extract_logprobs`] — parse the provider's raw response into a
//!   normalized [`TokenLogprobs`] list.
//! * [`calculate_weighted_summed_score`] — turn a raw integer score and the
//!   token log probabilities into a confidence-weighted score, mirroring
//!   deepeval's g-eval fix.

use serde_json::Value;

use super::TokenLogprobs;

/// Extract per-token log probabilities from a provider's raw response.
///
/// The OpenAI chat-completions wire format places them at
/// `choices[0].logprobs.content`, an array of objects shaped like
/// `{ "token": "...", "logprob": -0.1, "top_logprobs": [{ "token": "...", "logprob": -0.2 }] }`.
/// Any missing or malformed field is skipped; `None` is returned when no
/// usable log probabilities are present.
pub fn extract_logprobs(raw: &Value) -> Option<Vec<TokenLogprobs>> {
    let content = raw
        .get("choices")?
        .get(0)?
        .get("logprobs")?
        .get("content")?;

    let entries = content.as_array()?;
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        let token = entry.get("token")?.as_str()?.to_owned();
        let logprob = entry.get("logprob")?.as_f64()?;
        let top_logprobs = entry
            .get("top_logprobs")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|t| {
                        Some(super::TokenLogprob {
                            token: t.get("token")?.as_str()?.to_owned(),
                            logprob: t.get("logprob")?.as_f64()?,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(TokenLogprobs {
            token,
            logprob,
            top_logprobs,
        });
    }
    Some(out)
}

/// Compute a confidence-weighted score from a raw integer score and the token
/// log probabilities, mirroring deepeval's g-eval fix.
///
/// The model is asked to emit an integer score (e.g. 0-10). The last generated
/// token whose text equals `str(raw_score)` is located; its `top_logprobs`
/// are then filtered to tokens that are decimal and have a log probability of
/// at least `ln(0.01)`. The surviving tokens are converted to linear
/// probabilities, grouped by their numeric value, and combined as a weighted
/// sum. If no usable tokens survive, the raw score is returned unchanged.
pub fn calculate_weighted_summed_score(raw_score: f64, logprobs: &[TokenLogprobs]) -> f64 {
    let target = raw_score.to_string();
    let Some(position) = logprobs.iter().rposition(|lp| lp.token.trim() == target) else {
        return raw_score;
    };

    let candidates = logprobs[position]
        .top_logprobs
        .iter()
        .filter(|t| t.logprob >= f64::ln(0.01))
        .filter_map(|t| {
            let score = t.token.trim().parse::<f64>().ok()?;
            Some((score, t.logprob.exp()))
        })
        .collect::<Vec<_>>();

    if candidates.is_empty() {
        return raw_score;
    }

    let total: f64 = candidates.iter().map(|(_, p)| p).sum();
    if total <= 0.0 {
        return raw_score;
    }

    candidates.iter().map(|(score, p)| score * p / total).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{TokenLogprob, TokenLogprobs};

    fn lp(token: &str, logprob: f64) -> TokenLogprob {
        TokenLogprob {
            token: token.to_owned(),
            logprob,
        }
    }

    #[test]
    fn extracts_openai_logprobs() {
        let raw = serde_json::json!({
            "choices": [{
                "logprobs": {
                    "content": [
                        { "token": "8", "logprob": -0.1,
                          "top_logprobs": [
                              { "token": "8", "logprob": -0.1 },
                              { "token": "9", "logprob": -1.5 }
                          ] }
                    ]
                }
            }]
        });
        let logprobs = extract_logprobs(&raw).unwrap();
        assert_eq!(logprobs.len(), 1);
        assert_eq!(logprobs[0].token, "8");
        assert_eq!(logprobs[0].top_logprobs.len(), 2);
    }

    #[test]
    fn extract_returns_none_without_logprobs() {
        let raw = serde_json::json!({ "choices": [{ "logprobs": null }] });
        assert!(extract_logprobs(&raw).is_none());
    }

    #[test]
    fn weighted_score_uses_top_logprobs() {
        // Raw score 8; the top alternatives are 8 (prob 0.9) and 9 (prob 0.1).
        let logprobs = vec![TokenLogprobs {
            token: "8".to_owned(),
            logprob: -0.1,
            top_logprobs: vec![lp("8", -0.105), lp("9", -2.3)],
        }];
        let score = calculate_weighted_summed_score(8.0, &logprobs);
        assert!((score - 8.1).abs() < 0.01, "got {score}");
    }

    #[test]
    fn weighted_score_falls_back_when_no_match() {
        let logprobs = vec![TokenLogprobs {
            token: "7".to_owned(),
            logprob: -0.1,
            top_logprobs: vec![lp("7", -0.1)],
        }];
        assert_eq!(calculate_weighted_summed_score(8.0, &logprobs), 8.0);
    }

    #[test]
    fn weighted_score_filters_low_probability_tokens() {
        // 9 has logprob below ln(0.01) ~ -4.6, so it is filtered out.
        let logprobs = vec![TokenLogprobs {
            token: "8".to_owned(),
            logprob: -0.1,
            top_logprobs: vec![lp("8", -0.1), lp("9", -5.0)],
        }];
        let score = calculate_weighted_summed_score(8.0, &logprobs);
        assert!((score - 8.0).abs() < 0.01, "got {score}");
    }
}

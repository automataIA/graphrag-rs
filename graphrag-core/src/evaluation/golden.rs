//! Strict evaluation primitives for the versioned GraphRAG benchmark.
//!
//! Missing rows and provider usage are never silently inferred.  The runner
//! writes raw records and this module validates them before calculating metrics.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// A stable benchmark question and its source-level gold annotations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BenchmarkQuestion {
    /// Stable identifier.
    pub id: String,
    /// User question.
    pub query: String,
    /// Source/chunk IDs supporting the answer.
    pub relevant_source_ids: Vec<String>,
    /// Terms expected in an answer, compared case-insensitively.
    pub expected_answer_terms: Vec<String>,
    /// Whether the corpus contains an answer.
    pub answerable: bool,
    /// Balanced category such as `factual` or `relational`.
    pub category: String,
    /// `it` or `en`.
    pub language: String,
}

/// One raw retrieval row in rank order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalRecord {
    /// Question ID.
    pub question_id: String,
    /// Retrieved source IDs in rank order.
    pub source_ids: Vec<String>,
    /// Scores corresponding one-to-one with `source_ids`.
    pub scores: Vec<f64>,
    /// Retrieval latency in milliseconds.
    pub latency_ms: f64,
}

/// Provider-measured token usage.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderTokenUsage {
    /// Prompt/input tokens.
    pub prompt_tokens: u64,
    /// Completion/output tokens.
    pub completion_tokens: u64,
    /// Provider-reported total.
    pub total_tokens: u64,
}

/// One raw generated answer row.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnswerRecord {
    /// Question ID.
    pub question_id: String,
    /// Provider answer text. Empty means abstention.
    pub answer: String,
    /// Source context supplied to the answer adapter; these are not provider citations.
    #[serde(default)]
    pub context_source_ids: Vec<String>,
    /// Actual provider usage, if reported.
    pub usage: Option<ProviderTokenUsage>,
    /// Generation latency in milliseconds.
    pub latency_ms: f64,
}

/// Complete input to the strict evaluator.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkEvaluationInput {
    /// Gold questions.
    pub questions: Vec<BenchmarkQuestion>,
    /// Raw retrieval rows.
    pub retrieval: Vec<RetrievalRecord>,
    /// Raw answer rows.
    pub answers: Vec<AnswerRecord>,
    /// Common top-k cutoff.
    pub top_k: usize,
}

/// Retrieval quality for one question.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetrievalMetrics {
    /// Recall over relevant source IDs at k. `None` for unanswerable queries.
    pub recall_at_k: Option<f64>,
    /// Reciprocal rank of the first relevant source.
    pub reciprocal_rank: Option<f64>,
    /// Precision of retrieved source IDs against the relevant set.
    pub retrieved_source_precision: f64,
    /// Raw retrieval latency in milliseconds.
    pub latency_ms: f64,
}

/// Answer quality for one question, separate from retrieval quality.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnswerMetrics {
    /// Fraction of expected terms present in the answer.
    pub expected_term_coverage: Option<f64>,
    /// Correct abstention for an unanswerable question.
    pub abstained: Option<bool>,
    /// Provider-measured usage, if reported.
    pub usage: Option<ProviderTokenUsage>,
    /// Raw answer latency in milliseconds.
    pub latency_ms: f64,
}

/// Per-question evaluation output.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkQuestionScore {
    /// Stable question ID.
    pub question_id: String,
    /// Retrieval quality metrics.
    pub retrieval: RetrievalMetrics,
    /// Answer quality metrics.
    pub answer: AnswerMetrics,
}

/// Aggregate benchmark metrics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkAggregate {
    /// Mean Recall@k over answerable questions.
    pub mean_recall_at_k: Option<f64>,
    /// Mean MRR over answerable questions.
    pub mean_reciprocal_rank: Option<f64>,
    /// Mean retrieved-source precision.
    pub mean_retrieved_source_precision: f64,
    /// Mean expected-term coverage.
    pub mean_expected_term_coverage: Option<f64>,
    /// Correct abstention rate over unanswerable questions.
    pub abstention_rate: Option<f64>,
    /// Mean retrieval latency.
    pub mean_retrieval_latency_ms: f64,
    /// Mean answer latency.
    pub mean_answer_latency_ms: f64,
    /// Number of answers with measured provider usage.
    pub measured_usage_count: usize,
    /// Number of answer rows without provider-reported usage.
    pub missing_usage_count: usize,
    /// Sum of measured prompt tokens, excluding rows without usage.
    pub measured_prompt_tokens: u64,
    /// Sum of measured completion tokens, excluding rows without usage.
    pub measured_completion_tokens: u64,
    /// Sum of measured total tokens, excluding rows without usage.
    pub measured_total_tokens: u64,
}

/// Strict evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BenchmarkEvaluationReport {
    /// Number of evaluated questions.
    pub question_count: usize,
    /// Common top-k cutoff.
    pub top_k: usize,
    /// Per-question details.
    pub questions: Vec<BenchmarkQuestionScore>,
    /// Aggregate details.
    pub aggregate: BenchmarkAggregate,
}

/// Validation/evaluation failure.
#[derive(Debug, Clone, thiserror::Error, PartialEq, Eq)]
pub enum EvaluationError {
    /// A required collection is empty.
    #[error("{0} is empty")]
    Empty(&'static str),
    /// IDs or row lengths do not align.
    #[error("{0}")]
    Alignment(String),
    /// A numeric field is invalid.
    #[error("invalid {field}: {reason}")]
    Numeric {
        /// Name of the invalid field.
        field: String,
        /// Why the value is invalid.
        reason: String,
    },
    /// Provider usage is internally inconsistent.
    #[error("inconsistent token usage for {0}")]
    TokenUsage(String),
}

/// Validate and evaluate a complete benchmark input.
pub fn evaluate_benchmark(
    input: &BenchmarkEvaluationInput,
) -> std::result::Result<BenchmarkEvaluationReport, EvaluationError> {
    if input.questions.is_empty() {
        return Err(EvaluationError::Empty("questions"));
    }
    if input.retrieval.is_empty() {
        return Err(EvaluationError::Empty("retrieval"));
    }
    if input.answers.is_empty() {
        return Err(EvaluationError::Empty("answers"));
    }
    if input.top_k == 0 {
        return Err(EvaluationError::Numeric {
            field: "top_k".to_string(),
            reason: "must be greater than zero".to_string(),
        });
    }
    let question_ids = unique_ids(input.questions.iter().map(|q| q.id.as_str()), "question")?;
    let retrieval_ids = unique_ids(
        input.retrieval.iter().map(|r| r.question_id.as_str()),
        "retrieval",
    )?;
    let answer_ids = unique_ids(
        input.answers.iter().map(|a| a.question_id.as_str()),
        "answer",
    )?;
    if question_ids != retrieval_ids || question_ids != answer_ids {
        return Err(EvaluationError::Alignment(
            "questions, retrieval, and answers must contain exactly the same IDs".to_string(),
        ));
    }
    let retrieval_by_id: HashMap<_, _> = input
        .retrieval
        .iter()
        .map(|row| (row.question_id.as_str(), row))
        .collect();
    let answers_by_id: HashMap<_, _> = input
        .answers
        .iter()
        .map(|row| (row.question_id.as_str(), row))
        .collect();

    let mut questions = Vec::with_capacity(input.questions.len());
    for question in &input.questions {
        let retrieval = retrieval_by_id[question.id.as_str()];
        let answer = answers_by_id[question.id.as_str()];
        validate_question(question)?;
        validate_retrieval(retrieval)?;
        validate_answer(answer)?;
        questions.push(BenchmarkQuestionScore {
            question_id: question.id.clone(),
            retrieval: score_retrieval(question, retrieval, input.top_k),
            answer: score_answer(question, answer),
        });
    }
    let aggregate = aggregate_scores(&questions, &input.questions)?;
    validate_aggregate(&aggregate)?;
    Ok(BenchmarkEvaluationReport {
        question_count: questions.len(),
        top_k: input.top_k,
        questions,
        aggregate,
    })
}

fn unique_ids<'a>(
    ids: impl Iterator<Item = &'a str>,
    kind: &str,
) -> std::result::Result<HashSet<String>, EvaluationError> {
    let mut result = HashSet::new();
    for id in ids {
        if id.trim().is_empty() {
            return Err(EvaluationError::Alignment(format!("{kind} ID is empty")));
        }
        if !result.insert(id.to_string()) {
            return Err(EvaluationError::Alignment(format!(
                "duplicate {kind} ID: {id}"
            )));
        }
    }
    Ok(result)
}

fn validate_question(question: &BenchmarkQuestion) -> std::result::Result<(), EvaluationError> {
    if question.id.trim().is_empty() || question.query.trim().is_empty() {
        return Err(EvaluationError::Alignment(format!(
            "question {} must have a non-empty ID and query",
            question.id
        )));
    }
    if !matches!(question.language.as_str(), "it" | "en") {
        return Err(EvaluationError::Alignment(format!(
            "question {} has unsupported language {}",
            question.id, question.language
        )));
    }
    if !matches!(
        question.category.as_str(),
        "factual" | "relational" | "cross_document" | "unanswerable"
    ) {
        return Err(EvaluationError::Alignment(format!(
            "question {} has unsupported category {}",
            question.id, question.category
        )));
    }
    unique_ids(
        question.relevant_source_ids.iter().map(String::as_str),
        "gold source",
    )?;
    if question.answerable && question.relevant_source_ids.is_empty() {
        return Err(EvaluationError::Alignment(format!(
            "answerable question {} has no relevant source IDs",
            question.id
        )));
    }
    if question.answerable && question.expected_answer_terms.is_empty() {
        return Err(EvaluationError::Alignment(format!(
            "answerable question {} has no expected answer terms",
            question.id
        )));
    }
    if !question.answerable
        && (!question.relevant_source_ids.is_empty() || !question.expected_answer_terms.is_empty())
    {
        return Err(EvaluationError::Alignment(format!(
            "unanswerable question {} must have empty gold sources and terms",
            question.id
        )));
    }
    Ok(())
}

fn validate_retrieval(row: &RetrievalRecord) -> std::result::Result<(), EvaluationError> {
    if row.source_ids.len() != row.scores.len() {
        return Err(EvaluationError::Alignment(format!(
            "retrieval row {} has mismatched source_ids and scores",
            row.question_id
        )));
    }
    unique_ids(
        row.source_ids.iter().map(String::as_str),
        "retrieved source",
    )?;
    validate_finite_nonnegative(
        row.latency_ms,
        format!("retrieval latency for {}", row.question_id),
    )?;
    for score in &row.scores {
        if !score.is_finite() {
            return Err(EvaluationError::Numeric {
                field: format!("retrieval score for {}", row.question_id),
                reason: "must be finite".to_string(),
            });
        }
    }
    Ok(())
}

fn validate_answer(row: &AnswerRecord) -> std::result::Result<(), EvaluationError> {
    validate_finite_nonnegative(
        row.latency_ms,
        format!("answer latency for {}", row.question_id),
    )?;
    unique_ids(
        row.context_source_ids.iter().map(String::as_str),
        "context source",
    )?;
    if let Some(usage) = &row.usage {
        let expected_total = usage
            .prompt_tokens
            .checked_add(usage.completion_tokens)
            .ok_or_else(|| EvaluationError::TokenUsage(row.question_id.clone()))?;
        if usage.total_tokens != expected_total {
            return Err(EvaluationError::TokenUsage(row.question_id.clone()));
        }
    }
    Ok(())
}

fn validate_finite_nonnegative(
    value: f64,
    field: String,
) -> std::result::Result<(), EvaluationError> {
    if !value.is_finite() || value < 0.0 {
        return Err(EvaluationError::Numeric {
            field,
            reason: "must be finite and non-negative".to_string(),
        });
    }
    Ok(())
}

fn score_retrieval(
    question: &BenchmarkQuestion,
    row: &RetrievalRecord,
    top_k: usize,
) -> RetrievalMetrics {
    let relevant: HashSet<&str> = question
        .relevant_source_ids
        .iter()
        .map(String::as_str)
        .collect();
    let retrieved: Vec<&str> = row
        .source_ids
        .iter()
        .take(top_k)
        .map(String::as_str)
        .collect();
    let hits = retrieved
        .iter()
        .filter(|id| relevant.contains(**id))
        .count();
    let first_rank = retrieved
        .iter()
        .position(|id| relevant.contains(*id))
        .map(|rank| rank + 1);
    RetrievalMetrics {
        recall_at_k: (!relevant.is_empty()).then(|| hits as f64 / relevant.len() as f64),
        reciprocal_rank: first_rank.map(|rank| 1.0 / rank as f64),
        retrieved_source_precision: if retrieved.is_empty() {
            0.0
        } else {
            hits as f64 / retrieved.len() as f64
        },
        latency_ms: row.latency_ms,
    }
}

fn score_answer(question: &BenchmarkQuestion, row: &AnswerRecord) -> AnswerMetrics {
    let answer_lower = row.answer.to_lowercase();
    let expected_term_coverage = (!question.expected_answer_terms.is_empty()).then(|| {
        question
            .expected_answer_terms
            .iter()
            .filter(|term| answer_lower.contains(&term.to_lowercase()))
            .count() as f64
            / question.expected_answer_terms.len() as f64
    });
    let abstained = (!question.answerable).then(|| is_abstention(&row.answer));
    AnswerMetrics {
        expected_term_coverage,
        abstained,
        usage: row.usage.clone(),
        latency_ms: row.latency_ms,
    }
}

fn is_abstention(answer: &str) -> bool {
    let normalized = answer.trim().to_lowercase();
    normalized.is_empty()
        || [
            "i don't know",
            "i do not know",
            "non lo so",
            "non posso rispondere",
            "insufficient evidence",
            "evidenza insufficiente",
        ]
        .iter()
        .any(|marker| normalized.contains(marker))
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let values: Vec<_> = values.collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn aggregate_scores(
    scores: &[BenchmarkQuestionScore],
    questions: &[BenchmarkQuestion],
) -> std::result::Result<BenchmarkAggregate, EvaluationError> {
    let answerable: HashSet<&str> = questions
        .iter()
        .filter(|q| q.answerable)
        .map(|q| q.id.as_str())
        .collect();
    let unanswerable: HashSet<&str> = questions
        .iter()
        .filter(|q| !q.answerable)
        .map(|q| q.id.as_str())
        .collect();
    let answerable_scores = scores
        .iter()
        .filter(|s| answerable.contains(s.question_id.as_str()));
    let mean_recall_at_k = mean(
        answerable_scores
            .clone()
            .filter_map(|s| s.retrieval.recall_at_k),
    );
    // A relevant source missed by the cutoff contributes zero to MRR.  Dropping
    // misses would measure only successful queries and inflate the aggregate.
    let mean_reciprocal_rank =
        mean(answerable_scores.map(|s| s.retrieval.reciprocal_rank.unwrap_or(0.0)));
    let abstention_rate = mean(
        scores
            .iter()
            .filter(|s| unanswerable.contains(s.question_id.as_str()))
            .filter_map(|s| s.answer.abstained.map(|value| value as u8 as f64)),
    );
    let answerable_precision = scores
        .iter()
        .filter(|s| answerable.contains(s.question_id.as_str()))
        .map(|s| s.retrieval.retrieved_source_precision)
        .collect::<Vec<_>>();
    let mut measured_prompt_tokens = 0_u64;
    let mut measured_completion_tokens = 0_u64;
    let mut measured_total_tokens = 0_u64;
    for score in scores {
        if let Some(usage) = &score.answer.usage {
            measured_prompt_tokens = measured_prompt_tokens
                .checked_add(usage.prompt_tokens)
                .ok_or_else(|| EvaluationError::TokenUsage(score.question_id.clone()))?;
            measured_completion_tokens = measured_completion_tokens
                .checked_add(usage.completion_tokens)
                .ok_or_else(|| EvaluationError::TokenUsage(score.question_id.clone()))?;
            measured_total_tokens = measured_total_tokens
                .checked_add(usage.total_tokens)
                .ok_or_else(|| EvaluationError::TokenUsage(score.question_id.clone()))?;
        }
    }
    Ok(BenchmarkAggregate {
        mean_recall_at_k,
        mean_reciprocal_rank,
        mean_retrieved_source_precision: answerable_precision.iter().sum::<f64>()
            / answerable_precision.len() as f64,
        mean_expected_term_coverage: mean(
            scores
                .iter()
                .filter_map(|s| s.answer.expected_term_coverage),
        ),
        abstention_rate,
        mean_retrieval_latency_ms: scores.iter().map(|s| s.retrieval.latency_ms).sum::<f64>()
            / scores.len() as f64,
        mean_answer_latency_ms: scores.iter().map(|s| s.answer.latency_ms).sum::<f64>()
            / scores.len() as f64,
        measured_usage_count: scores.iter().filter(|s| s.answer.usage.is_some()).count(),
        missing_usage_count: scores.iter().filter(|s| s.answer.usage.is_none()).count(),
        measured_prompt_tokens,
        measured_completion_tokens,
        measured_total_tokens,
    })
}

fn validate_aggregate(aggregate: &BenchmarkAggregate) -> std::result::Result<(), EvaluationError> {
    let values = [
        aggregate.mean_retrieved_source_precision,
        aggregate.mean_retrieval_latency_ms,
        aggregate.mean_answer_latency_ms,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(EvaluationError::Numeric {
            field: "aggregate metric".to_string(),
            reason: "must be finite".to_string(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input() -> BenchmarkEvaluationInput {
        BenchmarkEvaluationInput {
            questions: vec![
                BenchmarkQuestion {
                    id: "q1".into(),
                    query: "Who?".into(),
                    relevant_source_ids: vec!["s1".into(), "s2".into()],
                    expected_answer_terms: vec!["Ada".into()],
                    answerable: true,
                    category: "factual".into(),
                    language: "en".into(),
                },
                BenchmarkQuestion {
                    id: "q2".into(),
                    query: "Unknown?".into(),
                    relevant_source_ids: vec![],
                    expected_answer_terms: vec![],
                    answerable: false,
                    category: "unanswerable".into(),
                    language: "it".into(),
                },
            ],
            retrieval: vec![
                RetrievalRecord {
                    question_id: "q1".into(),
                    source_ids: vec!["s3".into(), "s1".into()],
                    scores: vec![0.9, 0.8],
                    latency_ms: 2.0,
                },
                RetrievalRecord {
                    question_id: "q2".into(),
                    source_ids: vec!["s3".into()],
                    scores: vec![0.1],
                    latency_ms: 3.0,
                },
            ],
            answers: vec![
                AnswerRecord {
                    question_id: "q1".into(),
                    answer: "Ada Lovelace".into(),
                    context_source_ids: vec!["s1".into()],
                    usage: Some(ProviderTokenUsage {
                        prompt_tokens: 10,
                        completion_tokens: 2,
                        total_tokens: 12,
                    }),
                    latency_ms: 4.0,
                },
                AnswerRecord {
                    question_id: "q2".into(),
                    answer: "I don't know".into(),
                    context_source_ids: vec![],
                    usage: None,
                    latency_ms: 1.0,
                },
            ],
            top_k: 2,
        }
    }

    #[test]
    fn computes_source_recall_mrr_and_abstention() {
        let report = evaluate_benchmark(&input()).unwrap();
        assert_eq!(report.questions[0].retrieval.recall_at_k, Some(0.5));
        assert_eq!(report.questions[0].retrieval.reciprocal_rank, Some(0.5));
        assert_eq!(
            report.questions[0].retrieval.retrieved_source_precision,
            0.5
        );
        assert_eq!(report.questions[1].answer.abstained, Some(true));
        assert_eq!(report.aggregate.measured_usage_count, 1);
        assert_eq!(report.aggregate.missing_usage_count, 1);
        assert_eq!(report.aggregate.measured_prompt_tokens, 10);
        assert_eq!(report.aggregate.measured_completion_tokens, 2);
        assert_eq!(report.aggregate.measured_total_tokens, 12);
    }

    #[test]
    fn rejects_empty_input() {
        let mut value = input();
        value.questions.clear();
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Empty("questions"))
        ));
        let mut value = input();
        value.retrieval.clear();
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Empty("retrieval"))
        ));
        let mut value = input();
        value.answers.clear();
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Empty("answers"))
        ));
    }

    #[test]
    fn rejects_alignment_and_duplicate_ids() {
        let mut value = input();
        value.retrieval.pop();
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Alignment(_))
        ));
        let mut value = input();
        value.retrieval[0].source_ids.push("s1".into());
        value.retrieval[0].scores.push(0.7);
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Alignment(_))
        ));
    }

    #[test]
    fn rejects_nonfinite_metrics_and_zero_top_k() {
        let mut value = input();
        value.top_k = 0;
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::Numeric { .. })
        ));
        let mut value = input();
        value.retrieval[0].scores[0] = f64::NAN;
        assert!(evaluate_benchmark(&value).is_err());
    }

    #[test]
    fn rejects_inconsistent_provider_tokens() {
        let mut value = input();
        value.answers[0].usage.as_mut().unwrap().total_tokens = 99;
        assert!(matches!(
            evaluate_benchmark(&value),
            Err(EvaluationError::TokenUsage(_))
        ));
    }

    #[test]
    fn rejects_aggregate_token_overflow() {
        let retrieval = RetrievalMetrics {
            recall_at_k: Some(1.0),
            reciprocal_rank: Some(1.0),
            retrieved_source_precision: 1.0,
            latency_ms: 1.0,
        };
        let scores = vec![
            BenchmarkQuestionScore {
                question_id: "q1".into(),
                retrieval: retrieval.clone(),
                answer: AnswerMetrics {
                    expected_term_coverage: Some(1.0),
                    abstained: None,
                    usage: Some(ProviderTokenUsage {
                        prompt_tokens: u64::MAX,
                        completion_tokens: 0,
                        total_tokens: u64::MAX,
                    }),
                    latency_ms: 1.0,
                },
            },
            BenchmarkQuestionScore {
                question_id: "q2".into(),
                retrieval,
                answer: AnswerMetrics {
                    expected_term_coverage: None,
                    abstained: Some(true),
                    usage: Some(ProviderTokenUsage {
                        prompt_tokens: 1,
                        completion_tokens: 0,
                        total_tokens: 1,
                    }),
                    latency_ms: 1.0,
                },
            },
        ];
        assert!(matches!(
            aggregate_scores(&scores, &input().questions),
            Err(EvaluationError::TokenUsage(_))
        ));
    }
}

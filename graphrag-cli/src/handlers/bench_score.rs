//! Pure scoring logic for `bench --gold`: keyword-group recall.

use serde::{Deserialize, Serialize};

/// Gold file: a list of questions with expected keyword groups.
#[derive(Debug, Deserialize)]
pub struct GoldFile {
    pub questions: Vec<GoldQuestion>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GoldQuestion {
    pub id: String,
    pub question: String,
    /// All groups must match; within a group any alias is enough.
    pub expected_keywords: Vec<Vec<String>>,
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Score {
    /// Fraction of keyword groups matched (1.0 when there are no groups).
    pub score: f32,
    pub passed: bool,
    pub matched_groups: usize,
    pub total_groups: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub total: usize,
    pub passed: usize,
    pub mean_score: f32,
    pub mean_latency_ms: f64,
}

/// Lowercase, strip Italian/Latin accents, turn punctuation into single spaces.
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars().flat_map(char::to_lowercase) {
        let c = match c {
            'à' | 'á' | 'â' | 'ä' => 'a',
            'è' | 'é' | 'ê' | 'ë' => 'e',
            'ì' | 'í' | 'î' | 'ï' => 'i',
            'ò' | 'ó' | 'ô' | 'ö' => 'o',
            'ù' | 'ú' | 'û' | 'ü' => 'u',
            other => other,
        };
        if c.is_alphanumeric() {
            out.push(c);
        } else if !out.ends_with(' ') {
            out.push(' ');
        }
    }
    format!(" {} ", out.trim())
}

/// Score `answer` against keyword groups. A keyword matches when it occurs on
/// the normalized text starting at a word boundary (a keyword may be a word
/// prefix, e.g. "ciuchin" matches "ciuchini"; mid-word hits do not count).
pub fn score_answer(answer: &str, groups: &[Vec<String>]) -> Score {
    let haystack = normalize(answer);
    let matched = groups
        .iter()
        .filter(|g| {
            g.iter().any(|kw| {
                let kw = normalize(kw);
                let kw = kw.trim();
                !kw.is_empty() && haystack.contains(&format!(" {kw}"))
            })
        })
        .count();
    let total = groups.len();
    let score = if total == 0 {
        1.0
    } else {
        matched as f32 / total as f32
    };
    Score {
        score,
        passed: matched == total,
        matched_groups: matched,
        total_groups: total,
    }
}

pub fn summarize(scores: &[Score], latencies_ms: &[u128]) -> Summary {
    let n = scores.len();
    let mean_score = if n == 0 {
        0.0
    } else {
        scores.iter().map(|s| s.score).sum::<f32>() / n as f32
    };
    let mean_latency_ms = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<u128>() as f64 / latencies_ms.len() as f64
    };
    Summary {
        total: n,
        passed: scores.iter().filter(|s| s.passed).count(),
        mean_score,
        mean_latency_ms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(groups: &[&[&str]]) -> Vec<Vec<String>> {
        groups
            .iter()
            .map(|a| a.iter().map(|s| s.to_string()).collect())
            .collect()
    }

    #[test]
    fn all_groups_match() {
        let s = score_answer(
            "Geppetto intaglia un burattino",
            &g(&[&["Geppetto"], &["burattino", "marionetta"]]),
        );
        assert!(s.passed);
        assert_eq!(s.score, 1.0);
        assert_eq!((s.matched_groups, s.total_groups), (2, 2));
    }

    #[test]
    fn partial_match_is_fractional_and_not_passed() {
        let s = score_answer("Geppetto", &g(&[&["Geppetto"], &["burattino"]]));
        assert!(!s.passed);
        assert_eq!(s.score, 0.5);
    }

    #[test]
    fn case_and_accent_insensitive() {
        let s = score_answer(
            "PERCHÈ il Paese dei Balocchi",
            &g(&[&["perche"], &["paese dei balocchi"]]),
        );
        assert!(s.passed);
        let s = score_answer("perché", &g(&[&["PERCHÈ"]]));
        assert!(s.passed);
    }

    #[test]
    fn hyphen_and_punctuation_are_spaces() {
        let s = score_answer("nel ventre del Pesce-cane.", &g(&[&["pesce cane"]]));
        assert!(s.passed);
        let s = score_answer("il pesce cane", &g(&[&["Pesce-cane"]]));
        assert!(s.passed);
    }

    #[test]
    fn no_match_in_middle_of_word() {
        let s = score_answer("un gatto", &g(&[&["atto"]]));
        assert!(!s.passed);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn prefix_of_word_matches() {
        let s = score_answer("i ciuchini rebuzzano", &g(&[&["ciuchin"]]));
        assert!(s.passed);
    }

    #[test]
    fn empty_groups_and_empty_keywords() {
        assert!(score_answer("x", &[]).passed);
        let s = score_answer("x", &g(&[&[""]]));
        assert!(!s.passed);
    }

    #[test]
    fn summary_aggregates() {
        let scores = vec![
            score_answer("a", &g(&[&["a"]])),
            score_answer("a", &g(&[&["a"], &["b"]])),
        ];
        let s = summarize(&scores, &[100, 300]);
        assert_eq!((s.total, s.passed), (2, 1));
        assert!((s.mean_score - 0.75).abs() < 1e-6);
        assert!((s.mean_latency_ms - 200.0).abs() < 1e-9);
        let e = summarize(&[], &[]);
        assert_eq!((e.total, e.mean_score), (0, 0.0));
    }

    #[test]
    fn gold_file_parses() {
        let j =
            r#"{"questions":[{"id":"q1","question":"?","expected_keywords":[["a","b"],["c"]]}]}"#;
        let gf: GoldFile = serde_json::from_str(j).unwrap();
        assert_eq!(gf.questions[0].expected_keywords.len(), 2);
    }
}

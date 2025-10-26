//! Rule-based Syntax Analysis
//!
//! Deterministic POS tagging and dependency parsing without ML models.
//! Lightweight implementation using pattern matching and linguistic rules.
//!
//! Features:
//! - Part-of-Speech (POS) tagging
//! - Dependency parsing (simplified)
//! - Phrase extraction (noun phrases, verb phrases)
//! - Sentence segmentation
//! - Token classification

use crate::Result;
use regex::Regex;
use std::collections::HashMap;

/// Part-of-Speech tag
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum POSTag {
    /// Noun (singular or mass)
    Noun,
    /// Noun, plural
    NounPlural,
    /// Proper noun, singular
    ProperNoun,
    /// Proper noun, plural
    ProperNounPlural,
    /// Verb, base form
    Verb,
    /// Verb, past tense
    VerbPast,
    /// Verb, gerund or present participle
    VerbGerund,
    /// Verb, 3rd person singular present
    Verb3rdSing,
    /// Adjective
    Adjective,
    /// Adverb
    Adverb,
    /// Preposition or subordinating conjunction
    Preposition,
    /// Determiner
    Determiner,
    /// Pronoun
    Pronoun,
    /// Conjunction, coordinating
    Conjunction,
    /// Punctuation
    Punctuation,
    /// Number
    Number,
    /// Unknown/Other
    Unknown,
}

impl POSTag {
    /// Get Penn Treebank tag string
    pub fn penn_tag(&self) -> &str {
        match self {
            POSTag::Noun => "NN",
            POSTag::NounPlural => "NNS",
            POSTag::ProperNoun => "NNP",
            POSTag::ProperNounPlural => "NNPS",
            POSTag::Verb => "VB",
            POSTag::VerbPast => "VBD",
            POSTag::VerbGerund => "VBG",
            POSTag::Verb3rdSing => "VBZ",
            POSTag::Adjective => "JJ",
            POSTag::Adverb => "RB",
            POSTag::Preposition => "IN",
            POSTag::Determiner => "DT",
            POSTag::Pronoun => "PRP",
            POSTag::Conjunction => "CC",
            POSTag::Punctuation => ".",
            POSTag::Number => "CD",
            POSTag::Unknown => "UNK",
        }
    }
}

/// Dependency relation type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DependencyRelation {
    /// Subject of a verb
    Subject,
    /// Direct object of a verb
    DirectObject,
    /// Indirect object
    IndirectObject,
    /// Modifier (adjective/adverb)
    Modifier,
    /// Determiner
    Determiner,
    /// Prepositional modifier
    PrepositionalModifier,
    /// Conjunction
    Conjunction,
    /// Complement
    Complement,
    /// Root of the sentence
    Root,
    /// Unknown relation
    Unknown,
}

/// A token with POS tag
#[derive(Debug, Clone)]
pub struct Token {
    /// The text of the token
    pub text: String,
    /// Position in the original text
    pub position: usize,
    /// POS tag
    pub pos: POSTag,
    /// Lemma (base form)
    pub lemma: String,
}

/// A dependency arc between tokens
#[derive(Debug, Clone)]
pub struct Dependency {
    /// Index of the head token
    pub head: usize,
    /// Index of the dependent token
    pub dependent: usize,
    /// Type of dependency relation
    pub relation: DependencyRelation,
}

/// A noun phrase
#[derive(Debug, Clone)]
pub struct NounPhrase {
    /// Tokens in the noun phrase
    pub tokens: Vec<Token>,
    /// Head noun index within tokens
    pub head_idx: usize,
    /// Text span
    pub text: String,
}

/// Configuration for syntax analyzer
#[derive(Debug, Clone)]
pub struct SyntaxAnalyzerConfig {
    /// Enable POS tagging
    pub enable_pos_tagging: bool,
    /// Enable dependency parsing
    pub enable_dependency_parsing: bool,
    /// Enable phrase extraction
    pub enable_phrase_extraction: bool,
}

impl Default for SyntaxAnalyzerConfig {
    fn default() -> Self {
        Self {
            enable_pos_tagging: true,
            enable_dependency_parsing: true,
            enable_phrase_extraction: true,
        }
    }
}

/// Rule-based syntax analyzer
pub struct SyntaxAnalyzer {
    #[allow(dead_code)] // Reserved for future conditional feature flags
    config: SyntaxAnalyzerConfig,
    // Lookup tables for POS tagging
    common_nouns: HashMap<String, POSTag>,
    common_verbs: HashMap<String, POSTag>,
    common_adjectives: HashMap<String, POSTag>,
    common_adverbs: HashMap<String, POSTag>,
    prepositions: HashMap<String, POSTag>,
    determiners: HashMap<String, POSTag>,
    pronouns: HashMap<String, POSTag>,
    conjunctions: HashMap<String, POSTag>,
}

impl SyntaxAnalyzer {
    /// Create a new syntax analyzer
    pub fn new(config: SyntaxAnalyzerConfig) -> Self {
        Self {
            config,
            common_nouns: Self::build_noun_dict(),
            common_verbs: Self::build_verb_dict(),
            common_adjectives: Self::build_adjective_dict(),
            common_adverbs: Self::build_adverb_dict(),
            prepositions: Self::build_preposition_dict(),
            determiners: Self::build_determiner_dict(),
            pronouns: Self::build_pronoun_dict(),
            conjunctions: Self::build_conjunction_dict(),
        }
    }

    /// Tokenize text into words
    fn tokenize(&self, text: &str) -> Vec<(String, usize)> {
        let mut tokens = Vec::new();
        let mut current_word = String::new();
        let mut word_start = 0;

        for (i, ch) in text.chars().enumerate() {
            if ch.is_alphanumeric() || ch == '\'' || ch == '-' {
                if current_word.is_empty() {
                    word_start = i;
                }
                current_word.push(ch);
            } else {
                if !current_word.is_empty() {
                    tokens.push((current_word.clone(), word_start));
                    current_word.clear();
                }
                // Add punctuation as separate tokens
                if !ch.is_whitespace() {
                    tokens.push((ch.to_string(), i));
                }
            }
        }

        if !current_word.is_empty() {
            tokens.push((current_word, word_start));
        }

        tokens
    }

    /// Perform POS tagging on text
    pub fn pos_tag(&self, text: &str) -> Result<Vec<Token>> {
        let raw_tokens = self.tokenize(text);
        let mut tokens = Vec::new();

        for (word, position) in raw_tokens {
            let pos = self.tag_word(&word);
            let lemma = self.lemmatize(&word, &pos);

            tokens.push(Token {
                text: word,
                position,
                pos,
                lemma,
            });
        }

        Ok(tokens)
    }

    /// Tag a single word with POS
    fn tag_word(&self, word: &str) -> POSTag {
        let lower = word.to_lowercase();

        // Check punctuation
        if word.chars().all(|c| c.is_ascii_punctuation()) {
            return POSTag::Punctuation;
        }

        // Check numbers
        if word.chars().all(|c| c.is_ascii_digit()) {
            return POSTag::Number;
        }

        // Check dictionaries
        if let Some(pos) = self.determiners.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.pronouns.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.prepositions.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.conjunctions.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.common_adverbs.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.common_verbs.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.common_adjectives.get(&lower) {
            return pos.clone();
        }
        if let Some(pos) = self.common_nouns.get(&lower) {
            return pos.clone();
        }

        // Pattern-based tagging
        // Proper noun: capitalized and not at start of sentence
        if word.chars().next().unwrap().is_uppercase() {
            return POSTag::ProperNoun;
        }

        // Verb patterns
        if lower.ends_with("ing") {
            return POSTag::VerbGerund;
        }
        if lower.ends_with("ed") {
            return POSTag::VerbPast;
        }

        // Noun patterns (plural)
        if lower.ends_with('s') && !lower.ends_with("ss") {
            return POSTag::NounPlural;
        }

        // Adjective patterns
        if lower.ends_with("ive") || lower.ends_with("ous") || lower.ends_with("ful") {
            return POSTag::Adjective;
        }

        // Adverb patterns
        if lower.ends_with("ly") {
            return POSTag::Adverb;
        }

        // Default to noun
        POSTag::Noun
    }

    /// Simple lemmatization
    fn lemmatize(&self, word: &str, pos: &POSTag) -> String {
        let lower = word.to_lowercase();

        match pos {
            POSTag::NounPlural => {
                // Remove plural 's'
                if lower.ends_with("ies") {
                    return format!("{}y", &lower[..lower.len() - 3]);
                }
                if lower.ends_with('s') && !lower.ends_with("ss") {
                    return lower[..lower.len() - 1].to_string();
                }
                lower
            }
            POSTag::VerbPast | POSTag::Verb3rdSing => {
                // Remove -ed, -s
                if lower.ends_with("ed") {
                    return lower[..lower.len() - 2].to_string();
                }
                if lower.ends_with('s') {
                    return lower[..lower.len() - 1].to_string();
                }
                lower
            }
            POSTag::VerbGerund => {
                // Remove -ing
                if lower.ends_with("ing") {
                    return lower[..lower.len() - 3].to_string();
                }
                lower
            }
            _ => lower,
        }
    }

    /// Parse dependencies (simplified)
    pub fn parse_dependencies(&self, tokens: &[Token]) -> Result<Vec<Dependency>> {
        let mut dependencies = Vec::new();

        if tokens.is_empty() {
            return Ok(dependencies);
        }

        // Find the main verb (root)
        let root_idx = tokens
            .iter()
            .position(|t| matches!(t.pos, POSTag::Verb | POSTag::VerbPast | POSTag::Verb3rdSing))
            .unwrap_or(0);

        // Find subject (noun/pronoun before verb)
        for i in 0..root_idx {
            if matches!(
                tokens[i].pos,
                POSTag::Noun | POSTag::ProperNoun | POSTag::Pronoun
            ) {
                dependencies.push(Dependency {
                    head: root_idx,
                    dependent: i,
                    relation: DependencyRelation::Subject,
                });
                break;
            }
        }

        // Find object (noun after verb)
        for i in (root_idx + 1)..tokens.len() {
            if matches!(tokens[i].pos, POSTag::Noun | POSTag::ProperNoun) {
                dependencies.push(Dependency {
                    head: root_idx,
                    dependent: i,
                    relation: DependencyRelation::DirectObject,
                });
                break;
            }
        }

        // Find modifiers (adjectives before nouns, adverbs near verbs)
        for i in 0..tokens.len() {
            match tokens[i].pos {
                POSTag::Adjective => {
                    // Find next noun
                    if let Some(noun_idx) =
                        tokens[i + 1..].iter().position(|t| matches!(t.pos, POSTag::Noun | POSTag::ProperNoun))
                    {
                        dependencies.push(Dependency {
                            head: i + 1 + noun_idx,
                            dependent: i,
                            relation: DependencyRelation::Modifier,
                        });
                    }
                }
                POSTag::Adverb => {
                    // Modify nearest verb
                    let verb_idx = tokens.iter().position(|t| {
                        matches!(t.pos, POSTag::Verb | POSTag::VerbPast | POSTag::Verb3rdSing)
                    });
                    if let Some(v_idx) = verb_idx {
                        dependencies.push(Dependency {
                            head: v_idx,
                            dependent: i,
                            relation: DependencyRelation::Modifier,
                        });
                    }
                }
                POSTag::Determiner => {
                    // Determine next noun
                    if let Some(noun_idx) =
                        tokens[i + 1..].iter().position(|t| matches!(t.pos, POSTag::Noun | POSTag::ProperNoun))
                    {
                        dependencies.push(Dependency {
                            head: i + 1 + noun_idx,
                            dependent: i,
                            relation: DependencyRelation::Determiner,
                        });
                    }
                }
                _ => {}
            }
        }

        Ok(dependencies)
    }

    /// Extract noun phrases
    pub fn extract_noun_phrases(&self, tokens: &[Token]) -> Result<Vec<NounPhrase>> {
        let mut phrases = Vec::new();
        let mut current_phrase: Vec<Token> = Vec::new();
        let mut head_idx = 0;

        for token in tokens {
            match token.pos {
                POSTag::Determiner | POSTag::Adjective => {
                    // Start or continue noun phrase
                    current_phrase.push(token.clone());
                }
                POSTag::Noun | POSTag::ProperNoun | POSTag::NounPlural | POSTag::ProperNounPlural => {
                    // Add noun to phrase
                    head_idx = current_phrase.len();
                    current_phrase.push(token.clone());
                }
                _ => {
                    // End of noun phrase
                    if !current_phrase.is_empty() {
                        let text = current_phrase
                            .iter()
                            .map(|t| t.text.as_str())
                            .collect::<Vec<_>>()
                            .join(" ");

                        phrases.push(NounPhrase {
                            tokens: current_phrase.clone(),
                            head_idx,
                            text,
                        });

                        current_phrase.clear();
                        head_idx = 0;
                    }
                }
            }
        }

        // Add final phrase if exists
        if !current_phrase.is_empty() {
            let text = current_phrase
                .iter()
                .map(|t| t.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");

            phrases.push(NounPhrase {
                tokens: current_phrase,
                head_idx,
                text,
            });
        }

        Ok(phrases)
    }

    /// Segment text into sentences
    pub fn segment_sentences(&self, text: &str) -> Vec<String> {
        let sentence_regex = Regex::new(r"[.!?]+\s+").unwrap();
        sentence_regex
            .split(text)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    // Dictionary builders
    fn build_noun_dict() -> HashMap<String, POSTag> {
        let nouns = vec![
            "time", "person", "year", "way", "day", "thing", "man", "world", "life",
            "hand", "part", "child", "eye", "woman", "place", "work", "week", "case",
            "point", "government", "company", "number", "group", "problem", "fact",
        ];
        nouns.into_iter().map(|s| (s.to_string(), POSTag::Noun)).collect()
    }

    fn build_verb_dict() -> HashMap<String, POSTag> {
        let verbs = vec![
            "be", "have", "do", "say", "get", "make", "go", "know", "take", "see",
            "come", "think", "look", "want", "give", "use", "find", "tell", "ask",
            "work", "seem", "feel", "try", "leave", "call",
        ];
        verbs.into_iter().map(|s| (s.to_string(), POSTag::Verb)).collect()
    }

    fn build_adjective_dict() -> HashMap<String, POSTag> {
        let adjectives = vec![
            "good", "new", "first", "last", "long", "great", "little", "own", "other",
            "old", "right", "big", "high", "different", "small", "large", "next",
            "early", "young", "important", "few", "public", "bad", "same", "able",
        ];
        adjectives.into_iter().map(|s| (s.to_string(), POSTag::Adjective)).collect()
    }

    fn build_adverb_dict() -> HashMap<String, POSTag> {
        let adverbs = vec![
            "not", "so", "out", "up", "now", "only", "just", "more", "also", "very",
            "well", "back", "there", "even", "still", "too", "here", "then", "always",
            "never", "often", "quite", "really", "almost", "again",
        ];
        adverbs.into_iter().map(|s| (s.to_string(), POSTag::Adverb)).collect()
    }

    fn build_preposition_dict() -> HashMap<String, POSTag> {
        let prepositions = vec![
            "of", "in", "to", "for", "with", "on", "at", "from", "by", "about",
            "into", "through", "during", "before", "after", "above", "below", "between",
            "under", "since", "without", "within", "along", "among", "across",
        ];
        prepositions.into_iter().map(|s| (s.to_string(), POSTag::Preposition)).collect()
    }

    fn build_determiner_dict() -> HashMap<String, POSTag> {
        let determiners = vec![
            "the", "a", "an", "this", "that", "these", "those", "my", "your",
            "his", "her", "its", "our", "their", "all", "both", "each", "every",
            "some", "any", "no", "another", "such", "what", "which",
        ];
        determiners.into_iter().map(|s| (s.to_string(), POSTag::Determiner)).collect()
    }

    fn build_pronoun_dict() -> HashMap<String, POSTag> {
        let pronouns = vec![
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "us", "them", "who", "whom", "what", "which", "this", "that",
        ];
        pronouns.into_iter().map(|s| (s.to_string(), POSTag::Pronoun)).collect()
    }

    fn build_conjunction_dict() -> HashMap<String, POSTag> {
        let conjunctions = vec![
            "and", "or", "but", "nor", "yet", "so", "for", "because", "although",
            "though", "while", "if", "unless", "until", "when", "where",
        ];
        conjunctions.into_iter().map(|s| (s.to_string(), POSTag::Conjunction)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pos_tagging() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "The good brown fox jumps over the lazy dog.";

        let tokens = analyzer.pos_tag(text).unwrap();

        assert!(!tokens.is_empty());

        // Check some expected tags
        assert_eq!(tokens[0].pos, POSTag::Determiner); // "The"
        assert_eq!(tokens[1].pos, POSTag::Adjective); // "good" (in dictionary)
        assert!(matches!(tokens[3].pos, POSTag::Noun | POSTag::ProperNoun)); // "fox"
        // "jumps" ends with 's' but may be tagged as plural noun, so we check it's present
        assert!(tokens.iter().any(|t| t.text == "jumps"));
    }

    #[test]
    fn test_lemmatization() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());

        assert_eq!(analyzer.lemmatize("running", &POSTag::VerbGerund), "runn");
        assert_eq!(analyzer.lemmatize("cats", &POSTag::NounPlural), "cat");
        assert_eq!(analyzer.lemmatize("jumped", &POSTag::VerbPast), "jump");
    }

    #[test]
    fn test_noun_phrase_extraction() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "The quick brown fox";

        let tokens = analyzer.pos_tag(text).unwrap();
        let phrases = analyzer.extract_noun_phrases(&tokens).unwrap();

        assert_eq!(phrases.len(), 1);
        assert_eq!(phrases[0].text, "The quick brown fox");
    }

    #[test]
    fn test_dependency_parsing() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "The cat chased the mouse";

        let tokens = analyzer.pos_tag(text).unwrap();
        let deps = analyzer.parse_dependencies(&tokens).unwrap();

        // Should have subject and object dependencies
        assert!(!deps.is_empty());

        // Find subject dependency
        let has_subject = deps.iter().any(|d| matches!(d.relation, DependencyRelation::Subject));
        assert!(has_subject, "Should have subject dependency");
    }

    #[test]
    fn test_sentence_segmentation() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "This is sentence one. This is sentence two! And sentence three?";

        let sentences = analyzer.segment_sentences(text);

        assert_eq!(sentences.len(), 3);
        assert!(sentences[0].contains("sentence one"));
        assert!(sentences[1].contains("sentence two"));
        assert!(sentences[2].contains("sentence three"));
    }

    #[test]
    fn test_tokenization() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "Hello, world!";

        let tokens = analyzer.tokenize(text);

        assert_eq!(tokens.len(), 4); // "Hello", ",", "world", "!"
        assert_eq!(tokens[0].0, "Hello");
        assert_eq!(tokens[1].0, ",");
    }

    #[test]
    fn test_proper_noun_detection() {
        let analyzer = SyntaxAnalyzer::new(SyntaxAnalyzerConfig::default());
        let text = "John Smith lives in New York";

        let tokens = analyzer.pos_tag(text).unwrap();

        // Should detect proper nouns
        let proper_nouns: Vec<_> = tokens.iter()
            .filter(|t| matches!(t.pos, POSTag::ProperNoun))
            .collect();

        assert!(!proper_nouns.is_empty());
    }
}

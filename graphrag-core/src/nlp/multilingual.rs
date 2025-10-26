//! Multilingual Support
//!
//! This module provides language detection and language-specific text processing:
//! - Automatic language detection using n-gram analysis
//! - Language-specific tokenization and normalization
//! - Multi-language entity extraction
//! - Cross-lingual entity linking
//!
//! ## Supported Languages
//!
//! - English (en)
//! - Spanish (es)
//! - French (fr)
//! - German (de)
//! - Chinese (zh)
//! - Japanese (ja)
//! - Korean (ko)
//! - Arabic (ar)
//! - Russian (ru)
//! - Portuguese (pt)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported languages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Language {
    /// English language
    English,
    /// Spanish language
    Spanish,
    /// French language
    French,
    /// German language
    German,
    /// Chinese language
    Chinese,
    /// Japanese language
    Japanese,
    /// Korean language
    Korean,
    /// Arabic language
    Arabic,
    /// Russian language
    Russian,
    /// Portuguese language
    Portuguese,
    /// Unknown or unsupported language
    Unknown,
}

impl Language {
    /// Get ISO 639-1 language code
    pub fn code(&self) -> &str {
        match self {
            Language::English => "en",
            Language::Spanish => "es",
            Language::French => "fr",
            Language::German => "de",
            Language::Chinese => "zh",
            Language::Japanese => "ja",
            Language::Korean => "ko",
            Language::Arabic => "ar",
            Language::Russian => "ru",
            Language::Portuguese => "pt",
            Language::Unknown => "unknown",
        }
    }

    /// Parse from ISO 639-1 code
    pub fn from_code(code: &str) -> Self {
        match code.to_lowercase().as_str() {
            "en" => Language::English,
            "es" => Language::Spanish,
            "fr" => Language::French,
            "de" => Language::German,
            "zh" => Language::Chinese,
            "ja" => Language::Japanese,
            "ko" => Language::Korean,
            "ar" => Language::Arabic,
            "ru" => Language::Russian,
            "pt" => Language::Portuguese,
            _ => Language::Unknown,
        }
    }

    /// Get language name
    pub fn name(&self) -> &str {
        match self {
            Language::English => "English",
            Language::Spanish => "Spanish",
            Language::French => "French",
            Language::German => "German",
            Language::Chinese => "Chinese",
            Language::Japanese => "Japanese",
            Language::Korean => "Korean",
            Language::Arabic => "Arabic",
            Language::Russian => "Russian",
            Language::Portuguese => "Portuguese",
            Language::Unknown => "Unknown",
        }
    }

    /// Check if language uses CJK (Chinese, Japanese, Korean) script
    pub fn is_cjk(&self) -> bool {
        matches!(self, Language::Chinese | Language::Japanese | Language::Korean)
    }

    /// Check if language is right-to-left
    pub fn is_rtl(&self) -> bool {
        matches!(self, Language::Arabic)
    }
}

/// Language detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// Detected language
    pub language: Language,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Alternative languages with scores
    pub alternatives: Vec<(Language, f32)>,
}

/// Language detector using n-gram frequency analysis
pub struct LanguageDetector {
    /// N-gram language models
    models: HashMap<Language, LanguageModel>,
}

/// Simple n-gram language model
struct LanguageModel {
    /// Character n-gram frequencies
    ngrams: HashMap<String, f32>,
    /// Total n-gram count
    total: f32,
}

impl LanguageModel {
    /// Create new language model
    fn new() -> Self {
        Self {
            ngrams: HashMap::new(),
            total: 0.0,
        }
    }

    /// Add training text
    fn train(&mut self, text: &str, n: usize) {
        let chars: Vec<char> = text.chars().collect();
        for window in chars.windows(n) {
            let ngram: String = window.iter().collect();
            *self.ngrams.entry(ngram).or_insert(0.0) += 1.0;
            self.total += 1.0;
        }
    }

    /// Calculate probability of text
    fn score(&self, text: &str, n: usize) -> f32 {
        let chars: Vec<char> = text.chars().collect();
        let mut score = 0.0;
        let mut count = 0;

        for window in chars.windows(n) {
            let ngram: String = window.iter().collect();
            if let Some(&freq) = self.ngrams.get(&ngram) {
                score += (freq / self.total).ln();
            } else {
                score += (1.0 / (self.total + 1.0)).ln(); // Smoothing
            }
            count += 1;
        }

        if count > 0 {
            score / count as f32
        } else {
            0.0
        }
    }
}

impl LanguageDetector {
    /// Create new language detector
    pub fn new() -> Self {
        let mut detector = Self {
            models: HashMap::new(),
        };

        // Initialize with basic language models
        detector.initialize_models();
        detector
    }

    /// Initialize language models with sample text
    fn initialize_models(&mut self) {
        // English
        let mut english_model = LanguageModel::new();
        english_model.train("the quick brown fox jumps over the lazy dog", 3);
        english_model.train("this is a test of the english language", 3);
        self.models.insert(Language::English, english_model);

        // Spanish
        let mut spanish_model = LanguageModel::new();
        spanish_model.train("el rápido zorro marrón salta sobre el perro perezoso", 3);
        spanish_model.train("esta es una prueba del idioma español", 3);
        self.models.insert(Language::Spanish, spanish_model);

        // French
        let mut french_model = LanguageModel::new();
        french_model.train("le renard brun rapide saute par-dessus le chien paresseux", 3);
        french_model.train("ceci est un test de la langue française", 3);
        self.models.insert(Language::French, french_model);

        // German
        let mut german_model = LanguageModel::new();
        german_model.train("der schnelle braune fuchs springt über den faulen hund", 3);
        german_model.train("dies ist ein test der deutschen sprache", 3);
        self.models.insert(Language::German, german_model);

        // Portuguese
        let mut portuguese_model = LanguageModel::new();
        portuguese_model.train("a rápida raposa marrom pula sobre o cão preguiçoso", 3);
        portuguese_model.train("este é um teste da língua portuguesa", 3);
        self.models.insert(Language::Portuguese, portuguese_model);

        // TODO: Add models for Chinese, Japanese, Korean, Arabic, Russian
        // These require proper training data with representative character sets
    }

    /// Detect language of text
    pub fn detect(&self, text: &str) -> DetectionResult {
        if text.trim().is_empty() {
            return DetectionResult {
                language: Language::Unknown,
                confidence: 0.0,
                alternatives: Vec::new(),
            };
        }

        // Quick heuristics for CJK and RTL languages
        if self.is_likely_chinese(text) {
            return DetectionResult {
                language: Language::Chinese,
                confidence: 0.9,
                alternatives: vec![
                    (Language::Japanese, 0.1),
                ],
            };
        }

        if self.is_likely_japanese(text) {
            return DetectionResult {
                language: Language::Japanese,
                confidence: 0.9,
                alternatives: vec![
                    (Language::Chinese, 0.1),
                ],
            };
        }

        if self.is_likely_korean(text) {
            return DetectionResult {
                language: Language::Korean,
                confidence: 0.95,
                alternatives: Vec::new(),
            };
        }

        if self.is_likely_arabic(text) {
            return DetectionResult {
                language: Language::Arabic,
                confidence: 0.95,
                alternatives: Vec::new(),
            };
        }

        if self.is_likely_russian(text) {
            return DetectionResult {
                language: Language::Russian,
                confidence: 0.9,
                alternatives: Vec::new(),
            };
        }

        // Score against all models
        let mut scores: Vec<(Language, f32)> = self
            .models
            .iter()
            .map(|(lang, model)| (*lang, model.score(text, 3)))
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if scores.is_empty() {
            return DetectionResult {
                language: Language::Unknown,
                confidence: 0.0,
                alternatives: Vec::new(),
            };
        }

        // Normalize scores to confidence (0.0 to 1.0)
        let max_score = scores[0].1;
        let min_score = scores.last().unwrap().1;
        let range = max_score - min_score;

        let confidence = if range > 0.0 {
            ((max_score - min_score) / range).clamp(0.0, 1.0)
        } else {
            0.5
        };

        DetectionResult {
            language: scores[0].0,
            confidence,
            alternatives: scores.into_iter().skip(1).take(3).collect(),
        }
    }

    /// Check if text is likely Chinese (simplified or traditional)
    fn is_likely_chinese(&self, text: &str) -> bool {
        let chinese_chars = text.chars().filter(|c| {
            let code = *c as u32;
            (0x4E00..=0x9FFF).contains(&code) // CJK Unified Ideographs
        }).count();

        chinese_chars as f32 / text.chars().count() as f32 > 0.3
    }

    /// Check if text is likely Japanese (hiragana/katakana present)
    fn is_likely_japanese(&self, text: &str) -> bool {
        let japanese_chars = text.chars().filter(|c| {
            let code = *c as u32;
            (0x3040..=0x309F).contains(&code) || // Hiragana
            (0x30A0..=0x30FF).contains(&code)    // Katakana
        }).count();

        japanese_chars > 0
    }

    /// Check if text is likely Korean (Hangul)
    fn is_likely_korean(&self, text: &str) -> bool {
        let korean_chars = text.chars().filter(|c| {
            let code = *c as u32;
            (0xAC00..=0xD7AF).contains(&code) // Hangul Syllables
        }).count();

        korean_chars as f32 / text.chars().count() as f32 > 0.3
    }

    /// Check if text is likely Arabic
    fn is_likely_arabic(&self, text: &str) -> bool {
        let arabic_chars = text.chars().filter(|c| {
            let code = *c as u32;
            (0x0600..=0x06FF).contains(&code) // Arabic
        }).count();

        arabic_chars as f32 / text.chars().count() as f32 > 0.3
    }

    /// Check if text is likely Russian (Cyrillic)
    fn is_likely_russian(&self, text: &str) -> bool {
        let cyrillic_chars = text.chars().filter(|c| {
            let code = *c as u32;
            (0x0400..=0x04FF).contains(&code) // Cyrillic
        }).count();

        cyrillic_chars as f32 / text.chars().count() as f32 > 0.3
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Language-specific text processor
pub struct MultilingualProcessor {
    detector: LanguageDetector,
}

impl MultilingualProcessor {
    /// Create new multilingual processor
    pub fn new() -> Self {
        Self {
            detector: LanguageDetector::new(),
        }
    }

    /// Detect language and return processor configuration
    pub fn process(&self, text: &str) -> ProcessedText {
        let detection = self.detector.detect(text);
        let normalized = self.normalize_text(text, detection.language);
        let tokens = self.tokenize(&normalized, detection.language);

        ProcessedText {
            original: text.to_string(),
            normalized,
            tokens,
            language: detection.language,
            confidence: detection.confidence,
        }
    }

    /// Normalize text based on language
    fn normalize_text(&self, text: &str, language: Language) -> String {
        let mut normalized = text.to_string();

        // Remove extra whitespace
        normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");

        // Language-specific normalization
        match language {
            Language::Arabic => {
                // Remove Arabic diacritics
                normalized = normalized.chars()
                    .filter(|c| {
                        let code = *c as u32;
                        !(0x064B..=0x0652).contains(&code) // Arabic diacritics
                    })
                    .collect();
            }
            Language::Chinese | Language::Japanese => {
                // Full-width to half-width conversion for ASCII characters
                normalized = normalized.chars()
                    .map(|c| {
                        let code = c as u32;
                        if (0xFF01..=0xFF5E).contains(&code) {
                            char::from_u32(code - 0xFEE0).unwrap_or(c)
                        } else {
                            c
                        }
                    })
                    .collect();
            }
            _ => {}
        }

        normalized
    }

    /// Tokenize text based on language
    fn tokenize(&self, text: &str, language: Language) -> Vec<String> {
        match language {
            Language::Chinese | Language::Japanese => {
                // Character-level tokenization for CJK
                // TODO: Implement proper word segmentation (e.g., jieba for Chinese)
                text.chars()
                    .filter(|c| !c.is_whitespace())
                    .map(|c| c.to_string())
                    .collect()
            }
            _ => {
                // Word-level tokenization
                text.split_whitespace()
                    .map(|s| s.to_string())
                    .collect()
            }
        }
    }
}

impl Default for MultilingualProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Processed text result
#[derive(Debug, Clone)]
pub struct ProcessedText {
    /// Original text
    pub original: String,
    /// Normalized text
    pub normalized: String,
    /// Tokens
    pub tokens: Vec<String>,
    /// Detected language
    pub language: Language,
    /// Detection confidence
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_language_codes() {
        assert_eq!(Language::English.code(), "en");
        assert_eq!(Language::Spanish.code(), "es");
        assert_eq!(Language::from_code("fr"), Language::French);
        assert_eq!(Language::from_code("unknown"), Language::Unknown);
    }

    #[test]
    fn test_cjk_detection() {
        assert!(Language::Chinese.is_cjk());
        assert!(Language::Japanese.is_cjk());
        assert!(Language::Korean.is_cjk());
        assert!(!Language::English.is_cjk());
    }

    #[test]
    fn test_rtl_detection() {
        assert!(Language::Arabic.is_rtl());
        assert!(!Language::English.is_rtl());
    }

    #[test]
    fn test_language_detection() {
        let detector = LanguageDetector::new();

        let result = detector.detect("This is English text");
        assert_eq!(result.language, Language::English);
        assert!(result.confidence > 0.0);

        let result = detector.detect("Esto es texto en español");
        assert_eq!(result.language, Language::Spanish);

        let result = detector.detect("Ceci est du texte français");
        assert_eq!(result.language, Language::French);
    }

    #[test]
    fn test_chinese_detection() {
        let detector = LanguageDetector::new();
        let result = detector.detect("这是中文文本");
        assert_eq!(result.language, Language::Chinese);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_japanese_detection() {
        let detector = LanguageDetector::new();
        let result = detector.detect("これは日本語のテキストです");
        assert_eq!(result.language, Language::Japanese);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_korean_detection() {
        let detector = LanguageDetector::new();
        let result = detector.detect("이것은 한국어 텍스트입니다");
        assert_eq!(result.language, Language::Korean);
        assert!(result.confidence > 0.8);
    }

    #[test]
    fn test_multilingual_processing() {
        let processor = MultilingualProcessor::new();

        let result = processor.process("This is a test");
        assert_eq!(result.language, Language::English);
        assert!(!result.tokens.is_empty());

        let result = processor.process("Esto es una prueba");
        assert_eq!(result.language, Language::Spanish);
    }

    #[test]
    fn test_text_normalization() {
        let processor = MultilingualProcessor::new();
        let result = processor.process("This   has   extra   spaces");
        assert_eq!(result.normalized, "This has extra spaces");
    }
}

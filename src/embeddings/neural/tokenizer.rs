//! Multi-threaded tokenization for neural embeddings

use std::collections::HashMap;
use crate::core::{Result, GraphRAGError};

/// Tokenized input structure
#[derive(Debug, Clone)]
pub struct TokenizedInput {
    pub input_ids: Vec<Vec<u32>>,
    pub attention_mask: Vec<Vec<u32>>,
    pub token_type_ids: Option<Vec<Vec<u32>>>,
    pub special_tokens_mask: Option<Vec<Vec<u32>>>,
}

/// Multi-threaded tokenizer for batch processing
pub struct MultiThreadTokenizer {
    vocab: HashMap<String, u32>,
    inverse_vocab: HashMap<u32, String>,
    max_length: usize,
    pad_token_id: u32,
    cls_token_id: u32,
    sep_token_id: u32,
    unk_token_id: u32,
}

impl MultiThreadTokenizer {
    pub fn new(max_length: usize) -> Result<Self> {
        let mut tokenizer = Self {
            vocab: HashMap::new(),
            inverse_vocab: HashMap::new(),
            max_length,
            pad_token_id: 0,
            cls_token_id: 101,
            sep_token_id: 102,
            unk_token_id: 100,
        };

        tokenizer.initialize_basic_vocab();
        Ok(tokenizer)
    }

    pub fn from_vocab_file(vocab_path: &str, max_length: usize) -> Result<Self> {
        let mut tokenizer = Self::new(max_length)?;
        tokenizer.load_vocab_from_file(vocab_path)?;
        Ok(tokenizer)
    }

    fn initialize_basic_vocab(&mut self) {
        // Initialize with basic BERT vocabulary
        let special_tokens = vec![
            ("[PAD]", 0),
            ("[UNK]", 100),
            ("[CLS]", 101),
            ("[SEP]", 102),
            ("[MASK]", 103),
        ];

        for (token, id) in special_tokens {
            self.vocab.insert(token.to_string(), id);
            self.inverse_vocab.insert(id, token.to_string());
        }

        // Add basic punctuation and common words
        let common_tokens = vec![
            (".", 1012), (",", 1010), ("!", 999), ("?", 1029),
            ("the", 1996), ("a", 1037), ("an", 2019), ("and", 1998),
            ("or", 2030), ("but", 2021), ("in", 1999), ("on", 2006),
            ("at", 2012), ("to", 2000), ("for", 2005), ("of", 1997),
            ("with", 2007), ("by", 2011), ("is", 2003), ("are", 2024),
            ("was", 2001), ("were", 2020), ("be", 2022), ("been", 2042),
        ];

        for (token, id) in common_tokens {
            self.vocab.insert(token.to_string(), id);
            self.inverse_vocab.insert(id, token.to_string());
        }

        // Fill remaining vocabulary with placeholder tokens
        for i in 200..30522 {
            if !self.inverse_vocab.contains_key(&i) {
                let token = format!("[UNUSED{i}]");
                self.vocab.insert(token.clone(), i);
                self.inverse_vocab.insert(i, token);
            }
        }
    }

    fn load_vocab_from_file(&mut self, vocab_path: &str) -> Result<()> {
        let content = std::fs::read_to_string(vocab_path).map_err(GraphRAGError::Io)?;

        self.vocab.clear();
        self.inverse_vocab.clear();

        for (id, line) in content.lines().enumerate() {
            let token = line.trim().to_string();
            let id = id as u32;

            self.vocab.insert(token.clone(), id);
            self.inverse_vocab.insert(id, token);
        }

        Ok(())
    }

    pub fn encode_batch(&self, texts: &[&str]) -> Result<TokenizedInput> {
        let mut input_ids = Vec::with_capacity(texts.len());
        let mut attention_masks = Vec::with_capacity(texts.len());

        for text in texts {
            let (ids, mask) = self.encode_single(text)?;
            input_ids.push(ids);
            attention_masks.push(mask);
        }

        Ok(TokenizedInput {
            input_ids,
            attention_mask: attention_masks,
            token_type_ids: None,
            special_tokens_mask: None,
        })
    }

    pub fn encode(&self, text: &str) -> Result<TokenizedInput> {
        let (input_ids, attention_mask) = self.encode_single(text)?;

        Ok(TokenizedInput {
            input_ids: vec![input_ids],
            attention_mask: vec![attention_mask],
            token_type_ids: None,
            special_tokens_mask: None,
        })
    }

    fn encode_single(&self, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
        // Tokenize the text
        let tokens = self.tokenize(text);

        // Convert tokens to IDs
        let mut token_ids = vec![self.cls_token_id]; // Start with [CLS]

        for token in tokens {
            let id = self.vocab.get(&token).unwrap_or(&self.unk_token_id);
            token_ids.push(*id);

            // Stop if we're approaching max length (need space for [SEP])
            if token_ids.len() >= self.max_length - 1 {
                break;
            }
        }

        // Add [SEP] token
        token_ids.push(self.sep_token_id);

        // Create attention mask (1 for real tokens, 0 for padding)
        let mut attention_mask = vec![1u32; token_ids.len()];

        // Pad to max_length
        while token_ids.len() < self.max_length {
            token_ids.push(self.pad_token_id);
            attention_mask.push(0);
        }

        // Truncate if necessary
        token_ids.truncate(self.max_length);
        attention_mask.truncate(self.max_length);

        Ok((token_ids, attention_mask))
    }

    fn tokenize(&self, text: &str) -> Vec<String> {
        // Simple whitespace + punctuation tokenization
        // In a real implementation, this would use WordPiece or BPE

        let mut tokens = Vec::new();
        let mut current_token = String::new();

        for ch in text.chars() {
            match ch {
                ' ' | '\t' | '\n' | '\r' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.to_lowercase());
                        current_token.clear();
                    }
                }
                '.' | ',' | '!' | '?' | ':' | ';' | '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.to_lowercase());
                        current_token.clear();
                    }
                    tokens.push(ch.to_string());
                }
                _ => {
                    current_token.push(ch);
                }
            }
        }

        if !current_token.is_empty() {
            tokens.push(current_token.to_lowercase());
        }

        tokens
    }

    pub fn decode(&self, token_ids: &[u32]) -> String {
        let tokens: Vec<String> = token_ids
            .iter()
            .filter_map(|&id| {
                if id == self.pad_token_id || id == self.cls_token_id || id == self.sep_token_id {
                    None
                } else {
                    self.inverse_vocab.get(&id).cloned()
                }
            })
            .collect();

        tokens.join(" ")
    }

    pub fn get_vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn get_max_length(&self) -> usize {
        self.max_length
    }

    pub fn get_pad_token_id(&self) -> u32 {
        self.pad_token_id
    }

    pub fn get_cls_token_id(&self) -> u32 {
        self.cls_token_id
    }

    pub fn get_sep_token_id(&self) -> u32 {
        self.sep_token_id
    }

    pub fn get_unk_token_id(&self) -> u32 {
        self.unk_token_id
    }
}

/// For production, here's how you would integrate with a real tokenizer:
///
/// ```rust
/// use tokenizers::{Tokenizer, models::bpe::BPE, pre_tokenizers::whitespace::Whitespace};
///
/// impl MultiThreadTokenizer {
///     pub fn from_pretrained(model_name: &str) -> Result<Self> {
///         let tokenizer = Tokenizer::from_pretrained(model_name, None)?;
///
///         Ok(Self {
///             tokenizer,
///             max_length: 512,
///         })
///     }
///
///     pub fn encode_batch_with_real_tokenizer(&self, texts: &[&str]) -> Result<TokenizedInput> {
///         let encodings = self.tokenizer.encode_batch(texts, true)?;
///
///         let input_ids = encodings.iter().map(|e| e.get_ids().to_vec()).collect();
///         let attention_masks = encodings.iter().map(|e| e.get_attention_mask().to_vec()).collect();
///
///         Ok(TokenizedInput {
///             input_ids,
///             attention_mask: attention_masks,
///             token_type_ids: Some(encodings.iter().map(|e| e.get_type_ids().to_vec()).collect()),
///             special_tokens_mask: Some(encodings.iter().map(|e| e.get_special_tokens_mask().to_vec()).collect()),
///         })
///     }
/// }
/// ```
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = MultiThreadTokenizer::new(512);
        assert!(tokenizer.is_ok());

        let tokenizer = tokenizer.unwrap();
        assert_eq!(tokenizer.get_max_length(), 512);
        assert!(tokenizer.get_vocab_size() > 0);
    }

    #[test]
    fn test_single_text_encoding() {
        let tokenizer = MultiThreadTokenizer::new(128).unwrap();
        let text = "Hello world! This is a test.";

        let result = tokenizer.encode(text);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.input_ids.len(), 1);
        assert_eq!(encoded.attention_mask.len(), 1);
        assert_eq!(encoded.input_ids[0].len(), 128);
        assert_eq!(encoded.attention_mask[0].len(), 128);

        // First token should be [CLS]
        assert_eq!(encoded.input_ids[0][0], tokenizer.get_cls_token_id());

        // Should have some attention (non-zero values)
        let attention_sum: u32 = encoded.attention_mask[0].iter().sum();
        assert!(attention_sum > 0);
    }

    #[test]
    fn test_batch_encoding() {
        let tokenizer = MultiThreadTokenizer::new(64).unwrap();
        let texts = vec!["Hello world", "This is another test", "Short"];

        let result = tokenizer.encode_batch(&texts);
        assert!(result.is_ok());

        let encoded = result.unwrap();
        assert_eq!(encoded.input_ids.len(), 3);
        assert_eq!(encoded.attention_mask.len(), 3);

        // All sequences should be same length
        for ids in &encoded.input_ids {
            assert_eq!(ids.len(), 64);
        }

        for mask in &encoded.attention_mask {
            assert_eq!(mask.len(), 64);
        }
    }

    #[test]
    fn test_tokenization() {
        let tokenizer = MultiThreadTokenizer::new(512).unwrap();
        let text = "Hello, world! How are you?";

        let result = tokenizer.encode(text).unwrap();
        let decoded = tokenizer.decode(&result.input_ids[0]);

        // Should contain recognizable tokens
        assert!(decoded.contains("hello"));
        assert!(decoded.contains("world"));
    }

    #[test]
    fn test_padding_and_truncation() {
        let tokenizer = MultiThreadTokenizer::new(10).unwrap(); // Very short max length

        // Test padding (short text)
        let short_text = "Hi";
        let encoded = tokenizer.encode(short_text).unwrap();
        assert_eq!(encoded.input_ids[0].len(), 10);

        // Count padding tokens
        let pad_count = encoded.input_ids[0].iter().filter(|&&id| id == tokenizer.get_pad_token_id()).count();
        assert!(pad_count > 0);

        // Test truncation (long text)
        let long_text = "This is a very long text that should definitely be truncated because it exceeds our maximum length";
        let encoded_long = tokenizer.encode(long_text).unwrap();
        assert_eq!(encoded_long.input_ids[0].len(), 10);

        // Should still start with [CLS] and end with [SEP]
        assert_eq!(encoded_long.input_ids[0][0], tokenizer.get_cls_token_id());
        assert_eq!(encoded_long.input_ids[0][9], tokenizer.get_sep_token_id());
    }
}
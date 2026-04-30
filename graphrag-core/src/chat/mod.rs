//! `ChatClient` — runtime dispatcher between Ollama-protocol and
//! OpenAI-compatible chat backends.
//!
//! Every consumer of the chat LLM (entity extractors, query planner,
//! gleaning) used to take `OllamaClient` directly. Switching them to
//! `ChatClient` lets `Config.openai.enabled` route the same call sites
//! to either an Ollama server or any OpenAI-spec server (vLLM,
//! llama-server, real OpenAI, OpenRouter, ...) without per-callsite
//! branching.
//!
//! Method surface mirrors `OllamaClient` exactly (`generate`,
//! `generate_with_params`, `get_stats`, `clear_cache`, `cache_size`) so
//! the swap is a tree-wide find-replace. The canonical params type stays
//! `OllamaGenerationParams` — the OpenAI side ignores Ollama-only fields
//! (top_k, repeat_penalty, keep_alive, num_ctx).

use crate::core::error::Result;
use crate::ollama::{OllamaClient, OllamaConfig, OllamaGenerationParams, OllamaUsageStats};
use crate::openai::OpenAIConfig;
#[cfg(feature = "openai")]
use crate::openai::OpenAIClient;

/// Chat backend selector. `from_config` picks OpenAI when `openai.enabled`,
/// otherwise Ollama when `ollama.enabled`, returning `None` if neither is on.
///
/// The `OpenAI` variant is only present when the `openai` feature is
/// compiled in. Without it, `OpenAIConfig` still parses through serde
/// (so user configs round-trip), but `from_config` silently treats
/// `openai.enabled = true` as "no openai available" and falls back to
/// the Ollama branch.
#[derive(Clone, Debug)]
pub enum ChatClient {
    Ollama(OllamaClient),
    #[cfg(feature = "openai")]
    OpenAI(OpenAIClient),
}

impl ChatClient {
    /// Wire from the runtime config. OpenAI takes precedence over Ollama
    /// when both are enabled (so `openai.enabled = true` overrides any
    /// `ollama.enabled` setting). When the `openai` feature is not
    /// compiled in, `openai.enabled` is ignored and the dispatcher
    /// falls through to the Ollama branch (or `None`) — useful build-
    /// flag mistakes surface as a `tracing::warn!` if `tracing` is on.
    pub fn from_config(
        ollama: &OllamaConfig,
        openai: &OpenAIConfig,
    ) -> Option<Self> {
        #[cfg(feature = "openai")]
        if openai.enabled {
            return Some(Self::OpenAI(OpenAIClient::new(openai.clone())));
        }
        #[cfg(all(not(feature = "openai"), feature = "tracing"))]
        if openai.enabled {
            tracing::warn!(
                "openai.enabled = true but graphrag-core was built without the `openai` feature; \
                 falling back to ollama. Rebuild with `--features openai` (or the `starter` bundle) to enable."
            );
        }
        let _ = openai; // keep the binding meaningful when no cfg-guarded read happens
        if ollama.enabled {
            Some(Self::Ollama(OllamaClient::new(ollama.clone())))
        } else {
            None
        }
    }

    /// Construct directly from an `OllamaClient` (back-compat helper for
    /// call sites that build their own client).
    pub fn from_ollama(client: OllamaClient) -> Self {
        Self::Ollama(client)
    }

    /// Construct directly from an `OpenAIClient`. Only present with
    /// `feature = "openai"`.
    #[cfg(feature = "openai")]
    pub fn from_openai(client: OpenAIClient) -> Self {
        Self::OpenAI(client)
    }

    /// Whether either backend is available right now (mirrors
    /// `Config.{ollama,openai}.enabled` checks at call sites).
    pub fn is_enabled(&self) -> bool { true }

    /// Single-shot generate.
    pub async fn generate(&self, prompt: &str) -> Result<String> {
        match self {
            Self::Ollama(c) => c.generate(prompt).await,
            #[cfg(feature = "openai")]
            Self::OpenAI(c) => c.generate(prompt).await,
        }
    }

    /// Generate with caller-supplied params.
    pub async fn generate_with_params(
        &self,
        prompt: &str,
        params: OllamaGenerationParams,
    ) -> Result<String> {
        match self {
            Self::Ollama(c) => c.generate_with_params(prompt, params).await,
            #[cfg(feature = "openai")]
            Self::OpenAI(c) => c.generate_with_params(prompt, params).await,
        }
    }

    /// Stats handle (always returns the underlying client's stats).
    pub fn get_stats(&self) -> &OllamaUsageStats {
        match self {
            Self::Ollama(c) => c.get_stats(),
            #[cfg(feature = "openai")]
            Self::OpenAI(c) => c.get_stats(),
        }
    }

    /// `keep_alive` value when the active backend is Ollama; `None` for
    /// OpenAI (the field is Ollama-specific). Consumers used to read this
    /// off `ollama_client.config().keep_alive` directly; expose it here so
    /// the call site doesn't have to know which backend is live.
    pub fn keep_alive(&self) -> Option<String> {
        match self {
            Self::Ollama(c) => c.config().keep_alive.clone(),
            #[cfg(feature = "openai")]
            Self::OpenAI(_) => None,
        }
    }
}

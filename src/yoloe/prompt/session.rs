use crate::yoloe::checkpoint::identity::Identity;
use crate::yoloe::predict_config::PredictConfig;
use crate::yoloe::prompt::controller::Controller;
use crate::yoloe::prompt::table::ScorerConfig;
use crate::yoloe::reprta::RepRta;
use crate::yoloe::usage::{EmbeddingTable, Usage};

/// Validation request derived from a configured YOLOE session.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidationRequest {
    /// Prompt source used for validation.
    pub usage: Usage,
    /// Static class vocabulary being evaluated.
    pub classes: Vec<String>,
    /// Prediction options used by validation.
    pub predict: PredictConfig,
}

/// High-level YOLOE prompt session state.
///
/// Carries the active prompt mode (text, visual, or prompt-free), its
/// precomputed prompt embeddings, and the prediction defaults shared across
/// forward passes. Constructed once and passed to [`crate::yoloe::Model`]
/// / [`crate::yoloe::Model`] inference methods.
#[derive(Debug, Clone)]
pub struct Session {
    /// Parsed checkpoint identity, when constructed from an official checkpoint name.
    pub checkpoint: Option<Identity>,
    /// Prompt controller for text, visual, and prompt-free states.
    pub prompts: Controller,
    /// Prediction defaults for YOLOE validation and inference.
    pub predict: PredictConfig,
    /// Region-prompt scorer configuration.
    pub scorer: ScorerConfig,
    pub(crate) prompt_table: Option<EmbeddingTable>,
    /// Loaded RepRTA text-prompt adapter, applied automatically to text-prompt
    /// embeddings when `config.rep_rta.enabled`, matching official inference.
    pub reprta: Option<RepRta>,
}

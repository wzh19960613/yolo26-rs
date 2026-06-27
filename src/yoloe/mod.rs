//! YOLOE prompt state and open-vocabulary inference.
//!
//! YOLOE adds open-vocabulary text prompts, visual prompts, prompt-free
//! vocabularies, and fine-tuning workflows on top of the dense prediction
//! family. This module records the active prompt state (text, visual, or
//! prompt-free) and drives the YOLOE-26 inference path.

// YOLOE has separate text, visual and prompt-free feature paths. The internal
// prelude below is intentionally broader than any single path, so suppress
// feature-combination unused import noise while keeping the public API explicit.
#![allow(unused_imports)]

pub mod checkpoint;
pub mod detect;
pub mod head;
pub mod prompt;
pub mod savpe;
pub mod segment;

mod config;
pub(crate) mod first_dim;
pub(crate) mod infer_lrpc_vocab_classes;
mod predict_config;
/// Re-parameterizable Region-Text Alignment adapter, reachable as
/// `yoloe::reprta` (and re-exported as `yoloe::RepRta`) so callers can borrow
/// `Option<&RepRta>` from a loaded model for `Session::text`.
pub mod reprta;
pub(crate) mod select_lrpc_indices;
/// YOLOE usage, checkpoint-kind, and prompt embedding types.
pub mod usage;
/// Visuals batching helpers for SAVPE input construction.
pub mod visuals;

// Stable public API: re-exported as `pub` so callers can reach them through
// `crate::yoloe::`.
pub use checkpoint::identity::Identity;
pub use checkpoint::layout::Layout;
pub use config::*;
pub use prompt::session::{Session, ValidationRequest};
pub use prompt::state::State;
pub use prompt::table::ScorerConfig;
#[cfg(feature = "yoloe-text")]
pub use prompt::text_encoder::ClipTextEncoder;
pub use prompt::text_prompts::Text;
pub use prompt::visual::{Visual, VisualKind};
pub use segment::model::train_config::PromptMode;
pub use usage::*;
pub use visuals::*;

/// Default square YOLOE input size.
pub const MODEL_INPUT_SIZE: usize = crate::model::MODEL_INPUT_SIZE;

/// YOLOE prediction options alias, mirroring the task roots.
pub type PredictOptions = crate::options::FilterOption;

/// Returns a builder for YOLOE [`Config`] with default settings, mirroring the
/// `config_builder()` entry point of the stable task roots.
pub fn config_builder() -> config::Builder {
    Config::builder()
}

// Internal implementation prelude: head parts, encoders, lrpc/savpe/rep_rta
// modules, prompt controllers and auxiliary configs. Kept crate-internal so
// the `yoloe` namespace does not leak these as a public API surface; internal
// code references them through their full paths (`crate::yoloe::head::...`).
pub(crate) use detect::head::*;
pub(crate) use head::contrastive::*;
pub(crate) use head::key_plan::*;
pub(crate) use head::lrpc::head::*;
pub(crate) use head::lrpc::official::*;
pub(crate) use head::lrpc::output::*;
pub(crate) use head::lrpc::pyramid::*;
pub(crate) use predict_config::*;
pub(crate) use prompt::controller::*;
/// [`RepRta`] is the only internal-prelude item exposed publicly, because
/// `Session::text` borrows `Option<&RepRta>` from a loaded model.
pub use reprta::RepRta;
pub(crate) use reprta::*;
pub(crate) use savpe::encoder::*;
pub(crate) use savpe::pooler::*;
pub(crate) use segment::head::*;

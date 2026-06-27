//! YOLOE prompt state, session, and prompt encoders.

pub mod visual_letterbox;

/// Prompt controller tracking active prompt state and embedding space.
pub mod controller;
/// Prompt session holding resolved checkpoint, prompts, and scorers.
pub mod session;
pub(crate) mod session_construct;
pub(crate) mod session_forward;
pub(crate) mod session_prompts;
pub(crate) mod session_train;
/// Prompt state enum (text/visual/prompt-free).
pub mod state;
/// Prompt embedding table and scorer configuration.
pub mod table;
/// CLIP text encoder for text-prompt class names (`yoloe-text` feature).
#[cfg(feature = "yoloe-text")]
pub mod text_encoder;
/// Text prompt class vocabulary.
pub mod text_prompts;
/// Visual prompt (box/mask) types.
pub mod visual;

pub use controller::Controller;
pub use session::Session;
pub use session::ValidationRequest;
pub use state::State;
pub use table::ScorerConfig;
#[cfg(feature = "yoloe-text")]
pub use text_encoder::ClipTextEncoder;
pub use text_prompts::Text;
pub use visual::{Visual, VisualKind};

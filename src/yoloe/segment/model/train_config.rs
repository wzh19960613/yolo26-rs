//! YOLOE prompt-alignment mode.
//!
//! [`PromptMode`] selects which prompt-alignment branch the network activates.
//! It is a network-layer concept used by both inference and training, so it
//! stays in the `yoloe` module even though the trainable `TrainConfig` lives
//! under `train::yoloe`.

/// Trainable YOLOE head training mode selecting the prompt-alignment loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PromptMode {
    /// Text-prompt contrastive alignment (RepRTA + BNContrastive).
    TextPrompt,
    /// Visual-prompt alignment (SAVPE encoder + BNContrastive).
    Visual,
    /// Prompt-free alignment (LRPC vocab/pf/loc).
    PromptFree,
}

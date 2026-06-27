//! Native YOLOE training: trainable segment model, prompt-aligned loss, and a
//! training [`Session`] that mirrors the regular YOLO26 trainers.

pub(crate) use crate::train::exports::*;

pub(crate) mod config;
pub(crate) mod detect_forward;
pub(crate) mod detect_output;
pub(crate) mod eval;
pub(crate) mod eval_common;
pub(crate) mod eval_config;
pub(crate) mod head;
pub(crate) mod loss;
pub(crate) mod mode;
pub(crate) mod model;
pub(crate) mod model_config;
pub(crate) mod output;
pub(crate) mod parts;
pub(crate) mod prompt_free;
pub(crate) mod prompt_free_head;
pub(crate) mod save;
/// YOLOE training session owning a [`Model`] and optimizer state.
pub mod session;
pub(crate) mod session_ext;

pub use config::Config;
pub(crate) use eval_common::*;
pub use loss::{LossReport, segmentation_loss};
pub use mode::Mode;
pub use model::Model;
pub use model_config::ModelConfig;
pub use parts::Parts;
pub use session::Session;

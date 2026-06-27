//! Head variant enum for [`Model`], selected by prompt mode.

use super::prompt_free_head::TrainablePromptFreeHead;
use crate::yoloe::head::lrpc::pyramid::Pyramid;
use crate::yoloe::segment::head::Head;

/// Head variant selected by [`crate::yoloe::segment::model::train_config::PromptMode`].
pub(crate) enum TrainableSegHead {
    /// Prompted `-seg` head (text/visual prompts); no LRPC.
    Prompted(Head),
    /// Prompt-free `-seg-pf` head with a 4585-class LRPC vocabulary.
    PromptFree(TrainablePromptFreeHead),
}

impl TrainableSegHead {
    /// Returns the prompt-free LRPC pyramid when this is the `-seg-pf` variant.
    pub(crate) fn prompt_free_lrpc(&self) -> Option<&Pyramid> {
        match self {
            Self::PromptFree(head) => Some(head.lrpc()),
            Self::Prompted(_) => None,
        }
    }
}

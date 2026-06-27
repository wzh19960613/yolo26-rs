//! YOLOE training mode aligned with official Ultralytics trainer families.

/// YOLOE training mode aligned with official Ultralytics trainer families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Fine-tune a pretrained YOLOE segmentation checkpoint using the official
    /// `YOLOEPESegTrainer` linear-probing recipe.
    FineTune,
    /// Linear-probe a pretrained YOLOE checkpoint with the official freeze recipe.
    LinearProbe,
    /// Train a text-prompt YOLOE segmentation model from scratch.
    FromScratch,
    /// Train the SAVPE visual-prompt module over a trained text-prompt model.
    Visual,
    /// Train the prompt-free classifier branch for LRPC checkpoints.
    PromptFree,
}

impl Mode {
    /// Returns the official Ultralytics trainer class for segmentation-first YOLOE.
    pub const fn official_segment_trainer(self) -> &'static str {
        match self {
            Self::FineTune | Self::LinearProbe => "YOLOEPESegTrainer",
            Self::FromScratch => "YOLOESegTrainerFromScratch",
            Self::Visual => "YOLOESegVPTrainer",
            Self::PromptFree => "YOLOEPEFreeTrainer",
        }
    }
}

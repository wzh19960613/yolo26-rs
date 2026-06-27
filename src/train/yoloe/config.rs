//! YOLOE training configuration.

use super::mode::Mode;
use super::parts::Parts;

/// YOLOE training configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Training mode.
    pub mode: Mode,
    /// Dataset supplies segmentation annotations.
    pub segment_annotations: bool,
    /// Trainable parameter family implied by the official recipe freeze list.
    pub trainable_parts: Parts,
    /// Whether the official recipe expects YOLO plus grounding data.
    pub requires_grounding_data: bool,
    /// Whether the official recipe uses `single_cls=True`.
    pub single_class: bool,
}

impl Config {
    /// Creates an official YOLOE fine-tuning config.
    ///
    /// Ultralytics documents custom YOLOE segmentation fine-tuning through
    /// `YOLOEPESegTrainer`, whose `get_model()` fuses prompt embeddings and
    /// re-enables gradients only on the final `cv3`/`one2one_cv3` projections.
    pub const fn fine_tune(segment_annotations: bool) -> Self {
        Self {
            mode: Mode::FineTune,
            segment_annotations,
            trainable_parts: Parts::ClassificationBranch,
            requires_grounding_data: false,
            single_class: false,
        }
    }

    /// Creates a linear-probing config.
    pub const fn linear_probe(segment_annotations: bool) -> Self {
        Self {
            mode: Mode::LinearProbe,
            segment_annotations,
            trainable_parts: Parts::ClassificationBranch,
            requires_grounding_data: false,
            single_class: false,
        }
    }

    /// Creates a from-scratch text-prompt config.
    pub const fn from_scratch(segment_annotations: bool) -> Self {
        Self {
            mode: Mode::FromScratch,
            segment_annotations,
            trainable_parts: Parts::All,
            requires_grounding_data: true,
            single_class: false,
        }
    }

    /// Creates a visual-prompt training config.
    pub const fn visual_prompt(segment_annotations: bool) -> Self {
        Self {
            mode: Mode::Visual,
            segment_annotations,
            trainable_parts: Parts::Savpe,
            requires_grounding_data: true,
            single_class: false,
        }
    }

    /// Creates a prompt-free training config.
    pub const fn prompt_free(segment_annotations: bool) -> Self {
        Self {
            mode: Mode::PromptFree,
            segment_annotations,
            trainable_parts: Parts::PromptFreeClassifier,
            requires_grounding_data: true,
            single_class: true,
        }
    }

    /// Returns the official Ultralytics trainer class represented by this config.
    pub const fn official_segment_trainer(&self) -> &'static str {
        self.mode.official_segment_trainer()
    }

    /// Validates official YOLOE training prerequisites for this crate's segment path.
    pub fn validate(&self) -> crate::Result<()> {
        if !self.segment_annotations {
            return Err(crate::Error::InvalidConfig(
                "official YOLOE training requires segmentation annotations".to_string(),
            ));
        }
        Ok(())
    }

    /// Returns true when a variable name is trainable under this official recipe.
    ///
    /// The names are the YOLOE-26 safetensors/VarMap names, e.g.
    /// `model.23.one2one_cv3.0.2.weight` or `model.23.savpe.cv1.0.0.conv.weight`.
    /// General user freeze rules and the always-frozen DFL layer should still be
    /// applied by the caller.
    pub fn allows_variable(&self, name: &str) -> bool {
        self.trainable_parts.allows_variable(name)
    }
}

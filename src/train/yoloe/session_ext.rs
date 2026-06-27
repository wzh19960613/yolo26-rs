//! Train-only extension methods on the inference YOLOE [`Session`].
//!
//! These methods are compiled only under `train` and extend the prompt
//! session with official YOLOE trainer-recipe selection. The ungated
//! validation methods stay in `yoloe::prompt::session_train`.

use crate::yoloe::prompt::session::Session;
use crate::yoloe::usage::BaseTask;

use super::{Config, Mode};

impl Session {
    /// Creates an official YOLOE training config for a segmentation-first trainer.
    pub fn recipe_config(&self, mode: Mode, segment_annotations: bool) -> crate::Result<Config> {
        if self.prompts.config.base_task() != BaseTask::Detect {
            return Err(crate::Error::Unsupported(
                "official YOLOE training uses segmentation-first checkpoints".to_string(),
            ));
        }
        if mode == Mode::Visual
            && (!self.prompts.config.visual_prompts || !self.prompts.config.savpe.enabled)
        {
            return Err(crate::Error::Unsupported(
                "official YOLOE visual-prompt training requires a SAVPE checkpoint".to_string(),
            ));
        }
        if mode == Mode::PromptFree
            && (!self.prompts.config.prompt_free || !self.prompts.config.lrpc)
        {
            return Err(crate::Error::Unsupported(
                "official YOLOE prompt-free training requires an LRPC prompt-free checkpoint"
                    .to_string(),
            ));
        }
        let config = match mode {
            Mode::FineTune => Config::fine_tune(segment_annotations),
            Mode::LinearProbe => Config::linear_probe(segment_annotations),
            Mode::FromScratch => Config::from_scratch(segment_annotations),
            Mode::Visual => Config::visual_prompt(segment_annotations),
            Mode::PromptFree => Config::prompt_free(segment_annotations),
        };
        config.validate()?;
        Ok(config)
    }
}

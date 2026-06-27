//! Native YOLOE segmentation training session: owns a [`Model`] and an
//! optimizer state, and exposes text/visual/prompt-free batch steps plus
//! checkpoint and optimizer-state helpers.

use std::path::Path;

use candle_core::{Tensor, Var};

use crate::yoloe::segment::model::train_config::PromptMode;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::BatchVisuals;

use super::loss::segmentation_loss;
use super::model::Model;
use super::{
    DetectionLossConfig, LossComponents, OptimizerConfig, OptimizerState, Report,
    SegmentationTargets, scalar_loss_value,
};

/// Native YOLOE segmentation train session.
///
/// The session owns a [`Model`] and reuses the same optimizer
/// grouping as the regular YOLO26 trainers. Text batches expect prompt
/// embeddings supplied by the caller, visual-prompt batches derive prompt
/// embeddings from SAVPE masks, and prompt-free batches use the internal LRPC
/// vocabulary without caller prompts.
pub struct Session {
    model: Model,
    optimizer: OptimizerState,
}

impl Session {
    /// Creates a YOLOE train session that updates all floating-point variables.
    pub fn new(model: Model, optimizer: OptimizerConfig) -> crate::Result<Self> {
        let vars = named_variables(&model)?;
        Self::from_named_variables(model, optimizer, vars)
    }

    /// Creates a YOLOE train session with an Ultralytics-style variable filter.
    pub fn new_with_variable_filter(
        model: Model,
        optimizer: OptimizerConfig,
        mut filter: impl FnMut(&str) -> bool,
    ) -> crate::Result<Self> {
        let vars = named_variables(&model)?
            .into_iter()
            .filter(|(name, _)| filter(name))
            .collect();
        Self::from_named_variables(model, optimizer, vars)
    }

    /// Returns the underlying YOLOE trainable model.
    pub const fn model(&self) -> &Model {
        &self.model
    }

    /// Returns the underlying YOLOE trainable model mutably.
    pub fn model_mut(&mut self) -> &mut Model {
        &mut self.model
    }

    /// Saves the current YOLOE weights to a safetensors checkpoint.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        self.model.save_safetensors(path.as_ref())
    }

    /// Saves the current YOLOE weights to an official `.pt` checkpoint that
    /// `torch.load` can read (requires the `pt` feature). Dispatches to
    /// [`super::Model::save_pt`].
    #[cfg(feature = "pt")]
    pub fn save_pt(&self, path: impl AsRef<Path>) -> crate::Result<()> {
        self.model.save_pt(path.as_ref())
    }

    /// Saves optimizer internal state to a safetensors sidecar.
    pub fn save_optimizer_state_safetensors(&self, path: impl AsRef<Path>) -> crate::Result<bool> {
        self.optimizer.save_state_safetensors(path)
    }

    /// Loads optimizer internal state from a safetensors sidecar.
    pub fn load_optimizer_state_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> crate::Result<()> {
        self.optimizer
            .load_state_safetensors(path, self.model.device())
    }

    /// Returns the current base optimizer learning rate.
    pub fn learning_rate(&self) -> f64 {
        self.optimizer.learning_rate()
    }

    /// Sets the current base optimizer learning rate.
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.optimizer.set_learning_rate(learning_rate);
    }

    /// Returns the optimizer momentum or Adam beta1 value, when applicable.
    pub fn momentum(&self) -> Option<f64> {
        self.optimizer.momentum()
    }

    /// Runs one text-prompt YOLOE segmentation training step.
    pub fn text_batch(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
    ) -> crate::Result<Report> {
        self.text_batch_with_loss_config(input, target, prompts, DetectionLossConfig::default())
    }

    /// Runs one text-prompt YOLOE step with custom loss gains.
    pub fn text_batch_with_loss_config(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<Report> {
        self.ensure_mode(PromptMode::TextPrompt, "text-prompt")?;
        let aligned = self.model.align_text_prompts(prompts)?;
        self.run_with_prompt_table(input, target, &aligned, loss_config)
    }

    /// Runs one visual-prompt YOLOE segmentation training step.
    pub fn visual_batch(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        visuals: &BatchVisuals,
        classes: Vec<String>,
    ) -> crate::Result<Report> {
        self.visual_batch_with_loss_config(
            input,
            target,
            visuals,
            classes,
            DetectionLossConfig::default(),
        )
    }

    /// Runs one visual-prompt YOLOE step with custom loss gains.
    pub fn visual_batch_with_loss_config(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        visuals: &BatchVisuals,
        classes: Vec<String>,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<Report> {
        self.ensure_mode(PromptMode::Visual, "visual-prompt")?;
        let prompts = self.model.encode_visual_prompts(input, visuals, classes)?;
        self.run_with_prompt_table(input, target, &prompts, loss_config)
    }

    /// Runs one prompt-free YOLOE segmentation training step.
    pub fn prompt_free_batch(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
    ) -> crate::Result<Report> {
        self.prompt_free_batch_with_loss_config(input, target, DetectionLossConfig::default())
    }

    /// Runs one prompt-free YOLOE step with custom loss gains.
    pub fn prompt_free_batch_with_loss_config(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<Report> {
        self.ensure_mode(PromptMode::PromptFree, "prompt-free")?;
        let output = self.model.forward_prompt_free_dense(input)?;
        let report = segmentation_loss(&output, target, &loss_config)?;
        self.step_loss(report.loss)
    }

    fn from_named_variables(
        model: Model,
        optimizer: OptimizerConfig,
        vars: Vec<(String, Var)>,
    ) -> crate::Result<Self> {
        if vars.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "YOLOE optimizer variable selection is empty".to_string(),
            ));
        }
        Ok(Self {
            model,
            optimizer: OptimizerState::new_grouped(vars, optimizer)?,
        })
    }

    fn run_with_prompt_table(
        &mut self,
        input: &Tensor,
        target: &SegmentationTargets,
        prompts: &EmbeddingTable,
        loss_config: DetectionLossConfig,
    ) -> crate::Result<Report> {
        let output = self.model.forward_dense(input, prompts)?;
        let report = segmentation_loss(&output, target, &loss_config)?;
        self.step_loss(report.loss)
    }

    fn step_loss(&mut self, loss: Tensor) -> crate::Result<Report> {
        let total = scalar_loss_value(&loss)?;
        self.optimizer.backward_step(&loss)?;
        Ok(Report {
            loss: total,
            components: LossComponents {
                total,
                ..Default::default()
            },
        })
    }

    fn ensure_mode(&self, expected: PromptMode, label: &str) -> crate::Result<()> {
        let actual = self.model.config().mode;
        if actual == expected {
            return Ok(());
        }
        Err(crate::Error::InvalidConfig(format!(
            "YOLOE {label} training requires mode {expected:?}, got {actual:?}"
        )))
    }
}

/// Collects the trainable variables of a YOLOE [`Model`] as `(name, Var)` pairs,
/// sorted by name for deterministic optimizer grouping.
fn named_variables(model: &Model) -> crate::Result<Vec<(String, Var)>> {
    let data = model.varmap.data();
    let guard = data.lock().map_err(|_| {
        crate::Error::InvalidConfig("YOLOE variable map lock was poisoned".to_string())
    })?;
    let mut vars: Vec<(String, Var)> = guard
        .iter()
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    vars.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(vars)
}

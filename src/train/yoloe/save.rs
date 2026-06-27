//! Identity save methods for [`Model`], split out so the main
//! model file stays under the per-file line cap.

use super::model::Model;
use crate::yoloe::segment::model::train_config::PromptMode;

impl Model {
    /// Saves all trainable variables to a safetensors file (YOLOE key layout).
    pub fn save_safetensors(&self, path: &std::path::Path) -> crate::Result<()> {
        Ok(self.varmap.save(path)?)
    }

    /// Saves current variables to an official `.pt` checkpoint that
    /// `torch.load` can read (requires the `pt` feature). The embedded
    /// `data.pkl` template for the model's prompt mode and scale is reused:
    /// [`PromptMode::PromptFree`] selects `yoloe-26*-seg-pf.pt`,
    /// otherwise `yoloe-26*-seg.pt`. Tensors absent from the model
    /// (`num_batches_tracked`, the prototype semantic head) are zero-filled.
    #[cfg(feature = "pt")]
    pub fn save_pt(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let tensors = self
            .variables()?
            .into_iter()
            .map(|(name, var)| (name, var.as_tensor().clone()))
            .collect();
        crate::pt_loader::save_pt(path, &tensors, self.task_str(), self.config.scale)
    }

    /// Returns the `.pt` template task id for the active prompt mode:
    /// `yoloe_seg` for text/visual prompts, `yoloe_seg_pf` for prompt-free.
    pub(crate) fn task_str(&self) -> &'static str {
        match self.config.mode {
            PromptMode::PromptFree => "yoloe_seg_pf",
            PromptMode::TextPrompt | PromptMode::Visual => "yoloe_seg",
        }
    }
}

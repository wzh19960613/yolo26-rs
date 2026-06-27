use std::path::Path;

use super::Session;

impl Session {
    /// Saves optimizer internal state to a safetensors sidecar.
    ///
    /// Returns `true` when the optimizer has state tensors and a file was
    /// written. Optimizers without internal state, such as plain SGD, return
    /// `false`.
    pub fn save_optimizer_state_safetensors(&self, path: impl AsRef<Path>) -> crate::Result<bool> {
        self.optimizer.save_state_safetensors(path)
    }

    /// Loads optimizer internal state from a safetensors sidecar.
    ///
    /// The sidecar must match the current optimizer kind, parameter grouping,
    /// and variable shapes.
    pub fn load_optimizer_state_safetensors(
        &mut self,
        path: impl AsRef<Path>,
    ) -> crate::Result<()> {
        let device = self.model().device().clone();
        self.optimizer.load_state_safetensors(path, &device)
    }
}

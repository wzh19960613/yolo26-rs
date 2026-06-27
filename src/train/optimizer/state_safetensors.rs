use std::collections::HashMap;
use std::path::Path;

use candle_core::{Device, Tensor};

use super::OptimizerState;

impl OptimizerState {
    pub(crate) fn save_state_safetensors(&self, path: impl AsRef<Path>) -> crate::Result<bool> {
        let tensors = self.state_tensors()?;
        if tensors.is_empty() {
            return Ok(false);
        }
        candle_core::safetensors::save(&tensors, path)?;
        Ok(true)
    }

    pub(crate) fn load_state_safetensors(
        &mut self,
        path: impl AsRef<Path>,
        device: &Device,
    ) -> crate::Result<()> {
        let tensors = candle_core::safetensors::load(path, device)?;
        self.load_state_tensors(&tensors)
    }

    fn state_tensors(&self) -> crate::Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();
        match self {
            Self::Single(opt) => {
                tensors.extend(opt.state_tensors("single")?);
            }
            Self::Grouped(groups) => {
                for (index, group) in groups.iter().enumerate() {
                    tensors.extend(group.optimizer.state_tensors(&format!("group{index}"))?);
                }
            }
        }
        Ok(tensors)
    }

    fn load_state_tensors(&mut self, tensors: &HashMap<String, Tensor>) -> crate::Result<()> {
        match self {
            Self::Single(opt) => opt.load_state_tensors("single", tensors),
            Self::Grouped(groups) => {
                for (index, group) in groups.iter_mut().enumerate() {
                    group
                        .optimizer
                        .load_state_tensors(&format!("group{index}"), tensors)?;
                }
                Ok(())
            }
        }
    }
}

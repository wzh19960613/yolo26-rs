use super::*;

impl Session {
    /// Creates a new train session.
    pub fn new(model: Model, optimizer: OptimizerConfig) -> crate::Result<Self> {
        let vars = model.named_variables()?;
        Self::from_named_variables(model, optimizer, vars)
    }
    /// Creates a train session whose optimizer only updates variables matching a name filter.
    pub fn new_with_variable_filter(
        model: Model,
        optimizer: OptimizerConfig,
        mut filter: impl FnMut(&str) -> bool,
    ) -> crate::Result<Self> {
        let vars = model
            .named_variables()?
            .into_iter()
            .filter(|(name, _)| filter(name))
            .collect();
        Self::from_named_variables(model, optimizer, vars)
    }
    fn from_named_variables(
        model: Model,
        optimizer: OptimizerConfig,
        vars: Vec<(String, Var)>,
    ) -> crate::Result<Self> {
        if vars.is_empty() {
            return Err(crate::Error::InvalidConfig(
                "optimizer variable selection is empty".to_string(),
            ));
        }
        Ok(Self {
            model,
            optimizer: OptimizerState::new_grouped(vars, optimizer)?,
            ema: None,
        })
    }
    /// Returns the trainable model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    pub(crate) fn with_ema_weights<T>(
        &self,
        f: impl FnOnce(&Self) -> crate::Result<T>,
    ) -> crate::Result<T> {
        let Some(ema) = &self.ema else {
            return f(self);
        };
        let named = self.model.named_variables()?;
        let live = named
            .iter()
            .map(|(name, var)| {
                Ok((
                    name.clone(),
                    crate::train::session::ema::tensor_snapshot(var.as_tensor())?,
                ))
            })
            .collect::<crate::Result<Vec<_>>>()?;
        ema.copy_to(&named)?;
        let result = f(self);
        let restore = restore_live_weights(&named, &live);
        match (result, restore) {
            (Ok(value), Ok(())) => Ok(value),
            (Err(err), Ok(())) => Err(err),
            (Ok(_), Err(err)) | (Err(_), Err(err)) => Err(err),
        }
    }

    /// Saves the checkpoint weights: the EMA shadow when EMA is active (so the
    /// saved best/last reflect the averaged weights, matching the official
    /// trainer), otherwise the live model variables.
    ///
    /// When the destination path ends with `.pt`, the weights are written as an
    /// official PyTorch checkpoint (requires the `pt` feature), reusing the
    /// embedded `data.pkl` template for the model's task and scale. Otherwise
    /// `.safetensors` is written.
    pub fn save_checkpoint_weights(&self, path: impl AsRef<std::path::Path>) -> crate::Result<()> {
        let path = path.as_ref();

        // Collect the weight tensors to save (EMA or live model).
        let tensors: std::collections::HashMap<String, candle_core::Tensor> =
            if let Some(ema) = &self.ema {
                ema.tensors()
                    .iter()
                    .map(|(name, tensor)| (name.clone(), tensor.clone()))
                    .collect()
            } else {
                let data = self.model.varmap.data().lock().map_err(|_| {
                    crate::Error::InvalidConfig("failed to lock trainable variable map".to_string())
                })?;
                data.iter()
                    .map(|(name, var)| (name.clone(), var.as_tensor().clone()))
                    .collect()
            };

        // Dispatch by extension: .pt → official checkpoint, else safetensors.
        #[cfg(feature = "pt")]
        if path.extension().and_then(|e| e.to_str()) == Some("pt") {
            let metadata = self.model.class_metadata();
            crate::pt_loader::save_pt_with_class_metadata(
                path,
                &tensors,
                self.model.task_str(),
                self.model.scale(),
                Some(&metadata),
            )?;
            return Ok(());
        }

        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }
}

fn restore_live_weights(
    named: &[(String, Var)],
    live: &[(String, candle_core::Tensor)],
) -> crate::Result<()> {
    for ((expected_name, tensor), (name, var)) in live.iter().zip(named) {
        if name != expected_name {
            return Err(crate::Error::InvalidConfig(format!(
                "failed to restore EMA evaluation weights: expected {expected_name}, got {name}"
            )));
        }
        var.set(tensor)?;
    }
    Ok(())
}

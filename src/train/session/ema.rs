//! Exponential moving average (EMA) of model weights.
//!
//! The official trainer maintains a `ModelEMA` shadow that is updated after
//! every optimizer step (`shadow = decay*shadow + (1-decay)*live`) and used for
//! validation, best-checkpoint selection and the final saved weights. This
//! module owns the shadow tensors and the per-step lerp; the training loop
//! creates/updates it and the checkpoint path saves the shadow when EMA is on.

use candle_core::{Tensor, Var};

/// Shadow copy of trainable weights kept in sync by exponential moving average.
pub(crate) struct ModelEma {
    max_decay: f32,
    tau: f32,
    updates: usize,
    shadow: Vec<(String, Tensor)>,
}

impl ModelEma {
    /// Creates an EMA whose shadow starts at the current live weights.
    pub(crate) fn new(max_decay: f32, named: &[(String, Var)]) -> crate::Result<Self> {
        if !max_decay.is_finite() || !(0.0..=1.0).contains(&max_decay) {
            return Err(crate::Error::InvalidConfig(format!(
                "EMA decay must be in [0, 1], got {max_decay}"
            )));
        }
        let shadow = named
            .iter()
            .map(|(name, var)| Ok((name.clone(), tensor_snapshot(var.as_tensor())?)))
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            max_decay,
            tau: 2000.0,
            updates: 0,
            shadow,
        })
    }

    /// Moves each shadow one EMA step toward the current live weights.
    ///
    /// `named` must use the same names in the same order as [`Self::new`].
    pub(crate) fn update(&mut self, named: &[(String, Var)]) -> crate::Result<()> {
        self.updates += 1;
        let decay = self.current_decay();
        for (i, (_name, var)) in named.iter().enumerate() {
            let live = var.as_tensor();
            let prev = &self.shadow[i].1;
            let lerped = ((prev * decay as f64)? + (live * (1.0 - decay as f64))?)?;
            self.shadow[i].1 = tensor_snapshot(&lerped)?;
        }
        Ok(())
    }

    /// Copies EMA shadow weights into live trainable variables.
    pub(crate) fn copy_to(&self, named: &[(String, Var)]) -> crate::Result<()> {
        if named.len() != self.shadow.len() {
            return Err(crate::Error::InvalidConfig(format!(
                "EMA variable count mismatch: expected {}, got {}",
                self.shadow.len(),
                named.len()
            )));
        }
        for ((expected_name, tensor), (name, var)) in self.shadow.iter().zip(named) {
            if name != expected_name {
                return Err(crate::Error::InvalidConfig(format!(
                    "EMA variable order mismatch: expected {expected_name}, got {name}"
                )));
            }
            var.set(tensor)?;
        }
        Ok(())
    }

    /// Returns all shadow tensors as `(name, tensor)` for serialization.
    pub(crate) fn tensors(&self) -> &[(String, Tensor)] {
        &self.shadow
    }

    fn current_decay(&self) -> f32 {
        self.max_decay * (1.0 - (-(self.updates as f32) / self.tau).exp())
    }
}

pub(crate) fn tensor_snapshot(tensor: &Tensor) -> crate::Result<Tensor> {
    Ok(tensor.affine(1.0, 0.0)?.detach())
}

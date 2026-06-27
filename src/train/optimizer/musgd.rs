use std::collections::HashMap;

use candle_core::{Tensor, Var, backprop::GradStore};

use crate::train::optimizer::musgd_math::{muon_update, sgd_update};

/// Parameters for the native MuSGD optimizer.
#[derive(Clone, Debug)]
pub struct ParamsMuSgd {
    /// Learning rate.
    pub lr: f64,
    /// Momentum factor used by both Muon and SGD branches.
    pub momentum: f64,
    /// Weight decay applied to the SGD branch.
    pub weight_decay: f64,
    /// Whether to use Nesterov momentum.
    pub nesterov: bool,
    /// Weight of the Muon update in hybrid parameter groups.
    pub muon: f64,
    /// Weight of the SGD update in hybrid parameter groups.
    pub sgd: f64,
}

impl Default for ParamsMuSgd {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            momentum: 0.9,
            weight_decay: 0.0,
            nesterov: true,
            muon: 0.5,
            sgd: 0.5,
        }
    }
}

pub(crate) struct MuSgdOptimizer {
    vars: Vec<MuSgdVar>,
    params: ParamsMuSgd,
}

struct MuSgdVar {
    name: String,
    var: Var,
    momentum: Var,
    sgd_momentum: Option<Var>,
    use_muon: bool,
}

impl MuSgdOptimizer {
    pub(crate) fn new_named(vars: Vec<(String, Var)>, params: ParamsMuSgd) -> crate::Result<Self> {
        validate_params(&params)?;
        let vars = vars
            .into_iter()
            .filter(|(_, var)| var.dtype().is_float())
            .map(|(name, var)| {
                let use_muon = matches!(var.dims().len(), 2 | 4);
                let momentum = Var::from_tensor(&var.zeros_like()?)?;
                let sgd_momentum = use_muon
                    .then(|| Var::from_tensor(&var.zeros_like()?))
                    .transpose()?;
                Ok(MuSgdVar {
                    name,
                    var,
                    momentum,
                    sgd_momentum,
                    use_muon,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self { vars, params })
    }

    pub(crate) fn step(&mut self, grads: &GradStore) -> crate::Result<()> {
        for var in self.vars.iter() {
            let Some(grad) = grads.get(&var.var) else {
                continue;
            };
            if var.use_muon {
                self.step_hybrid(var, grad)?;
            } else {
                self.step_sgd(var, grad, self.params.lr)?;
            }
        }
        Ok(())
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    pub(crate) fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub(crate) fn momentum(&self) -> f64 {
        self.params.momentum
    }

    pub(crate) fn set_momentum(&mut self, momentum: f64) {
        self.params.momentum = momentum;
    }

    pub(crate) fn state_tensors(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut tensors = Vec::new();
        for var in &self.vars {
            let key = format!("{prefix}.musgd.var.{}", var.name);
            tensors.push((format!("{key}.momentum"), var.momentum.as_tensor().clone()));
            if let Some(momentum) = var.sgd_momentum.as_ref() {
                tensors.push((format!("{key}.sgd_momentum"), momentum.as_tensor().clone()));
            }
        }
        tensors
    }

    pub(crate) fn load_state_tensors(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
    ) -> crate::Result<()> {
        for var in &self.vars {
            let key = format!("{prefix}.musgd.var.{}", var.name);
            set_state(&var.momentum, tensors, &format!("{key}.momentum"))?;
            if let Some(momentum) = var.sgd_momentum.as_ref() {
                set_state(momentum, tensors, &format!("{key}.sgd_momentum"))?;
            }
        }
        Ok(())
    }

    fn step_hybrid(&self, var: &MuSgdVar, grad: &Tensor) -> crate::Result<()> {
        let muon_update = muon_update(grad, var.var.as_tensor(), &var.momentum, &self.params)?;
        let muon_next =
            (var.var.as_tensor() - (muon_update * (self.params.lr * self.params.muon))?)?;
        var.var.set(&muon_next)?;
        let sgd_momentum = var.sgd_momentum.as_ref().ok_or_else(|| {
            crate::Error::InvalidConfig("MuSGD hybrid state is missing SGD momentum".to_string())
        })?;
        let sgd_update = sgd_update(grad, var.var.as_tensor(), sgd_momentum, &self.params)?;
        let next = (var.var.as_tensor() - (sgd_update * (self.params.lr * self.params.sgd))?)?;
        var.var.set(&next)?;
        Ok(())
    }

    fn step_sgd(&self, var: &MuSgdVar, grad: &Tensor, lr: f64) -> crate::Result<()> {
        let update = sgd_update(grad, var.var.as_tensor(), &var.momentum, &self.params)?;
        let next = (var.var.as_tensor() - (update * lr)?)?;
        var.var.set(&next)?;
        Ok(())
    }
}

fn set_state(var: &Var, tensors: &HashMap<String, Tensor>, key: &str) -> crate::Result<()> {
    let tensor = tensors
        .get(key)
        .ok_or_else(|| crate::Error::InvalidConfig(format!("missing optimizer state {key}")))?;
    if tensor.shape() != var.shape() {
        return Err(crate::Error::InvalidTensor(format!(
            "optimizer state {key} shape mismatch: expected {:?}, got {:?}",
            var.dims(),
            tensor.dims()
        )));
    }
    var.set(&tensor.to_dtype(var.dtype())?)?;
    Ok(())
}

fn validate_params(params: &ParamsMuSgd) -> crate::Result<()> {
    for (name, value) in [
        ("lr", params.lr),
        ("momentum", params.momentum),
        ("weight_decay", params.weight_decay),
        ("muon", params.muon),
        ("sgd", params.sgd),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(crate::Error::InvalidConfig(format!(
                "MuSGD {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

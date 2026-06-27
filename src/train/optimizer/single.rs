use std::collections::HashMap;

use candle_core::{Tensor, Var, backprop::GradStore};
use candle_nn::Optimizer;

use super::{AdamWOptimizer, MuSgdOptimizer, OptimizerConfig};

pub(crate) enum SingleOptimizer {
    Sgd(candle_nn::SGD),
    AdamW(AdamWOptimizer),
    MuSgd(MuSgdOptimizer),
}

impl SingleOptimizer {
    pub(crate) fn new_named(
        vars: Vec<(String, Var)>,
        config: OptimizerConfig,
    ) -> crate::Result<Self> {
        match config {
            OptimizerConfig::Sgd { learning_rate } => {
                let vars = vars.into_iter().map(|(_, var)| var).collect();
                Ok(Self::Sgd(candle_nn::SGD::new(vars, learning_rate)?))
            }
            OptimizerConfig::AdamW { params } => {
                Ok(Self::AdamW(AdamWOptimizer::new_named(vars, params)?))
            }
            OptimizerConfig::MuSgd { params } => {
                Ok(Self::MuSgd(MuSgdOptimizer::new_named(vars, params)?))
            }
        }
    }

    pub(crate) fn step(&mut self, grads: &GradStore) -> crate::Result<()> {
        match self {
            Self::Sgd(opt) => Ok(opt.step(grads)?),
            Self::AdamW(opt) => Ok(opt.step(grads)?),
            Self::MuSgd(opt) => opt.step(grads),
        }
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        match self {
            Self::Sgd(opt) => opt.learning_rate(),
            Self::AdamW(opt) => opt.learning_rate(),
            Self::MuSgd(opt) => opt.learning_rate(),
        }
    }

    pub(crate) fn set_learning_rate(&mut self, learning_rate: f64) {
        match self {
            Self::Sgd(opt) => opt.set_learning_rate(learning_rate),
            Self::AdamW(opt) => opt.set_learning_rate(learning_rate),
            Self::MuSgd(opt) => opt.set_learning_rate(learning_rate),
        }
    }

    pub(crate) fn momentum(&self) -> Option<f64> {
        match self {
            Self::Sgd(_) => None,
            // PyTorch AdamW stores beta1 in the `betas` param-group field,
            // not in `momentum`; Ultralytics only warms groups containing a
            // `momentum` key, so AdamW beta1 stays fixed during warmup.
            Self::AdamW(_) => None,
            Self::MuSgd(opt) => Some(opt.momentum()),
        }
    }

    pub(crate) fn set_momentum(&mut self, momentum: f64) {
        match self {
            Self::Sgd(_) => {}
            Self::AdamW(_) => {}
            Self::MuSgd(opt) => opt.set_momentum(momentum),
        }
    }

    pub(crate) fn state_tensors(&self, prefix: &str) -> crate::Result<Vec<(String, Tensor)>> {
        match self {
            Self::Sgd(_) => Ok(Vec::new()),
            Self::AdamW(opt) => opt.state_tensors(prefix),
            Self::MuSgd(opt) => Ok(opt.state_tensors(prefix)),
        }
    }

    pub(crate) fn load_state_tensors(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
    ) -> crate::Result<()> {
        match self {
            Self::Sgd(_) => Ok(()),
            Self::AdamW(opt) => opt.load_state_tensors(prefix, tensors),
            Self::MuSgd(opt) => opt.load_state_tensors(prefix, tensors),
        }
    }
}

pub(crate) fn no_decay_config(mut config: OptimizerConfig) -> OptimizerConfig {
    match &mut config {
        OptimizerConfig::Sgd { .. } => {}
        OptimizerConfig::AdamW { params } => params.weight_decay = 0.0,
        OptimizerConfig::MuSgd { params } => params.weight_decay = 0.0,
    }
    config
}

pub(crate) fn scale_learning_rate(mut config: OptimizerConfig, scale: f64) -> OptimizerConfig {
    match &mut config {
        OptimizerConfig::Sgd { learning_rate } => *learning_rate *= scale,
        OptimizerConfig::AdamW { params } => params.lr *= scale,
        OptimizerConfig::MuSgd { params } => params.lr *= scale,
    }
    config
}

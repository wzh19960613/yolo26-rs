use crate::model::ImageSize;
pub(crate) use crate::train::exports::*;

pub(crate) mod adamw;
pub(crate) mod musgd;
pub(crate) mod musgd_math;
pub(crate) mod single;
pub(crate) mod state;
pub(crate) mod state_safetensors;
pub(crate) mod var_groups;

pub(crate) use adamw::*;
pub use musgd::*;
pub(crate) use single::*;
pub(crate) use state::*;
pub(crate) use state_safetensors::*;
pub(crate) use var_groups::*;

use candle_nn::ParamsAdamW;

use musgd::ParamsMuSgd;

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub enum OptimizerConfig {
    /// Stochastic gradient descent without momentum.
    Sgd {
        /// Learning rate.
        learning_rate: f64,
    },
    /// AdamW optimizer.
    AdamW {
        /// AdamW parameters.
        params: ParamsAdamW,
    },
    /// Official MuSGD optimizer.
    MuSgd {
        /// MuSGD parameters.
        params: ParamsMuSgd,
    },
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self::AdamW {
            params: ParamsAdamW::default(),
        }
    }
}

/// Official optimizer family selected by Ultralytics-style configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OfficialOptimizer {
    /// Stochastic gradient descent.
    Sgd,
    /// AdamW optimizer.
    AdamW,
    /// Official MuSGD optimizer with Muon-style updates.
    MuSgd,
}

/// Result of resolving Ultralytics `optimizer=auto`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AutoOptimizerSelection {
    /// Optimizer selected by the official heuristic.
    pub optimizer: OfficialOptimizer,
    /// Learning rate selected by the official heuristic.
    pub learning_rate: f64,
    /// Momentum or Adam beta1 selected by the official heuristic.
    pub momentum: f64,
    /// Warmup bias learning rate selected by the official heuristic.
    pub warmup_bias_lr: f64,
}

impl AutoOptimizerSelection {
    /// Resolves Ultralytics `optimizer=auto` for a class count and iteration count.
    pub fn ultralytics(classes_count: usize, iterations: usize) -> crate::Result<Self> {
        if classes_count == 0 {
            return Err(crate::Error::InvalidConfig(
                "optimizer=auto requires at least one class".to_string(),
            ));
        }
        if iterations > 10_000 {
            Ok(Self {
                optimizer: OfficialOptimizer::MuSgd,
                learning_rate: 0.01,
                momentum: 0.9,
                warmup_bias_lr: 0.0,
            })
        } else {
            Ok(Self {
                optimizer: OfficialOptimizer::AdamW,
                learning_rate: (0.002 * 5.0 / (4 + classes_count) as f64 * 1_000_000.0).round()
                    / 1_000_000.0,
                momentum: 0.9,
                // Ultralytics resets `warmup_bias_lr` to 0.0 when
                // `optimizer=auto` selects AdamW so all groups warm up from
                // zero together.
                warmup_bias_lr: 0.0,
            })
        }
    }

    /// Converts a supported selection into this crate's optimizer config.
    pub fn optimizer_config(
        self,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> crate::Result<OptimizerConfig> {
        match self.optimizer {
            OfficialOptimizer::AdamW => Ok(OptimizerConfig::AdamW {
                params: ParamsAdamW {
                    lr: self.learning_rate,
                    beta1: self.momentum,
                    beta2,
                    eps,
                    weight_decay,
                },
            }),
            OfficialOptimizer::Sgd => Ok(OptimizerConfig::Sgd {
                learning_rate: self.learning_rate,
            }),
            OfficialOptimizer::MuSgd => Ok(OptimizerConfig::MuSgd {
                params: ParamsMuSgd {
                    lr: self.learning_rate,
                    momentum: self.momentum,
                    weight_decay,
                    nesterov: true,
                    ..Default::default()
                },
            }),
        }
    }
}

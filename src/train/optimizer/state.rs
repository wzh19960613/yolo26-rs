use candle_core::{Tensor, Var, backprop::GradStore};

use super::OptimizerConfig;
use super::RunnerConfig;
use crate::train::optimizer::single::{SingleOptimizer, no_decay_config, scale_learning_rate};
use crate::train::optimizer::var_groups::{OptimizerGroupRole, optimizer_var_groups};

pub(crate) enum OptimizerState {
    Single(SingleOptimizer),
    Grouped(Vec<OptimizerGroup>),
}

pub(crate) struct OptimizerGroup {
    role: OptimizerGroupRole,
    lr_scale: f64,
    pub(crate) optimizer: SingleOptimizer,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OptimizerStepSettings {
    pub(crate) learning_rate: f64,
    pub(crate) momentum: Option<f64>,
}

impl OptimizerState {
    pub(crate) fn new_grouped(
        named_vars: Vec<(String, Var)>,
        config: OptimizerConfig,
    ) -> crate::Result<Self> {
        let use_high_lr = matches!(config, OptimizerConfig::MuSgd { .. });
        let groups = optimizer_var_groups(named_vars, use_high_lr);
        if groups.len() == 1 && groups[0].role == OptimizerGroupRole::Main {
            return Self::new(
                groups
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        crate::Error::InvalidConfig(
                            "optimizer var group unexpectedly empty".to_string(),
                        )
                    })?
                    .vars,
                config,
            );
        }
        groups
            .into_iter()
            .map(|group| {
                OptimizerGroup::new(group.role, group.lr_scale, group.vars, config.clone())
            })
            .collect::<crate::Result<Vec<_>>>()
            .map(Self::Grouped)
    }

    pub(crate) fn new(vars: Vec<(String, Var)>, config: OptimizerConfig) -> crate::Result<Self> {
        Ok(Self::Single(SingleOptimizer::new_named(vars, config)?))
    }

    pub(crate) fn backward_step(&mut self, loss: &Tensor) -> crate::Result<()> {
        let grads = loss.backward()?;
        self.step(&grads)
    }

    pub(crate) fn step_with_grads(&mut self, grads: &GradStore) -> crate::Result<()> {
        self.step(grads)
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        match self {
            Self::Single(opt) => opt.learning_rate(),
            Self::Grouped(groups) => groups[0].base_learning_rate(),
        }
    }

    pub(crate) fn group_learning_rates(&self) -> Vec<f64> {
        match self {
            Self::Single(opt) => vec![opt.learning_rate()],
            Self::Grouped(groups) => groups
                .iter()
                .map(|group| group.optimizer.learning_rate())
                .collect(),
        }
    }

    pub(crate) fn bias_learning_rate(&self) -> Option<f64> {
        self.groups()
            .find(|group| group.role == OptimizerGroupRole::Bias)
            .map(OptimizerGroup::base_learning_rate)
    }

    pub(crate) fn set_learning_rate(&mut self, learning_rate: f64) {
        match self {
            Self::Single(opt) => opt.set_learning_rate(learning_rate),
            Self::Grouped(groups) => {
                for group in groups {
                    group.set_base_learning_rate(learning_rate);
                }
            }
        }
    }

    pub(crate) fn set_bias_learning_rate(&mut self, learning_rate: f64) {
        if let Self::Grouped(groups) = self {
            for group in groups
                .iter_mut()
                .filter(|group| group.role == OptimizerGroupRole::Bias)
            {
                group.set_base_learning_rate(learning_rate);
            }
        }
    }

    pub(crate) fn momentum(&self) -> Option<f64> {
        match self {
            Self::Single(opt) => opt.momentum(),
            Self::Grouped(groups) => groups[0].optimizer.momentum(),
        }
    }

    pub(crate) fn set_momentum(&mut self, momentum: f64) {
        match self {
            Self::Single(opt) => opt.set_momentum(momentum),
            Self::Grouped(groups) => {
                for group in groups {
                    group.optimizer.set_momentum(momentum);
                }
            }
        }
    }

    pub(crate) fn apply_step_settings(
        &mut self,
        config: &RunnerConfig,
        step: usize,
        warmup_steps: usize,
        target_lr: f64,
    ) -> crate::Result<OptimizerStepSettings> {
        let learning_rate = config.step_learning_rate(step, warmup_steps, target_lr)?;
        let bias_learning_rate = config.step_bias_learning_rate(
            step,
            warmup_steps,
            target_lr,
            self.bias_learning_rate(),
        )?;
        let momentum = config.step_momentum(step, warmup_steps, self.momentum())?;
        self.set_learning_rate(learning_rate);
        if let Some(learning_rate) = bias_learning_rate {
            self.set_bias_learning_rate(learning_rate);
        }
        if let Some(momentum) = momentum {
            self.set_momentum(momentum);
        }
        Ok(OptimizerStepSettings {
            learning_rate,
            momentum,
        })
    }

    fn step(&mut self, grads: &GradStore) -> crate::Result<()> {
        match self {
            Self::Single(opt) => opt.step(grads),
            Self::Grouped(groups) => {
                for group in groups {
                    group.optimizer.step(grads)?;
                }
                Ok(())
            }
        }
    }

    fn groups(&self) -> impl Iterator<Item = &OptimizerGroup> {
        match self {
            Self::Single(_) => [].iter(),
            Self::Grouped(groups) => groups.iter(),
        }
    }
}

impl OptimizerGroup {
    fn new(
        role: OptimizerGroupRole,
        lr_scale: f64,
        vars: Vec<(String, Var)>,
        config: OptimizerConfig,
    ) -> crate::Result<Self> {
        let config = if role == OptimizerGroupRole::Main {
            config
        } else {
            no_decay_config(config)
        };
        Ok(Self {
            role,
            lr_scale,
            optimizer: SingleOptimizer::new_named(vars, scale_learning_rate(config, lr_scale))?,
        })
    }

    fn base_learning_rate(&self) -> f64 {
        self.optimizer.learning_rate() / self.lr_scale
    }

    fn set_base_learning_rate(&mut self, learning_rate: f64) {
        self.optimizer
            .set_learning_rate(learning_rate * self.lr_scale);
    }
}

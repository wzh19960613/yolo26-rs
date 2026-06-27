use std::collections::HashMap;

use candle_core::{Device, Tensor, Var, backprop::GradStore};
use candle_nn::ParamsAdamW;

pub(crate) struct AdamWOptimizer {
    vars: Vec<AdamWVar>,
    step_t: usize,
    params: ParamsAdamW,
}

struct AdamWVar {
    name: String,
    var: Var,
    first_moment: Var,
    second_moment: Var,
}

impl AdamWOptimizer {
    pub(crate) fn new_named(vars: Vec<(String, Var)>, params: ParamsAdamW) -> crate::Result<Self> {
        validate_params(&params)?;
        let vars = vars
            .into_iter()
            .filter(|(_, var)| var.dtype().is_float())
            .map(|(name, var)| {
                let first_moment = Var::from_tensor(&var.zeros_like()?)?;
                let second_moment = Var::from_tensor(&var.zeros_like()?)?;
                Ok(AdamWVar {
                    name,
                    var,
                    first_moment,
                    second_moment,
                })
            })
            .collect::<crate::Result<Vec<_>>>()?;
        Ok(Self {
            vars,
            step_t: 0,
            params,
        })
    }

    pub(crate) fn step(&mut self, grads: &GradStore) -> crate::Result<()> {
        self.step_t += 1;
        let scale_m = 1.0 / (1.0 - self.params.beta1.powi(self.step_t as i32));
        let scale_v = 1.0 / (1.0 - self.params.beta2.powi(self.step_t as i32));
        for var in self.vars.iter() {
            let Some(grad) = grads.get(&var.var) else {
                continue;
            };
            self.step_var(var, grad, scale_m, scale_v)?;
        }
        Ok(())
    }

    pub(crate) fn learning_rate(&self) -> f64 {
        self.params.lr
    }

    pub(crate) fn set_learning_rate(&mut self, lr: f64) {
        self.params.lr = lr;
    }

    pub(crate) fn params(&self) -> &ParamsAdamW {
        &self.params
    }

    pub(crate) fn set_params(&mut self, params: ParamsAdamW) {
        self.params = params;
    }

    pub(crate) fn state_tensors(&self, prefix: &str) -> crate::Result<Vec<(String, Tensor)>> {
        let step_t = u32::try_from(self.step_t).map_err(|_| {
            crate::Error::InvalidConfig("AdamW step count exceeds u32 sidecar range".to_string())
        })?;
        let mut tensors = vec![(
            format!("{prefix}.adamw.step_t"),
            Tensor::from_vec(vec![step_t], (1,), &Device::Cpu)?,
        )];
        for var in &self.vars {
            let key = format!("{prefix}.adamw.var.{}", var.name);
            tensors.push((
                format!("{key}.first_moment"),
                var.first_moment.as_tensor().clone(),
            ));
            tensors.push((
                format!("{key}.second_moment"),
                var.second_moment.as_tensor().clone(),
            ));
        }
        Ok(tensors)
    }

    pub(crate) fn load_state_tensors(
        &mut self,
        prefix: &str,
        tensors: &HashMap<String, Tensor>,
    ) -> crate::Result<()> {
        self.step_t = read_step_t(tensors, &format!("{prefix}.adamw.step_t"))?;
        for var in &self.vars {
            let key = format!("{prefix}.adamw.var.{}", var.name);
            set_state(&var.first_moment, tensors, &format!("{key}.first_moment"))?;
            set_state(&var.second_moment, tensors, &format!("{key}.second_moment"))?;
        }
        Ok(())
    }

    fn step_var(
        &self,
        var: &AdamWVar,
        grad: &Tensor,
        scale_m: f64,
        scale_v: f64,
    ) -> crate::Result<()> {
        let m = var.first_moment.as_tensor();
        let v = var.second_moment.as_tensor();
        let next_m = ((m * self.params.beta1)? + (grad * (1.0 - self.params.beta1))?)?;
        let next_v = ((v * self.params.beta2)? + (grad.sqr()? * (1.0 - self.params.beta2))?)?;
        let m_hat = (&next_m * scale_m)?;
        let v_hat = (&next_v * scale_v)?;
        let decay = self.params.lr * self.params.weight_decay;
        let next = (var.var.as_tensor() * (1.0 - decay))?;
        let adjusted = (m_hat / (v_hat.sqrt()? + self.params.eps)?)?;
        let next = (next - (adjusted * self.params.lr)?)?;
        var.first_moment.set(&next_m)?;
        var.second_moment.set(&next_v)?;
        var.var.set(&next)?;
        Ok(())
    }
}

fn read_step_t(tensors: &HashMap<String, Tensor>, key: &str) -> crate::Result<usize> {
    let tensor = tensors
        .get(key)
        .ok_or_else(|| crate::Error::InvalidConfig(format!("missing optimizer state {key}")))?;
    if tensor.dims() != [1] {
        return Err(crate::Error::InvalidTensor(format!(
            "optimizer state {key} must have shape [1], got {:?}",
            tensor.dims()
        )));
    }
    let value = tensor.to_vec1::<u32>()?[0];
    Ok(value as usize)
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

fn validate_params(params: &ParamsAdamW) -> crate::Result<()> {
    for (name, value) in [
        ("lr", params.lr),
        ("beta1", params.beta1),
        ("beta2", params.beta2),
        ("eps", params.eps),
        ("weight_decay", params.weight_decay),
    ] {
        if !value.is_finite() || value < 0.0 {
            return Err(crate::Error::InvalidConfig(format!(
                "AdamW {name} must be finite and non-negative"
            )));
        }
    }
    Ok(())
}

use candle_core::{Tensor, Var};

use super::ParamsMuSgd;

pub(crate) fn sgd_update(
    grad: &Tensor,
    var: &Tensor,
    momentum: &Var,
    params: &ParamsMuSgd,
) -> crate::Result<Tensor> {
    let grad = if params.weight_decay == 0.0 {
        grad.clone()
    } else {
        (grad + &(var * params.weight_decay)?)?
    };
    let next_momentum = ((momentum.as_tensor() * params.momentum)? + &grad)?;
    momentum.set(&next_momentum)?;
    if params.nesterov {
        Ok((grad + (next_momentum * params.momentum)?)?)
    } else {
        Ok(next_momentum)
    }
}

pub(crate) fn muon_update(
    grad: &Tensor,
    var: &Tensor,
    momentum: &Var,
    params: &ParamsMuSgd,
) -> crate::Result<Tensor> {
    let effective_grad = if params.weight_decay == 0.0 {
        grad.clone()
    } else {
        (grad + &(var * params.weight_decay)?)?
    };
    let beta = params.momentum;
    let next_momentum = ((momentum.as_tensor() * beta)? + (&effective_grad * (1.0 - beta))?)?;
    momentum.set(&next_momentum)?;
    let update = if params.nesterov {
        ((&next_momentum * beta)? + (&effective_grad * (1.0 - beta))?)?
    } else {
        next_momentum
    };
    let shape = effective_grad.dims().to_vec();
    let (matrix, scale) = muon_matrix_and_scale(&update, &shape)?;
    let update = zero_power_newton_schulz5(&matrix)?;
    Ok((update * scale)?.reshape(shape)?)
}

fn muon_matrix_and_scale(update: &Tensor, shape: &[usize]) -> crate::Result<(Tensor, f64)> {
    match shape {
        [rows, cols] => Ok((
            update.clone(),
            ((*rows as f64 / *cols as f64).max(1.0)).sqrt(),
        )),
        [out, inn, height, width] => Ok((
            update.reshape((*out, inn * height * width))?,
            ((*height as f64 / *width as f64).max(1.0)).sqrt(),
        )),
        _ => Err(crate::Error::InvalidConfig(
            "MuSGD Muon update requires 2D or 4D parameters".to_string(),
        )),
    }
}

fn zero_power_newton_schulz5(matrix: &Tensor) -> crate::Result<Tensor> {
    let dims = matrix.dims();
    let (rows, cols) = (dims[0], dims[1]);
    let norm = (matrix.sqr()?.sum_all()?.sqrt()? + 1e-7)?;
    let mut x = matrix.broadcast_div(&norm)?;
    let transposed = rows > cols;
    if transposed {
        x = x.t()?;
    }
    for _ in 0..5 {
        let a_mat = x.matmul(&x.t()?)?;
        let b_mat = ((&a_mat * -4.7750)? + (a_mat.matmul(&a_mat)? * 2.0315)?)?;
        x = ((&x * 3.4445)? + b_mat.matmul(&x)?)?;
    }
    if transposed { Ok(x.t()?) } else { Ok(x) }
}

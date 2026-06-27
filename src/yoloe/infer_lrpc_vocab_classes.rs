use candle_core::Tensor;
use candle_nn::{Linear, VarBuilder};

pub(crate) fn infer_lrpc_vocab_classes(
    raw_weight: &Tensor,
    feature_dim: usize,
) -> crate::Result<usize> {
    match raw_weight.dims() {
        [classes, input, 1, 1] if *input == feature_dim => Ok(*classes),
        [classes, input] if *input == feature_dim => Ok(*classes),
        dims => Err(crate::Error::InvalidTensor(format!(
            "YOLOE LRPC vocab weight must have input dim {feature_dim}, got {dims:?}"
        ))),
    }
}

pub(crate) fn load_lrpc_vocab_linear(
    vb: VarBuilder,
    feature_dim: usize,
    classes: usize,
) -> crate::Result<Linear> {
    let raw_weight = vb
        .get_unchecked("weight")
        .or_else(|_| vb.get((classes, feature_dim, 1, 1), "weight"))?;
    let weight = match raw_weight.dims() {
        [out, input, 1, 1] if (*out, *input) == (classes, feature_dim) => {
            raw_weight.reshape((classes, feature_dim))?
        }
        [out, input] if (*out, *input) == (classes, feature_dim) => raw_weight,
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC vocab weight must have shape [{classes}, {feature_dim}, 1, 1] or [{classes}, {feature_dim}], got {dims:?}"
            )));
        }
    };
    let bias = vb.get(classes, "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

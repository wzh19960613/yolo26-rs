use candle_core::Tensor;
use candle_nn::VarBuilder;

pub(crate) fn select_lrpc_indices(
    proposals: &[f32],
    confidence_threshold: f32,
    max_proposals: Option<usize>,
) -> Vec<usize> {
    let mut selected = proposals
        .iter()
        .enumerate()
        .filter_map(|(idx, logit)| {
            if confidence_threshold == 0.0 || sigmoid_scalar(*logit) > confidence_threshold {
                Some(idx)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    if let Some(max_proposals) = max_proposals {
        selected.sort_by(|a, b| proposals[*b].total_cmp(&proposals[*a]));
        selected.truncate(max_proposals);
        selected.sort_unstable();
    }
    selected
}

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub(crate) fn max_index(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(idx, _)| idx)
        .unwrap_or(0)
}

pub(crate) fn pad_last_dim(tensor: &Tensor, target: usize) -> crate::Result<Tensor> {
    let dims = tensor.dims();
    let Some((&last, prefix)) = dims.split_last() else {
        return Err(crate::Error::InvalidTensor(
            "YOLOE padding requires a tensor with at least one dimension".to_string(),
        ));
    };
    if last == target {
        return Ok(tensor.clone());
    }
    if last > target {
        return Err(crate::Error::InvalidTensor(format!(
            "YOLOE cannot pad tensor last dim {last} down to {target}"
        )));
    }
    let mut pad_dims = prefix.to_vec();
    pad_dims.push(target - last);
    let zeros = Tensor::zeros(pad_dims, tensor.dtype(), tensor.device())?;
    Ok(Tensor::cat(&[tensor, &zeros], tensor.rank() - 1)?)
}

pub(crate) fn pad_last_dim_with(
    tensor: &Tensor,
    target: usize,
    value: f64,
) -> crate::Result<Tensor> {
    let dims = tensor.dims();
    let Some((&last, prefix)) = dims.split_last() else {
        return Err(crate::Error::InvalidTensor(
            "YOLOE padding requires a tensor with at least one dimension".to_string(),
        ));
    };
    if last == target {
        return Ok(tensor.clone());
    }
    if last > target {
        return Err(crate::Error::InvalidTensor(format!(
            "YOLOE cannot pad tensor last dim {last} down to {target}"
        )));
    }
    let mut pad_dims = prefix.to_vec();
    pad_dims.push(target - last);
    let padding = Tensor::zeros(pad_dims, tensor.dtype(), tensor.device())?.affine(0.0, value)?;
    Ok(Tensor::cat(&[tensor, &padding], tensor.rank() - 1)?)
}

pub(crate) fn infer_yoloe_head_prefix(names: &[String]) -> String {
    for marker in [
        ".one2one_cv2.",
        ".one2one_cv3.",
        ".one2one_cv5.",
        ".one2one_cv4.",
        ".proto.",
    ] {
        if let Some(prefix) = names
            .iter()
            .find_map(|name| name.find(marker).map(|idx| name[..idx].to_string()))
        {
            return prefix;
        }
    }
    "model.23".to_string()
}

pub(crate) fn has_tensor_prefix(names: &[String], prefix: &str) -> bool {
    names.iter().any(|name| name.starts_with(prefix))
}

pub(crate) fn has_tensor_name(names: &[String], key: &str) -> bool {
    names.iter().any(|name| name == key)
}

pub(crate) fn has_official_bn_contrastive(names: &[String], head_prefix: &str) -> bool {
    (0..3).all(|i| {
        let prefix = format!("{head_prefix}.one2one_cv4.{i}");
        has_tensor_name(names, &format!("{prefix}.bias"))
            && has_tensor_name(names, &format!("{prefix}.logit_scale"))
            && has_tensor_prefix(names, &format!("{prefix}.norm."))
    })
}

pub(crate) fn has_official_savpe(names: &[String], head_prefix: &str) -> bool {
    let prefix = format!("{head_prefix}.savpe");
    (0..3).all(|i| {
        has_tensor_name(names, &format!("{prefix}.cv1.{i}.0.conv.weight"))
            && has_tensor_name(names, &format!("{prefix}.cv1.{i}.1.conv.weight"))
            && has_tensor_name(names, &format!("{prefix}.cv2.{i}.0.conv.weight"))
    }) && has_tensor_name(names, &format!("{prefix}.cv3.weight"))
        && has_tensor_name(names, &format!("{prefix}.cv4.weight"))
        && has_tensor_name(names, &format!("{prefix}.cv5.weight"))
        && has_tensor_name(names, &format!("{prefix}.cv6.0.conv.weight"))
        && has_tensor_name(names, &format!("{prefix}.cv6.1.weight"))
}

pub(crate) fn has_official_reprta(names: &[String], head_prefix: &str) -> bool {
    let prefix = format!("{head_prefix}.reprta.m");
    has_tensor_name(names, &format!("{prefix}.w12.weight"))
        && has_tensor_name(names, &format!("{prefix}.w12.bias"))
        && has_tensor_name(names, &format!("{prefix}.w3.weight"))
        && has_tensor_name(names, &format!("{prefix}.w3.bias"))
}

pub(crate) fn has_official_lrpc(names: &[String], head_prefix: &str) -> bool {
    (0..3).all(|i| {
        let prefix = format!("{head_prefix}.lrpc.{i}");
        has_tensor_name(names, &format!("{prefix}.vocab.weight"))
            && has_tensor_name(names, &format!("{prefix}.pf.weight"))
            && has_tensor_name(names, &format!("{prefix}.loc.weight"))
    })
}

pub(crate) fn has_bn_contrastive_tensors(vb: &VarBuilder) -> bool {
    let branch = vb.pp("one2one_cv4").pp("0");
    branch.contains_tensor("bias")
        && branch.contains_tensor("logit_scale")
        && branch.pp("norm").contains_tensor("weight")
}

pub(crate) fn contains_marker(name: &str, marker: &str) -> bool {
    name.to_ascii_lowercase().contains(marker)
}

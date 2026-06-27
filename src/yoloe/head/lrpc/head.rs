use candle_core::Tensor;

use crate::yoloe::head::lrpc::output::LrpcOutput;
use crate::yoloe::prompt::table::ScorerConfig;
use crate::yoloe::select_lrpc_indices::{max_index, pad_last_dim, select_lrpc_indices};
use crate::yoloe::usage::EmbeddingTable;

/// Lazy Region-Prompt Contrast prompt-free scoring head.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LrpcHead {
    /// Proposal filter confidence threshold.
    pub confidence_threshold: f32,
    /// Optional maximum number of proposals to keep per image.
    pub max_proposals: Option<usize>,
    /// Prompt scorer options for vocabulary projection.
    pub scorer: ScorerConfig,
}

impl Default for LrpcHead {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.001,
            max_proposals: None,
            scorer: ScorerConfig::default(),
        }
    }
}

impl LrpcHead {
    /// Runs prompt-free proposal filtering and vocabulary scoring.
    pub fn forward(
        &self,
        cls_features: &Tensor,
        loc_features: &Tensor,
        proposal_logits: &Tensor,
        vocabulary: &EmbeddingTable,
    ) -> crate::Result<LrpcOutput> {
        if !(self.confidence_threshold.is_finite() && self.confidence_threshold >= 0.0) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE LRPC confidence threshold must be finite and non-negative".to_string(),
            ));
        }
        let (batch, dim, height, width) = cls_features.dims4()?;
        let (loc_batch, loc_channels, loc_h, loc_w) = loc_features.dims4()?;
        let (pf_batch, pf_channels, pf_h, pf_w) = proposal_logits.dims4()?;
        if (loc_batch, loc_h, loc_w) != (batch, height, width) {
            return Err(crate::Error::InvalidTensor(
                "YOLOE LRPC localization features must share batch and spatial shape with class features"
                    .to_string(),
            ));
        }
        if (pf_batch, pf_channels, pf_h, pf_w) != (batch, 1, height, width) {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC proposal logits must have shape [{batch}, 1, {height}, {width}]"
            )));
        }
        if dim != vocabulary.dim()? {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC class feature dim {dim} does not match vocabulary dim {}",
                vocabulary.dim()?
            )));
        }

        let spatial = height * width;
        let mut selected_indices = Vec::with_capacity(batch);
        let mut cls_selected = Vec::with_capacity(batch);
        let mut loc_selected = Vec::with_capacity(batch);
        let mut max_selected = 0usize;

        for b in 0..batch {
            let proposals = proposal_logits
                .narrow(0, b, 1)?
                .reshape((spatial,))?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?;
            let mut indices =
                select_lrpc_indices(&proposals, self.confidence_threshold, self.max_proposals);
            if indices.is_empty() {
                indices.push(max_index(&proposals));
            }
            max_selected = max_selected.max(indices.len());
            let index_tensor = Tensor::new(
                indices.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
                cls_features.device(),
            )?;
            let cls_b = cls_features
                .narrow(0, b, 1)?
                .reshape((dim, spatial))?
                .transpose(0, 1)?
                .contiguous()?
                .index_select(&index_tensor, 0)?;
            let loc_b = loc_features
                .narrow(0, b, 1)?
                .reshape((loc_channels, spatial))?
                .contiguous()?
                .index_select(&index_tensor, 1)?;
            selected_indices.push(indices);
            cls_selected.push(cls_b);
            loc_selected.push(loc_b);
        }

        let mut batch_scores = Vec::with_capacity(batch);
        let mut batch_boxes = Vec::with_capacity(batch);
        let mut batch_valid = Vec::with_capacity(batch);
        for b in 0..batch {
            let scores = vocabulary
                .score_features(&cls_selected[b], self.scorer)?
                .transpose(0, 1)?;
            let boxes = loc_selected[b].clone();
            let selected = selected_indices[b].len();
            batch_scores.push(pad_last_dim(&scores, max_selected)?.unsqueeze(0)?);
            batch_boxes.push(pad_last_dim(&boxes, max_selected)?.unsqueeze(0)?);
            let mut valid = vec![1.0f32; selected];
            valid.resize(max_selected, 0.0);
            batch_valid.push(Tensor::new(valid, cls_features.device())?.unsqueeze(0)?);
        }

        Ok(LrpcOutput {
            boxes: Tensor::cat(&batch_boxes.iter().collect::<Vec<_>>(), 0)?,
            scores: Tensor::cat(&batch_scores.iter().collect::<Vec<_>>(), 0)?,
            valid: Tensor::cat(&batch_valid.iter().collect::<Vec<_>>(), 0)?,
            selected_indices,
        })
    }
}

pub(crate) fn maybe_normalize(tensor: &Tensor, normalize: bool) -> crate::Result<Tensor> {
    if normalize {
        l2_normalize_last_dim(tensor)
    } else {
        Ok(tensor.clone())
    }
}

pub(crate) fn l2_normalize_last_dim(tensor: &Tensor) -> crate::Result<Tensor> {
    if tensor.rank() == 0 {
        return Err(crate::Error::InvalidTensor(
            "YOLOE prompt tensors must have at least one dimension".to_string(),
        ));
    }
    l2_normalize_dim(tensor, tensor.rank() - 1)
}

pub(crate) fn l2_normalize_dim(tensor: &Tensor, dim: usize) -> crate::Result<Tensor> {
    let norm = tensor.sqr()?.sum_keepdim(dim)?.sqrt()?.affine(1.0, 1e-12)?;
    Ok(tensor.broadcast_div(&norm)?)
}

pub(crate) fn linear_last_dim(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
) -> crate::Result<Tensor> {
    let (out_dim, in_dim) = weight.dims2()?;
    let mut output = match input.dims() {
        [rows, dim] if *dim == in_dim => input
            .matmul(&weight.transpose(0, 1)?)?
            .reshape((*rows, out_dim))?,
        [_, dim] => {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE linear input dim {dim} does not match weight dim {in_dim}"
            )));
        }
        [batch, rows, dim] if *dim == in_dim => input
            .reshape((*batch * *rows, *dim))?
            .matmul(&weight.transpose(0, 1)?)?
            .reshape((*batch, *rows, out_dim))?,
        [_, _, dim] => {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE linear input dim {dim} does not match weight dim {in_dim}"
            )));
        }
        dims => {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE linear expects [N, dim] or [batch, N, dim], got {dims:?}"
            )));
        }
    };
    if let Some(bias) = bias {
        if bias.dims() != [out_dim] {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE linear bias must have shape [{out_dim}], got {:?}",
                bias.dims()
            )));
        }
        let bias = match output.dims() {
            [_, _] => bias.reshape((1, out_dim))?,
            [_, _, _] => bias.reshape((1, 1, out_dim))?,
            dims => {
                return Err(crate::Error::InvalidTensor(format!(
                    "YOLOE linear output must be rank 2 or 3 to apply bias, got {dims:?}"
                )));
            }
        };
        output = output.broadcast_add(&bias)?;
    }
    Ok(output)
}

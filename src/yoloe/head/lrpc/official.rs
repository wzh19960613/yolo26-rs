use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv2d, Linear};

use crate::yoloe::head::lrpc::output::OfficialOutput;
use crate::yoloe::select_lrpc_indices::{max_index, pad_last_dim, select_lrpc_indices};

/// Official YOLOE `LRPCHead` adapter for prompt-free checkpoints.
pub struct Official {
    pub(crate) vocab: Linear,
    pub(crate) pf: Conv2d,
    pub(crate) loc: Conv2d,
    pub(crate) enabled: bool,
    pub(crate) classes: usize,
    pub(crate) feature_dim: usize,
    pub(crate) proposal_channels: usize,
    pub(crate) loc_feature_dim: usize,
    pub(crate) box_channels: usize,
}

impl Official {
    /// Returns whether proposal filtering is enabled.
    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    /// Number of prompt-free classes projected by this head.
    pub const fn classes(&self) -> usize {
        self.classes
    }

    /// Feature dimension accepted by this head.
    pub const fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Number of channels emitted by the proposal filter.
    pub const fn proposal_channels(&self) -> usize {
        self.proposal_channels
    }

    /// Number of localization feature channels accepted by this head.
    pub const fn loc_feature_dim(&self) -> usize {
        self.loc_feature_dim
    }

    /// Number of localization channels emitted by this head.
    pub const fn box_channels(&self) -> usize {
        self.box_channels
    }

    /// Runs official-style prompt-free proposal filtering and vocabulary projection.
    pub fn forward(
        &self,
        cls_feat: &Tensor,
        loc_feat: &Tensor,
        confidence_threshold: f32,
    ) -> crate::Result<OfficialOutput> {
        if !(confidence_threshold.is_finite() && confidence_threshold >= 0.0) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE LRPC confidence threshold must be finite and non-negative".to_string(),
            ));
        }
        let (batch, feature_dim, height, width) = cls_feat.dims4()?;
        if feature_dim != self.feature_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC class feature dim {feature_dim} does not match head feature_dim {}",
                self.feature_dim
            )));
        }
        let (loc_batch, loc_feature_dim, loc_h, loc_w) = loc_feat.dims4()?;
        if loc_feature_dim != self.loc_feature_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC localization feature dim {loc_feature_dim} does not match head loc_feature_dim {}",
                self.loc_feature_dim
            )));
        }
        if (loc_batch, loc_h, loc_w) != (batch, height, width) {
            return Err(crate::Error::InvalidTensor(
                "YOLOE LRPC localization features must share batch and spatial shape with class features"
                    .to_string(),
            ));
        }
        let boxes = self.loc.forward(loc_feat)?;
        let spatial = height * width;
        if !self.enabled {
            let logits = self
                .vocab
                .forward(
                    &cls_feat
                        .reshape((batch, feature_dim, spatial))?
                        .transpose(1, 2)?,
                )?
                .transpose(1, 2)?;
            return Ok(OfficialOutput {
                boxes,
                scores: logits,
                selected_indices: vec![(0..spatial).collect(); batch],
            });
        }

        let proposal_logits = self.pf.forward(cls_feat)?.narrow(1, 0, 1)?;
        let mut selected_indices = Vec::with_capacity(batch);
        let mut selected_scores = Vec::with_capacity(batch);
        let mut max_selected = 0usize;
        for b in 0..batch {
            let proposals = proposal_logits
                .narrow(0, b, 1)?
                .reshape((spatial,))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            let mut indices = select_lrpc_indices(&proposals, confidence_threshold, None);
            if indices.is_empty() {
                indices.push(max_index(&proposals));
            }
            max_selected = max_selected.max(indices.len());
            let index_tensor = Tensor::new(
                indices.iter().map(|idx| *idx as u32).collect::<Vec<_>>(),
                cls_feat.device(),
            )?;
            let features = cls_feat
                .narrow(0, b, 1)?
                .reshape((feature_dim, spatial))?
                .transpose(0, 1)?
                .contiguous()?
                .index_select(&index_tensor, 0)?;
            selected_scores.push(self.vocab.forward(&features)?.transpose(0, 1)?);
            selected_indices.push(indices);
        }
        let mut batch_scores = Vec::with_capacity(batch);
        for scores in selected_scores {
            batch_scores.push(pad_last_dim(&scores, max_selected)?.unsqueeze(0)?);
        }
        Ok(OfficialOutput {
            boxes,
            scores: Tensor::cat(&batch_scores.iter().collect::<Vec<_>>(), 0)?,
            selected_indices,
        })
    }

    /// Training-only dense LRPC forward over every anchor.
    ///
    /// Returns box distances `[batch, 4, H, W]` and vocabulary logits
    /// `[batch, classes, H, W]`, matching the tensors consumed by the regular
    /// detection/segmentation loss before any proposal filtering.
    pub fn forward_dense_train(
        &self,
        cls_feat: &Tensor,
        loc_feat: &Tensor,
    ) -> crate::Result<(Tensor, Tensor)> {
        let (batch, feature_dim, height, width) = cls_feat.dims4()?;
        if feature_dim != self.feature_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC dense class feature dim {feature_dim} does not match head feature_dim {}",
                self.feature_dim
            )));
        }
        let (loc_batch, loc_feature_dim, loc_h, loc_w) = loc_feat.dims4()?;
        if loc_feature_dim != self.loc_feature_dim {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE LRPC dense localization feature dim {loc_feature_dim} does not match head loc_feature_dim {}",
                self.loc_feature_dim
            )));
        }
        if (loc_batch, loc_h, loc_w) != (batch, height, width) {
            return Err(crate::Error::InvalidTensor(
                "YOLOE LRPC dense localization features must share batch and spatial shape with class features"
                    .to_string(),
            ));
        }
        let spatial = height * width;
        let boxes = self.loc.forward(loc_feat)?;
        let vocab = self
            .vocab
            .forward(
                &cls_feat
                    .reshape((batch, feature_dim, spatial))?
                    .transpose(1, 2)?,
            )?
            .transpose(1, 2)?
            .reshape((batch, self.classes, height, width))?;
        Ok((boxes, vocab))
    }
}

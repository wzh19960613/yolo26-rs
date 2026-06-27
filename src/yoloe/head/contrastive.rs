use candle_core::Tensor;
use candle_nn::{BatchNorm, BatchNormConfig, ModuleT, VarBuilder, batch_norm};

use crate::yoloe::head::lrpc::head::l2_normalize_last_dim;
use crate::yoloe::prompt::table::ScorerConfig;
use crate::yoloe::usage::EmbeddingTable;

/// Contrastive feature-map scorer used by YOLOE detection heads.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contrastive {
    /// Prompt scorer options.
    pub scorer: ScorerConfig,
    /// Additive class-logit bias.
    pub bias: f32,
}

impl Default for Contrastive {
    fn default() -> Self {
        Self {
            scorer: ScorerConfig::default(),
            bias: -10.0,
        }
    }
}

impl Contrastive {
    /// Scores a `[batch, dim, height, width]` feature map with prompt embeddings.
    pub fn forward(self, feature_map: &Tensor, prompts: &EmbeddingTable) -> crate::Result<Tensor> {
        let (batch, dim, height, width) = feature_map.dims4()?;
        if dim != prompts.dim()? {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE contrastive feature dim {dim} does not match prompt dim {}",
                prompts.dim()?
            )));
        }
        let spatial = height * width;
        let features = feature_map
            .reshape((batch, dim, spatial))?
            .transpose(1, 2)?;
        let scores = prompts.score_features(&features, self.scorer)?;
        Ok(scores
            .transpose(1, 2)?
            .reshape((batch, prompts.class_count(), height, width))?
            .affine(1.0, self.bias as f64)?)
    }
}

#[derive(Clone, Debug)]

pub(crate) struct BnContrastive {
    norm: BatchNorm,
    logit_scale: Tensor,
    bias: Tensor,
}

impl BnContrastive {
    pub(crate) fn load(vb: VarBuilder, embed_dim: usize) -> crate::Result<Self> {
        let norm = batch_norm(
            embed_dim,
            BatchNormConfig {
                eps: 1e-3,
                ..BatchNormConfig::default()
            },
            vb.pp("norm"),
        )?;
        Ok(Self {
            norm,
            logit_scale: vb.get((), "logit_scale")?,
            bias: vb.get(1, "bias")?,
        })
    }

    pub(crate) fn forward(
        &self,
        feature_map: &Tensor,
        prompts: &EmbeddingTable,
    ) -> crate::Result<Tensor> {
        self.forward_impl(feature_map, prompts, false)
    }

    /// Training-time scoring: BatchNorm runs in train mode so batch statistics
    /// update the running buffers and gradients flow through both the prompt
    /// table and the head parameters. Mirrors the inference math otherwise.
    #[cfg(feature = "train")]
    pub(crate) fn forward_dense(
        &self,
        feature_map: &Tensor,
        prompts: &EmbeddingTable,
    ) -> crate::Result<Tensor> {
        self.forward_impl(feature_map, prompts, true)
    }

    fn forward_impl(
        &self,
        feature_map: &Tensor,
        prompts: &EmbeddingTable,
        bn_train: bool,
    ) -> crate::Result<Tensor> {
        let (batch, dim, height, width) = feature_map.dims4()?;
        if dim != prompts.dim()? {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE BN contrastive feature dim {dim} does not match prompt dim {}",
                prompts.dim()?
            )));
        }
        let normalized = self.norm.forward_t(feature_map, bn_train)?;
        let spatial = height * width;
        let features = normalized
            .reshape((batch, dim, spatial))?
            .transpose(1, 2)?
            .reshape((batch * spatial, dim))?;
        let weights = l2_normalize_last_dim(&prompts.embeddings)?;
        // Align prompt embeddings to the feature dtype/device before matmul:
        // embeddings are commonly F32 while the model may run in F16, and on a
        // GPU the embeddings may still be on CPU until first use.
        let weights = weights
            .to_dtype(features.dtype())?
            .to_device(features.device())?;
        let scores = features
            .matmul(&weights.transpose(0, 1)?)?
            .reshape((batch, spatial, prompts.class_count()))?
            .transpose(1, 2)?
            .reshape((batch, prompts.class_count(), height, width))?;
        Ok(scores
            .broadcast_mul(&self.logit_scale.exp()?)?
            .broadcast_add(&self.bias.reshape((1, 1, 1, 1))?)?)
    }
}

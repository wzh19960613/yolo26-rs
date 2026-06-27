use candle_core::Tensor;

use crate::yoloe::head::lrpc::head::maybe_normalize;
use crate::yoloe::usage::{EmbeddingSpace, EmbeddingTable};

impl EmbeddingTable {
    /// Creates a validated prompt embedding table.
    pub fn new(embeddings: Tensor, classes: Vec<String>) -> crate::Result<Self> {
        let (rows, dim) = embeddings.dims2().map_err(crate::Error::from)?;
        if rows == 0 || dim == 0 {
            return Err(crate::Error::InvalidTensor(
                "YOLOE prompt embeddings must have non-empty [classes, dim] shape".to_string(),
            ));
        }
        if classes.len() != rows {
            return Err(crate::Error::InvalidConfig(format!(
                "YOLOE prompt class count {} does not match embedding rows {}",
                classes.len(),
                rows
            )));
        }
        if classes.iter().any(|class| class.trim().is_empty()) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt classes must not contain empty labels".to_string(),
            ));
        }
        Ok(Self {
            classes,
            embeddings,
        })
    }

    /// Number of prompt classes.
    pub fn class_count(&self) -> usize {
        self.classes.len()
    }

    /// Prompt embedding dimension.
    pub fn dim(&self) -> crate::Result<usize> {
        Ok(self.embeddings.dims2()?.1)
    }

    /// Returns the descriptor used by prompt-state APIs.
    pub fn embedding_space(&self) -> crate::Result<EmbeddingSpace> {
        EmbeddingSpace::new(self.dim()?, self.classes.clone())
    }

    /// Scores dense region features against this prompt table.
    pub fn score_features(
        &self,
        region_features: &Tensor,
        config: ScorerConfig,
    ) -> crate::Result<Tensor> {
        config.score(region_features, self)
    }
}

/// Prompt scorer options for YOLOE open-vocabulary logits.
#[derive(Debug, Clone, Copy, PartialEq)]

pub struct ScorerConfig {
    /// L2-normalize region features and prompt embeddings before dot-product scoring.
    pub normalize: bool,
    /// Multiplicative scale applied to output logits.
    pub logit_scale: f32,
}

impl Default for ScorerConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            logit_scale: 1.0,
        }
    }
}

impl ScorerConfig {
    /// Scores `[N, dim]` or `[batch, anchors, dim]` region features against prompts.
    pub fn score(
        self,
        region_features: &Tensor,
        prompts: &EmbeddingTable,
    ) -> crate::Result<Tensor> {
        if !(self.logit_scale.is_finite() && self.logit_scale > 0.0) {
            return Err(crate::Error::InvalidConfig(
                "YOLOE prompt logit_scale must be finite and greater than zero".to_string(),
            ));
        }
        let prompt_dim = prompts.dim()?;
        match region_features.dims() {
            [_, dim] if *dim == prompt_dim => {
                let features = maybe_normalize(region_features, self.normalize)?;
                let weights = maybe_normalize(&prompts.embeddings, self.normalize)?;
                // Embeddings (text/visual/LRPC) are typically F32 and may live
                // on a different device/dtype than the model features. Align
                // them to the features so matmul never trips a dtype mismatch.
                let weights = weights
                    .to_dtype(features.dtype())?
                    .to_device(features.device())?;
                Ok(features
                    .matmul(&weights.transpose(0, 1)?)?
                    .affine(self.logit_scale as f64, 0.0)?)
            }
            [_, dim] => Err(crate::Error::InvalidTensor(format!(
                "YOLOE feature dim {dim} does not match prompt dim {prompt_dim}"
            ))),
            [batch, anchors, dim] if *dim == prompt_dim => {
                let flat = region_features.reshape((*batch * *anchors, *dim))?;
                let scored = self.score(&flat, prompts)?;
                Ok(scored.reshape((*batch, *anchors, prompts.class_count()))?)
            }
            [_, _, dim] => Err(crate::Error::InvalidTensor(format!(
                "YOLOE feature dim {dim} does not match prompt dim {prompt_dim}"
            ))),
            dims => Err(crate::Error::InvalidTensor(format!(
                "YOLOE prompt scoring expects [N, dim] or [batch, anchors, dim] features, got {dims:?}"
            ))),
        }
    }
}

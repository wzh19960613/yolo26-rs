use candle_core::Tensor;

use crate::yoloe::head::lrpc::head::l2_normalize_last_dim;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::Visuals;

/// Semantic-Activated Visual Prompt Encoder style prompt pooler.
///
/// **This is NOT the official SAVPE module.** It is a lightweight
/// masked-average-pool fallback used when only a single dense embedding map is
/// available (official SAVPE needs three feature scales); its output is not
/// numerically equivalent to official SAVPE. Prefer
/// [`Encoder`](super::Encoder) when the
/// three-scale features are available.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pooler {
    /// Whether to L2-normalize generated visual prompt embeddings.
    pub normalize: bool,
}

impl Default for Pooler {
    fn default() -> Self {
        Self { normalize: true }
    }
}

impl Pooler {
    /// Aggregates prompt masks over an embedding map into `[batch, prompts, dim]` embeddings.
    pub fn encode(&self, embedding_map: &Tensor, prompt_masks: &Tensor) -> crate::Result<Tensor> {
        let (batch, dim, height, width) = embedding_map.dims4()?;
        let mask_dims = prompt_masks.dims();
        let prompts = match mask_dims {
            [mask_batch, prompts, mask_h, mask_w]
                if (*mask_batch, *mask_h, *mask_w) == (batch, height, width) =>
            {
                *prompts
            }
            _ => {
                return Err(crate::Error::InvalidTensor(format!(
                    "YOLOE SAVPE masks must have shape [batch, prompts, {height}, {width}], got {mask_dims:?}"
                )));
            }
        };
        if prompts == 0 {
            return Ok(Tensor::zeros(
                (batch, 0, dim),
                embedding_map.dtype(),
                embedding_map.device(),
            )?);
        }
        let spatial = height * width;
        let features = embedding_map
            .reshape((batch, dim, spatial))?
            .transpose(1, 2)?;
        let masks = prompt_masks.reshape((batch, prompts, spatial))?;
        let denom = masks
            .sum_keepdim(2)?
            .affine(1.0, 1e-12)?
            .to_dtype(embedding_map.dtype())?;
        let weights = masks
            .to_dtype(embedding_map.dtype())?
            .broadcast_div(&denom)?;
        let pooled = weights.matmul(&features)?;
        if self.normalize {
            l2_normalize_last_dim(&pooled)
        } else {
            Ok(pooled)
        }
    }

    /// Aggregates a single image's [`Visuals`] into a prompt embedding table.
    pub fn encode_single_image_table(
        &self,
        embedding_map: &Tensor,
        visuals: &Visuals,
        classes: Vec<String>,
    ) -> crate::Result<EmbeddingTable> {
        let encoded = self.encode(embedding_map, &visuals.tensor)?;
        if encoded.dim(0)? != 1 {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE SAVPE table export requires batch size 1, got {}",
                encoded.dim(0)?
            )));
        }
        EmbeddingTable::new(encoded.squeeze(0)?, classes)
    }
}

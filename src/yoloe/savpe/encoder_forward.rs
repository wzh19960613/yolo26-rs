use candle_core::{Module, Tensor};

use crate::yoloe::head::lrpc::head::l2_normalize_last_dim;
use crate::yoloe::savpe::encoder::Encoder;
use crate::yoloe::usage::EmbeddingTable;
use crate::yoloe::visuals::Visuals;

impl Encoder {
    /// Encodes `[batch, prompts, height, width]` visual prompt masks from three feature scales.
    pub fn forward(&self, features: &[&Tensor], prompt_masks: &Tensor) -> crate::Result<Tensor> {
        if features.len() != 3 {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE SAVPE expects 3 feature maps, got {}",
                features.len()
            )));
        }
        let (batch, _, height, width) = features[0].dims4()?;
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
                (batch, 0, self.embed_dim),
                features[0].dtype(),
                features[0].device(),
            )?);
        }

        let mut attention_inputs = Vec::with_capacity(features.len());
        let mut prompt_inputs = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            attention_inputs.push(self.cv2[i].forward(feature, height, width)?);
            prompt_inputs.push(self.cv1[i].forward(feature, height, width)?);
        }
        let attention_refs = attention_inputs.iter().collect::<Vec<_>>();
        let prompt_refs = prompt_inputs.iter().collect::<Vec<_>>();
        let attention_base = self.cv4.forward(&Tensor::cat(&attention_refs, 1)?)?;
        let prompt_features = self.cv3.forward(&Tensor::cat(&prompt_refs, 1)?)?;
        let (_, embed_dim, prompt_h, prompt_w) = prompt_features.dims4()?;
        if (embed_dim, prompt_h, prompt_w) != (self.embed_dim, height, width) {
            return Err(crate::Error::InvalidTensor(
                "YOLOE SAVPE prompt feature shape does not match configured output".to_string(),
            ));
        }

        let pixels = height * width;
        let attention = attention_base
            .reshape((batch, 1, self.attention_channels, height, width))?
            .broadcast_as((batch, prompts, self.attention_channels, height, width))?
            .reshape((batch * prompts, self.attention_channels, height, width))?;
        let masks = prompt_masks.to_dtype(features[0].dtype())?.reshape((
            batch * prompts,
            1,
            height,
            width,
        ))?;
        let mask_attention = self.cv5.forward(&masks)?;
        let refined = self.cv6_1.forward(
            &self
                .cv6_0
                .forward(&Tensor::cat(&[&attention, &mask_attention], 1)?)?,
        )?;
        let refined = refined.reshape((batch, prompts, self.attention_channels, pixels))?;
        let mask = prompt_masks
            .to_dtype(refined.dtype())?
            .reshape((batch, prompts, 1, pixels))?;
        let missing_mask = mask.affine(-1.0, 1.0)?;
        let neg = Tensor::new(-1.0e4f32, refined.device())?.to_dtype(refined.dtype())?;
        let logits = refined
            .broadcast_mul(&mask)?
            .broadcast_add(&missing_mask.broadcast_mul(&neg)?)?;
        let weights = candle_nn::ops::softmax(&logits, 3)?;
        let grouped_features = prompt_features
            .reshape((
                batch,
                self.attention_channels,
                self.embed_dim / self.attention_channels,
                pixels,
            ))?
            .transpose(2, 3)?;
        let aggregated = weights.transpose(1, 2)?.matmul(&grouped_features)?;
        l2_normalize_last_dim(&aggregated.transpose(1, 2)?.reshape((
            batch,
            prompts,
            self.embed_dim,
        ))?)
    }

    /// Encodes a single image's [`Visuals`] into a prompt embedding table.
    ///
    /// `visuals.tensor` is `[1, classes, h, w]`; it is passed straight to
    /// [`Self::forward`] as the `[batch, prompts, h, w]` mask (axis-1 is the
    /// class/prompt axis).
    pub fn encode_single_image_table(
        &self,
        features: &[&Tensor],
        visuals: &Visuals,
        classes: Vec<String>,
    ) -> crate::Result<EmbeddingTable> {
        let encoded = self.forward(features, &visuals.tensor)?;
        if encoded.dim(0)? != 1 {
            return Err(crate::Error::InvalidTensor(format!(
                "YOLOE SAVPE table export requires batch size 1, got {}",
                encoded.dim(0)?
            )));
        }
        EmbeddingTable::new(encoded.squeeze(0)?, classes)
    }
}

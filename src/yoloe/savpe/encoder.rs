use candle_core::Tensor;
use candle_nn::{Conv2d, VarBuilder};

use crate::network::blocks::ConvBlock;

pub(crate) struct SavpeFeaturePath {
    pub(crate) conv0: ConvBlock,
    pub(crate) conv1: Option<ConvBlock>,
    pub(crate) scale: usize,
}

impl SavpeFeaturePath {
    pub(crate) fn two_conv(
        vb: VarBuilder,
        input_channels: usize,
        hidden_channels: usize,
        scale: usize,
    ) -> crate::Result<Self> {
        Ok(Self {
            conv0: ConvBlock::load(vb.pp("0"), input_channels, hidden_channels, 3, 1, 1, true)?,
            conv1: Some(ConvBlock::load(
                vb.pp("1"),
                hidden_channels,
                hidden_channels,
                3,
                1,
                1,
                true,
            )?),
            scale,
        })
    }

    pub(crate) fn one_conv(
        vb: VarBuilder,
        input_channels: usize,
        hidden_channels: usize,
        scale: usize,
    ) -> crate::Result<Self> {
        Ok(Self {
            conv0: ConvBlock::load(vb.pp("0"), input_channels, hidden_channels, 1, 1, 1, true)?,
            conv1: None,
            scale,
        })
    }

    pub(crate) fn forward(
        &self,
        input: &Tensor,
        output_h: usize,
        output_w: usize,
    ) -> crate::Result<Tensor> {
        let mut out = self.conv0.forward(input)?;
        if let Some(conv1) = self.conv1.as_ref() {
            out = conv1.forward(&out)?;
        }
        if self.scale > 1 {
            out.upsample_nearest2d(output_h, output_w)
                .map_err(crate::Error::from)
        } else {
            Ok(out)
        }
    }
}

/// Official YOLOE Spatial-Aware Visual Prompt Embedding encoder.
///
/// This module mirrors Ultralytics `SAVPE`: feature projections from the three
/// detection scales are fused, prompt masks guide a lightweight spatial
/// attention branch, and the final output is L2-normalized visual prompt
/// embeddings with shape `[batch, prompts, embed_dim]`.
pub struct Encoder {
    pub(crate) cv1: Vec<SavpeFeaturePath>,
    pub(crate) cv2: Vec<SavpeFeaturePath>,
    pub(crate) cv3: Conv2d,
    pub(crate) cv4: Conv2d,
    pub(crate) cv5: Conv2d,
    pub(crate) cv6_0: ConvBlock,
    pub(crate) cv6_1: Conv2d,
    pub(crate) attention_channels: usize,
    pub(crate) hidden_channels: usize,
    pub(crate) embed_dim: usize,
}

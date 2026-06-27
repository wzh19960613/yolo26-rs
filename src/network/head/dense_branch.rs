use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::network::blocks::ConvBlock;

pub struct DenseBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    cv2: candle_nn::Conv2d,
    out_channels: usize,
}

impl DenseBranch {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        hidden: usize,
        out_channels: usize,
        cfg: Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            cv0: ConvBlock::load(vb.pp("0"), channels, hidden, 3, 1, 1, true)?,
            cv1: ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1, 1, true)?,
            cv2: conv2d(hidden, out_channels, 1, cfg, vb.pp("2"))?,
            out_channels,
        })
    }

    pub fn forward(&self, feature: &Tensor, batch: usize, spatial: usize) -> Result<Tensor> {
        let x = self.cv0.forward(feature)?;
        let x = self.cv1.forward(&x)?;
        self.cv2
            .forward(&x)?
            .reshape((batch, self.out_channels, spatial))
    }

    pub(crate) fn forward_branches(branches: &[Self], features: &[&Tensor]) -> Result<Tensor> {
        let batch = features[0].dim(0)?;
        let mut outputs = Vec::with_capacity(features.len());
        for (i, feature) in features.iter().enumerate() {
            let (_, _, h, w) = feature.dims4()?;
            outputs.push(branches[i].forward(feature, batch, h * w)?);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 2)
    }
}

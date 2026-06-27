use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::network::blocks::ConvBlock;

pub struct PoseBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    keypoints: candle_nn::Conv2d,
    out_channels: usize,
}

impl PoseBranch {
    pub fn load(
        vb: VarBuilder,
        keypoints_vb: VarBuilder,
        channels: usize,
        hidden: usize,
        out_channels: usize,
        cfg: Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            cv0: ConvBlock::load(vb.pp("0"), channels, hidden, 3, 1, 1, true)?,
            cv1: ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1, 1, true)?,
            keypoints: conv2d(hidden, out_channels, 1, cfg, keypoints_vb)?,
            out_channels,
        })
    }

    pub fn forward(&self, feature: &Tensor, batch: usize, spatial: usize) -> Result<Tensor> {
        let x = self.cv0.forward(feature)?;
        let x = self.cv1.forward(&x)?;
        self.keypoints
            .forward(&x)?
            .reshape((batch, self.out_channels, spatial))
    }
}

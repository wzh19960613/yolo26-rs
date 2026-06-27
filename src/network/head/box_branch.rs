use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::network::blocks::ConvBlock;

pub struct BoxBranch {
    cv0: ConvBlock,
    cv1: ConvBlock,
    cv2: candle_nn::Conv2d,
}

impl BoxBranch {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        hidden: usize,
        reg_max: usize,
        cfg: Conv2dConfig,
    ) -> Result<Self> {
        Ok(Self {
            cv0: ConvBlock::load(vb.pp("0"), channels, hidden, 3, 1, 1, true)?,
            cv1: ConvBlock::load(vb.pp("1"), hidden, hidden, 3, 1, 1, true)?,
            cv2: conv2d(hidden, 4 * reg_max, 1, cfg, vb.pp("2"))?,
        })
    }

    pub fn forward(&self, feature: &Tensor, batch: usize, spatial: usize) -> Result<Tensor> {
        self.forward_map(feature)?.reshape((batch, 4, spatial))
    }

    pub fn forward_map(&self, feature: &Tensor) -> Result<Tensor> {
        let bx = self.cv0.forward(feature)?;
        let bx = self.cv1.forward(&bx)?;
        self.cv2.forward(&bx)
    }
}

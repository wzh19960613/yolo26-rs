use candle_core::{Module, Result, Tensor};
use candle_nn::{VarBuilder, linear};

use crate::network::blocks::ConvBlock;

pub(crate) struct Head {
    conv: ConvBlock,
    linear: candle_nn::Linear,
}

impl Head {
    pub(crate) fn load(vb: VarBuilder, input_channels: usize, nc: usize) -> Result<Self> {
        Ok(Self {
            conv: ConvBlock::load(vb.pp("conv"), input_channels, 1280, 1, 1, 1, true)?,
            linear: linear(1280, nc, vb.pp("linear"))?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.forward_logits(x)?;
        candle_nn::ops::softmax(&logits, 1)
    }

    pub(crate) fn forward_logits(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        let (_, _, h, w) = x.dims4()?;
        let x = x.avg_pool2d((h, w))?.flatten_from(1)?;
        self.linear.forward(&x)
    }
}

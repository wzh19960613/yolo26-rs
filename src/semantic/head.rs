use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2dConfig, VarBuilder, conv2d};

use crate::network::blocks::ConvBlock;

pub(crate) struct Head {
    conv: ConvBlock,
    classifier: candle_nn::Conv2d,
}

impl Head {
    pub(crate) fn load(vb: VarBuilder, input_channels: &[usize], nc: usize) -> Result<Self> {
        let c_mid = input_channels[0];
        Ok(Self {
            conv: ConvBlock::load(vb.pp("classifier").pp("0"), c_mid, c_mid, 3, 1, 1, true)?,
            classifier: conv2d(
                c_mid,
                nc,
                1,
                Conv2dConfig::default(),
                vb.pp("classifier").pp("1"),
            )?,
        })
    }

    pub(crate) fn forward(&self, features: &[&Tensor]) -> Result<Tensor> {
        let x = self.conv.forward(features[0])?;
        self.classifier.forward(&x)
    }
}

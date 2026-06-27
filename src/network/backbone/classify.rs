use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::super::blocks::{C2psa, C3k2, C3k2Config, ConvBlock};
use crate::Scale;

pub(crate) struct Classify {
    l0: ConvBlock,
    l1: ConvBlock,
    l2: C3k2,
    l3: ConvBlock,
    l4: C3k2,
    l5: ConvBlock,
    l6: C3k2,
    l7: ConvBlock,
    l8: C3k2,
    l9: C2psa,
}

impl Classify {
    pub(crate) fn load(vb: VarBuilder, scale: Scale) -> Result<Self> {
        let c0 = scale.channel(64);
        let c1 = scale.channel(128);
        let c2 = scale.channel(256);
        let c4 = scale.channel(512);
        let c7 = scale.channel(1024);
        let n = scale.repeat(2);
        let c3k_24 = scale.c3k_all();

        Ok(Self {
            l0: ConvBlock::load(vb.pp("0"), 3, c0, 3, 2, 1, true)?,
            l1: ConvBlock::load(vb.pp("1"), c0, c1, 3, 2, 1, true)?,
            l2: C3k2::load(
                vb.pp("2"),
                c1,
                c2,
                C3k2Config::new(n, c3k_24, 0.25, true, false),
            )?,
            l3: ConvBlock::load(vb.pp("3"), c2, c2, 3, 2, 1, true)?,
            l4: C3k2::load(
                vb.pp("4"),
                c2,
                c4,
                C3k2Config::new(n, c3k_24, 0.25, true, false),
            )?,
            l5: ConvBlock::load(vb.pp("5"), c4, c4, 3, 2, 1, true)?,
            l6: C3k2::load(
                vb.pp("6"),
                c4,
                c4,
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
            l7: ConvBlock::load(vb.pp("7"), c4, c7, 3, 2, 1, true)?,
            l8: C3k2::load(
                vb.pp("8"),
                c7,
                c7,
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
            l9: C2psa::load(vb.pp("9"), c7, c7, n)?,
        })
    }

    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.l0.forward(x)?;
        let x = self.l1.forward(&x)?;
        let x = self.l2.forward(&x)?;
        let x = self.l3.forward(&x)?;
        let x = self.l4.forward(&x)?;
        let x = self.l5.forward(&x)?;
        let x = self.l6.forward(&x)?;
        let x = self.l7.forward(&x)?;
        let x = self.l8.forward(&x)?;
        self.l9.forward(&x)
    }
}

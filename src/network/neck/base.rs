use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::super::backbone;
use super::super::blocks::{C3k2, C3k2Config, ConvBlock};
use crate::Scale;

pub(crate) struct Output {
    pub(crate) small: Tensor,
    pub(crate) medium: Tensor,
    pub(crate) large: Tensor,
}

pub(crate) struct Base {
    l13: C3k2,
    l16: C3k2,
    l17: ConvBlock,
    l19: C3k2,
    l20: ConvBlock,
    l22: C3k2,
}

impl Base {
    pub(crate) fn load(vb: VarBuilder, scale: Scale) -> Result<Self> {
        let c256 = scale.channel(256);
        let c512 = scale.channel(512);
        let c1024 = scale.channel(1024);
        let n = scale.repeat(2);

        let p3_ch = c512;
        let p4_ch = c512;
        let p5_ch = c1024;
        let l12_in = p5_ch + p4_ch;
        let l15_in = c512 + p3_ch;
        let l18_in = c256 + c512;
        let l21_in = c512 + p5_ch;

        Ok(Self {
            l13: C3k2::load(
                vb.pp("13"),
                l12_in,
                c512,
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
            l16: C3k2::load(
                vb.pp("16"),
                l15_in,
                c256,
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
            l17: ConvBlock::load(vb.pp("17"), c256, c256, 3, 2, 1, true)?,
            l19: C3k2::load(
                vb.pp("19"),
                l18_in,
                c512,
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
            l20: ConvBlock::load(vb.pp("20"), c512, c512, 3, 2, 1, true)?,
            l22: C3k2::load(
                vb.pp("22"),
                l21_in,
                c1024,
                C3k2Config::new(1, true, 0.5, true, true),
            )?,
        })
    }

    pub(crate) fn forward(&self, bb: &backbone::base::Output) -> Result<Output> {
        let (_, _, h4, w4) = bb.p4.dims4()?;
        let l11 = bb.p5.upsample_nearest2d(h4, w4)?;
        let l12 = Tensor::cat(&[&l11, &bb.p4], 1)?;
        let l13 = self.l13.forward(&l12)?;

        let (_, _, h3, w3) = bb.p3.dims4()?;
        let l14 = l13.upsample_nearest2d(h3, w3)?;
        let l15 = Tensor::cat(&[&l14, &bb.p3], 1)?;
        let l16 = self.l16.forward(&l15)?;

        let l17 = self.l17.forward(&l16)?;
        let l18 = Tensor::cat(&[&l17, &l13], 1)?;
        let l19 = self.l19.forward(&l18)?;

        let l20 = self.l20.forward(&l19)?;
        let l21 = Tensor::cat(&[&l20, &bb.p5], 1)?;
        let l22 = self.l22.forward(&l21)?;

        Ok(Output {
            small: l16,
            medium: l19,
            large: l22,
        })
    }
}

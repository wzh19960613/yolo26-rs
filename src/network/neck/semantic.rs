use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::super::backbone::base::Output as BackboneOutput;
use super::super::blocks::{C3k2, C3k2Config};
use crate::Scale;

pub(crate) struct Output {
    pub(crate) small: Tensor,
    pub(crate) medium: Tensor,
}

pub(crate) struct Semantic {
    l13: C3k2,
    l16: C3k2,
}

impl Semantic {
    pub(crate) fn load(vb: VarBuilder, scale: Scale) -> Result<Self> {
        let c512 = scale.channel(512);
        let c1024 = scale.channel(1024);
        let n = scale.repeat(2);

        let p3_ch = c512;
        let p4_ch = c512;
        let p5_ch = c1024;
        let l12_in = p5_ch + p4_ch;
        let l15_in = c512 + p3_ch;

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
                scale.channel(256),
                C3k2Config::new(n, true, 0.5, true, false),
            )?,
        })
    }

    pub(crate) fn forward(&self, bb: &BackboneOutput) -> Result<Output> {
        let (_, _, h4, w4) = bb.p4.dims4()?;
        let l11 = bb.p5.upsample_nearest2d(h4, w4)?;
        let l12 = Tensor::cat(&[&l11, &bb.p4], 1)?;
        let l13 = self.l13.forward(&l12)?;

        let (_, _, h3, w3) = bb.p3.dims4()?;
        let l14 = l13.upsample_nearest2d(h3, w3)?;
        let l15 = Tensor::cat(&[&l14, &bb.p3], 1)?;
        let l16 = self.l16.forward(&l15)?;

        Ok(Output {
            small: l16,
            medium: l13,
        })
    }
}

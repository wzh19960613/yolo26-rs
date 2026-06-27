use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::bottleneck::Bottleneck;
use super::psa_block::PsaBlock;

pub struct AttnBranch {
    bottleneck: Bottleneck,
    psa: PsaBlock,
}

impl AttnBranch {
    pub fn load(vb: VarBuilder, channels: usize, shortcut: bool) -> Result<Self> {
        Ok(Self {
            bottleneck: Bottleneck::load(vb.pp("0"), channels, channels, shortcut, (3, 3), 0.5)?,
            psa: PsaBlock::load(vb.pp("1"), channels, (channels / 64).max(1))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.psa.forward(&self.bottleneck.forward(x)?)
    }
}

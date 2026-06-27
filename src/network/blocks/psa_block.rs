use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::attention::Attention;
use super::conv_block::ConvBlock;

pub struct PsaBlock {
    attn: Attention,
    ffn0: ConvBlock,
    ffn1: ConvBlock,
}

impl PsaBlock {
    pub fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(vb.pp("attn"), dim, num_heads)?,
            ffn0: ConvBlock::load(vb.pp("ffn").pp("0"), dim, dim * 2, 1, 1, 1, true)?,
            ffn1: ConvBlock::load(vb.pp("ffn").pp("1"), dim * 2, dim, 1, 1, 1, false)?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = (x + self.attn.forward(x)?)?;
        let y = self.ffn1.forward(&self.ffn0.forward(&x)?)?;
        &x + y
    }
}

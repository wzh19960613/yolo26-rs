use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::conv_block::ConvBlock;
use super::psa_block::PsaBlock;

pub struct C2psa {
    cv1: ConvBlock,
    cv2: ConvBlock,
    blocks: Vec<PsaBlock>,
    split: usize,
}

impl C2psa {
    pub fn load(vb: VarBuilder, c_in: usize, c_out: usize, repeats: usize) -> Result<Self> {
        let split = c_in / 2;
        let heads = (split / 64).max(1);
        let mut blocks = Vec::with_capacity(repeats);
        for i in 0..repeats {
            blocks.push(PsaBlock::load(vb.pp("m").pp(i.to_string()), split, heads)?);
        }
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c_in, split * 2, 1, 1, 1, true)?,
            cv2: ConvBlock::load(vb.pp("cv2"), split * 2, c_out, 1, 1, 1, true)?,
            blocks,
            split,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let a = y.narrow(1, 0, self.split)?;
        let mut b = y.narrow(1, self.split, self.split)?;
        for block in &self.blocks {
            b = block.forward(&b)?;
        }
        self.cv2.forward(&Tensor::cat(&[&a, &b], 1)?)
    }
}

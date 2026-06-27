use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::bottleneck::Bottleneck;
use super::conv_block::ConvBlock;

pub struct C3k {
    cv1: ConvBlock,
    cv2: ConvBlock,
    cv3: ConvBlock,
    blocks: Vec<Bottleneck>,
}

impl C3k {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        repeats: usize,
        shortcut: bool,
    ) -> Result<Self> {
        let hidden = c_out / 2;
        let mut blocks = Vec::with_capacity(repeats);
        for i in 0..repeats {
            blocks.push(Bottleneck::load(
                vb.pp("m").pp(i.to_string()),
                hidden,
                hidden,
                shortcut,
                (3, 3),
                1.0,
            )?);
        }
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c_in, hidden, 1, 1, 1, true)?,
            cv2: ConvBlock::load(vb.pp("cv2"), c_in, hidden, 1, 1, 1, true)?,
            cv3: ConvBlock::load(vb.pp("cv3"), hidden * 2, c_out, 1, 1, 1, true)?,
            blocks,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut a = self.cv1.forward(x)?;
        for block in &self.blocks {
            a = block.forward(&a)?;
        }
        let b = self.cv2.forward(x)?;
        self.cv3.forward(&Tensor::cat(&[&a, &b], 1)?)
    }
}

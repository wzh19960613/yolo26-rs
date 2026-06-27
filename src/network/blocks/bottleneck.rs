use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::conv_block::ConvBlock;

pub struct Bottleneck {
    cv1: ConvBlock,
    cv2: ConvBlock,
    shortcut: bool,
}

impl Bottleneck {
    pub fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        shortcut: bool,
        kernels: (usize, usize),
        expansion: f32,
    ) -> Result<Self> {
        let hidden = (c_out as f32 * expansion) as usize;
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c_in, hidden, kernels.0, 1, 1, true)?,
            cv2: ConvBlock::load(vb.pp("cv2"), hidden, c_out, kernels.1, 1, 1, true)?,
            shortcut: shortcut && c_in == c_out,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv2.forward(&self.cv1.forward(x)?)?;
        if self.shortcut { x + y } else { Ok(y) }
    }
}

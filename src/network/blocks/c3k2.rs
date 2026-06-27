use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::{attn_branch::AttnBranch, bottleneck::Bottleneck, c3k::C3k, conv_block::ConvBlock};

enum Branch {
    Bottleneck(Bottleneck),
    C3k(C3k),
    Attention(Box<AttnBranch>),
}

impl Branch {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Bottleneck(block) => block.forward(x),
            Self::C3k(block) => block.forward(x),
            Self::Attention(block) => block.forward(x),
        }
    }
}

pub struct C3k2 {
    cv1: ConvBlock,
    cv2: ConvBlock,
    branches: Vec<Branch>,
}

#[derive(Clone, Copy)]
pub(crate) struct C3k2Config {
    repeats: usize,
    c3k: bool,
    expansion: f32,
    shortcut: bool,
    attention: bool,
}

impl C3k2Config {
    pub(crate) const fn new(
        repeats: usize,
        c3k: bool,
        expansion: f32,
        shortcut: bool,
        attention: bool,
    ) -> Self {
        Self {
            repeats,
            c3k,
            expansion,
            shortcut,
            attention,
        }
    }
}

impl C3k2 {
    pub(crate) fn load(
        vb: VarBuilder,
        c_in: usize,
        c_out: usize,
        config: C3k2Config,
    ) -> Result<Self> {
        let hidden = (c_out as f32 * config.expansion) as usize;
        let mut branches = Vec::with_capacity(config.repeats);
        for i in 0..config.repeats {
            let branch_vb = vb.pp("m").pp(i.to_string());
            let branch = if config.attention {
                Branch::Attention(Box::new(AttnBranch::load(
                    branch_vb,
                    hidden,
                    config.shortcut,
                )?))
            } else if config.c3k {
                Branch::C3k(C3k::load(branch_vb, hidden, hidden, 2, config.shortcut)?)
            } else {
                Branch::Bottleneck(Bottleneck::load(
                    branch_vb,
                    hidden,
                    hidden,
                    config.shortcut,
                    (3, 3),
                    0.5,
                )?)
            };
            branches.push(branch);
        }
        Ok(Self {
            cv1: ConvBlock::load(vb.pp("cv1"), c_in, hidden * 2, 1, 1, 1, true)?,
            cv2: ConvBlock::load(
                vb.pp("cv2"),
                hidden * (2 + config.repeats),
                c_out,
                1,
                1,
                1,
                true,
            )?,
            branches,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.cv1.forward(x)?;
        let mut chunks = y.chunk(2, 1)?;
        // `chunk(2, 1)` seeds `chunks` with two elements, so the last index is
        // always valid without an `Option` unwrap.
        for branch in &self.branches {
            let last_index = chunks.len() - 1;
            let next = branch.forward(&chunks[last_index])?;
            chunks.push(next);
        }
        let refs: Vec<&Tensor> = chunks.iter().collect();
        self.cv2.forward(&Tensor::cat(&refs, 1)?)
    }
}

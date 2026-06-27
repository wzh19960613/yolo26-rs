use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use super::conv_block::ConvBlock;

pub struct Attention {
    qkv: ConvBlock,
    proj: ConvBlock,
    pe: ConvBlock,
    num_heads: usize,
    key_dim: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    pub fn load(vb: VarBuilder, dim: usize, num_heads: usize) -> Result<Self> {
        let head_dim = dim / num_heads;
        let key_dim = head_dim / 2;
        let qkv_dim = dim + 2 * num_heads * key_dim;
        Ok(Self {
            qkv: ConvBlock::load(vb.pp("qkv"), dim, qkv_dim, 1, 1, 1, false)?,
            proj: ConvBlock::load(vb.pp("proj"), dim, dim, 1, 1, 1, false)?,
            pe: ConvBlock::load(vb.pp("pe"), dim, dim, 3, 1, dim, false)?,
            num_heads,
            key_dim,
            head_dim,
            scale: (key_dim as f64).powf(-0.5),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, c, h, w) = x.dims4()?;
        let n = h * w;
        let qkv = self.qkv.forward(x)?.reshape((
            b,
            self.num_heads,
            self.key_dim * 2 + self.head_dim,
            n,
        ))?;

        let q = qkv.narrow(2, 0, self.key_dim)?;
        let k = qkv.narrow(2, self.key_dim, self.key_dim)?;
        let v = qkv.narrow(2, self.key_dim * 2, self.head_dim)?;
        let attn = (q.transpose(2, 3)?.matmul(&k)? * self.scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = v.matmul(&attn.transpose(2, 3)?)?.reshape((b, c, h, w))?;
        let pe = self.pe.forward(&v.reshape((b, c, h, w))?)?;
        self.proj.forward(&(out + pe)?)
    }
}
